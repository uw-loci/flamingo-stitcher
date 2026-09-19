"""The cold-start ETA's registration term, and the cache axes around it.

The 2026-09-18 7x7 predicted 5237 s and took 33697 s (6.4x low). Registration
was 71.8% of that run and contributed ~74 s to the prediction, because the
prior modelled load and fuse and nothing else. These tests pin the term that
fixes it, and the two cache axes that decide whether a *measured* total is
reused or thrown away.
"""

import dataclasses

import pytest

from flamingo_stitcher.pipeline import (
    StitchingConfig,
    _grid_shape,
    _rough_overlap_fraction,
    _rough_register_seconds,
    build_timing_key,
    rough_run_seconds,
)
from flamingo_stitcher.timing_cache import StitchingTimingKey


class _Tile:
    def __init__(self, x, y, planes=643, frame=2048):
        self.x_mm, self.y_mm = x, y
        self.n_planes = planes
        self.frame_width = self.frame_height = frame
        self.channels = [3]
        self.illumination_sides = [0, 1]


def _grid(cols=7, rows=7, pitch=1.8235, **kw):
    return [
        _Tile(1.333 + c * pitch, 14.037 + r * pitch, **kw)
        for r in range(rows)
        for c in range(cols)
    ]


def _config(**kw):
    cfg = StitchingConfig()
    cfg.pixel_size_um = 1.0475
    cfg.downsample_xy = cfg.downsample_z = 1
    cfg.flat_field_correction = True
    cfg.output_format = "imaris"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------
# Layout, read from the stage positions
# --------------------------------------------------------------------------


def test_the_grid_shape_comes_from_the_stage_positions():
    assert _grid_shape(_grid(7, 7)) == (7, 7)
    assert _grid_shape(_grid(5, 3)) == (3, 5)
    assert _grid_shape([_Tile(1.0, 1.0)]) == (1, 1)


def test_nominally_equal_positions_are_one_column():
    """Stage coordinates differ in the last decimals between rows.

    Without rounding, every tile becomes its own column and the seam count is
    wrong in the direction that matters (too few).
    """
    tiles = _grid(3, 2)
    tiles[3].x_mm += 1e-6
    tiles[4].y_mm -= 2e-7
    assert _grid_shape(tiles) == (2, 3)


def test_overlap_is_measured_not_assumed():
    cfg = _config()
    # 2048 px at 1.0475 um = 2.1453 mm frame; a 1.8235 mm step is 15% overlap.
    assert _rough_overlap_fraction(_grid(), cfg) == pytest.approx(0.15, abs=0.005)
    wide = _grid(pitch=1.0727)  # half the frame
    assert _rough_overlap_fraction(wide, cfg) == pytest.approx(0.50, abs=0.01)


def test_an_unreadable_layout_falls_back_to_fifteen_percent():
    cfg = _config()
    assert _rough_overlap_fraction([_Tile(1.0, 1.0)], cfg) == 0.15
    assert _rough_overlap_fraction(_grid(), _config(pixel_size_um=0.0)) == 0.15


# --------------------------------------------------------------------------
# The registration term itself
# --------------------------------------------------------------------------


def test_the_prior_now_lands_within_a_factor_of_two_of_the_measured_run():
    """The 2026-09-18 7x7: 33697 s measured, 5237 s predicted before this."""
    predicted = rough_run_seconds(_grid(), _config())
    assert 33697 / predicted < 2.0, f"{predicted:.0f}s is still far low"
    assert predicted / 33697 < 2.0, f"{predicted:.0f}s now overshoots"


def test_registration_dominates_the_prior_for_a_large_mosaic():
    """It was 71.8% of the measured run and ~1.4% of the old prediction."""
    tiles, cfg = _grid(), _config()
    share = _rough_register_seconds(tiles, cfg) / rough_run_seconds(tiles, cfg)
    assert 0.5 < share < 0.95


def test_skipping_registration_returns_the_old_estimate_exactly():
    """A skip-registration run must be unchanged by this, byte for byte."""
    tiles = _grid()
    assert _rough_register_seconds(tiles, _config(skip_registration=True)) == 0.0
    assert rough_run_seconds(tiles, _config(skip_registration=True)) == pytest.approx(
        5237.0, abs=1.0
    )


def test_the_term_scales_with_seams_not_tiles():
    """A 1xN strip has N-1 seams; an NxN grid has 2N(N-1). Cost follows seams.

    Pairwise registrations run serially, so the seam count is the multiplier.
    """
    cfg = _config()
    strip = _rough_register_seconds(_grid(cols=7, rows=1), cfg)
    grid = _rough_register_seconds(_grid(cols=7, rows=7), cfg)
    assert grid / strip == pytest.approx(84 / 6, rel=0.01)


def test_the_term_scales_with_overlap_and_depth():
    cfg = _config()
    base = _rough_register_seconds(_grid(), cfg)
    assert _rough_register_seconds(_grid(planes=1286), cfg) == pytest.approx(
        base * 2, rel=0.01
    )
    # Twice the overlap is twice the voxels compared per seam.
    # 2048 px x 1.0475 um = 2.14528 mm frame; a 1.5017 mm step is 30% overlap,
    # twice the 15% the base grid has.
    assert _rough_register_seconds(_grid(pitch=1.5017), cfg) / base == pytest.approx(
        2.0, rel=0.05
    )


def test_z_refinement_doubles_it():
    cfg, tiles = _config(), _grid()
    plain = _rough_register_seconds(tiles, cfg)
    refined = _rough_register_seconds(tiles, _config(registration_z_refine=True))
    assert refined == pytest.approx(plain * 2, rel=0.01)


def test_a_single_tile_needs_no_registration():
    assert _rough_register_seconds([_Tile(1.0, 1.0)], _config()) == 0.0


# --------------------------------------------------------------------------
# Cache key: illumination fusion
# --------------------------------------------------------------------------


def _key(**kw):
    base = dict(
        n_tiles=49, n_channels=1, n_pyramid_levels=-1, n_timepoints=1,
        output_format="imaris", fusion_method="cosine", skip_registration=False,
        planes_per_tile=643, flat_field=True, source_drive="D:", dest_drive="G:",
    )
    base.update(kw)
    return StitchingTimingKey(**base)


def test_illumination_fusion_modes_do_not_share_a_cache_entry():
    """content costs ~40x split per tile; averaging them fits neither."""
    keys = {
        m: _key(illumination_fusion=m).serialize()
        for m in ("max", "split", "blend", "content")
    }
    assert len(set(keys.values())) == 4


def test_a_max_run_keeps_the_key_it_always_had():
    """Existing users' learned timings must survive this axis being added."""
    serialized = _key(illumination_fusion="max").serialize()
    assert "|if=" not in serialized
    assert serialized.endswith("src=D:|dst=G:")


def test_the_config_reaches_the_key():
    cfg = _config(illumination_fusion="split")
    key = build_timing_key(_grid(), cfg, acquisition_dir="D:/a", output_dir="G:/b")
    assert key.illumination_fusion == "split"
    assert "|if=split" in key.serialize()
