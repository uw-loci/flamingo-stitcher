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
    # Every run these tests were fitted to chose seams by SSIM; cell match
    # (v0.13.15, the default since) is priced by its own factor.
    cfg.registration_seam_choice = "ssim"
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
    """The 2026-09-18 7x7: 33697 s measured, 5237 s predicted before this.

    Configures the registration binning the run actually used (the n7 profile's
    {z:1, y:2, x:2}), NOT the dataclass default {z:2, y:4, x:4}. Binning is now
    part of the model — a default run bins harder and genuinely costs less — so
    a test asserting against a measured run has to describe that run. It passed
    with the default before only because the old model ignored binning
    entirely.
    """
    cfg = _config()
    cfg.registration_binning = {"z": 1, "y": 2, "x": 2}
    for t in (tiles := _grid()):
        t.z_step_mm = 0.010  # 10 um, as acquired
    predicted = rough_run_seconds(tiles, cfg)
    assert 33697 / predicted < 2.0, f"{predicted:.0f}s is still far low"
    assert predicted / 33697 < 2.0, f"{predicted:.0f}s now overshoots"


def test_binning_harder_makes_the_estimate_smaller():
    """A default run bins 4x in XY where the n7 profile bins 2x, so it really
    does register on fewer voxels. The model has to say so."""
    tiles = _grid()
    for t in tiles:
        t.z_step_mm = 0.010
    light = _config()
    light.registration_binning = {"z": 1, "y": 2, "x": 2}
    heavy = _config()
    heavy.registration_binning = {"z": 2, "y": 4, "x": 4}
    assert _rough_register_seconds(tiles, heavy) < _rough_register_seconds(tiles, light)


def test_registration_dominates_the_prior_for_a_large_mosaic():
    """It was 71.8% of the measured run and ~1.4% of the old prediction.

    Uses the 09-18 run's own geometry (Z step and registration binning), so
    the share is judged against the run it was measured on.
    """
    tiles, ds_xy, ds_z, _reg = RUN_0918
    cfg = _cfg(ds_xy, ds_z)
    share = _rough_register_seconds(tiles, cfg) / rough_run_seconds(tiles, cfg)
    assert 0.5 < share < 0.95


def test_skipping_registration_leaves_only_load_and_fuse():
    """Without registration the prior is the load and fuse terms alone.

    The 2026-09-18 run spent 33697 - 24187 = 9510 s outside registration. The
    first prior gave 5237 s for that part (0.55x); after the 2026-10-27
    recalibration (load 260 MB/s, fuse 0.325 s per output unit) it gives 12346
    s (1.30x).
    """
    tiles = _grid()
    assert _rough_register_seconds(tiles, _config(skip_registration=True)) == 0.0
    assert rough_run_seconds(tiles, _config(skip_registration=True)) == pytest.approx(
        12346.0, abs=1.0
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


# --------------------------------------------------------------------------
# Recalibration, 2026-10-12: the term must MOVE with downsample
# --------------------------------------------------------------------------


class _T:
    """A tile with a real Z step, which the binning rule needs."""

    def __init__(self, x, y, planes, z_step_um, frame=2048):
        self.x_mm, self.y_mm = x, y
        self.n_planes = planes
        self.frame_width = self.frame_height = frame
        self.channels = [3]
        self.illumination_sides = [0, 1]
        self.z_step_mm = z_step_um / 1000.0


def _acq(cols, rows, planes, pitch_mm, z_step_um):
    return [
        _T(1.0 + c * pitch_mm, 1.0 + r * pitch_mm, planes, z_step_um)
        for r in range(rows)
        for c in range(cols)
    ]


def _cfg(ds_xy=1, ds_z=1, **kw):
    cfg = _config(**kw)
    cfg.downsample_xy, cfg.downsample_z = ds_xy, ds_z
    cfg.registration_binning = {"z": 1, "y": 2, "x": 2}
    return cfg


# The three runs the constant is fitted to. Overlap differs between them and is
# read from the stage positions, not assumed.
RUN_0918 = (_acq(7, 7, 643, 1.8235, 10.0), 1, 1, 24187)
RUN_1007_2X = (_acq(6, 7, 2999, 1.93, 2.5), 2, 1, 5242)
RUN_1007_8X = (_acq(6, 7, 2999, 1.93, 2.5), 8, 1, 1184)


@pytest.mark.parametrize(
    "tiles_, ds_xy, ds_z, measured",
    [RUN_0918, RUN_1007_2X, RUN_1007_8X],
    ids=["2026-09-18 XY=1", "2026-10-07 XY=2", "2026-10-07 XY=8"],
)
def test_the_term_is_within_a_small_factor_of_every_measured_run(
    tiles_, ds_xy, ds_z, measured
):
    """Before this, the same model gave 6.4x, 3.2x and 9.2x."""
    predicted = _rough_register_seconds(tiles_, _cfg(ds_xy, ds_z))
    assert 0.6 < predicted / measured < 1.7, (
        f"predicted {predicted:.0f}s against {measured}s measured"
    )


def test_downsample_actually_changes_the_estimate():
    """The bug: two runs of the same acquisition at XY=2 and XY=8 differed by
    2.9x in reality and by 3% in the prior, because the term had no downsample
    in it at all."""
    tiles_ = _acq(6, 7, 2999, 1.93, 2.5)
    at2 = _rough_register_seconds(tiles_, _cfg(2, 1))
    at8 = _rough_register_seconds(tiles_, _cfg(8, 1))
    assert at2 / at8 > 2.0, f"XY=2 {at2:.0f}s vs XY=8 {at8:.0f}s is barely a change"
    # ...but not the naive ds^2 = 16x either: registration bins LESS hard when
    # the strip is already thin, which partly cancels the downsample.
    assert at2 / at8 < 8.0


def test_registration_always_works_at_about_the_reference_z_step():
    """Z binning is a physical target, so the compared plane count is stack
    depth / 10 um whatever the sampling. A 2.5 um stack and a 10 um stack of the
    same DEPTH must cost the same."""
    fine = _acq(6, 7, 2999, 1.93, 2.5)      # 7497 um deep
    coarse = _acq(6, 7, 750, 1.93, 10.0)    # 7500 um deep
    a = _rough_register_seconds(fine, _cfg(1, 1))
    b = _rough_register_seconds(coarse, _cfg(1, 1))
    assert a == pytest.approx(b, rel=0.01)


def test_a_z_downsample_reduces_it_only_while_z_is_finer_than_the_reference():
    """At a 2.5 um step, Z=4 lands exactly on the 10 um target, so registration
    was ALREADY binning to that and the estimate must not fall further."""
    tiles_ = _acq(6, 7, 2999, 1.93, 2.5)
    assert _rough_register_seconds(tiles_, _cfg(8, 4)) == pytest.approx(
        _rough_register_seconds(tiles_, _cfg(8, 1)), rel=0.01
    )


def test_the_whole_run_prior_is_now_within_a_factor_of_two():
    for tiles_, ds_xy, ds_z, _reg, total in (
        (*RUN_0918[:3], RUN_0918[3], 33697),
        (*RUN_1007_2X[:3], RUN_1007_2X[3], 23573),
        (*RUN_1007_8X[:3], RUN_1007_8X[3], 8071),
    ):
        cfg = _cfg(ds_xy, ds_z)
        cfg.output_format = "imaris" if ds_xy == 1 else "ome-zarr"
        prior = rough_run_seconds(tiles_, cfg)
        assert 0.5 < prior / total < 2.0, (
            f"XY={ds_xy}: prior {prior:.0f}s against {total}s measured"
        )


# --------------------------------------------------------------------------
# Recalibration against the n7 rig logs, 2026-10-27
# --------------------------------------------------------------------------


def test_destripe_is_priced_per_light_path_plane():
    """Run 25 (10-21) destriped 84 stacks of 2999 planes at ~440 s each.

    Before, the prior had no destripe term and that run took 2.3x its estimate.
    """
    from flamingo_stitcher.pipeline import rough_phase_seconds

    tiles = _grid(6, 7, pitch=1.931, planes=2999)
    off = rough_phase_seconds(tiles, _config(downsample_xy=2))
    on = rough_phase_seconds(tiles, _config(downsample_xy=2, destripe=True))
    extra = on["preprocess"] - off["preprocess"]
    assert extra == pytest.approx(84 * 440, rel=0.1)


def test_cell_match_seam_choice_registers_about_six_times_faster():
    tiles = _grid(6, 7, pitch=1.931, planes=2999)
    for t in tiles:
        t.z_step_mm = 0.0025
    cells = _rough_register_seconds(
        tiles, _config(downsample_xy=2, registration_seam_choice="cells")
    )
    ssim = _rough_register_seconds(tiles, _config(downsample_xy=2))
    assert cells / ssim == pytest.approx(0.16)


def test_ket_trial1_fuse_and_load_match_the_rig():
    """Runs 29/30: preprocess 8060/8152 s, fuse + write 14742/15142 s (Apply)."""
    from flamingo_stitcher.pipeline import rough_phase_seconds

    tiles = _grid(6, 7, pitch=1.931, planes=2999)
    p = rough_phase_seconds(
        tiles,
        _config(downsample_xy=2, output_format="ome-tiff", intensity_adjustment="apply"),
    )
    assert p["preprocess"] == pytest.approx(8100, rel=0.05)
    assert p["fuse"] == pytest.approx(14900, rel=0.05)
