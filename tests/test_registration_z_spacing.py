"""Registration Z is a spacing in µm, resolved per acquisition, and every
number the user is shown follows it.

2026-10-18: on a 2.5 µm full-Z Ket_Trial1 run "Reg. binning Z 1x" meant 10 µm
(each step was REFERENCE_Z_STEP_UM), so 1x averaged 4 planes and there was no
way to register on every plane. The setting is now a spacing with "every plane"
(0), and one function -- resolve_registration_z -- feeds the run, the axial
bound, the ETA (cold-start model, cache key, progress bands) and the memory
estimate, so none of them can disagree with what the run does.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flamingo_stitcher import pipeline as P
from flamingo_stitcher.pipeline import (
    RawTileInfo,
    StitchingConfig,
    StitchingPipeline,
    build_timing_key,
    estimate_memory_usage,
    estimate_registration_peak_bytes,
    progress_bands,
    registration_z_choices,
    remap_progress,
    resolve_registration_z,
)


def _ket_tiles(planes=2999, z_step_um=2.5):
    """Ket_Trial1's geometry, metadata only: 6x7, 2048^2, 10% overlap."""
    tiles = []
    for iy in range(7):
        for ix in range(6):
            tiles.append(
                RawTileInfo(
                    folder=Path(f"/x/t{ix}{iy}"),
                    x_mm=1.74 + 1.931 * ix,
                    y_mm=13.41 + 1.931 * iy,
                    z_min_mm=14.965,
                    z_max_mm=14.965 + z_step_um / 1000.0 * (planes - 1),
                    n_planes=planes,
                    raw_files={3: {0: Path("a"), 1: Path("b")}},
                    channels=[3],
                    illumination_sides=[0, 1],
                    frame_width=2048,
                    frame_height=2048,
                )
            )
    return tiles


def _cfg(**kw):
    c = StitchingConfig(
        downsample_xy=2, downsample_z=1, pixel_size_um=1.0475,
        registration_binning={"z": 2, "y": 1, "x": 1},
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# --------------------------------------------------------------------------
# The resolver
# --------------------------------------------------------------------------


class TestResolve:
    def test_every_plane_is_possible_at_any_step(self):
        for voxel in (2.5, 10.0, 0.7):
            rz = resolve_registration_z(_cfg(registration_z_spacing_um=0.0), voxel)
            assert rz.bin_z == 1 and rz.spacing_um == pytest.approx(voxel)

    def test_a_spacing_is_physical(self):
        assert resolve_registration_z(_cfg(registration_z_spacing_um=10.0), 2.5).bin_z == 4
        assert resolve_registration_z(_cfg(registration_z_spacing_um=10.0), 10.0).bin_z == 1

    def test_a_spacing_between_multiples_rounds_to_whole_planes(self):
        rz = resolve_registration_z(_cfg(registration_z_spacing_um=20.0), 3.0)
        assert rz.bin_z == 7 and rz.spacing_um == pytest.approx(21.0)

    def test_older_configs_mean_what_they_meant(self):
        # No spacing set: "z: 1" was ~10 µm, which is 4 planes at 2.5 µm --
        # exactly what the 2026-10-18 log showed.
        rz = resolve_registration_z(
            _cfg(registration_z_spacing_um=None, registration_binning={"z": 1}), 2.5
        )
        assert rz.bin_z == 4
        assert "read as 10 µm per step" in rz.source

    def test_the_source_says_where_it_came_from(self):
        assert "every plane" in resolve_registration_z(
            _cfg(registration_z_spacing_um=0.0), 2.5
        ).source
        assert "20 µm" in resolve_registration_z(
            _cfg(registration_z_spacing_um=20.0), 2.5
        ).source

    def test_unknown_voxel_does_not_bin(self):
        assert resolve_registration_z(_cfg(registration_z_spacing_um=20.0), 0.0).bin_z == 1

    def test_choices_start_at_every_plane_and_double(self):
        assert registration_z_choices(2.5) == [
            (0.0, 1), (5.0, 2), (10.0, 4), (20.0, 8), (40.0, 16), (80.0, 32)
        ]


# --------------------------------------------------------------------------
# The run uses it; the Z-refinement pass really is every plane
# --------------------------------------------------------------------------


def _effective(cfg, voxel_z, **kw):
    import numpy as np

    pipe = StitchingPipeline(cfg)
    vol = np.zeros((40, 16, 16), dtype=np.uint16)
    ti = SimpleNamespace(n_planes=40)
    return pipe._effective_registration_binning(
        [(vol, ti)], [], {}, voxel_size_um={"z": voxel_z, "y": 2.0, "x": 2.0}, **kw
    )


def test_the_run_bins_what_the_setting_says():
    assert _effective(_cfg(registration_z_spacing_um=0.0), 2.5)["z"] == 1
    assert _effective(_cfg(registration_z_spacing_um=10.0), 2.5)["z"] == 4


def test_the_refine_pass_is_every_plane_at_2_5_um():
    # Read as a spacing, refine's z=1 meant 10 µm: 4 planes at 2.5 µm, so
    # "Refine Z alignment" registered no finer than the main pass it refines.
    out = _effective(
        _cfg(registration_z_spacing_um=20.0), 2.5,
        configured={"z": 1, "y": 4, "x": 4}, z_in_planes=True,
    )
    assert out["z"] == 1


def test_the_log_says_planes_spacing_and_source(caplog):
    import logging

    with caplog.at_level(logging.INFO):
        _effective(_cfg(registration_z_spacing_um=10.0), 2.5)
    assert "Registration Z: 4 planes averaged of 2.5 µm = 10 µm" in caplog.text


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------


class TestTime:
    def test_every_plane_costs_eight_times_twenty_um_at_2_5(self):
        tiles = _ket_tiles()
        every = P._rough_register_seconds(tiles, _cfg(registration_z_spacing_um=0.0))
        twenty = P._rough_register_seconds(tiles, _cfg(registration_z_spacing_um=20.0))
        assert every / twenty == pytest.approx(8.0, rel=1e-6)

    def test_the_same_spacing_costs_the_same_at_any_step(self):
        # Physical: 20 µm compares the same planes whether acquired at 2.5 or 10.
        fine = P._rough_register_seconds(
            _ket_tiles(planes=2999, z_step_um=2.5), _cfg(registration_z_spacing_um=20.0)
        )
        coarse = P._rough_register_seconds(
            _ket_tiles(planes=750, z_step_um=10.0), _cfg(registration_z_spacing_um=20.0)
        )
        assert fine == pytest.approx(coarse, rel=0.01)

    def test_refine_is_costed_as_every_plane_not_a_flat_doubling(self):
        tiles = _ket_tiles()
        base = _cfg(registration_z_spacing_um=20.0)
        on = _cfg(registration_z_spacing_um=20.0, registration_z_refine=True)
        extra = P._rough_register_seconds(tiles, on) - P._rough_register_seconds(tiles, base)
        # every plane at XY bin 4 vs 8 planes at XY bin 1: 8 / 16 = 0.5x of main
        main = P._rough_register_seconds(tiles, base)
        assert extra > 0
        assert extra != pytest.approx(main)  # not the old "x2"

    def test_the_cache_key_separates_spacings_but_keeps_old_entries(self):
        tiles = _ket_tiles()
        k20 = build_timing_key(tiles, _cfg(registration_z_spacing_um=20.0)).serialize()
        k0 = build_timing_key(tiles, _cfg(registration_z_spacing_um=0.0)).serialize()
        k10 = build_timing_key(tiles, _cfg(registration_z_spacing_um=10.0)).serialize()
        assert "rz=" not in k20  # existing cache entries still match
        assert k0.endswith("|rz=2.5") and k10.endswith("|rz=10")
        skip = build_timing_key(
            tiles, _cfg(registration_z_spacing_um=0.0, skip_registration=True)
        ).serialize()
        assert "rz=" not in skip

    def test_progress_bands_follow_the_phase_shares(self):
        a, b = progress_bands({"preprocess": 30, "register": 60, "fuse": 10})
        assert (a, b) == pytest.approx((0.3, 0.9))
        # end points and band edges map where they should
        assert remap_progress(0, (a, b)) == 0.0
        assert remap_progress(P._PCT_REGISTER, (a, b)) == pytest.approx(a)
        assert remap_progress(P._PCT_FUSE, (a, b)) == pytest.approx(b)
        assert remap_progress(100, (a, b)) == pytest.approx(1.0)

    def test_without_shares_the_bands_are_unchanged(self):
        bands = progress_bands({})
        for pct in (0, 20, 45, 60, 70, 95, 100):
            assert remap_progress(pct, bands) == pytest.approx(pct / 100.0)


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


class TestMemory:
    def test_registration_is_in_the_estimate(self):
        tiles = _ket_tiles()
        est = estimate_memory_usage(tiles, [3], _cfg(registration_z_spacing_um=0.0))
        assert est["registration_gb"] > 30  # one seam: 2999 x 1024 x ~102 px
        assert est["streaming_gb"] >= est["registration_gb"]

    def test_it_scales_with_the_planes_compared(self):
        tiles = _ket_tiles()
        b0 = estimate_registration_peak_bytes(tiles, _cfg(registration_z_spacing_um=0.0))
        b20 = estimate_registration_peak_bytes(tiles, _cfg(registration_z_spacing_um=20.0))
        assert b0 > 4 * b20

    def test_skipped_registration_costs_nothing(self):
        tiles = _ket_tiles()
        assert estimate_registration_peak_bytes(
            tiles, _cfg(registration_z_spacing_um=0.0, skip_registration=True)
        ) == 0

    def test_registration_can_be_what_limits_the_peak(self):
        # Heavy XY downsample shrinks preprocess and fusion far more than the
        # strip registration keeps (binning backs off to hold ~32 px).
        tiles = _ket_tiles()
        est = estimate_memory_usage(
            tiles, [3], _cfg(registration_z_spacing_um=0.0, downsample_xy=1,
                             registration_binning={"z": 2, "y": 1, "x": 1})
        )
        assert est["registration_gb"] > 0
        if est["registration_gb"] > est["preprocess_gb"]:
            assert est["limited_by"] == "registration"


def test_the_yaml_key_is_read(monkeypatch):
    import flamingo_stitcher.config_loader as loader

    monkeypatch.setattr(
        loader, "get_stitching_defaults",
        lambda: {"registration": {"z_spacing_um": 0}},
    )
    c = StitchingConfig()
    loader.apply_stitching_yaml_to_config(c)
    assert c.registration_z_spacing_um == 0.0


# --------------------------------------------------------------------------
# Peak refinement (registration_upsample_factor)
# --------------------------------------------------------------------------


class TestPeakRefinement:
    def test_it_costs_a_little_time_and_no_memory(self):
        tiles = _ket_tiles()
        base = _cfg(registration_z_spacing_um=10.0)
        fine = _cfg(registration_z_spacing_um=10.0, registration_upsample_factor=10)
        ratio = P._rough_register_seconds(tiles, fine) / P._rough_register_seconds(tiles, base)
        assert ratio == pytest.approx(P._ROUGH_UPSAMPLE_TIME_FACTOR)
        assert estimate_registration_peak_bytes(tiles, fine) == (
            estimate_registration_peak_bytes(tiles, base)
        )

    def test_the_cache_key_separates_it_only_when_set(self):
        tiles = _ket_tiles()
        assert "up=" not in build_timing_key(tiles, _cfg(registration_z_spacing_um=20.0)).serialize()
        assert build_timing_key(
            tiles, _cfg(registration_z_spacing_um=20.0, registration_upsample_factor=10)
        ).serialize().endswith("|up=10")

    def test_the_log_says_the_z_step_per_seam(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            _effective(_cfg(registration_z_spacing_um=10.0, registration_upsample_factor=10), 2.5)
        assert "steps of 1 µm" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.INFO):
            _effective(_cfg(registration_z_spacing_um=10.0), 2.5)
        assert "steps of 5 µm" in caplog.text and "multiview-stitcher default" in caplog.text


def test_writer_progress_stays_in_the_write_band():
    seen = []
    pipe = StitchingPipeline(StitchingConfig())
    pipe._progress_fn = lambda pct, msg: seen.append((pct, msg))
    pipe._writer_progress(0, "Preparing")
    pipe._writer_progress(50, "Computing channel 1/1 (3130 Z-planes)...")
    pipe._writer_progress(100, "done")
    assert [p for p, _ in seen] == [P._PCT_WRITE, (P._PCT_WRITE + P._PCT_METADATA) // 2, P._PCT_METADATA]
    # classified as writing, not fusion
    assert all(pipe._classify_phase(m) == "write" for _, m in seen)
