"""Unit tests for the pure tile-border artifact detector.

Build two overlapping tiles as crops of a shared base volume (so their overlap
is physically identical), optionally inject a known intensity step, and assert
the detector flags exactly what it should.
"""

from __future__ import annotations

import numpy as np
import pytest

from flamingo_stitcher.border_qc import (
    BorderQCParams,
    detect_border_steps,
    find_neighbor_pairs,
)

Z, Y, TX = 12, 40, 32
PITCH = 20  # tile X pitch in px -> overlap W = TX - PITCH = 12
W = TX - PITCH


def _base(seed=0, ramp=True, texture=8.0):
    """A wide (Z, Y, X+PITCH) base volume with a smooth X ramp + mild texture."""
    rng = np.random.default_rng(seed)
    wide = TX + PITCH
    x = np.linspace(1000, 1400, wide) if ramp else np.full(wide, 1000.0)
    vol = np.broadcast_to(x, (Z, Y, wide)).astype(np.float32).copy()
    vol += rng.normal(0, texture, size=vol.shape).astype(np.float32)
    return vol


def _pair(base):
    """Left tile A and right tile B, cropped so they overlap by W px."""
    a = base[:, :, 0:TX].copy()
    b = base[:, :, PITCH : PITCH + TX].copy()
    return a, b


def test_clean_pair_not_flagged():
    a, b = _pair(_base())
    res = detect_border_steps(a, b, "x", W, params=BorderQCParams(mode="full"))
    assert not res.flagged


def test_injected_step_flagged_full():
    a, b = _pair(_base())
    z0, z1, y0, y1, C = 3, 9, 10, 30, 2000.0
    b[z0:z1, y0:y1, :] += C  # bright step over a known Z/Y rectangle
    res = detect_border_steps(a, b, "x", W, params=BorderQCParams(mode="full"))
    assert res.flagged
    # area ~ (z1-z0)*(y1-y0); allow slack for bg mask / component filter.
    expected = (z1 - z0) * (y1 - y0)
    assert 0.6 * expected <= res.area_px <= expected
    assert res.z_index_range is not None
    z_lo, z_hi = res.z_index_range
    assert z_lo <= z0 + 1 and z_hi >= z1 - 2
    assert abs(res.median_step_counts - C) < 300


def test_smooth_gradient_not_flagged():
    # A continuous ramp across the seam: both tiles sample it identically.
    a, b = _pair(_base(ramp=True, texture=0.0))
    res = detect_border_steps(a, b, "x", W, params=BorderQCParams(mode="full"))
    assert not res.flagged


def test_bio_edge_crossing_seam_not_flagged():
    base = _base(ramp=False, texture=2.0)
    # A sharp bright bar inside the overlap region, seen by BOTH tiles.
    base[:, :, PITCH + 2 : PITCH + 5] += 3000.0
    a, b = _pair(base)
    res = detect_border_steps(a, b, "x", W, params=BorderQCParams(mode="full"))
    assert not res.flagged


def test_noise_only_not_flagged():
    rng = np.random.default_rng(1)
    a = rng.normal(1000, 60, size=(Z, Y, TX)).astype(np.float32)
    b = rng.normal(1000, 60, size=(Z, Y, TX)).astype(np.float32)
    res = detect_border_steps(a, b, "x", W, params=BorderQCParams(mode="full"))
    assert not res.flagged


def test_abutting_step_flagged():
    base = _base(ramp=False, texture=4.0)
    a = base[:, :, 0:TX].copy()
    b = base[:, :, TX : 2 * TX].copy()  # no overlap (abutting)
    b += 1500.0  # whole right tile brighter -> seam step at the boundary
    res = detect_border_steps(
        a, b, "x", 0, params=BorderQCParams(mode="full", refine_shift=False)
    )
    assert res.flagged
    assert res.note == "abutting"


def test_background_excluded():
    # Signal only in the left half of Y; inject a step where A is background.
    base = _base(ramp=False, texture=2.0)
    base[:, :20, :] = 0.0  # background band
    a, b = _pair(base)
    b[:, :20, :] += 3000.0  # step lives in the background region only
    res = detect_border_steps(a, b, "x", W, params=BorderQCParams(mode="full"))
    assert not res.flagged


def test_mip_mode_reports_length():
    a, b = _pair(_base())
    y0, y1, C = 8, 28, 2500.0
    b[4:7, y0:y1, :] += C  # step over a few Z planes; MIP should surface it
    res = detect_border_steps(a, b, "x", W, params=BorderQCParams(mode="mip"))
    assert res.flagged
    assert res.border_length_px is not None
    assert 0.6 * (y1 - y0) <= res.border_length_px <= (y1 - y0)


def test_shift_refinement_aligns():
    # Structured overlap; roll B in Y by 2 px. With refinement the tiles
    # re-align and a clean pair stays clean.
    base = _base(ramp=False, texture=0.0)
    base += np.sin(np.linspace(0, 6, Y))[None, :, None].astype(np.float32) * 200
    a, b = _pair(base)
    b = np.roll(b, shift=2, axis=1)
    res = detect_border_steps(
        a, b, "x", W, params=BorderQCParams(mode="full", refine_shift=True)
    )
    assert abs(res.used_shift[1]) >= 1
    assert not res.flagged


def test_y_seam_orientation():
    # Same as the injected-step test but along Y (B below A).
    rng = np.random.default_rng(2)
    wide = Y + PITCH
    base = np.broadcast_to(
        np.linspace(1000, 1200, wide)[:, None], (wide, TX)
    ).astype(np.float32)
    base = np.broadcast_to(base, (Z, wide, TX)).copy()
    base += rng.normal(0, 6, base.shape).astype(np.float32)
    a = base[:, 0:Y, :].copy()
    b = base[:, PITCH : PITCH + Y, :].copy()
    b[3:9, :, 8:24] += 2000.0
    res = detect_border_steps(a, b, "y", W, params=BorderQCParams(mode="full"))
    assert res.flagged


class _T:
    """Minimal duck-typed tile for find_neighbor_pairs."""

    def __init__(self, x, y, z=10.0):
        self.x_mm = x
        self.y_mm = y
        self.z_min_mm = z


def test_find_neighbor_pairs_grid():
    # 3x2 grid, pitch 1.0 in both axes.
    tiles = [_T(x, y) for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0)]
    pairs = find_neighbor_pairs(tiles)
    axes = [ax for _, _, ax in pairs]
    # 2 rows x 2 x-neighbors = 4 X-seams; 3 cols x 1 y-neighbor = 3 Y-seams.
    assert axes.count("x") == 4
    assert axes.count("y") == 3


def test_end_to_end_flags_injected_seam(tmp_path):
    """Real discover→preprocess, then run the orchestrator on the preprocessed
    tiles: exactly the tile pair with an injected border step is flagged."""
    from _synth_acq import write_synth_acquisition

    from flamingo_stitcher import border_qc
    from flamingo_stitcher.pipeline import (
        StitchingConfig,
        StitchingPipeline,
        discover_tiles,
    )

    acq = write_synth_acquisition(
        tmp_path / "acq",
        grid=(2, 2),
        overlap=0.15,
        n_planes=16,
        channels=(1,),
        illum_sides=(0,),
        frame_size=(64, 64),
        pixel_size_um=0.406,
        inject_border_step={
            "tile": (0, 0),
            "edge": "right",
            "magnitude": 8000.0,
            "z_slice": (4, 12),
            "along_slice": (15, 35),  # interior Y, clear of the vertical seam
        },
    )
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.downsample_xy = 1
    cfg.downsample_z = 1
    cfg.camera_x_inverted = False
    cfg.flat_field_correction = False
    cfg.skip_registration = True

    tiles = discover_tiles(acq)
    pipe = StitchingPipeline(cfg)
    ctd = pipe._load_and_preprocess(tiles, [1])
    present = [t for _v, t in ctd[1]]

    report = border_qc.run_border_qc(
        ctd,
        present,
        pixel_size_um=0.406,
        ds_xy=1,
        ds_z=1,
        z_step_um=5.0,
        reg_channel=1,
        params=border_qc.BorderQCParams(mode="full"),
    )

    assert report.n_pairs_flagged == 1
    pr = report.pairs[0]
    assert pr.axis == "x"
    names = {pr.tile_a_name, pr.tile_b_name}
    assert "X0.00_Y0.00" in names  # the injected tile
    # flagged Z overlaps the injected [4, 12) range
    z0, z1 = pr.result.z_index_range
    assert z0 <= 6 and z1 >= 9


def test_streaming_qc_reuses_spill_when_registration_skipped(tmp_path):
    """With registration skipped + border QC on, the reference channel must be
    materialized to disk ONCE: QC's own spill is handed to the fusion loop
    instead of being deleted and re-preprocessed. Regression guard for the
    doubled-preprocess bug (QC → qc_chNN, fusion → chNN)."""
    pytest.importorskip("multiview_stitcher")
    pytest.importorskip("tifffile")
    from _synth_acq import write_synth_acquisition

    from flamingo_stitcher.pipeline import (
        StitchingConfig,
        StitchingPipeline,
        discover_tiles,
    )

    acq = write_synth_acquisition(
        tmp_path / "acq",
        grid=(2, 2),
        overlap=0.2,
        n_planes=8,
        channels=(3,),
        illum_sides=(0,),
        frame_size=(48, 48),
    )
    assert len(discover_tiles(acq)) == 4

    cfg = StitchingConfig.with_yaml_defaults()
    cfg.skip_registration = True
    cfg.streaming_mode = True
    cfg.output_format = "ome-tiff"
    cfg.resource_guard_enabled = False
    cfg.flat_field_correction = False
    cfg.reg_channel = 3
    cfg.border_qc_enabled = True
    cfg.border_qc_mode = "mip"
    cfg.output_chunksize = {"z": 4, "y": 16, "x": 16}

    pipe = StitchingPipeline(cfg)
    calls = {"n": 0}
    orig = pipe._materialize_tiles_to_disk

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    pipe._materialize_tiles_to_disk = counting
    pipe.run(acq, tmp_path / "out")

    # One materialize pass for the single (reference) channel — not two.
    assert calls["n"] == 1, f"expected 1 materialize pass, got {calls['n']}"


def test_find_neighbor_pairs_skips_gap():
    # Missing middle tile in a row -> the gap yields no X-pair across it.
    tiles = [_T(0.0, 0.0), _T(2.0, 0.0)]  # gap of 2*pitch
    # With only these two and pitch=2.0, they are "adjacent" at that pitch;
    # add a third to establish pitch=1.0 so the 2.0 gap is skipped.
    tiles = [_T(0.0, 0.0), _T(1.0, 0.0), _T(3.0, 0.0)]
    pairs = find_neighbor_pairs(tiles)
    xpairs = [(i, j) for i, j, ax in pairs if ax == "x"]
    assert (0, 1) in xpairs
    assert (1, 2) not in xpairs  # 2.0 gap > 1.5*pitch
