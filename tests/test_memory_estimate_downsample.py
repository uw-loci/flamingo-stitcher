"""Memory estimate must keep responding as downsample is turned up.

Two symptoms were reported from the rig: in-memory and streaming showed the
*same* number, and going from 4x to 8x XY downsample changed nothing. Both came
from the preprocess term -- modelled at a flat 4 native-resolution workers
regardless of available RAM -- clamping both peaks via ``max()``. The estimate
now mirrors the runtime worker picker (which caps by available RAM) and reports
which term the peak is pinned to.
"""

from pathlib import Path

import pytest

from flamingo_stitcher.pipeline import (
    RawTileInfo,
    StitchingConfig,
    compute_iso_downsample,
    estimate_memory_usage,
)


def _tiles(n_x=5, n_y=4, n_planes=800, frame=2048):
    return [
        RawTileInfo(
            folder=Path("."),
            x_mm=0.83 + ix * 0.42,
            y_mm=4.82 + iy * 0.42,
            z_min_mm=12.0,
            z_max_mm=13.6,
            n_planes=n_planes,
            frame_width=frame,
            frame_height=frame,
        )
        for iy in range(n_y)
        for ix in range(n_x)
    ]


def _config(**kw):
    base = dict(
        pixel_size_um=0.2551,
        z_step_um=2.0,
        downsample_xy=1,
        downsample_z=1,
        skip_registration=True,
    )
    base.update(kw)
    return StitchingConfig(**base)


def test_in_memory_estimate_keeps_falling_past_4x(monkeypatch):
    """4x -> 8x must not be a no-op: the output/tile terms still shrink."""
    tiles = _tiles()
    est = {
        ds: estimate_memory_usage(tiles, [3], _config(downsample_xy=ds))
        for ds in (1, 2, 4, 8)
    }
    peaks = [est[ds]["in_memory_gb"] for ds in (1, 2, 4, 8)]
    assert peaks == sorted(peaks, reverse=True), peaks
    assert peaks[3] < peaks[2], "4x -> 8x left the estimate unchanged"

    # Output size keeps quartering with XY regardless of any floor.
    for ds in (2, 4, 8):
        assert est[ds]["output_gb"] < est[ds // 2]["output_gb"]


def test_preprocess_term_tracks_available_ram(monkeypatch):
    """The estimate models the worker count the RUN will use, not a flat 4."""
    import flamingo_stitcher.pipeline as mod

    tiles = _tiles(n_planes=100)
    config = _config()

    def _cap(_per_worker_bytes, n):
        monkeypatch.setattr(mod, "_avail_worker_cap", lambda _b: n)
        return estimate_memory_usage(tiles, [3], config)

    one = _cap(0, 1)
    four = _cap(0, 4)
    assert one["preprocess_workers"] == 1
    assert four["preprocess_workers"] == 4
    # Reported values are rounded to 0.1 GB, so allow for that rounding.
    assert four["preprocess_gb"] == pytest.approx(one["preprocess_gb"] * 4, rel=0.05)


def test_explicit_worker_override_ignores_the_ram_cap(monkeypatch):
    """A pinned preprocess_workers is honoured (clamped to 8), as at run time."""
    import flamingo_stitcher.pipeline as mod

    monkeypatch.setattr(mod, "_avail_worker_cap", lambda _b: 1)
    est = estimate_memory_usage(
        _tiles(n_planes=100), [3], _config(preprocess_workers=6)
    )
    assert est["preprocess_workers"] == 6


def test_limited_by_names_the_pinning_term():
    """A flat estimate must be explainable, not just unresponsive."""
    tiles = _tiles(n_planes=100)
    est = estimate_memory_usage(tiles, [3], _config(downsample_xy=1))
    assert est["limited_by"] in {"preprocess", "fusion", "output", "tiles"}


def test_z_downsample_lowers_the_peak():
    """Z reduction is a streamed slab mean, so turning it up must SAVE memory.

    While it was a whole-volume float32 zoom, raising Z downsample raised the
    predicted peak (21.5 -> 34.4 GB on the reference job) -- the opposite of
    what the control is for.
    """
    tiles = _tiles()
    peaks = [
        estimate_memory_usage(tiles, [3], _config(downsample_xy=4, downsample_z=dz))[
            "in_memory_gb"
        ]
        for dz in (1, 2, 4)
    ]
    assert peaks[1] < peaks[0], peaks
    assert peaks[2] <= peaks[1], peaks

    # Output keeps halving with Z regardless.
    outs = [
        estimate_memory_usage(tiles, [3], _config(downsample_xy=4, downsample_z=dz))[
            "output_gb"
        ]
        for dz in (1, 2, 4, 8)
    ]
    assert outs == sorted(outs, reverse=True), outs


def test_chunks_are_sized_against_the_final_grid():
    """Auto-chunking must keep the fused block near one tile pitch.

    A fixed 256-px chunk covers 2x the sample area per downsample step, so the
    number of tiles overlapping each fused block -- and with it the float64
    fusion working set -- used to CLIMB as downsample went up.
    """
    from flamingo_stitcher.pipeline import resolve_output_chunksize

    tiles = _tiles()
    heavy = _config(downsample_xy=16)
    chunks = resolve_output_chunksize(heavy, tiles)
    assert chunks["x"] < 256 and chunks["y"] < 256
    assert chunks["x"] >= 64, "must not chunk below the useful floor"

    # Fusion working set must not grow as XY downsample is turned up.
    fusion = [
        estimate_memory_usage(tiles, [3], _config(downsample_xy=ds))["fusion_gb"]
        for ds in (4, 8, 16, 32)
    ]
    assert max(fusion[1:]) <= fusion[0], fusion

    # Opting out returns the configured chunks verbatim.
    manual = _config(downsample_xy=16, auto_output_chunksize=False)
    assert resolve_output_chunksize(manual, tiles) == {"z": 128, "y": 256, "x": 256}


def test_chunks_never_exceed_the_output_extent():
    from flamingo_stitcher.pipeline import resolve_output_chunksize

    chunks = resolve_output_chunksize(
        _config(), _tiles(), out_shape=(10, 90, 40)
    )
    assert chunks == {"z": 10, "y": 90, "x": 40}


def test_iso_can_pick_the_new_heavy_factors():
    """iso's allowed factors must match the widened UI choices."""
    # Very fine XY against a coarse Z step needs more than 8x to reach cubic.
    ds_xy, ds_z = compute_iso_downsample(0.2551, 8.0)
    assert ds_xy in (1, 2, 4, 8, 16, 32)
    assert ds_z in (1, 2, 4, 8, 16)
    assert ds_xy > 8, f"expected a heavy XY factor, got {ds_xy}"

    # A near-isotropic acquisition must still stay at 1x/1x.
    assert compute_iso_downsample(0.5, 0.5) == (1, 1)
