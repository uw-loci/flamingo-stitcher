"""Align light paths: the second path's stack is measured against the first
and moved in whole voxels before the two are merged (2026-10-29, doubling in
the early rows of Ket_Trial1)."""

import numpy as np
import pytest
from scipy import ndimage

from flamingo_stitcher import side_align
from flamingo_stitcher.pipeline import fuse_illumination_sides

VOXEL = (2.5, 1.05, 1.05)  # z, y, x in um, as Ket_Trial1


def _cells(shape, seed=0, n=1500):
    """Sparse blurred cells (~3 px across, a few planes deep) on background."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, np.float32)
    idx = tuple(rng.integers(0, s, n) for s in shape)
    vol[idx] = 1.0
    vol = ndimage.gaussian_filter(vol, (1.5, 2.0, 2.0))
    vol = 100 + 3000 * vol / vol.max()
    return (vol + rng.normal(0, 3, shape)).astype(np.uint16)


def _move(vol, t):
    """second(x + t) == first(x): second is first moved by -t."""
    return np.roll(vol, shift=t, axis=(0, 1, 2))


@pytest.fixture(scope="module")
def tile():
    return _cells((160, 192, 192))


def test_measures_the_shift_between_paths(tile):
    t = (3, 4, -6)
    second = _move(tile, t)
    res = side_align.measure(tile, second, VOXEL, 0.1)
    assert res.applied, res.reason
    assert res.shift == t
    assert res.r_with > res.r_without


def test_nothing_to_line_up_is_not_applied():
    flat = np.full((160, 128, 128), 100, np.uint16)
    res = side_align.measure(flat, flat.copy(), VOXEL, 0.1)
    assert not res.applied
    assert res.as_applied() is None


def test_noise_alone_is_not_applied():
    rng = np.random.default_rng(5)
    a = (100 + rng.normal(0, 3, (160, 128, 128))).astype(np.uint16)
    b = (100 + rng.normal(0, 3, (160, 128, 128))).astype(np.uint16)
    assert not side_align.measure(a, b, VOXEL, 0.1).applied


def test_aligned_max_fusion_is_not_doubled(tile):
    t = (3, 4, -6)
    second = _move(tile, t)
    inner = (slice(10, -10), slice(10, -10), slice(10, -10))
    doubled = fuse_illumination_sides({0: tile, 1: second}, method="max")
    fixed = fuse_illumination_sides({0: tile, 1: second}, method="max", align=t)
    assert not np.array_equal(doubled[inner], tile[inner])
    np.testing.assert_array_equal(fixed[inner], tile[inner])


def test_split_with_alignment_matches_the_unshifted_result(tile):
    t = (2, -3, 5)
    second = _move(tile, t)
    kw = dict(method="split", axis=-1, low_side=0)
    want = fuse_illumination_sides({0: tile, 1: tile}, **kw)
    got = fuse_illumination_sides({0: tile, 1: second}, align=t, **kw)
    inner = (slice(10, -10), slice(10, -10), slice(10, -10))
    np.testing.assert_array_equal(got[inner], want[inner])


def test_uncovered_voxels_keep_the_first_path():
    first = np.full((4, 6, 6), 7, np.uint16)
    second = np.full((4, 6, 6), 9, np.uint16)
    # z + 2 is past the stack for the last two planes
    assert (side_align.shifted_plane(second, 3, (2, 0, 0), first[3]) == 7).all()
    p = side_align.shifted_plane(second, 0, (0, 2, -1), first[0])
    assert (p[:4, 1:] == 9).all()
    assert (p[4:, :] == 7).all() and (p[:, :1] == 7).all()


def test_no_shift_keeps_the_original_merge(tile):
    a = fuse_illumination_sides({0: tile, 1: tile}, method="max", align=(0, 0, 0))
    b = fuse_illumination_sides({0: tile, 1: tile}, method="max")
    np.testing.assert_array_equal(a, b)


def test_shifts_of_many_planes_are_measured(tile):
    t = (12, -10, 8)
    res = side_align.measure(tile, _move(tile, t), VOXEL, 0.1)
    assert res.applied and res.shift == t


def test_pipeline_measures_logs_and_returns_the_shift(tile, caplog):
    import logging
    from types import SimpleNamespace

    from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

    cfg = StitchingConfig(pixel_size_um=VOXEL[1])
    cfg.z_step_um = VOXEL[0]
    pipe = StitchingPipeline(cfg)
    t = (3, 4, -6)
    vols = {0: tile, 1: _move(tile, t)}
    one = SimpleNamespace(tile_index=(0, 0), x_mm=11.4, y_mm=25.0, z_step_mm=0.0025)
    two = SimpleNamespace(tile_index=(1, 0), x_mm=9.5, y_mm=25.0, z_step_mm=0.0025)

    assert pipe._light_path_shift(vols, one, 3) is None  # option off
    cfg.illumination_align = True
    with caplog.at_level(logging.INFO):
        assert pipe._light_path_shift(vols, two, 3) == t
        assert pipe._light_path_shift(vols, one, 3) == t
        pipe._log_light_path_shifts()
    text = caplog.text
    assert "applied" in text and "dz +3 planes (+7.5 µm)" in text
    # the summary lists X000 Y000 (imaged first) before X001 Y000
    summary = text[text.index("Align light paths: 2 of 2"):]
    assert summary.index("X000 Y000") < summary.index("X001 Y000")
