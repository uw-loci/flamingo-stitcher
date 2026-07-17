"""Unit tests for the post-registration shift clamp.

multiview-stitcher's phase correlation bounds a pairwise shift to the tile size,
not the overlap, so a low-content tile can be flung ~a full tile away and open a
gap. `_clamp_registration_shifts` reverts any tile whose correction exceeds the
expected overlap (or an explicit bound) to its stage position. These tests drive
the pure clamp with hand-built translation params.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("multiview_stitcher")
from multiview_stitcher import param_utils  # noqa: E402

from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline  # noqa: E402


def _params(translations):
    # translation is (z, y, x) in µm; MVS puts it in the affine's last column.
    return [param_utils.affine_from_translation(np.asarray(t, float)) for t in translations]


def _trans(param):
    arr = np.asarray(param)
    if arr.ndim == 3:
        arr = arr[0]
    return arr[:3, 3]


def _tile_data(xs, ys, shape=(8, 100, 100)):
    vol = np.zeros(shape, np.uint16)  # only .shape is read
    return [(vol, SimpleNamespace(x_mm=x, y_mm=y, z_min_mm=0.0)) for x, y in zip(xs, ys)]


# 3 tiles in a row: pitch 80 µm, frame 100 µm (voxel 1 µm) -> overlap 20 µm.
_VOXEL = {"z": 1.0, "y": 1.0, "x": 1.0}


def test_auto_bound_reverts_overbudget_tile():
    pipe = StitchingPipeline(StitchingConfig())  # max_registration_shift_um=0 -> auto
    td = _tile_data([0.0, 0.08, 0.16], [0.0, 0.0, 0.0])
    params = _params([(0, 2, 3), (0, 5, 60), (0, 1, -4)])  # middle exceeds 20 µm
    out = pipe._clamp_registration_shifts(params, td, _VOXEL)
    assert np.max(np.abs(_trans(out[0]))) == pytest.approx(3, abs=1e-6)  # kept
    assert np.allclose(_trans(out[1]), 0.0)  # reverted to stage
    assert np.max(np.abs(_trans(out[2]))) == pytest.approx(4, abs=1e-6)  # kept


def test_explicit_bound_overrides_auto():
    cfg = StitchingConfig()
    cfg.max_registration_shift_um = 10.0
    pipe = StitchingPipeline(cfg)
    td = _tile_data([0.0, 0.08], [0.0, 0.0])  # overlap 20 µm, but bound is 10
    params = _params([(0, 0, 8), (0, 0, 12)])
    out = pipe._clamp_registration_shifts(params, td, _VOXEL)
    assert np.max(np.abs(_trans(out[0]))) == pytest.approx(8, abs=1e-6)  # 8 < 10 kept
    assert np.allclose(_trans(out[1]), 0.0)  # 12 > 10 reverted


def test_noop_when_all_within_bound():
    pipe = StitchingPipeline(StitchingConfig())
    td = _tile_data([0.0, 0.08, 0.16], [0.0, 0.0, 0.0])
    params = _params([(0, 1, 1), (0, 2, 2), (0, -3, 1)])  # all << 20 µm
    out = pipe._clamp_registration_shifts(params, td, _VOXEL)
    for p_in, p_out in zip(params, out):
        assert np.allclose(_trans(p_in), _trans(p_out))


def test_empty_params_returns_empty():
    pipe = StitchingPipeline(StitchingConfig())
    assert pipe._clamp_registration_shifts([], _tile_data([0.0], [0.0]), _VOXEL) == []
