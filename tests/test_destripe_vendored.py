"""The vendored pystripe stripe filter (`_pystripe_core`).

Destriping runs the vendored `filter_streaks` (numpy/scipy/pywt/scikit-image
only) instead of importing the full pystripe package, whose dcimg/imageio/tqdm
imports kept failing to bundle in the frozen build. These tests lock in:
  * the vendored function runs and preserves shape/dtype (needs pywt), and
  * it is BIT-IDENTICAL to upstream pystripe.core.filter_streaks when that is
    importable (skipped otherwise — pystripe is not a runtime/CI dependency).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

pytest.importorskip("pywt", reason="vendored destripe backend needs pywt")

from flamingo_stitcher._pystripe_core import filter_streaks as vendored  # noqa: E402

_HAVE_PYSTRIPE = importlib.util.find_spec("pystripe") is not None
_CONFIGS = [
    ((96, 96), [32, 32], 4, "db2"),
    ((64, 80), [128, 256], 7, "db3"),   # the pipeline defaults
    ((50, 60), [16, 0], 3, "db2"),      # foreground-only
    ((40, 72), [0, 64], 2, "db1"),      # background-only
]


def test_runs_and_preserves_shape_dtype():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 4000, size=(48, 56), dtype=np.uint16)
    out = np.asarray(vendored(img, sigma=[128, 256], level=7, wavelet="db3"))
    assert out.shape == img.shape
    assert out.dtype == np.uint16


def _striped(orient, seed):
    rng = np.random.default_rng(seed)
    base = rng.uniform(600, 1200, size=(5, 160, 160)).astype(np.float32)
    s = 500 * np.sin(np.arange(160) / 1.5)
    stripe = s[None, :, None] if orient == "horizontal" else s[None, None, :]
    return np.clip(base + stripe, 0, 65535).astype(np.uint16)


def _stripe_power(vol, orient):
    proj = np.asarray(vol).astype(float).mean(axis=0)
    p = proj.mean(axis=1) if orient == "horizontal" else proj.mean(axis=0)
    return float((np.abs(np.fft.rfft(p - p.mean())) ** 2)[1:].sum())


@pytest.mark.parametrize("orient", ["horizontal", "vertical"])
def test_auto_detect_and_remove_either_orientation(orient):
    # The v0.9.5 bug: pystripe only removes horizontal stripes, but destriping
    # runs in the camera frame where these stripes are vertical -> nothing removed.
    # "auto" must detect the orientation and remove it either way.
    from flamingo_stitcher.pipeline import _detect_stripe_axis, destripe_volume

    vol = _striped(orient, seed=3)
    assert _detect_stripe_axis(vol) == orient
    out = destripe_volume(vol, direction="auto")
    p_in = _stripe_power(vol, orient)
    p_out = _stripe_power(out, orient)
    assert p_out < p_in * 0.4, f"{orient} stripes not removed ({p_out/p_in:.2f})"


def test_wrong_direction_removes_nothing():
    # Forcing the perpendicular axis (the pre-fix behaviour) leaves stripes intact.
    from flamingo_stitcher.pipeline import destripe_volume

    vol = _striped("vertical", seed=4)
    out = destripe_volume(vol, direction="horizontal")  # wrong axis
    p_in = _stripe_power(vol, "vertical")
    p_out = _stripe_power(out, "vertical")
    assert p_out > p_in * 0.8, "horizontal filter should not touch vertical stripes"


@pytest.mark.skipif(not _HAVE_PYSTRIPE, reason="pystripe not importable here")
def test_bit_identical_to_upstream_pystripe():
    from pystripe.core import filter_streaks as upstream

    rng = np.random.default_rng(7)
    for shp, sig, lvl, wv in _CONFIGS:
        img = rng.integers(0, 4000, size=shp, dtype=np.uint16)
        a = np.asarray(upstream(img, sigma=sig, level=lvl, wavelet=wv))
        b = np.asarray(vendored(img, sigma=sig, level=lvl, wavelet=wv))
        np.testing.assert_array_equal(a, b, err_msg=f"{shp} sig={sig} lvl={lvl} {wv}")
