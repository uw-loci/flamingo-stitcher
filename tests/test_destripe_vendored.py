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


def test_params_reach_the_filter():
    """destripe_params must actually change filtering (not be silently ignored)."""
    from flamingo_stitcher.pipeline import destripe_volume

    vol = _striped("horizontal", seed=9)
    default = destripe_volume(vol, direction="horizontal")
    narrow = destripe_volume(
        vol, direction="horizontal",
        params={"sigma_foreground": 8, "sigma_background": 8},
    )
    assert not np.array_equal(default, narrow), "sigma override had no effect"

    # sigma 0/0 disables the filter entirely (upstream's documented behaviour)
    off = destripe_volume(
        vol, direction="horizontal",
        params={"sigma_foreground": 0, "sigma_background": 0},
    )
    assert np.abs(off.astype(int) - vol.astype(int)).max() <= 1

    # a partial dict overrides only what it names
    assert destripe_volume(
        vol, direction="horizontal", params={"level": 3}
    ).shape == vol.shape


def test_settings_dialog_defaults_match_pipeline():
    """The dialog's DEFAULTS must not drift from the pipeline/YAML defaults."""
    pytest.importorskip("PyQt5", reason="dialog defaults live in a Qt module")
    from flamingo_stitcher.config_loader import get_stitching_value
    from flamingo_stitcher.gui.destripe_settings_dialog import DEFAULTS

    sigma = get_stitching_value("destripe", "sigma", default=[128, 256])
    assert DEFAULTS["sigma_foreground"] == float(sigma[0])
    assert DEFAULTS["sigma_background"] == float(sigma[1])
    assert DEFAULTS["level"] == get_stitching_value("destripe", "level", default=7)
    assert DEFAULTS["wavelet"] == get_stitching_value(
        "destripe", "wavelet", default="db2"
    )
    assert DEFAULTS["threshold"] is None  # Otsu


@pytest.mark.skipif(not _HAVE_PYSTRIPE, reason="pystripe not importable here")
def test_bit_identical_to_upstream_pystripe():
    from pystripe.core import filter_streaks as upstream

    rng = np.random.default_rng(7)
    for shp, sig, lvl, wv in _CONFIGS:
        img = rng.integers(0, 4000, size=shp, dtype=np.uint16)
        a = np.asarray(upstream(img, sigma=sig, level=lvl, wavelet=wv))
        b = np.asarray(vendored(img, sigma=sig, level=lvl, wavelet=wv))
        np.testing.assert_array_equal(a, b, err_msg=f"{shp} sig={sig} lvl={lvl} {wv}")


# --- Acquisition-wide stripe axis + frame-size-aware wavelet depth ---------


def _axis_tile(size=128, planes=4, orient="vertical", content=0.0, seed=0):
    """Tile with stripes along one axis, plus optional anatomical content."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.zeros((size, size), np.float32) + 50.0
    if content:
        # Banding that runs PERPENDICULAR to the real stripes, i.e. the kind of
        # anatomy that can outvote them in a per-tile detector.
        base = base + content * np.sin(yy / 5.0) * (xx > size * 0.6)
    stripes = 300 * np.sin((xx if orient == "vertical" else yy) / 2.0)
    plane = np.clip(base + stripes + rng.normal(0, 15, (size, size)), 0, 65535)
    return np.repeat(plane[None].astype(np.uint16), planes, axis=0)


class TestAcquisitionWideStripeAxis:
    """Stripe orientation is set by the beam/camera frame, so it is ONE
    property of the acquisition -- not something to re-guess per tile."""

    def _pipeline(self, direction="auto"):
        from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

        return StitchingPipeline(config=StitchingConfig(destripe_direction=direction))

    def test_direction_is_locked_after_enough_votes(self):
        pipe = self._pipeline()
        vol = _axis_tile(orient="vertical")

        for _ in range(pipe._DESTRIPE_AXIS_VOTES):
            assert pipe._resolve_destripe_direction(vol) == "vertical"

        assert pipe._destripe_axis_locked == "vertical"

    def test_locked_axis_survives_a_tile_that_votes_the_other_way(self):
        """The whole point: one odd tile can no longer get the wrong filter."""
        pipe = self._pipeline()
        vertical = _axis_tile(orient="vertical")
        for _ in range(pipe._DESTRIPE_AXIS_VOTES):
            pipe._resolve_destripe_direction(vertical)

        horizontal = _axis_tile(orient="horizontal")
        assert pipe._resolve_destripe_direction(horizontal) == "vertical"

    def test_explicit_direction_bypasses_voting_entirely(self):
        pipe = self._pipeline(direction="horizontal")
        vol = _axis_tile(orient="vertical")

        assert pipe._resolve_destripe_direction(vol) == "horizontal"
        assert pipe._destripe_axis_locked is None  # never voted

    def test_low_confidence_tiles_barely_count(self):
        from flamingo_stitcher.pipeline import _stripe_axis_vote

        _, strong = _stripe_axis_vote(_axis_tile(orient="vertical"))
        _, weak = _stripe_axis_vote(
            np.full((4, 128, 128), 100, dtype=np.uint16)  # featureless
        )
        assert strong > weak

    def test_reset_clears_the_lock_for_a_new_run(self):
        pipe = self._pipeline()
        vol = _axis_tile(orient="vertical")
        for _ in range(pipe._DESTRIPE_AXIS_VOTES):
            pipe._resolve_destripe_direction(vol)
        assert pipe._destripe_axis_locked is not None

        pipe._reset_destripe_axis()
        assert pipe._destripe_axis_locked is None


class TestLevelClampedToFrameSize:
    """`level` is a fixed config value (default 7) but the usable wavelet depth
    is set by the frame. Past it, pywt warns that "all coefficients will
    experience boundary effects" and the filter's behaviour becomes erratic:
    measured on a 256px frame with content, level=7 removed 14.7% of the stripe
    power where the frame's max level of 6 removed 59.0%. Clamping keeps the
    transform in the regime where coefficients are meaningful. It is a GUARD,
    not a guaranteed quality win -- on a 128px frame the deeper level happened
    to score better -- so these tests lock the clamping behaviour, not a
    universal improvement.
    """

    def test_clamps_and_says_so_when_the_frame_is_too_small(self, caplog):
        from flamingo_stitcher.pipeline import destripe_volume
        from flamingo_stitcher._pystripe_core import max_level

        vol = _axis_tile(size=128, planes=2, orient="vertical")
        usable = max_level(128, "db2")
        assert usable < 7, "test premise: 7 must be too deep for a 128px frame"

        with caplog.at_level("INFO"):
            destripe_volume(
                vol, direction="vertical", max_workers=1, params={"level": 7}
            )

        assert "clamping" in caplog.text
        assert f"level={usable}" in caplog.text

    def test_leaves_the_requested_level_alone_when_it_fits(self, caplog):
        from flamingo_stitcher.pipeline import destripe_volume

        vol = _axis_tile(size=1024, planes=2, orient="vertical")
        with caplog.at_level("INFO"):
            destripe_volume(
                vol, direction="vertical", max_workers=1, params={"level": 5}
            )

        assert "level=5" in caplog.text
        assert "clamping" not in caplog.text

    def test_clamped_run_still_removes_stripes(self):
        from flamingo_stitcher.pipeline import destripe_volume

        vol = _axis_tile(size=128, planes=2, orient="vertical")

        def power(v):
            p = np.asarray(v).astype(np.float32).mean(axis=(0, 1))
            f = np.abs(np.fft.rfft(p - p.mean())) ** 2
            return float(f[max(1, len(f) // 8):].sum())

        out = destripe_volume(
            vol, direction="vertical", max_workers=1, params={"level": 7}
        )
        assert power(out) < power(vol) * 0.5
