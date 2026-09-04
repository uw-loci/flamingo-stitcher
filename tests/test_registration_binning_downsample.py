"""registration_binning must not multiply with the downsample already applied.

The config value is expressed against NATIVE tiles — "z=2, xy=4" means half the
raw planes and a quarter of the raw pixels, which is how the defaults were
tuned. But registration runs on the PREPROCESSED spill, which the pipeline has
already downsampled, so the two compound.

The 7x7 run on 2026-09-03 (XY=8x downsample, default xy=4 binning) registered at
32x: a 2048 px frame became 64 px and a 15% overlap became a 9.6 px strip. Phase
correlation on ten pixels returns a confident wrong shift rather than failing —
31 of 84 seams came back at quality 0.12-0.34 against a 0.4 threshold, the
coverage gate saw 44%, and all 49 tiles were placed by stage position.

Run: python -m pytest tests/test_registration_binning_downsample.py -q
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from flamingo_stitcher.pipeline import (  # noqa: E402
    MIN_REGISTRATION_OVERLAP_PX,
    REGISTRATION_OVERLAP_TARGET_PX,
    StitchingConfig,
    StitchingPipeline,
)

NATIVE, PLANES = 2048, 640
PITCH_MM = 1.8225  # 15% overlap on a 2048 px frame at ~1.0475 µm/px


def _tile_data(downsample_xy, downsample_z=1, n=4):
    """Tiles as the spill holds them: already downsampled."""
    vol = np.zeros(
        (PLANES // downsample_z, NATIVE // downsample_xy, NATIVE // downsample_xy),
        np.uint16,
    )
    out = []
    for i in range(n):
        row, col = divmod(i, 2)
        out.append(
            (
                vol,
                SimpleNamespace(
                    x_mm=col * PITCH_MM,
                    y_mm=row * PITCH_MM,
                    z_min_mm=16.931,
                    n_planes=PLANES,
                    frame_width=NATIVE,
                    frame_height=NATIVE,
                ),
            )
        )
    return out


def _extent(downsample_xy):
    # One frame's physical coverage — independent of the downsample.
    return {"x": NATIVE * 1.0475, "y": NATIVE * 1.0475, "z": PLANES * 10.0}


def _binning(config, downsample_xy, downsample_z=1, caplog=None):
    pipe = StitchingPipeline(config)
    td = _tile_data(downsample_xy, downsample_z)
    tiles = [ti for _v, ti in td]
    ctx = caplog.at_level(logging.INFO, logger=pipe.logger.name) if caplog else None
    if ctx:
        with ctx:
            return pipe._effective_registration_binning(td, tiles, _extent(downsample_xy))
    return pipe._effective_registration_binning(td, tiles, _extent(downsample_xy))


class TestItStopsMultiplying:
    def test_full_resolution_is_unchanged(self):
        """The tuned default must survive a 1x run exactly."""
        assert _binning(StitchingConfig(), 1) == {"z": 2, "y": 4, "x": 4}

    def test_the_run_that_failed_no_longer_bins_on_top(self):
        """XY=8x with xy=4 configured: nothing left to bin."""
        out = _binning(StitchingConfig(), 8)
        assert out["x"] == 1 and out["y"] == 1

    def test_the_case_that_already_worked_is_not_made_more_expensive(self):
        """XY=2x kept binning 4 and registered 11 of 12 seams at a 38 px strip.

        Discounting purely by the downsample would drop it to 2 and register on
        4x the pixels for margin the evidence says is not needed. The configured
        value is a CEILING, not a target to climb back down from.
        """
        out = _binning(StitchingConfig(), 2)
        assert out["x"] == 4 and out["y"] == 4
        assert 0.15 * (NATIVE / 2) / out["x"] == pytest.approx(38.4)

    def test_a_middling_downsample_bins_as_hard_as_the_strip_allows(self):
        out = _binning(StitchingConfig(), 4)
        assert out["x"] == 2
        assert 0.15 * (NATIVE / 4) / out["x"] == pytest.approx(38.4)

    def test_the_configured_value_is_never_exceeded(self):
        """More binning than asked for would register on fewer pixels than the
        setting allows, which is the user's call, not ours."""
        cfg = StitchingConfig(registration_binning={"z": 2, "y": 2, "x": 2})
        for ds in (1, 2, 4, 8):
            out = _binning(cfg, ds)
            assert out["x"] <= 2 and out["y"] <= 2, (ds, out)

    def test_z_is_discounted_on_its_own_axis(self):
        out = _binning(StitchingConfig(), 1, downsample_z=2)
        assert out["z"] == 1, "z=2 configured against a 2x Z downsample leaves 1"

    def test_z_is_untouched_when_z_is_not_downsampled(self):
        # The failing run was Z=1x: the Z binning was never the problem.
        assert _binning(StitchingConfig(), 8, downsample_z=1)["z"] == 2

    def test_it_never_returns_zero_or_negative(self):
        for ds in (1, 2, 4, 8, 16, 32):
            out = _binning(StitchingConfig(), ds)
            assert all(v >= 1 for v in out.values()), (ds, out)


class TestTheOverlapFloor:
    def test_the_failing_run_now_clears_the_floor(self):
        """9.6 px was the whole problem; unbinned it is 38 px."""
        out = _binning(StitchingConfig(), 8)
        strip = 0.15 * (NATIVE / 8) / out["x"]
        assert strip >= MIN_REGISTRATION_OVERLAP_PX, f"{strip:.1f} px"

    def test_a_larger_configured_binning_is_still_held_to_the_strip(self):
        out = _binning(StitchingConfig(registration_binning={"z": 2, "y": 8, "x": 8}), 4)
        strip = 0.15 * (NATIVE / 4) / out["x"]
        assert strip >= MIN_REGISTRATION_OVERLAP_PX, f"{strip:.1f} px with {out}"

    def test_every_downsample_lands_at_or_above_the_target_when_it_can(self):
        for ds in (1, 2, 4, 8):
            out = _binning(StitchingConfig(), ds)
            strip = 0.15 * (NATIVE / ds) / out["x"]
            assert strip >= REGISTRATION_OVERLAP_TARGET_PX, (ds, out, strip)

    def test_it_warns_when_even_unbinned_is_too_thin(self, caplog):
        out = _binning(StitchingConfig(), 32, caplog=caplog)
        assert out["x"] == 1
        assert "px" in caplog.text and "downsample" in caplog.text


class TestItSaysWhatItDid:
    def test_a_change_is_logged_with_both_values(self, caplog):
        _binning(StitchingConfig(), 8, caplog=caplog)
        assert "Registration binning" in caplog.text
        assert "already downsampled" in caplog.text

    def test_no_change_is_not_announced(self, caplog):
        _binning(StitchingConfig(), 1, caplog=caplog)
        assert "Registration binning" not in caplog.text


class TestDoesNotKillTheRun:
    def test_no_tiles_returns_the_configured_value(self):
        pipe = StitchingPipeline(StitchingConfig())
        assert pipe._effective_registration_binning([], [], {}) == {
            "z": 2, "y": 4, "x": 4
        }

    def test_unmeasurable_tile_info_returns_the_configured_value(self):
        pipe = StitchingPipeline(StitchingConfig())
        td = [(np.zeros((4, 8, 8)), SimpleNamespace())]
        assert pipe._effective_registration_binning(td, [], {}) == {
            "z": 2, "y": 4, "x": 4
        }

    def test_missing_extent_still_discounts_the_downsample(self):
        """The overlap floor needs the extent; the discount does not."""
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data(8)
        out = pipe._effective_registration_binning(td, [ti for _v, ti in td], None)
        assert out["x"] == 1
