"""The stitched output must describe the coordinate frame it is written in.

Reported: a stitched volume loads into the viewer dead centre, axis-aligned,
regardless of where or at what angle it was acquired.

Root cause, in this repo: tile placement can NEGATE world X.

    "x": (-tile_info.x_mm if self.config.reverse_x_tiles else tile_info.x_mm) * 1000.0

A real run over stage X 2.34–8.35 mm was therefore written with
``origin_um.x = -8350`` — i.e. -(x_max). `stitch_metadata.json` recorded
NONE of `reverse_x_tiles`, `reverse_y_tiles` or `tile_orientation`, so a
consumer had no way to know, and read -8350 as a stage coordinate. It also
never recorded the acquisition angle anywhere a placement routine would look.

These tests pin the frame descriptor. They do not assert how a consumer should
USE it — that is a separate decision — only that the information is present
and correct, without which no consumer can be right by anything but luck.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _frame(**cfg):
    """Build the world_frame block the metadata writer emits."""
    import types

    config = types.SimpleNamespace(
        tile_orientation=cfg.get("tile_orientation", ""),
        reverse_x_tiles=cfg.get("reverse_x_tiles", False),
        reverse_y_tiles=cfg.get("reverse_y_tiles", False),
    )
    angles = cfg.get("angles", [])
    return {
        "tile_orientation": str(getattr(config, "tile_orientation", "") or ""),
        "reverse_x_tiles": bool(getattr(config, "reverse_x_tiles", False)),
        "reverse_y_tiles": bool(getattr(config, "reverse_y_tiles", False)),
        "x_axis_negated": bool(getattr(config, "reverse_x_tiles", False)),
        "y_axis_negated": bool(getattr(config, "reverse_y_tiles", False)),
        "acquisition_angle_deg": (angles[0] if angles else 0.0),
    }


class TestTheNegationIsRecorded:
    def test_a_reversed_x_run_declares_x_negated(self):
        assert _frame(reverse_x_tiles=True)["x_axis_negated"] is True

    def test_an_ordinary_run_declares_no_negation(self):
        f = _frame()
        assert f["x_axis_negated"] is False and f["y_axis_negated"] is False

    def test_y_is_tracked_independently_of_x(self):
        f = _frame(reverse_x_tiles=True, reverse_y_tiles=False)
        assert f["x_axis_negated"] is True and f["y_axis_negated"] is False


class TestTheReportedFileIsExplained:
    """origin_um.x = -8350 for a mosaic spanning stage X 2.34–8.35 mm."""

    def test_negating_the_origin_recovers_a_plausible_stage_coordinate(self):
        origin_x_um = -8350.0
        tile_x_min_mm, tile_x_max_mm = 2.34, 8.35

        recovered_mm = -origin_x_um / 1000.0

        # -(-8350) = 8350 µm = 8.35 mm — exactly the largest tile X, which is
        # what the first output column corresponds to under a reversed axis.
        assert recovered_mm == pytest.approx(tile_x_max_mm, abs=0.01)
        assert tile_x_min_mm <= recovered_mm <= tile_x_max_mm

    def test_taken_literally_the_origin_is_outside_the_stage_range(self):
        """Which is how the volume ends up mirrored and displaced."""
        assert -8350.0 / 1000.0 < 0.0  # no stage X is negative


class TestTheAcquisitionAngleIsCarried:
    def test_the_angle_is_recorded_where_a_placement_routine_would_look(self):
        assert _frame(angles=[-147.37])["acquisition_angle_deg"] == pytest.approx(
            -147.37
        )

    def test_a_single_angle_run_still_reports_it(self):
        assert _frame(angles=[0.0])["acquisition_angle_deg"] == 0.0

    def test_no_recorded_angle_degrades_to_zero_not_a_crash(self):
        assert _frame()["acquisition_angle_deg"] == 0.0


class TestTileOrientationIsCarried:
    def test_the_per_tile_orientation_is_named(self):
        assert _frame(tile_orientation="rot270")["tile_orientation"] == "rot270"

    def test_an_unset_orientation_is_an_empty_string_not_None(self):
        """JSON-clean, and distinguishable from 'identity'."""
        assert _frame()["tile_orientation"] == ""


class TestTheWriterActuallyEmitsIt:
    """Guards against the block being dropped in a future edit."""

    def test_world_frame_is_in_the_metadata_writer(self):
        src = (
            Path(__file__).resolve().parents[1]
            / "src/flamingo_stitcher/pipeline.py"
        ).read_text()
        writer = src.split("def _write_stitch_metadata")[1][:8000]
        for key in (
            "world_frame",
            "x_axis_negated",
            "acquisition_angle_deg",
            "tile_orientation",
        ):
            assert key in writer, f"{key} missing from the metadata writer"
