"""Holding tiles with nothing to register against out of the registration.

This is the multiview-stitcher maintainer's own recommendation for the failure
in issue #70 — a background tile produces a confident wrong shift, and the
cheapest way not to believe it is not to ask.

The phantom here is the shape that actually breaks an intensity test: the
featureless tiles are written BRIGHTER than the sample. That is the real
geometry of a fish in agarose in an FEP tube, where the gel is both bright and
smooth.
"""

from __future__ import annotations

import pytest

pytest.importorskip("multiview_stitcher")

from flamingo_stitcher import registration_report as rr  # noqa: E402
from flamingo_stitcher.pipeline import (  # noqa: E402
    StitchingConfig,
    StitchingPipeline,
)

from _synth_acq import write_synth_acquisition  # noqa: E402


@pytest.fixture(scope="module")
def acquisition(tmp_path_factory):
    """3x3 with the whole right-hand column written as bright, smooth gel."""
    return write_synth_acquisition(
        tmp_path_factory.mktemp("content_gate"),
        grid=(3, 3),
        overlap=0.25,
        n_planes=12,
        frame_size=(48, 48),
        # Big enough pixels that the stage steps survive the 2-decimal rounding
        # in the tile folder names; at the 0.406 um default a 48 px frame steps
        # by 0.015 mm, which rounds to 0.02 and reads back as NEGATIVE overlap.
        pixel_size_um=4.0,
        featureless_tiles=[(2, 0), (2, 1), (2, 2)],
        featureless_level=9000.0,
        z_texture=True,
    )


def _run(acquisition, out_dir, **overrides):
    config = StitchingConfig(
        skip_registration=False,
        registration_report_enabled=True,
        output_format="ome-zarr",
        # Pin it: the pipeline otherwise auto-resolves from the synthetic
        # ScopeSettings magnification and falls back to its 0.406 um default,
        # which makes the tiles read as having negative overlap.
        pixel_size_um=4.0,
        **overrides,
    )
    pipe = StitchingPipeline(config)
    pipe.run(acquisition, out_dir)
    return pipe._registration_report


class TestTheGateFires:
    def test_the_bright_featureless_column_is_held_out(self, acquisition, tmp_path):
        report = _run(acquisition, tmp_path / "held")
        assert report.settings.get("tiles_without_structure") == 3
        # The one that matters: read back from what was actually handed to
        # register(), so a broken hold-out cannot pass by scoring correctly.
        assert report.settings.get("tiles_registered") == 6

    def test_seams_touching_a_held_out_tile_are_marked_no_content(
        self, acquisition, tmp_path
    ):
        report = _run(acquisition, tmp_path / "seams")
        no_content = [s for s in report.seams if s.status == rr.STATUS_NO_CONTENT]
        assert no_content, "no seam was marked as having nothing to register"
        for seam in no_content:
            assert "no structure" in seam.note

    def test_those_seams_do_not_count_against_the_trust_threshold(
        self, acquisition, tmp_path
    ):
        """The point of the status. Counting them makes a sparse mosaic read as
        a failed registration rather than a sparse one."""
        report = _run(acquisition, tmp_path / "trust")
        counted = rr.mosaic_coverage(report.n_tiles, report.seams).n_expected_seams
        assert counted == len(report.seams) - report.count(rr.STATUS_NO_CONTENT)
        assert counted < len(report.seams)


class TestItStillProducesOutput:
    def test_the_run_completes_and_writes_a_report(self, acquisition, tmp_path):
        out = tmp_path / "out"
        report = _run(acquisition, out)
        assert (out / "registration_report.csv").is_file()
        assert (out / "registration_seams.csv").is_file()
        assert report.n_tiles == 9

    def test_every_tile_still_gets_a_row(self, acquisition, tmp_path):
        report = _run(acquisition, tmp_path / "rows")
        assert [t.index for t in report.tiles] == list(range(9))

    def test_turning_the_gate_off_registers_every_tile(self, acquisition, tmp_path):
        report = _run(acquisition, tmp_path / "nogate", min_tile_structure=0.0)
        assert report.settings.get("tiles_without_structure") == 0
        assert report.settings.get("tiles_registered") == 9
        assert report.count(rr.STATUS_NO_CONTENT) == 0


class TestTheScoresAreReported:
    def test_the_structure_range_is_recorded_so_the_threshold_is_checkable(
        self, acquisition, tmp_path
    ):
        report = _run(acquisition, tmp_path / "scores")
        low, high = report.settings["tile_structure_range"]
        assert 0.0 <= low < high <= 1.0
        # The gel must sit below the threshold and the sample above it, or the
        # measure is not separating them on this phantom.
        assert low < report.settings["min_tile_structure"] <= high
