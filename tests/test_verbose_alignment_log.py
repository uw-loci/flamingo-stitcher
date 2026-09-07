"""The alignment evidence belongs in the log, not only in the CSVs.

`format_report_text` elides: five worst corrections, ten unused seams, "... and
32 more (see registration_seams.csv)". That is the right length for a run that
worked. Three runs in a row this week were diagnosed only by opening the CSVs —
and the CSVs are on the acquisition machine while the log is what gets shared.

So the full tables go in the log: every tile with how it got where it is, every
adjacent pair with what was measured across it.

Run: python -m pytest tests/test_verbose_alignment_log.py -q
"""

from __future__ import annotations

import pytest

from flamingo_stitcher import registration_report as rr  # noqa: E402
from flamingo_stitcher.pipeline import StitchingConfig  # noqa: E402


def _report(n_tiles=30, n_seams=50):
    """Bigger than every truncation limit in format_report_text."""
    report = rr.RegistrationReport(ran=True, applied=True, transform_key="registered")
    for i in range(n_tiles):
        report.tiles.append(
            rr.TileShift(
                index=i,
                name="a_very_long_acquisition_folder_name",
                x_mm=float(i),
                y_mm=2.0 * i,
                dz_um=float(i),
                dy_um=-float(i),
                dx_um=0.5 * i,
            )
        )
    for i in range(n_seams):
        report.seams.append(
            rr.SeamResult(
                index_a=i % n_tiles,
                index_b=(i + 1) % n_tiles,
                tile_a="a_very_long_acquisition_folder_name",
                tile_b="a_very_long_acquisition_folder_name",
                axis="x",
                status=rr.STATUS_REGISTERED,
                quality=0.8,
                dz_um=1.0,
                dy_um=2.0,
                dx_um=3.0,
                residual_px=0.4,
                overlap_frac=0.15,
            )
        )
    return report


class TestNothingIsElided:
    def test_every_tile_gets_a_row(self):
        lines = rr.format_verbose_alignment(_report())
        body = [ln for ln in lines if ln.startswith("  ") and "stage X" not in ln]
        assert sum(1 for ln in body if "a_very_long" in ln or "T" in ln) >= 30

    def test_every_seam_gets_a_row(self):
        report = _report(n_tiles=30, n_seams=50)
        text = "\n".join(rr.format_verbose_alignment(report))
        assert text.count("registered") >= 50

    def test_it_never_says_and_n_more(self):
        """The exact phrase the summary uses to stop short."""
        text = "\n".join(rr.format_verbose_alignment(_report()))
        assert "more (see" not in text

    def test_the_counts_are_stated(self):
        text = "\n".join(rr.format_verbose_alignment(_report(12, 20)))
        assert "TILE PLACEMENT (12 tiles)" in text
        assert "SEAM MEASUREMENTS (20 adjacent pairs)" in text


class TestItNamesTilesUsefully:
    def test_labels_replace_the_repeated_folder_name(self):
        """All 49 tiles of a single-workflow run share one folder name."""
        report = _report(3, 2)
        labels = {0: "X000 Y000", 1: "X001 Y000", 2: "X002 Y000"}
        text = "\n".join(rr.format_verbose_alignment(report, labels=labels))
        assert "X001 Y000" in text
        assert "a_very_long_ac" not in text.split("SEAM MEASUREMENTS")[0]

    def test_without_labels_it_falls_back_to_the_tile_name(self):
        text = "\n".join(rr.format_verbose_alignment(_report(2, 1)))
        assert "a_very_long_ac" in text


class TestItSaysHowEachTileGotThere:
    def test_the_placement_column_is_rendered(self):
        report = _report(3, 1)
        placement = {0: "registered", 1: "carried r1", 2: "no neighbour"}
        text = "\n".join(
            rr.format_verbose_alignment(report, placement=placement)
        )
        assert "carried r1" in text and "no neighbour" in text

    def test_registered_is_the_default_when_nothing_is_passed(self):
        text = "\n".join(rr.format_verbose_alignment(_report(2, 1)))
        assert "registered" in text

    def test_a_clamped_axis_is_called_out_on_its_own_row(self):
        """A clamped axis was NOT measured — the row has to say so, because a
        legend further up is not read when scanning 49 rows."""
        report = _report(1, 0)
        report.tiles[0].clamped_z = True
        report.tiles[0].clamped_x = True
        text = "\n".join(rr.format_verbose_alignment(report))
        assert "clamped ZX" in text
        assert "kept stage position" in text


class TestUnits:
    def test_the_residual_column_is_labelled_micrometres(self):
        """multiview-stitcher's edge_residuals are physical units; the CSV's
        `residual_px` name is a misnomer kept for compatibility."""
        text = "\n".join(rr.format_verbose_alignment(_report(2, 1)))
        assert "resid µm" in text
        assert "resid px" not in text


class TestMissingValuesDoNotBreakTheTable:
    def test_a_seam_with_no_measurement_still_renders(self):
        report = rr.RegistrationReport(ran=True)
        report.seams.append(
            rr.SeamResult(
                index_a=0,
                index_b=1,
                tile_a="a",
                tile_b="b",
                axis="y",
                status=rr.STATUS_NO_CONTENT,
                quality=None,
                residual_px=None,
                overlap_frac=None,
            )
        )
        text = "\n".join(rr.format_verbose_alignment(report))
        assert "no_content" in text

    def test_an_empty_report_renders_headers_only(self):
        lines = rr.format_verbose_alignment(rr.RegistrationReport())
        assert any("TILE PLACEMENT (0 tiles)" in ln for ln in lines)


class TestTheSwitch:
    def test_it_is_on_by_default(self):
        """Off by default would mean discovering after a 17-hour run that the
        evidence was truncated."""
        assert StitchingConfig().verbose_alignment_log is True

    def test_it_can_be_turned_off(self):
        assert StitchingConfig(verbose_alignment_log=False).verbose_alignment_log is False
