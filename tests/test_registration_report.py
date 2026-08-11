"""The registration report's contract: shape, units, and honesty.

The honesty assertions are the point. Border QC once printed a clamped search
limit as though it were a measurement, and 57% of the flagged seams in a real
run turned out to be unmeasured while the report read like a tidy small offset.
These tests pin the three rules that prevent a repeat: a clamped axis is not a
result, an unknown number is an empty cell, and a run that did not register
still says so in writing.

Run: python3 -m pytest tests/test_registration_report.py -q
"""

from __future__ import annotations

import csv
import io
from types import SimpleNamespace

import numpy as np
import pytest

from flamingo_stitcher import registration_report as rr

_VOXEL = {"z": 10.0, "y": 0.8, "x": 0.8}


def _tile(name, x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(
        folder=SimpleNamespace(name=name), x_mm=x, y_mm=y, z_min_mm=z
    )


def _affine(dz=0.0, dy=0.0, dx=0.0, with_time_axis=False):
    mat = np.eye(4)
    mat[0, 3], mat[1, 3], mat[2, 3] = dz, dy, dx
    return mat[None, ...] if with_time_axis else mat


def _row_grid(n=2, pitch=0.9):
    """n tiles in a row, so find_neighbor_pairs sees n-1 X-seams."""
    return [_tile(f"X{i * pitch:.2f}_Y0.00", x=i * pitch) for i in range(n)]


def _rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def _report(tiles, params, **kwargs):
    return rr.build_report(
        tiles=tiles, params=params, voxel_size_um=_VOXEL, **kwargs
    )


class TestTileCsv:
    def test_the_header_is_the_contract(self):
        report = _report(_row_grid(1), [_affine()])
        assert _rows(rr.tile_rows_csv(report))[0].keys() == dict.fromkeys(
            rr.TILE_CSV_HEADER
        ).keys()

    def test_one_row_per_tile_in_input_order(self):
        tiles = [_tile("A"), _tile("B", x=0.9), _tile("C", x=1.8)]
        report = _report(tiles, [_affine()] * 3)
        assert [r["tile_name"] for r in _rows(rr.tile_rows_csv(report))] == [
            "A",
            "B",
            "C",
        ]

    def test_micron_and_voxel_columns_describe_the_same_shift(self):
        report = _report(_row_grid(1), [_affine(dz=25.0, dy=4.0, dx=-1.6)])
        row = _rows(rr.tile_rows_csv(report))[0]
        assert float(row["dz_um"]) == pytest.approx(25.0)
        assert float(row["dz_frames"]) == pytest.approx(2.5)  # 25 / 10 µm
        assert float(row["dy_px"]) == pytest.approx(5.0)  # 4 / 0.8 µm
        assert float(row["dx_px"]) == pytest.approx(-2.0)

    def test_a_leading_time_axis_is_tolerated(self):
        # Real multiview-stitcher params are (t, 4, 4); tests hand it a bare 4x4.
        report = _report(_row_grid(1), [_affine(dz=7.0, with_time_axis=True)])
        assert float(_rows(rr.tile_rows_csv(report))[0]["dz_um"]) == pytest.approx(7.0)

    def test_the_folder_name_is_the_row_key(self):
        # RawTileInfo.tile_index is None for folder-layout acquisitions, so it
        # cannot be the identifier; folder.name always exists.
        tiles = [SimpleNamespace(folder=SimpleNamespace(name="X1.00_Y2.00"),
                                 x_mm=1.0, y_mm=2.0, z_min_mm=0.0, tile_index=None)]
        report = _report(tiles, [_affine()])
        assert _rows(rr.tile_rows_csv(report))[0]["tile_name"] == "X1.00_Y2.00"

    def test_a_tile_without_a_param_is_noted_not_silently_zero(self):
        report = _report(_row_grid(2), [_affine(dx=1.0)])  # only one param
        rows = _rows(rr.tile_rows_csv(report))
        assert "no registration param" in rows[1]["note"]

    def test_an_empty_report_still_writes_the_header(self):
        report = _report([], [])
        assert rr.tile_rows_csv(report).strip() == ",".join(rr.TILE_CSV_HEADER)


class TestClampHonesty:
    def _clamped(self, **flags):
        return SimpleNamespace(
            index=0, dz_um=97.0, dy_um=2.0, dx_um=71.0, whole_matrix=False, **flags
        )

    def test_clamped_flags_are_independent_per_axis(self):
        report = _report(
            _row_grid(1),
            [_affine()],
            clamp_records=[self._clamped(clamped_z=True, clamped_xy=False)],
        )
        row = _rows(rr.tile_rows_csv(report))[0]
        assert (row["clamped_z"], row["clamped_y"], row["clamped_x"]) == (
            "yes",
            "no",
            "no",
        )

    def test_lateral_clamping_marks_both_x_and_y(self):
        # They revert together because they come out of one correlation peak;
        # the report must not imply Y was independently trusted.
        report = _report(
            _row_grid(1),
            [_affine()],
            clamp_records=[self._clamped(clamped_z=False, clamped_xy=True)],
        )
        row = _rows(rr.tile_rows_csv(report))[0]
        assert (row["clamped_y"], row["clamped_x"]) == ("yes", "yes")

    def test_the_rejected_value_is_preserved_not_replaced_by_the_zero(self):
        # "We measured +97 µm and did not believe it" is a different finding
        # from "we measured nothing", and only one of them points at a cause.
        report = _report(
            _row_grid(1),
            [_affine()],
            clamp_records=[self._clamped(clamped_z=True, clamped_xy=False)],
        )
        row = _rows(rr.tile_rows_csv(report))[0]
        assert float(row["dz_um_before_clamp"]) == pytest.approx(97.0)
        assert float(row["dz_um"]) == pytest.approx(0.0)

    def test_an_unclamped_tile_leaves_the_before_columns_empty(self):
        report = _report(
            _row_grid(1),
            [_affine(dx=1.0)],
            clamp_records=[self._clamped(clamped_z=False, clamped_xy=False)],
        )
        row = _rows(rr.tile_rows_csv(report))[0]
        assert row["dz_um_before_clamp"] == ""

    def test_the_text_headlines_the_clamp_and_says_not_measured(self):
        report = _report(
            _row_grid(1),
            [_affine()],
            clamp_records=[self._clamped(clamped_z=True, clamped_xy=False)],
        )
        text = rr.format_report_text(report)
        assert "hit a clamp bound" in text
        assert "NOT MEASURED" in text

    def test_clamped_tiles_are_excluded_from_the_summary_statistics(self):
        # A clamped tile contributes a 0, and averaging that in would make a
        # badly misaligned run look better than a mildly misaligned one.
        tiles = _row_grid(2)
        report = _report(
            tiles,
            [_affine(dx=4.0), _affine()],
            clamp_records=[
                SimpleNamespace(
                    index=1,
                    dz_um=0.0,
                    dy_um=0.0,
                    dx_um=99.0,
                    clamped_xy=True,
                    clamped_z=False,
                    whole_matrix=False,
                )
            ],
        )
        assert "1 unclamped tiles" in rr.format_report_text(report)


class TestSeamCsv:
    def test_the_header_is_the_contract(self):
        report = _report(_row_grid(2), [_affine()] * 2)
        assert _rows(rr.seam_rows_csv(report))[0].keys() == dict.fromkeys(
            rr.SEAM_CSV_HEADER
        ).keys()

    def test_every_expected_pair_gets_a_row_even_when_unregistered(self):
        # Driven by the stage grid, not the graph. A pair registration threw
        # away is the row most worth having, and it is not in the graph.
        tiles = _row_grid(4)  # 3 X-seams
        report = _report(tiles, [_affine()] * 4)
        assert len(_rows(rr.seam_rows_csv(report))) == 3

    def test_status_counts_partition_the_expected_pairs(self):
        report = _report(_row_grid(4), [_affine()] * 4)
        total = sum(
            report.count(s)
            for s in (
                rr.STATUS_REGISTERED,
                rr.STATUS_PRUNED,
                rr.STATUS_BELOW_QUALITY,
                rr.STATUS_DROPPED,
                rr.STATUS_NOT_RUN,
            )
        )
        assert total == report.n_expected_pairs

    def test_an_unknown_quality_is_an_empty_cell_not_a_zero(self):
        # 0.0 means "the overlap did not correlate at all", which is a finding.
        # An empty cell means nobody could recover the number. Writing 0 for
        # both destroys the distinction in every spreadsheet downstream.
        report = _report(_row_grid(2), [_affine()] * 2)
        assert _rows(rr.seam_rows_csv(report))[0]["quality"] == ""

    def test_the_overlap_fraction_comes_from_the_stage_positions(self):
        tiles = _row_grid(2, pitch=0.9)
        report = _report(
            tiles, [_affine()] * 2, frame_extent_um={"x": 1000.0, "y": 1000.0}
        )
        assert float(_rows(rr.seam_rows_csv(report))[0]["overlap_frac"]) == (
            pytest.approx(0.1)
        )

    def test_an_empty_report_still_writes_the_header(self):
        report = _report([], [])
        assert rr.seam_rows_csv(report).strip() == ",".join(rr.SEAM_CSV_HEADER)


class TestSkippedRun:
    def test_it_still_lists_every_tile_and_seam(self):
        report = rr.skipped_report("skip_registration is on", tiles=_row_grid(3))
        assert len(_rows(rr.tile_rows_csv(report))) == 3
        assert len(_rows(rr.seam_rows_csv(report))) == 2

    def test_every_seam_reads_not_run(self):
        report = rr.skipped_report("gated off", tiles=_row_grid(3))
        assert {s.status for s in report.seams} == {rr.STATUS_NOT_RUN}

    def test_the_text_names_the_reason(self):
        text = rr.format_report_text(
            rr.skipped_report("measured overlap 3% is below the 5% gate")
        )
        assert "DID NOT RUN" in text
        assert "below the 5% gate" in text

    def test_it_says_unmeasured_rather_than_zero(self):
        # A file of zeroes reads as "registration found nothing to fix".
        text = rr.format_report_text(rr.skipped_report("disabled"))
        assert "not zero, unmeasured" in text


class TestZRefineSummary:
    def test_a_rejected_correction_is_reported_as_a_floor(self):
        report = _report(
            _row_grid(1),
            [_affine()],
            z_refine=rr.ZRefineSummary(
                ran=True,
                binning={"z": 1},
                upsample_factor=10,
                search_range_um=40.0,
                n_hit_search_limit=3,
            ),
        )
        text = rr.format_report_text(report)
        assert "AT the search limit" in text
        assert "floors, not measurements" in text

    def test_a_disabled_pass_says_why(self):
        report = _report(
            _row_grid(1),
            [_affine()],
            z_refine=rr.ZRefineSummary(ran=False, reason="overlap below the gate"),
        )
        assert "overlap below the gate" in rr.format_report_text(report)


class TestWriteReport:
    def test_it_puts_three_files_in_the_given_directory(self, tmp_path):
        report = _report(_row_grid(2), [_affine()] * 2)
        written = rr.write_report(tmp_path, report)
        assert set(written) == {"tiles_csv", "seams_csv", "text"}
        assert {p.name for p in tmp_path.iterdir()} == {
            rr.TILE_CSV_NAME,
            rr.SEAM_CSV_NAME,
            rr.TEXT_NAME,
        }

    def test_json_is_opt_in(self, tmp_path):
        report = _report(_row_grid(2), [_affine()] * 2)
        written = rr.write_report(tmp_path, report, write_json=True)
        assert (tmp_path / rr.JSON_NAME).exists() and "json" in written

    def test_an_unwritable_destination_warns_and_does_not_raise(self, tmp_path):
        # Evidence about a run must never be the thing that fails the run.
        blocked = tmp_path / "not_a_dir"
        blocked.write_text("I am a file")
        warnings = []
        written = rr.write_report(
            blocked,
            _report(_row_grid(1), [_affine()]),
            logger=SimpleNamespace(warning=warnings.append),
        )
        assert written == {}
        assert len(warnings) == 3


class TestRobustness:
    def test_a_malformed_reg_dict_never_raises(self):
        report = _report(
            _row_grid(2),
            [_affine()] * 2,
            reg_dict={"pairwise_registration": None, "groupwise_resolution": 7},
        )
        assert report.n_expected_pairs == 1

    def test_a_param_that_cannot_be_read_is_noted(self):
        report = _report(_row_grid(1), ["not an affine"])
        assert "no registration param" in report.tiles[0].note

    def test_translation_from_param_returns_none_rather_than_guessing(self):
        assert rr.translation_from_param(object()) is None

    def test_json_round_trips_the_status_counts(self):
        report = _report(_row_grid(3), [_affine()] * 3)
        payload = rr.report_to_json(report, acquisition="acq")
        assert payload["acquisition"] == "acq"
        assert sum(payload["seam_status_counts"].values()) == report.n_expected_pairs
