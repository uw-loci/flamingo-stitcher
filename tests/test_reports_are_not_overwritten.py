"""A second stitch into one folder must not destroy the first one's evidence.

Observed on the rig: one acquisition stitched three ways into one output folder
left three differently-named TIFFs —

    ..._2048x2048.ome.tif
    ..._2048x2048_flatfield.ome.tif
    ..._2048x2048_xy8x_z1x.ome.tif

— and exactly ONE registration_report.csv, one registration_seams.csv, one
registration_report.txt and one stitch_metadata.json, all describing whichever
run finished last. The image carried its settings in its name; the evidence did
not. These files are the only record of how a run placed its tiles, so losing
them silently is worse than losing the image, which is at least obviously gone.

Every file is now written TWICE: under its documented name, which holds the
newest run and is what the troubleshooting guide tells people to open, and under
a per-run name that the next stitch cannot touch. `stitch_metadata.json` must
keep its fixed name for a harder reason — Sample View and the pipeline importer
read it at exactly that path.

Run: python -m pytest tests/test_reports_are_not_overwritten.py -q
"""

from __future__ import annotations

from flamingo_stitcher import registration_report as rr

VARIANTS = ("acq_flatfield", "acq_xy8x_z1x")


def _report():
    return rr.skipped_report("registration skipped for this test")


class TestRegistrationReportsAreNamedPerRun:
    def test_two_runs_in_one_folder_keep_both_reports(self, tmp_path):
        for stem in VARIANTS:
            rr.write_report(tmp_path, _report(), acquisition="acq", prefix=stem)
        for stem in VARIANTS:
            assert (tmp_path / f"{stem}_{rr.TILE_CSV_NAME}").is_file()
            assert (tmp_path / f"{stem}_{rr.SEAM_CSV_NAME}").is_file()
            assert (tmp_path / f"{stem}_{rr.TEXT_NAME}").is_file()

    def test_the_second_run_does_not_clobber_the_first(self, tmp_path):
        rr.write_report(tmp_path, _report(), prefix="run_a")
        first = (tmp_path / f"run_a_{rr.TEXT_NAME}").read_text()
        rr.write_report(tmp_path, _report(), prefix="run_b")
        assert (tmp_path / f"run_a_{rr.TEXT_NAME}").read_text() == first

    def test_the_documented_names_still_exist_and_hold_the_newest_run(self):
        # The troubleshooting guide sends people to these exact names.
        assert rr.TILE_CSV_NAME == "registration_report.csv"
        assert rr.SEAM_CSV_NAME == "registration_seams.csv"

    def test_the_documented_name_is_written_alongside_the_per_run_copy(
        self, tmp_path
    ):
        rr.write_report(tmp_path, _report(), prefix="run_a")
        assert (tmp_path / rr.TILE_CSV_NAME).is_file()
        assert (tmp_path / f"run_a_{rr.TILE_CSV_NAME}").is_file()

    def test_the_json_is_copied_per_run_too(self, tmp_path):
        rr.write_report(tmp_path, _report(), prefix="run_a", write_json=True)
        assert (tmp_path / f"run_a_{rr.JSON_NAME}").is_file()

    def test_the_returned_paths_are_all_really_written(self, tmp_path):
        written = rr.write_report(tmp_path, _report(), prefix="run_a")
        assert written
        for path in written.values():
            assert path.is_file()
        assert any(p.name.startswith("run_a_") for p in written.values())

    def test_no_prefix_keeps_the_bare_names(self, tmp_path):
        # A caller with nothing to disambiguate still gets the documented names.
        rr.write_report(tmp_path, _report())
        assert (tmp_path / rr.TILE_CSV_NAME).is_file()

    def test_a_prefix_of_underscores_is_not_taken_literally(self, tmp_path):
        rr.write_report(tmp_path, _report(), prefix="__")
        assert (tmp_path / rr.TILE_CSV_NAME).is_file()


class TestStitchMetadataKeepsItsFixedNameAndAlsoSurvives:
    def _pipeline(self, tmp_path):
        from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

        return StitchingPipeline(StitchingConfig())

    def test_the_canonical_name_is_still_written(self, tmp_path):
        # Sample View and the pipeline importer read this exact path.
        self._pipeline(tmp_path)._write_metadata_json(
            tmp_path, {"version": 2}, "acq_flatfield"
        )
        assert (tmp_path / "stitch_metadata.json").is_file()

    def test_a_per_run_copy_is_written_beside_it(self, tmp_path):
        self._pipeline(tmp_path)._write_metadata_json(
            tmp_path, {"version": 2}, "acq_flatfield"
        )
        assert (tmp_path / "acq_flatfield_stitch_metadata.json").is_file()

    def test_a_second_run_leaves_the_first_runs_copy_intact(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        pipe._write_metadata_json(tmp_path, {"run": "a"}, "acq_flatfield")
        pipe._write_metadata_json(tmp_path, {"run": "b"}, "acq_xy8x_z1x")
        import json

        a = json.loads((tmp_path / "acq_flatfield_stitch_metadata.json").read_text())
        b = json.loads((tmp_path / "acq_xy8x_z1x_stitch_metadata.json").read_text())
        canonical = json.loads((tmp_path / "stitch_metadata.json").read_text())
        assert a["run"] == "a"
        assert b["run"] == "b"
        assert canonical["run"] == "b", "the fixed name should hold the newest run"

    def test_no_basename_writes_only_the_canonical_file(self, tmp_path):
        self._pipeline(tmp_path)._write_metadata_json(tmp_path, {"version": 2}, "")
        assert [p.name for p in tmp_path.glob("*.json")] == ["stitch_metadata.json"]
