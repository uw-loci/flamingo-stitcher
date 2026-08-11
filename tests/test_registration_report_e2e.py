"""Registration, end to end, through the real pipeline on a synthetic acquisition.

The unit tests pin the report's shape; this file pins that it is actually
produced, lands next to the stitched output, and survives every path — in-memory
and streaming, registration on and off, gated and ungated.

The last test is the one that matters most: a deliberate Z error injected into a
tile's pixels while its metadata claims the shared start depth, which is exactly
what a Z stage error looks like on disk. If registration cannot recover that,
none of the rest is worth having.

Run: python3 -m pytest tests/test_registration_report_e2e.py -q
"""

from __future__ import annotations

import csv
import unittest

import pytest

from _synth_acq import write_synth_acquisition

try:
    from flamingo_stitcher import registration_report as rr
    from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

    import multiview_stitcher  # noqa: F401

    _HAVE = True
except Exception:  # pragma: no cover - environment without the optional deps
    _HAVE = False


def _config(**overrides):
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.skip_registration = False
    cfg.streaming_mode = True
    cfg.output_format = "ome-tiff"
    cfg.resource_guard_enabled = False
    cfg.flat_field_correction = False
    cfg.reg_channel = 1
    cfg.output_chunksize = {"z": 4, "y": 16, "x": 16}
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _run(acq, out, cfg):
    StitchingPipeline(cfg).run(acq, out)
    return out


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@unittest.skipUnless(_HAVE, "multiview-stitcher not installed")
class TestTheReportIsWritten(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_three_files_land_beside_stitch_metadata(self):
        acq = write_synth_acquisition(
            self.tmp / "acq", grid=(2, 2), overlap=0.2, n_planes=8, frame_size=(48, 48)
        )
        out = _run(acq, self.tmp / "out", _config())
        # Anchored on the metadata file: the report has to travel with the store.
        assert (out / "stitch_metadata.json").exists()
        for name in (rr.TILE_CSV_NAME, rr.SEAM_CSV_NAME, rr.TEXT_NAME):
            assert (out / name).exists(), name

    def test_the_tile_csv_has_the_documented_header_and_one_row_per_tile(self):
        acq = write_synth_acquisition(
            self.tmp / "acq", grid=(2, 2), overlap=0.2, n_planes=8, frame_size=(48, 48)
        )
        out = _run(acq, self.tmp / "out", _config())
        rows = _rows(out / rr.TILE_CSV_NAME)
        assert list(rows[0]) == list(rr.TILE_CSV_HEADER)
        assert len(rows) == 4

    def test_the_seam_csv_covers_every_stage_neighbour_pair(self):
        # A 2x2 grid has 4 adjacencies. Rows come from the stage grid, so this
        # count must not depend on how many seams registration actually used.
        acq = write_synth_acquisition(
            self.tmp / "acq", grid=(2, 2), overlap=0.2, n_planes=8, frame_size=(48, 48)
        )
        out = _run(acq, self.tmp / "out", _config())
        assert len(_rows(out / rr.SEAM_CSV_NAME)) == 4

    def test_the_in_memory_path_writes_it_too(self):
        acq = write_synth_acquisition(
            self.tmp / "acq", grid=(2, 2), overlap=0.2, n_planes=8, frame_size=(48, 48)
        )
        out = _run(acq, self.tmp / "out", _config(streaming_mode=False))
        assert (out / rr.TILE_CSV_NAME).exists()

    def test_a_skipped_run_still_writes_a_file_naming_the_reason(self):
        # The ambiguity this closes: for months 'Skip registration' was on and
        # the output gave no sign of it. A file of zeroes would be no better, so
        # the text has to say the tiles were never registered.
        acq = write_synth_acquisition(
            self.tmp / "acq", grid=(2, 2), overlap=0.2, n_planes=8, frame_size=(48, 48)
        )
        out = _run(acq, self.tmp / "out", _config(skip_registration=True))
        text = (out / rr.TEXT_NAME).read_text(encoding="utf-8")
        assert "DID NOT RUN" in text
        assert "Skip registration" in text
        assert len(_rows(out / rr.TILE_CSV_NAME)) == 4

    def test_turning_the_report_off_writes_nothing_but_keeps_the_metadata(self):
        acq = write_synth_acquisition(
            self.tmp / "acq", grid=(2, 2), overlap=0.2, n_planes=8, frame_size=(48, 48)
        )
        out = _run(
            acq, self.tmp / "out", _config(registration_report_enabled=False)
        )
        assert (out / "stitch_metadata.json").exists()
        assert not (out / rr.TILE_CSV_NAME).exists()

    def test_json_is_written_when_asked_for(self):
        acq = write_synth_acquisition(
            self.tmp / "acq", grid=(2, 2), overlap=0.2, n_planes=8, frame_size=(48, 48)
        )
        out = _run(acq, self.tmp / "out", _config(registration_report_json=True))
        assert (out / rr.JSON_NAME).exists()


@unittest.skipUnless(_HAVE, "multiview-stitcher not installed")
class TestTheOverlapGate(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_too_little_overlap_skips_registration_and_says_so(self):
        acq = write_synth_acquisition(
            self.tmp / "acq", grid=(2, 1), overlap=0.02, n_planes=8, frame_size=(48, 48)
        )
        out = _run(acq, self.tmp / "out", _config())
        text = (out / rr.TEXT_NAME).read_text(encoding="utf-8")
        assert "DID NOT RUN" in text
        assert "overlap" in text.lower()
        assert {r["status"] for r in _rows(out / rr.SEAM_CSV_NAME)} == {
            rr.STATUS_NOT_RUN
        }

    def test_enough_overlap_is_not_gated(self):
        # Grid-aligned numbers on purpose. Tile folder names quantize the stage
        # position to 0.01 mm, so at 48 px x 0.406 µm a "20%" overlap lands as a
        # 1 px GAP once discovery reads it back. 64 px at 1.0 µm with a 0.375
        # overlap steps exactly 0.04 mm, so the requested overlap survives the
        # round trip.
        acq = write_synth_acquisition(
            self.tmp / "acq",
            grid=(2, 1),
            overlap=0.375,
            n_planes=8,
            frame_size=(64, 64),
            pixel_size_um=1.0,
        )
        out = _run(acq, self.tmp / "out", _config(pixel_size_um=1.0))
        assert "DID NOT RUN" not in (out / rr.TEXT_NAME).read_text(encoding="utf-8")

    def test_the_gate_threshold_is_configurable(self):
        acq = write_synth_acquisition(
            self.tmp / "acq", grid=(2, 1), overlap=0.1, n_planes=8, frame_size=(48, 48)
        )
        out = _run(
            acq, self.tmp / "out", _config(min_registration_overlap_frac=0.5)
        )
        assert "DID NOT RUN" in (out / rr.TEXT_NAME).read_text(encoding="utf-8")


@unittest.skipUnless(_HAVE, "multiview-stitcher not installed")
class TestZRecovery(unittest.TestCase):
    """Can registration find a Z error that is really there?

    Registration only ever sees stage metadata claiming both tiles start at the
    same depth; one tile's pixels come from three planes deeper. That is the
    disagreement, and recovering it is the point of this work.

    The phantom parameters are load-bearing, and each one cost a wrong answer
    before it was right:

      * ``pixel_size_um=1.0`` with ``overlap=0.375`` on a 64 px frame steps
        exactly 0.04 mm, so the 0.01 mm quantization in the tile folder name
        does not shift the content relative to the metadata. A 2 px mismatch
        there is enough to destroy a bead correlation.
      * ``tile_orientation="identity"``: the default empty value keeps a legacy
        X flip, which mirrors tile content relative to stage placement. That is
        the Liara ghost-duplicate failure, and no registration can fix a
        reflection — it has to be excluded, not registered around.
      * ``z_texture=True``: the default phantom is nearly separable in Z.
    """

    OFFSET_PLANES = 3
    Z_STEP_UM = 5.0
    EXPECTED_UM = OFFSET_PLANES * Z_STEP_UM  # 15.0

    def setUp(self):
        import tempfile
        from pathlib import Path

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _acq(self):
        return write_synth_acquisition(
            self.tmp / "acq",
            grid=(1, 2),
            overlap=0.375,
            n_planes=32,
            frame_size=(64, 64),
            pixel_size_um=1.0,
            z_step_um=self.Z_STEP_UM,
            z_texture=True,
            tile_z_offset_planes={(1, 0): self.OFFSET_PLANES},
        )

    def _z_config(self, **overrides):
        return _config(
            downsample_z=1,
            downsample_xy=1,
            pixel_size_um=1.0,
            tile_orientation="identity",
            **overrides,
        )

    def _dz_um(self, out):
        return [float(r["dz_um"]) for r in _rows(out / rr.TILE_CSV_NAME)]

    def test_registration_recovers_the_injected_offset(self):
        # The headline claim: with registration simply turned ON, the 3-plane
        # error is found. No refinement needed at the default Z binning, which
        # already resolves one plane.
        out = _run(self._acq(), self.tmp / "out", self._z_config())
        spread = max(self._dz_um(out)) - min(self._dz_um(out))
        assert abs(spread - self.EXPECTED_UM) <= self.Z_STEP_UM, (
            f"expected ~{self.EXPECTED_UM} µm of Z disagreement, got {spread}"
        )

    def test_a_coarsely_binned_first_pass_misses_it(self):
        # Guards the premise of the next test. At z binning 8 the first pass can
        # only express multiples of 40 µm, so a 15 µm error rounds to nothing.
        out = _run(
            self._acq(),
            self.tmp / "out",
            self._z_config(registration_binning={"z": 8, "y": 4, "x": 4}),
        )
        spread = max(self._dz_um(out)) - min(self._dz_um(out))
        assert spread < self.Z_STEP_UM, f"expected the coarse pass to miss it, got {spread}"

    def test_the_refinement_recovers_what_the_coarse_pass_missed(self):
        out = _run(
            self._acq(),
            self.tmp / "out",
            self._z_config(
                registration_binning={"z": 8, "y": 4, "x": 4},
                registration_z_refine=True,
            ),
        )
        spread = max(self._dz_um(out)) - min(self._dz_um(out))
        assert abs(spread - self.EXPECTED_UM) <= self.Z_STEP_UM, (
            f"refinement should have recovered ~{self.EXPECTED_UM} µm, got {spread}"
        )
        assert "Z refinement:" in (out / rr.TEXT_NAME).read_text(encoding="utf-8")

    def test_the_refinement_adds_nothing_when_the_first_pass_was_already_right(self):
        # It must not invent a correction to justify itself.
        out = _run(
            self._acq(), self.tmp / "out", self._z_config(registration_z_refine=True)
        )
        spread = max(self._dz_um(out)) - min(self._dz_um(out))
        assert abs(spread - self.EXPECTED_UM) <= self.Z_STEP_UM

    def test_z_refine_off_says_it_did_not_run(self):
        out = _run(self._acq(), self.tmp / "out", self._z_config())
        assert "Z refinement: not run" in (out / rr.TEXT_NAME).read_text(
            encoding="utf-8"
        )

    def test_a_failing_refinement_pass_does_not_cost_the_first_pass(self):
        # The refinement is an improvement, not a dependency. If pass 2 throws,
        # the run must still be registered by pass 1 rather than fall all the
        # way back to stage positions.
        cfg = self._z_config(registration_z_refine=True)
        cfg.registration_z_refine_binning = {"z": "not an int"}
        out = _run(self._acq(), self.tmp / "out", cfg)
        text = (out / rr.TEXT_NAME).read_text(encoding="utf-8")
        assert "DID NOT RUN" not in text
        assert "Z refinement: not run" in text
        spread = max(self._dz_um(out)) - min(self._dz_um(out))
        assert abs(spread - self.EXPECTED_UM) <= self.Z_STEP_UM


@unittest.skipUnless(_HAVE, "multiview-stitcher not installed")
class TestDefaultsAreUnchanged(unittest.TestCase):
    def test_the_new_registration_options_are_off_by_default(self):
        # Nothing here may change an existing run's output. The refinement is
        # opt-in and the upsample sentinel leaves MVS's own default alone.
        cfg = StitchingConfig()
        assert cfg.registration_z_refine is False
        assert cfg.registration_upsample_factor == 0
        assert StitchingPipeline._pairwise_reg_kwargs(0) is None
        assert StitchingPipeline._pairwise_reg_kwargs(10) == {"upsample_factor": 10}
