"""The stitched file is named after the dataset, not after whatever is above it.

Two layouts put the descriptive name in different places:

* **Subfolder-per-tile** (multi-acquisition) — ``<sample>/<date>/X4.00_Y12.00/``.
  The acquisition folder is a bare date, so the sample folder is prepended:
  ``2026-04-05.ome.zarr`` alone does not say which sample it is, and two
  samples stitched into one output folder would collide.
* **Flat** (Single Workflow, the C++ server's native output) —
  ``<whatever>/<dataset>/*.raw``. The acquisition folder IS the dataset.

Prepending the parent unconditionally — correct for the first — gave the second
a file named after a directory that says nothing about the data, sitting inside
a ``<dataset>_stitched`` folder that already had the right name. The reported
symptom: the parent folder's name used as the stitched file name, where the
stitched FOLDER's name was what it should have been.

Run: python -m pytest tests/test_output_basename_layout.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flamingo_stitcher.pipeline import (  # noqa: E402
    StitchingConfig,
    StitchingPipeline,
    acquisition_is_flat,
)


def _pipeline(**overrides):
    return StitchingPipeline(StitchingConfig(**overrides))


def _flat(tmp_path, dataset="MyDataset", parent="RawData", ext=".raw"):
    """<parent>/<dataset>/*.raw — the Single Workflow layout."""
    acq = tmp_path / parent / dataset
    acq.mkdir(parents=True)
    (acq / f"X000_Y000{ext}").write_bytes(b"")
    (acq / f"X001_Y000{ext}").write_bytes(b"")
    return acq


def _foldered(tmp_path, sample="OrganoidV2", date="2026-04-05"):
    """<sample>/<date>/X../ — the multi-acquisition layout."""
    acq = tmp_path / sample / date
    for name in ("X4.00_Y12.00", "X6.00_Y12.00"):
        (acq / name).mkdir(parents=True)
        (acq / name / "tile.raw").write_bytes(b"")
    return acq


class TestTheLayoutIsRecognised:
    def test_tile_files_directly_inside_is_flat(self, tmp_path):
        assert acquisition_is_flat(_flat(tmp_path))

    def test_coordinate_named_subfolders_are_not(self, tmp_path):
        assert not acquisition_is_flat(_foldered(tmp_path))

    def test_a_dated_flat_acquisition_is_still_flat(self, tmp_path):
        # discover_flat_tiles also reads <dataset>/<date>/*.raw. The dataset
        # name is still the acquisition folder's own, so the rule is unchanged.
        acq = tmp_path / "MyDataset"
        (acq / "2026-04-05").mkdir(parents=True)
        (acq / "2026-04-05" / "X000_Y000.raw").write_bytes(b"")
        assert acquisition_is_flat(acq)

    @pytest.mark.parametrize("ext", [".raw", ".tif", ".tiff", ".btf"])
    def test_every_tile_extension_counts(self, tmp_path, ext):
        assert acquisition_is_flat(_flat(tmp_path / ext.strip("."), ext=ext))

    def test_an_empty_directory_is_not_flat(self, tmp_path):
        # Nothing to go on. The old behaviour is the safer default: a redundant
        # prefix is ugly, a missing one can collide two samples' dates.
        empty = tmp_path / "nothing"
        empty.mkdir()
        assert not acquisition_is_flat(empty)

    def test_a_missing_directory_is_not_an_exception(self, tmp_path):
        # Naming must never be the thing that fails a run.
        assert not acquisition_is_flat(tmp_path / "gone")


class TestFlatAcquisitionsAreNamedAfterThemselves:
    def test_the_parent_does_not_reach_the_filename(self, tmp_path):
        acq = _flat(tmp_path, dataset="MyDataset", parent="RawData")
        assert _pipeline()._build_output_basename(acq) == "MyDataset"

    def test_the_file_matches_the_stitched_folder(self, tmp_path):
        # The GUI writes into "<acq name>_stitched". The store inside it should
        # carry the same name — that mismatch is what was reported.
        acq = _flat(tmp_path, dataset="Embryo_47")
        folder = f"{acq.name}_stitched"
        base = _pipeline()._build_output_basename(acq)
        assert folder.startswith(base)

    def test_processing_tags_are_still_appended(self, tmp_path):
        # Different settings must still resolve to different filenames, or a
        # re-run silently clobbers a result produced another way.
        acq = _flat(tmp_path, dataset="MyDataset")
        assert (
            _pipeline(destripe=True)._build_output_basename(acq)
            == "MyDataset_destripe"
        )

    def test_a_dataset_directly_under_a_drive_root_is_unchanged(self, tmp_path):
        acq = tmp_path / "MyDataset"
        acq.mkdir()
        (acq / "X000_Y000.raw").write_bytes(b"")
        assert _pipeline()._build_output_basename(acq) == "MyDataset"


class TestFolderedAcquisitionsStillGetTheirSampleName:
    """Unchanged: a bare date needs the sample folder to mean anything."""

    def test_the_sample_is_prepended(self, tmp_path):
        acq = _foldered(tmp_path, sample="OrganoidV2", date="2026-04-05")
        assert _pipeline()._build_output_basename(acq) == "OrganoidV2_2026-04-05"

    def test_tags_are_appended_after_it(self, tmp_path):
        acq = _foldered(tmp_path)
        assert _pipeline(destripe=True)._build_output_basename(acq).endswith(
            "_destripe"
        )

    def test_two_samples_on_the_same_date_do_not_collide(self, tmp_path):
        # The reason the prefix exists. Stitching both into one output folder
        # would otherwise write one over the other.
        a = _foldered(tmp_path / "a", sample="OrganoidV2", date="2026-04-05")
        b = _foldered(tmp_path / "b", sample="OrganoidV7", date="2026-04-05")
        pipe = _pipeline()
        assert pipe._build_output_basename(a) != pipe._build_output_basename(b)


class TestTheExistingOutputGuardAgrees:
    """`expected_output_path` exists so a re-run can detect a prior result.

    It shares `_build_output_basename` with the run, so the two cannot drift —
    but only if the naming stays a pure function of the path and settings.
    """

    def test_it_predicts_the_flat_name(self, tmp_path):
        acq = _flat(tmp_path, dataset="MyDataset", parent="RawData")
        out = _pipeline().expected_output_path(acq, tmp_path / "out")
        assert out.name.startswith("MyDataset")
        assert "RawData" not in out.name

    def test_it_predicts_the_foldered_name(self, tmp_path):
        acq = _foldered(tmp_path)
        out = _pipeline().expected_output_path(acq, tmp_path / "out")
        assert out.name.startswith("OrganoidV2_2026-04-05")

    def test_it_is_stable_across_calls(self, tmp_path):
        acq = _flat(tmp_path)
        pipe = _pipeline()
        first = pipe.expected_output_path(acq, tmp_path / "out")
        assert first == pipe.expected_output_path(acq, tmp_path / "out")
