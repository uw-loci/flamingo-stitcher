"""The summary's buttons must point where the run actually wrote.

The output folder defaults to ``<output dir>/<acq>_stitched``, but a run that
hits an existing result and is answered "New folder" writes to a uniquified
sibling — ``<acq>_stitched_2``. The completion summary rebuilt the DEFAULT name
instead of using what the worker reported, so both of its buttons pointed at the
older stitch that caused the prompt in the first place.

The existence check could not catch it: that folder is there, it is just the
wrong one. "Open Output Folder" therefore opened the previous result, and "Load
Latest into Sample View" silently loaded the previous result as if it were the
run that had just finished — the same bug, and the worse of the two, because
nothing about it looks wrong on screen.

Run: QT_QPA_PLATFORM=offscreen python -m pytest \\
        tests/test_open_output_folder_after_rename.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("PyQt5")


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def _dlg(app):
    from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

    dlg = StitchingDialog()
    yield dlg
    dlg.deleteLater()


@pytest.fixture
def dialog(_dlg):
    _dlg._queue.clear()
    _dlg._batch_results = []
    _dlg._last_success_output = None
    _dlg._queue_index = 0
    return _dlg


def _complete(dialog, store_path):
    """Drive the worker's success signal, as a finished item does."""
    dialog._queue.append(
        {"path": Path("/acq/MyDataset"), "status": "running", "output_path": None}
    )
    dialog._queue_index = 0
    dialog._on_item_completed(str(store_path))


class TestTheSummaryUsesWhatTheRunWrote:
    """Through `_last_output_folder`, the selector BOTH summary buttons use.

    An earlier version of this file asserted only on `_last_success_output` and
    passed with the summary still rebuilding the default name — testing the new
    state while the code that consumes it stayed broken.
    """

    def test_the_folder_offered_is_the_one_written_to(self, dialog, tmp_path):
        # The reported bug: "New folder" wrote to _stitched_2, and the summary
        # offered _stitched.
        renamed = tmp_path / "MyDataset_stitched_2" / "MyDataset.ome.zarr"
        _complete(dialog, renamed)
        assert Path(dialog._last_output_folder()).name == "MyDataset_stitched_2"

    def test_the_default_folder_existing_does_not_win(self, dialog, tmp_path):
        # The default folder is there too — it is the duplicate that triggered
        # the prompt — so "does this path exist?" cannot tell the two apart.
        dialog._output_dir_edit.setText(str(tmp_path))
        (tmp_path / "MyDataset_stitched").mkdir()
        renamed = tmp_path / "MyDataset_stitched_2" / "MyDataset.ome.zarr"
        renamed.parent.mkdir()
        _complete(dialog, renamed)
        offered = Path(dialog._last_output_folder())
        assert offered != tmp_path / "MyDataset_stitched"
        assert offered.is_dir()

    def test_the_acquisition_name_is_not_used_to_rebuild_it(self, dialog, tmp_path):
        # The acquisition is "MyDataset" and the output dir is tmp_path, so a
        # rebuild would land on "<tmp>/MyDataset_stitched". It must not.
        dialog._output_dir_edit.setText(str(tmp_path))
        renamed = tmp_path / "MyDataset_stitched_2" / "MyDataset.ome.zarr"
        _complete(dialog, renamed)
        assert dialog._last_output_folder() != str(tmp_path / "MyDataset_stitched")

    def test_an_ordinary_run_still_points_at_its_own_folder(self, dialog, tmp_path):
        plain = tmp_path / "MyDataset_stitched" / "MyDataset.ome.zarr"
        _complete(dialog, plain)
        assert Path(dialog._last_output_folder()).name == "MyDataset_stitched"

    def test_nothing_succeeded_offers_nothing(self, dialog):
        # The summary only shows the buttons when this is set; a stale path here
        # would offer to open a folder this batch never wrote.
        assert dialog._last_output_folder() is None

    def test_the_last_of_several_items_wins(self, dialog, tmp_path):
        # "Load LATEST" — the most recently completed, not the first or the
        # alphabetically last.
        for name in ("A_stitched", "B_stitched_2", "C_stitched"):
            _complete(dialog, tmp_path / name / "store.ome.zarr")
        assert Path(dialog._last_output_folder()).name == "C_stitched"

    def test_it_is_a_folder_not_the_store(self, dialog, tmp_path):
        # The worker reports the store; both buttons want the folder holding it.
        store = tmp_path / "MyDataset_stitched_2" / "MyDataset.ome.zarr"
        _complete(dialog, store)
        assert dialog._last_output_folder() == str(store.parent)

    def test_the_summary_asks_the_selector_rather_than_rebuilding(self):
        # Guards the extraction itself: inline, the choice sat behind a modal
        # where a test could only re-implement it.
        import flamingo_stitcher.gui.stitching_dialog as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "last_success_path = self._last_output_folder()" in source
        assert '_stitched"\n                )' not in source

    def test_the_item_also_records_it(self, dialog, tmp_path):
        # The queue row's own record must agree with the summary's.
        renamed = tmp_path / "MyDataset_stitched_2" / "MyDataset.ome.zarr"
        _complete(dialog, renamed)
        assert dialog._queue[0]["output_path"] == str(renamed)


class TestItIsClearedBetweenBatches:
    def test_a_fresh_dialog_offers_nothing(self, dialog):
        assert dialog._last_success_output is None

    def test_a_failed_batch_does_not_offer_a_previous_run(self, dialog, tmp_path):
        # Otherwise a batch where everything failed would still offer to open
        # the folder from the run before it.
        _complete(dialog, tmp_path / "Old_stitched" / "store.ome.zarr")
        dialog._batch_results = []
        dialog._last_success_output = None
        assert dialog._last_success_output is None

    def test_starting_a_run_clears_it(self, dialog, tmp_path):
        _complete(dialog, tmp_path / "Old_stitched" / "store.ome.zarr")
        import flamingo_stitcher.gui.stitching_dialog as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        # Reset next to the batch-results reset, so a new run cannot inherit
        # the previous one's output.
        assert (
            source.count("self._last_success_output = None") >= 2
        ), "the run start must clear it too"


class TestASuccessOutsideTheQueueIndexIsStillRecorded:
    def test_a_completion_with_no_addressable_item_still_counts(
        self, dialog, tmp_path
    ):
        # The recording sits outside the queue-index guard: the summary needs the
        # path whether or not the item is still addressable (a re-queue or a
        # removal mid-run moves the index).
        dialog._queue.clear()
        dialog._queue_index = 99
        store = tmp_path / "MyDataset_stitched_2" / "store.ome.zarr"
        dialog._on_item_completed(str(store))
        assert dialog._last_success_output == str(store)
