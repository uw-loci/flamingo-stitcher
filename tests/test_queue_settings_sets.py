"""Each queued acquisition keeps its own settings (0.14.0).

Agreed with the user 2026-11-05/06: a 'Microscope' column; settings sets A, B,
C (a second microscope starts its own letter); selecting rows shows their
settings and edits apply to them; rows from different microscopes cannot be
edited together; while a run goes on the queue stays usable and only the
running acquisition's settings are locked (grayed out and folded away).

Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/test_queue_settings_sets.py -q
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from flamingo_stitcher import run_lock  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp, monkeypatch, tmp_path):
    from flamingo_stitcher import scope_profiles
    from flamingo_stitcher.gui.stitching_dialog import NativeStitchingDialog

    def fake_resolve(path):
        name = "liara" if "liara" in str(path) else "n7"
        return name, ("20" if name == "liara" else "6.2"), {}, "test"

    monkeypatch.setattr(scope_profiles, "resolve_for_acquisition", fake_resolve)
    # A run remembers each microscope in a user file; keep tests out of it.
    monkeypatch.setattr(scope_profiles, "seen_path", lambda: tmp_path / "seen.json")
    monkeypatch.setattr(scope_profiles, "profiles_path", lambda: tmp_path / "profiles.json")
    d = NativeStitchingDialog()
    for name in ("n7_a", "n7_b", "liara_c"):
        (tmp_path / name).mkdir()
        d._add_path_to_queue(tmp_path / name)
    yield d
    run_lock.release(d)
    d.deleteLater()


def _select(d, *rows):
    d._queue_table.clearSelection()
    for r in rows:
        for c in range(d._queue_table.columnCount()):
            d._queue_table.item(r, c).setSelected(True)


def _cell(d, r, c):
    return d._queue_table.item(r, c).text()


def test_microscope_column_and_letters(dialog):
    d = dialog
    headers = [d._queue_table.horizontalHeaderItem(c).text()
               for c in range(d._queue_table.columnCount())]
    assert headers == ["Status", "Directory", "Microscope", "Tiles", "Planes",
                       "Raw size", "Est. time", "Settings"]
    assert [_cell(d, r, 2) for r in range(3)] == ["n7, 6.2x", "n7, 6.2x", "liara, 20x"]
    assert [_cell(d, r, 7) for r in range(3)] == ["Set A", "Set A", "Set B"]
    assert "2 settings sets" in d._queue_summary_label.text()


def test_editing_one_row_splits_it_and_selection_shows_each_rows_settings(dialog):
    d = dialog
    _select(d, 1)
    i8 = d._downsample_xy_combo.findData(8)
    d._downsample_xy_combo.setCurrentIndex(i8)
    d._write_settings_to_selection()
    assert [_cell(d, r, 7) for r in range(3)] == ["Set A", "Set C", "Set B"]
    assert d._queue[1]["config"].downsample_xy == 8
    assert d._queue[0]["config"].downsample_xy != 8
    _select(d, 0)
    assert d._downsample_xy_combo.currentData() == d._queue[0]["config"].downsample_xy
    _select(d, 1)
    assert d._downsample_xy_combo.currentData() == 8


def test_two_microscopes_selected_gray_out_everything(dialog):
    d = dialog
    _select(d, 1, 2)
    assert not d._config_container.isEnabled()
    assert not d._settings_tabs.widget(0).isEnabled()
    note = d._selection_note.text()
    assert "different microscopes or objectives" in note and "liara, 20x" in note
    assert d._run_summary.message  # the panel says why instead of showing tags
    _select(d, 0, 1)  # one microscope: editable again
    assert d._config_container.isEnabled()


def test_running_row_locks_and_folds_while_pending_rows_stay_editable(dialog):
    d = dialog
    d._batch_running = True
    d._queue_index = 0
    d._queue[0]["status"] = "stitching"
    d._update_queue_table()
    _select(d, 0)
    assert not d._config_container.isEnabled()
    assert not d._proc_toggle.isChecked()  # settings folded away
    assert "Locked" in d._selection_note.text()
    _select(d, 1)
    assert d._config_container.isEnabled()
    assert d._proc_toggle.isChecked()
    assert d._add_btn.isEnabled()  # the queue stays usable during a run
    d._batch_running = False


def test_each_acquisition_runs_with_its_own_settings(dialog, monkeypatch):
    import flamingo_stitcher.worker as worker_mod

    d = dialog
    _select(d, 1)
    d._downsample_xy_combo.setCurrentIndex(d._downsample_xy_combo.findData(8))
    d._write_settings_to_selection()
    _select(d, 0)  # the window now shows row 0's settings
    seen = {}

    class FakeWorker:
        def __init__(self, config, **kw):
            seen["config"] = config
            seen["channels"] = kw.get("channels")
            for sig in ("progress", "log_message", "completed", "error",
                        "memory_warning", "finished"):
                setattr(self, sig, type("S", (), {"connect": lambda *_: None})())

        def start(self):
            pass

    monkeypatch.setattr(worker_mod, "StitchingWorker", FakeWorker)
    d._batch_config = d._build_config()
    d._queue[1]["channels_text"] = "3"
    d._start_item_worker(d._queue[1], str(d._queue[1]["path"]))
    assert seen["config"].downsample_xy == 8
    assert seen["channels"] == [3]


def test_new_microscope_does_not_inherit_a_typed_frame_size(dialog, tmp_path):
    d = dialog
    _select(d, 0)
    d._frame_size_combo.setCurrentIndex(d._frame_size_combo.findData(2048))
    d._write_settings_to_selection()
    assert d._queue[0]["config"].frame_width == 2048
    (tmp_path / "liara_d").mkdir()
    _select(d, 0)
    d._add_path_to_queue(tmp_path / "liara_d")  # joins liara's Set B, not A
    assert d._queue[3]["set"] == "B"
    assert d._queue[3]["config"].frame_width is None
