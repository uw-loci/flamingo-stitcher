"""One stitching run at a time across the app's tabs.

Until 0.14.0, 'Single Workflow', 'Multi-Acquisition' and 'Multi-View' could
each run at once. Two runs compete for memory (one large run peaked near 79 GB
of 191), cores and, before 0.14.0, the same scratch files. User, 2026-11-06:
"don't allow runs in two different tabs at the same time".

Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/test_run_lock.py -q
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flamingo_stitcher import run_lock  # noqa: E402


class _Owner:
    def __init__(self):
        self.calls = 0

    def changed(self):
        self.calls += 1


@pytest.fixture(autouse=True)
def _free_lock():
    run_lock._holder = None
    yield
    run_lock._holder = None


def test_second_holder_is_refused_until_release():
    a, b = _Owner(), _Owner()
    assert run_lock.acquire(a, "Single Workflow")
    assert not run_lock.acquire(b, "Multi-Acquisition")
    assert run_lock.holder_label(exclude=b) == "Single Workflow"
    assert run_lock.holder_label(exclude=a) is None  # a's own run is not "another"
    run_lock.release(b)  # not the holder: no effect
    assert run_lock.holder_label() == "Single Workflow"
    run_lock.release(a)
    assert run_lock.acquire(b, "Multi-Acquisition")


def test_listeners_hear_start_and_end_and_closed_ones_drop_out():
    a, b = _Owner(), _Owner()
    run_lock.subscribe(b.changed)
    run_lock.acquire(a, "Single Workflow")
    run_lock.release(a)
    assert b.calls == 2
    del b  # weakly held: a closed dialog does not keep firing
    run_lock.acquire(a, "Single Workflow")


# --- In the dialogs -----------------------------------------------------------

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def tabs(qapp):
    from flamingo_stitcher.gui.stitching_dialog import (
        NativeStitchingDialog,
        StitchingDialog,
    )

    single, multi = NativeStitchingDialog(), StitchingDialog()
    yield single, multi
    for d in (single, multi):
        run_lock.release(d)
        d.deleteLater()


def _queue_one(dialog, tmp_path):
    dialog._queue = [{"path": tmp_path, "status": "pending", "tiles": None,
                      "error": None, "output_path": None}]
    dialog._update_action_buttons()


def test_run_all_grays_out_in_the_other_tab(tabs, tmp_path):
    single, multi = tabs
    _queue_one(multi, tmp_path)
    assert multi._run_btn.isEnabled()
    assert run_lock.acquire(single, single._run_label)
    assert not multi._run_btn.isEnabled()
    assert "'Single Workflow'" in multi._run_btn.toolTip()
    run_lock.release(single)
    assert multi._run_btn.isEnabled()
    assert multi._run_btn.toolTip() == multi._run_tip


def test_run_in_the_other_tab_is_refused_with_its_name(tabs, tmp_path, monkeypatch):
    single, multi = tabs
    _queue_one(multi, tmp_path)
    said = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda parent, title, text: said.append(text))
    run_lock.acquire(single, single._run_label)
    multi._on_run()
    assert not multi._batch_running
    assert said and "'Single Workflow'" in said[0]
