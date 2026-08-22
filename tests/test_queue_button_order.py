"""The queue row reads left to right in the order the work is done.

Add... is the first thing anyone must click: with an empty queue there is
nothing to discover and nothing to run. Leading the row with Discover Tiles
pointed the eye at a button that was disabled anyway, and left the one usable
action third in line.

Exactly one button flashes at a time, pointing at whichever step is next —
Add... while the queue is empty, Discover Tiles once there is something to
discover. Two things flashing is noise, not guidance.

Run: QT_QPA_PLATFORM=offscreen python -m pytest \\
        tests/test_queue_button_order.py -q
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
    """One dialog for the module — constructing it costs seconds."""
    from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

    dlg = StitchingDialog()
    yield dlg
    dlg.deleteLater()


@pytest.fixture
def dialog(_dlg):
    """The shared dialog, returned to a freshly-opened state each test."""
    _dlg._queue.clear()
    _dlg._batch_running = False
    _dlg._discover_needed = False
    _dlg._update_action_buttons()
    return _dlg


def _order(dlg):
    """Queue-row buttons in left-to-right order, by LAYOUT index.

    Not by ``pos().x()``: a dialog that was never shown has not been laid out,
    so every button reports x=0 and a sort on that is stable — it hands back
    whatever order the test listed them in. An earlier version of this file did
    exactly that and passed with the buttons deliberately reordered.
    """
    from PyQt5.QtWidgets import QHBoxLayout

    buttons = {
        "add": dlg._add_btn,
        "add_folder": dlg._add_folder_btn,
        "discover": dlg._discover_btn,
        "orientation": dlg._orientation_btn,
        "requeue": dlg._requeue_btn,
        "remove": dlg._remove_btn,
    }
    row = next(
        layout
        for layout in dlg.findChildren(QHBoxLayout)
        if layout.indexOf(dlg._add_btn) >= 0
    )
    placed = [(row.indexOf(btn), name) for name, btn in buttons.items()]
    assert all(index >= 0 for index, _ in placed), "a button left the queue row"
    return [name for _index, name in sorted(placed)]


def _queue_one(dlg, status="pending"):
    dlg._queue.append({"path": Path("/tmp/acq"), "status": status})
    dlg._discover_needed = True
    dlg._update_action_buttons()


class TestTheRowReadsInWorkingOrder:
    def test_add_comes_first(self, dialog):
        assert _order(dialog)[0] == "add"

    def test_add_all_in_folder_stays_beside_it(self, dialog):
        order = _order(dialog)
        assert order.index("add_folder") == order.index("add") + 1

    def test_discover_follows_the_two_add_buttons(self, dialog):
        order = _order(dialog)
        assert order.index("discover") > order.index("add_folder")

    def test_the_rest_follow_discover(self, dialog):
        order = _order(dialog)
        for name in ("orientation", "requeue", "remove"):
            assert order.index(name) > order.index("discover")


class TestAnEmptyQueuePointsAtAdd:
    def test_add_is_the_flashing_call_to_action(self, dialog):
        # The user's first move, and until this it was the only enabled button
        # in the row with nothing drawing the eye to it.
        assert dialog._cta_button is dialog._add_btn
        assert dialog._cta_flash_timer.isActive()

    def test_it_is_actually_styled_not_just_tracked(self, dialog):
        assert dialog._add_btn.styleSheet() != ""

    def test_discover_is_disabled_with_nothing_to_discover(self, dialog):
        assert not dialog._discover_btn.isEnabled()

    def test_the_state_is_painted_at_construction(self, app):
        # Deliberately NOT the shared fixture: that calls
        # `_update_action_buttons` itself, which is the very thing under test.
        # `_update_action_buttons` used to run only on a queue change, so a
        # freshly opened dialog showed Discover Tiles enabled over an empty
        # queue with nothing pointing at Add.
        from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

        fresh = StitchingDialog()
        try:
            assert fresh._cta_button is fresh._add_btn
            assert not fresh._discover_btn.isEnabled()
        finally:
            fresh.deleteLater()


class TestQueueingSomethingMovesTheCallToAction:
    def test_discover_takes_over(self, dialog):
        _queue_one(dialog)
        assert dialog._cta_button is dialog._discover_btn
        assert dialog._cta_flash_timer.isActive()

    def test_add_stops_flashing(self, dialog):
        # Only one at a time.
        _queue_one(dialog)
        assert dialog._add_btn.styleSheet() == ""

    def test_emptying_the_queue_hands_it_back(self, dialog):
        _queue_one(dialog)
        dialog._queue.clear()
        dialog._update_action_buttons()
        assert dialog._cta_button is dialog._add_btn
        assert dialog._discover_btn.styleSheet() == ""

    def test_a_completed_discover_stops_the_flashing(self, dialog):
        _queue_one(dialog)
        dialog._discover_needed = False
        dialog._update_action_buttons()
        assert dialog._cta_button is None
        assert not dialog._cta_flash_timer.isActive()

    def test_discover_keeps_its_accent_when_not_flashing(self, dialog):
        # It is still the notable action in the row; it should not drop to a
        # plain button just because nothing is owed right now.
        _queue_one(dialog)
        dialog._discover_needed = False
        dialog._update_action_buttons()
        assert "1E88E5" in dialog._discover_btn.styleSheet()

    def test_nothing_flashes_while_a_batch_runs(self, dialog):
        # Every queue action is disabled mid-run; flashing one would be telling
        # the user to press something that cannot be pressed. The empty-queue
        # case is the one this guard actually protects — a queued run reaches
        # the same answer through Discover being disabled.
        dialog._batch_running = True
        dialog._update_action_buttons()
        try:
            assert dialog._cta_button is None
            assert not dialog._cta_flash_timer.isActive()
        finally:
            dialog._batch_running = False

    def test_nothing_flashes_mid_run_with_items_queued_either(self, dialog):
        _queue_one(dialog)
        dialog._batch_running = True
        dialog._update_action_buttons()
        try:
            assert dialog._cta_button is None
        finally:
            dialog._batch_running = False


class TestOnlyOneThingEverFlashes:
    def test_the_timer_drives_exactly_one_button(self, dialog):
        _queue_one(dialog)
        dialog._on_cta_flash_tick()
        styled = [
            b
            for b in (dialog._add_btn, dialog._discover_btn)
            if b.styleSheet() != ""
        ]
        assert styled == [dialog._discover_btn]

    def test_a_tick_with_no_target_is_not_an_error(self, dialog):
        dialog._stop_cta_flash()
        dialog._on_cta_flash_tick()  # must not raise
