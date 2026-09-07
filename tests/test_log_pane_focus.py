"""The window belongs to the settings before a run, and to the log during one.

Before a run the operator is choosing settings and there is no output to read,
so the Log pane starts closed and does not remember otherwise. Once a run
starts, the settings are locked anyway and the output is the only thing that
matters, so the log opens and Processing Options closes.

Closing Processing Options is what makes the auto-open safe. dbe3d43 REMOVED an
earlier auto-open for two reasons: it overrode the user's remembered preference,
and on small screens it pushed the always-visible Progress section (laid out
BELOW the log) off the bottom. The first reason is gone — there is no remembered
preference any more. The second is handled by taking the log's height back from
the settings pane rather than from the bottom of the window.

Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/test_log_pane_focus.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DIALOG = REPO / "src" / "flamingo_stitcher" / "gui" / "stitching_dialog.py"
SOURCE = DIALOG.read_text(encoding="utf-8")


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(SOURCE)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == method_name
                ):
                    return child
    raise AssertionError(f"{class_name}.{method_name} not found")


def _setting_keys(method: ast.AST) -> set:
    keys = set()
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("setValue", "value"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            if isinstance(node.args[0].value, str):
                keys.add(node.args[0].value)
    return keys


class TestItStartsClosed:
    @pytest.mark.parametrize(
        "dialog", ["StitchingDialog", "NativeStitchingDialog"]
    )
    def test_neither_dialog_persists_the_open_state(self, dialog):
        """A remembered 'open' is what the old auto-open was accused of
        overriding. There is nothing to override once nothing is remembered."""
        assert "log_expanded" not in _setting_keys(_method(dialog, "_save_settings"))
        assert "log_expanded" not in _setting_keys(
            _method(dialog, "_restore_settings")
        )

    @pytest.mark.parametrize(
        "dialog", ["StitchingDialog", "NativeStitchingDialog"]
    )
    def test_restore_forces_it_closed(self, dialog):
        """Not merely 'defaults to closed' — set closed, every launch."""
        body = ast.dump(_method(dialog, "_restore_settings"))
        assert "_log_toggle" in body
        source = ast.get_source_segment(SOURCE, _method(dialog, "_restore_settings"))
        assert "self._log_toggle.setChecked(False)" in source

    def test_the_widget_is_built_collapsed(self):
        assert "self._log_group.setVisible(False)" in SOURCE


class TestItOpensOnRun:
    def test_the_run_handler_calls_it(self):
        source = ast.get_source_segment(SOURCE, _method("StitchingDialog", "_on_run"))
        assert "_focus_log_for_run()" in source

    def test_it_is_called_only_after_every_pre_flight_check(self):
        """A run the user backs out of must not rearrange their window."""
        source = ast.get_source_segment(SOURCE, _method("StitchingDialog", "_on_run"))
        call = source.index("_focus_log_for_run()")
        for guard in (
            "_confirm_orientation_known",
            "_confirm_pixel_size",
            "_confirm_resource_headroom",
        ):
            assert source.index(guard) < call, guard

    def test_it_opens_the_log_and_closes_processing_options(self):
        source = ast.get_source_segment(
            SOURCE, _method("StitchingDialog", "_focus_log_for_run")
        )
        assert "self._log_toggle.setChecked(True)" in source
        assert "self._proc_toggle.setChecked(False)" in source

    def test_processing_options_closes_before_the_log_opens(self):
        """Free the space first, so the window is never momentarily taller —
        which on a small screen is what pushed Progress off the bottom."""
        source = ast.get_source_segment(
            SOURCE, _method("StitchingDialog", "_focus_log_for_run")
        )
        assert source.index("_proc_toggle.setChecked(False)") < source.index(
            "_log_toggle.setChecked(True)"
        )

    def test_it_resizes_once_at_the_end(self):
        source = ast.get_source_segment(
            SOURCE, _method("StitchingDialog", "_focus_log_for_run")
        )
        assert source.count("self.resize(") == 1
        assert source.index("_log_toggle.setChecked(True)") < source.index(
            "self.resize("
        )


class TestProgressStaysReachable:
    def test_progress_is_still_laid_out_below_the_log(self):
        """The reason the ordering matters is recorded in the method's own
        docstring; if the layout order ever changes, that rationale is stale."""
        assert SOURCE.index("self._log_group.setVisible(False)") < SOURCE.index(
            'QGroupBox("Progress")'
        )


# ---------------------------------------------------------------------------
# The live widgets. The AST checks above pin the wiring; these pin the effect,
# which is the part a refactor can quietly invert.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    # Scoped to the fixture, not the module: the AST checks above need no Qt,
    # and a module-level importorskip would silently take them down with it.
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def dialog(qapp):
    from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

    d = StitchingDialog()
    yield d
    d.deleteLater()


class TestTheLiveWidgets:
    def test_a_fresh_dialog_has_the_log_closed(self, dialog):
        assert dialog._log_toggle.isChecked() is False
        # isHidden(), not isVisible(): a widget inside a dialog that was never
        # show()n reports isVisible() == False whatever its own state, so the
        # visible case cannot be asserted that way and the hidden case would
        # pass for the wrong reason.
        assert dialog._log_group.isHidden() is True

    def test_focusing_for_a_run_swaps_them(self, dialog):
        """Processing Options is itself a collapsible pane whose state is
        remembered (`proc_options_expanded`), so set it explicitly rather than
        assuming what a fresh dialog restored."""
        dialog._proc_toggle.setChecked(True)
        dialog._focus_log_for_run()
        assert dialog._log_toggle.isChecked() is True
        assert dialog._proc_toggle.isChecked() is False

    def test_it_closes_processing_options_even_if_already_closed(self, dialog):
        dialog._proc_toggle.setChecked(False)
        dialog._focus_log_for_run()
        assert dialog._proc_toggle.isChecked() is False
        assert dialog._log_toggle.isChecked() is True

    def test_the_toggle_label_follows_the_state(self, dialog):
        """The caret is the only cue that the pane is collapsible."""
        assert dialog._log_toggle.text().startswith("\u25b6")
        dialog._focus_log_for_run()
        assert dialog._log_toggle.text().startswith("\u25bc")

    def test_it_is_idempotent(self, dialog):
        """A second queue item must not toggle anything back."""
        dialog._focus_log_for_run()
        dialog._focus_log_for_run()
        assert dialog._log_toggle.isChecked() is True
        assert dialog._proc_toggle.isChecked() is False

    def test_the_user_can_reopen_the_settings_afterwards(self, dialog):
        """Stated in the docstring as deliberate: adding to the queue mid-run
        has to work, so neither pane is held."""
        dialog._focus_log_for_run()
        dialog._proc_toggle.setChecked(True)
        assert dialog._proc_toggle.isChecked() is True
        assert dialog._log_toggle.isChecked() is True

    def test_reopening_settings_does_not_close_the_log(self, dialog):
        dialog._focus_log_for_run()
        dialog._proc_toggle.setChecked(True)
        assert dialog._log_group.isHidden() is False
