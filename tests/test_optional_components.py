"""'Set up flat-field…' moved to the Options tab (0.14.0).

User, 2026-11-05: not next to 'Run All' and 'Cancel'; the Options tab holds
extra things, "as long as we direct people to it when something isn't active
yet". Grayed-out controls otherwise explain themselves in their tooltip only;
a missing install is the exception and gets a line that links to the tab.

Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/test_optional_components.py -q
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def missing(monkeypatch):
    """Flat-field and deconvolution both missing."""
    import flamingo_stitcher.deconvolution as decon
    import flamingo_stitcher.flat_field as ff

    monkeypatch.setattr(ff, "is_available", lambda: False)
    monkeypatch.setattr(decon, "is_available", lambda: False)


@pytest.fixture
def dialog(qapp):
    from flamingo_stitcher.gui.stitching_dialog import NativeStitchingDialog

    d = NativeStitchingDialog()
    yield d
    d.deleteLater()


def test_setup_button_is_not_on_the_run_row(dialog):
    run_row = dialog._run_btn.parentWidget()
    assert dialog._setup_env_btn.parentWidget() is not run_row or \
        dialog._setup_env_btn.parentWidget().layout().indexOf(dialog._setup_env_btn) < 0


def test_missing_installs_point_to_the_options_tab(dialog, missing, monkeypatch):
    opened = []
    monkeypatch.setattr(dialog, "_open_options_tab", lambda: opened.append(True))
    dialog.set_options_tab_available(True)
    for note in (dialog._flat_field_install_note, dialog._deconv_install_note):
        assert not note.isHidden()
        assert "Options tab" in note.text()
    assert dialog._setup_env_btn.isHidden()
    dialog._flat_field_install_note.linkActivated.emit("options")
    assert opened == [True]


def test_without_an_options_tab_the_button_shows_beside_flat_field(dialog, missing):
    dialog.set_options_tab_available(False)
    assert not dialog._setup_env_btn.isHidden()
    assert "Options tab" not in dialog._flat_field_install_note.text()


def test_no_note_once_installed(dialog, monkeypatch):
    import flamingo_stitcher.deconvolution as decon
    import flamingo_stitcher.flat_field as ff

    monkeypatch.setattr(ff, "is_available", lambda: True)
    monkeypatch.setattr(decon, "is_available", lambda: True)
    dialog.set_options_tab_available(True)
    assert dialog._flat_field_install_note.isHidden()
    assert dialog._deconv_install_note.isHidden()


def test_options_box_shows_status_and_the_right_button(qapp, missing, monkeypatch):
    from flamingo_stitcher import preprocessing_env
    from flamingo_stitcher.gui.optional_components import OptionalComponentsBox

    monkeypatch.setattr(preprocessing_env, "is_built", lambda: False)
    box = OptionalComponentsBox()
    assert box._status["flat"].text() == "Not installed"
    assert box._status["deconv"].text() == "Not available in this build"
    assert box.setup_btn.text() == "Set up flat-field…"
    monkeypatch.setattr(preprocessing_env, "is_built", lambda: True)
    box.refresh()
    assert box.setup_btn.text() == "Reinstall flat-field…"
    box.deleteLater()


def test_options_panel_has_the_box(qapp):
    from flamingo_stitcher.gui.options_panel import OptionsPanel

    panel = OptionsPanel()
    assert panel.components_box.setup_btn.text() in (
        "Set up flat-field…", "Reinstall flat-field…")
    panel.deleteLater()
