"""Tests for showing release notes in the update dialog."""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QMainWindow,
    QTabWidget,
    QTextEdit,
)

from flamingo_stitcher.gui import updater  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class _FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self, *a):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_parses_release_body(monkeypatch):
    payload = {
        "tag_name": "v9.9.9",
        "html_url": "https://example/release",
        "body": "## Highlights\n- Fixed a seam bug\n- Faster fusion",
        "assets": [
            {
                "name": "FlamingoStitcher-Setup-9.9.9.exe",
                "browser_download_url": "https://example/dl.exe",
                "size": 1234,
            }
        ],
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _FakeResp(payload)
    )
    info = updater.fetch_latest_release_info("https://api/whatever")
    assert info["latest_version"] == "9.9.9"
    assert "Fixed a seam bug" in info["body"]
    assert info["asset_url"] == "https://example/dl.exe"


def _panel(qapp):
    win = QMainWindow()
    tabs = QTabWidget()
    win.setCentralWidget(tabs)

    class _Settings:
        def value(self, key, default=None, **k):
            return default

        def setValue(self, *a, **k):
            pass

    panel = updater.UpdatePanel(win, tabs, _Settings())
    tabs.addTab(panel, "Updates")
    return win, tabs, panel


def test_dialog_shows_notes_box(qapp):
    win, tabs, panel = _panel(qapp)
    panel._latest_notes = "## What changed\n- A scrollable notes box\n- Line two"
    panel.show_update_dialog("9.9.9")
    dlg = panel._pending_dialog
    assert dlg is not None
    boxes = dlg.findChildren(QTextEdit)
    assert boxes, "expected a release-notes text box"
    text = boxes[0].toPlainText()
    assert "scrollable notes box" in text
    assert boxes[0].isReadOnly()
    dlg.reject()


def test_dialog_without_notes_has_no_box(qapp):
    win, tabs, panel = _panel(qapp)
    panel._latest_notes = ""
    panel.show_update_dialog("9.9.8")  # different version (dedupe is per-version)
    dlg = panel._pending_dialog
    assert dlg is not None
    assert not dlg.findChildren(QTextEdit)
    dlg.reject()
