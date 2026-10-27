"""The log panel keeps only its newest lines.

2026-10-27: the panel had held every run since the app was opened (three full
stitches) when the window froze at the end of a run. The full log stays in the
file under <config dir>/logs.
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


def test_the_log_panel_drops_its_oldest_lines(qapp):
    from flamingo_stitcher.gui import stitching_dialog as sd

    d = sd.StitchingDialog()
    try:
        doc = d._log_text.document()
        assert doc.maximumBlockCount() == sd.LOG_PANEL_MAX_LINES
        doc.setMaximumBlockCount(50)
        for i in range(200):
            d._append_log(f"line {i}")
        assert doc.blockCount() <= 50
        assert "line 199" in d._log_text.toPlainText()
        assert "line 0\n" not in d._log_text.toPlainText()
    finally:
        d.deleteLater()
