"""Release notes must show in full, whatever characters they contain.

The 0.13.12 notes contained "<name>_registration_map.png". The update dialog
rendered them with Qt's Markdown, which passes raw HTML through, so <name>
became an unknown element and the user saw the first line followed by blank
space.
"""

import pytest

pytest.importorskip("PyQt5")


def test_angle_brackets_do_not_swallow_the_notes():
    from PyQt5.QtWidgets import QApplication, QTextEdit

    from flamingo_stitcher.gui.updater import render_release_notes

    _app = QApplication.instance() or QApplication([])
    notes = (
        "New: a registration map with every stitch.\n"
        "<name>_registration_map.png is written next to the report:\n"
        "- tiles tinted by Z\n- a red X on links the solve did not use\n"
    )
    box = QTextEdit()
    render_release_notes(box, notes)
    text = box.toPlainText()
    assert "<name>_registration_map.png" in text
    assert "red X on links" in text
