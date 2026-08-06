"""Stray mouse-wheel scrolls must not rewrite settings.

Qt's default: a combo box or spin box under the cursor eats wheel events, so
scrolling a tall settings dialog quietly changes whatever happened to be under
the pointer. In a stitching dialog that means a changed output format, pixel
size or fusion mode discovered hours later in the result.

`stitching_dialog` already carried a bespoke `_NoScrollDoubleSpinBox` for the
XY-pixel-size box, which proves the failure mode — but it covered only
QDoubleSpinBox, leaving every dropdown, plain spin box and slider exposed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QEvent, QPoint, Qt  # noqa: E402
from PyQt5.QtGui import QWheelEvent  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from flamingo_stitcher.gui._wheel_guard import install_wheel_guard  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _wheel(widget):
    return QWheelEvent(
        QPoint(5, 5), widget.mapToGlobal(QPoint(5, 5)), QPoint(0, -120),
        QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
    )


@pytest.fixture
def panel(app):
    w = QWidget()
    layout = QVBoxLayout(w)
    w.combo = QComboBox()
    w.combo.addItems(["zarr", "ome-tiff", "imaris"])
    w.spin = QSpinBox()
    w.spin.setRange(0, 100)
    w.dspin = QDoubleSpinBox()
    w.dspin.setRange(0.0, 10.0)
    w.dspin.setValue(0.406)
    w.slider = QSlider(Qt.Horizontal)
    w.slider.setRange(0, 100)
    w.slider.setValue(50)
    for c in (w.combo, w.spin, w.dspin, w.slider):
        layout.addWidget(c)
    return w


def test_every_value_widget_is_guarded(panel):
    assert install_wheel_guard(panel) == 4


def test_scrolling_an_unfocused_dropdown_does_not_change_it(app, panel):
    install_wheel_guard(panel)
    panel.combo.setCurrentIndex(0)

    app.sendEvent(panel.combo, _wheel(panel.combo))

    assert panel.combo.currentIndex() == 0


@pytest.mark.parametrize("name,getter", [
    ("spin", lambda w: w.spin.value()),
    ("dspin", lambda w: w.dspin.value()),
    ("slider", lambda w: w.slider.value()),
])
def test_scrolling_an_unfocused_spin_or_slider_does_not_change_it(
    app, panel, name, getter
):
    install_wheel_guard(panel)
    before = getter(panel)

    app.sendEvent(getattr(panel, name), _wheel(getattr(panel, name)))

    assert getter(panel) == before


def test_the_event_is_left_unaccepted_so_the_dialog_still_scrolls(app, panel):
    """Ignoring (not swallowing) is what lets the scroll area receive it."""
    install_wheel_guard(panel)
    event = _wheel(panel.combo)

    app.sendEvent(panel.combo, event)

    assert not event.isAccepted()


def test_focus_policy_is_relaxed_so_a_scroll_cannot_grab_focus(panel):
    """Otherwise the first scroll focuses the widget and the second edits it."""
    panel.combo.setFocusPolicy(Qt.WheelFocus)
    install_wheel_guard(panel)

    assert panel.combo.focusPolicy() == Qt.StrongFocus


def test_a_focused_widget_still_scrolls_normally(app, panel):
    """Deliberate editing must keep working."""
    install_wheel_guard(panel)
    panel.show()
    panel.spin.setValue(5)
    panel.spin.setFocus()
    if not panel.spin.hasFocus():
        pytest.skip("no focus in this headless environment")

    app.sendEvent(panel.spin, _wheel(panel.spin))

    assert panel.spin.value() != 5


def test_installing_twice_reuses_one_filter(panel):
    install_wheel_guard(panel)
    first = panel._wheel_guard
    install_wheel_guard(panel)

    assert panel._wheel_guard is first


def test_a_broken_root_never_raises():
    class _Bad:
        def findChildren(self, *_a, **_k):
            raise RuntimeError("no")

    assert install_wheel_guard(_Bad()) == 0


# --------------------------------------------------------------------------- #
# Discovery must say what it found. Until now it logged only a tile COUNT, so
# the channel set — which decides what gets written and how long the run takes
# — was invisible until run time.
# --------------------------------------------------------------------------- #
class _Tile:
    def __init__(self, channels, sides=(0,)):
        self.channels = list(channels)
        self.raw_files = {c: {s: None for s in sides} for c in channels}


def _describe(tiles):
    from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

    stub = StitchingDialog.__new__(StitchingDialog)
    return StitchingDialog._describe_discovered(stub, tiles)


def test_discovery_line_leads_with_the_channel_count_not_the_index():
    """'channel 3' for one channel reads as three channels."""
    text = _describe([_Tile([3]), _Tile([3])])

    assert "Channels: 1" in text
    assert "640" in text  # named laser, so the index is unambiguous


def test_discovery_line_reports_a_single_illumination_side():
    text = _describe([_Tile([3], sides=(0,))])

    assert "1 illumination side (I0)" in text


def test_discovery_line_reports_two_illumination_sides():
    text = _describe([_Tile([3], sides=(0, 1))])

    assert "2 illumination sides" in text
    assert "I0" in text and "I1" in text


def test_multi_channel_discovery_counts_them_all():
    text = _describe([_Tile([0, 3])])

    assert "Channels: 2" in text


def test_a_tile_with_no_metadata_does_not_raise():
    class _Empty:
        channels = []
        raw_files = {}

    text = _describe([_Empty()])

    assert "illumination sides unknown" in text
