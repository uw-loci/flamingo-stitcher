"""Settings tabs glow after 'Discover Tiles' until the user opens them.

User, 2026-11-10: tabs not clicked before stitching should blink, "not
painfully obviously, but 'there is something here to look at'", a light blue
halo, for tabs not opened since the last Discover Tiles. Also: 'Intensity
adjustment' shows 'Active' instead of 'Apply', so choosing it does not read as
an action, and is renamed 'Tile brightness matching' ("vague").

Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/test_tab_attention.py -q
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QTabWidget, QWidget  # noqa: E402

from flamingo_stitcher import run_lock  # noqa: E402
from flamingo_stitcher.gui import tab_attention  # noqa: E402
from flamingo_stitcher.gui.tab_attention import GLOW_TOOLTIP, TabAttentionGlow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def tabs(qapp):
    t = QTabWidget()
    t.addTab(QWidget(), "Image quality")
    t.addTab(QWidget(), "Registration and overlap")
    t.tabBar().setTabToolTip(1, "plain tip")
    t.resize(600, 200)
    t.show()
    qapp.processEvents()
    yield t
    t.deleteLater()


def test_nothing_glows_until_armed(tabs):
    g = TabAttentionGlow(tabs)
    assert not g.pending and g.strength() == 0.0
    assert not g._timer.isActive()


def test_each_tab_glows_until_it_is_clicked(tabs):
    g = TabAttentionGlow(tabs)
    g.arm()
    assert g.pending == {0, 1} and g._timer.isActive()
    assert tabs.tabBar().tabToolTip(1) == GLOW_TOOLTIP
    tabs.tabBarClicked.emit(1)
    assert g.pending == {0}
    assert tabs.tabBar().tabToolTip(1) == "plain tip"  # the tab's own tip comes back
    tabs.tabBarClicked.emit(0)  # clicking the tab already showing also counts
    assert not g.pending and not g._timer.isActive()


def test_switching_tabs_from_the_keyboard_counts(tabs):
    g = TabAttentionGlow(tabs)
    g.arm()
    tabs.setCurrentIndex(1)
    assert g.pending == {0}


def test_a_new_discover_rearms_tabs_already_opened(tabs):
    g = TabAttentionGlow(tabs)
    g.arm()
    g.acknowledge(0)
    g.acknowledge(1)
    g.arm()
    assert g.pending == {0, 1}


def test_the_pulse_is_gentle_and_never_opaque(tabs, qapp, monkeypatch):
    monkeypatch.setattr(tab_attention, "PERIOD_S", 0.2)
    g = TabAttentionGlow(tabs)
    g.arm()
    seen = []
    t0 = time.time()
    while time.time() - t0 < 0.3:
        seen.append(g.strength())
        qapp.processEvents()
        time.sleep(0.005)
    assert min(seen) < 0.1 and max(seen) > 0.9  # it rises and falls
    assert tab_attention.PEAK_ALPHA < 255  # strongest frame is still see-through
    assert all(0.0 <= s <= 1.0 for s in seen)


def test_suppressed_hides_the_glow_but_keeps_it_owed(tabs):
    busy = {"on": True}
    g = TabAttentionGlow(tabs, suppressed=lambda: busy["on"])
    g.arm()
    assert g.strength() == 0.0 and not g.is_glowing(0)
    busy["on"] = False
    assert g.is_glowing(0) and g.pending == {0, 1}


# --- In the stitching dialog -----------------------------------------------------


@pytest.fixture
def dialog(qapp, monkeypatch, tmp_path):
    from flamingo_stitcher import scope_profiles
    from flamingo_stitcher.gui.stitching_dialog import NativeStitchingDialog

    monkeypatch.setattr(scope_profiles, "resolve_for_acquisition",
                        lambda p: ("n7", "6.2", {}, "test"))
    monkeypatch.setattr(scope_profiles, "seen_path", lambda: tmp_path / "seen.json")
    d = NativeStitchingDialog()
    yield d
    run_lock.release(d)
    d.deleteLater()


def test_discover_tiles_arms_both_tabs(dialog, monkeypatch, tmp_path):
    d = dialog
    (tmp_path / "acq").mkdir()
    d._add_path_to_queue(tmp_path / "acq")
    monkeypatch.setattr(d, "_discover_tiles_for_path", lambda p: ["tile"])
    monkeypatch.setattr(d, "_describe_discovered", lambda tiles: "")
    monkeypatch.setattr(d, "_warn_on_discovery_issues", lambda pending: None)
    assert not d._tab_glow.pending
    try:
        d._on_discover()
    except Exception:
        pass  # later steps read real tiles; the tabs are armed before them
    assert d._tab_glow.pending == {0, 1}
    d._settings_tabs.tabBarClicked.emit(0)
    assert d._tab_glow.pending == {1}


def test_a_flashing_next_step_button_pauses_the_glow(dialog):
    d = dialog
    d._tab_glow.arm()
    d._flash_cta(d._add_btn)
    assert not d._tab_glow.is_glowing(0)
    d._stop_cta_flash()
    assert d._tab_glow.is_glowing(0)


def test_intensity_adjustment_says_active_and_still_stores_apply(dialog):
    combo = dialog._intensity_adjust_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert labels == ["Off", "Measure only", "Active"]
    combo.setCurrentIndex(2)
    assert combo.currentData() == "apply"  # saved settings and configs keep working
    assert "'Active'" in combo.toolTip() and "'Apply'" not in combo.toolTip()
    from PyQt5.QtWidgets import QLabel

    labels = {w.text() for w in dialog.findChildren(QLabel)}
    assert "Tile brightness matching:" in labels
    assert "Intensity adjustment:" not in labels
