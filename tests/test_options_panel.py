"""The Options tab.

What matters here is the round trip: what the panel shows is what gets saved,
what gets saved is what a run loads, and a profile for one microscope never
leaks into another. The widget layout is not the contract.

Dialog construction is class-scoped: building QWidgets per test segfaults
pytest even under QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")

from flamingo_stitcher import scope_profiles as sp  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "scope_profiles.json"
    monkeypatch.setattr(sp, "profiles_path", lambda: path)
    return path


@pytest.fixture
def panel(qapp, store):
    from flamingo_stitcher.gui.options_panel import OptionsPanel

    return OptionsPanel()


class TestRoundTrip:
    def test_every_tunable_gets_a_control(self, panel):
        assert set(panel._spins) == set(sp.TUNABLE_FIELDS)

    def test_controls_start_at_the_pipeline_defaults(self, panel):
        from flamingo_stitcher.pipeline import StitchingConfig

        blank = StitchingConfig()
        for field, spin in panel._spins.items():
            assert spin.value() == pytest.approx(float(getattr(blank, field)))

    def test_saving_writes_a_profile_a_run_can_load(self, panel):
        panel._new_scope_edit.setText("Liara")
        panel._spins["min_registered_seam_frac"].setValue(0.2)
        panel._on_save()
        values, source = sp.load_profile("Liara", None)
        assert values["min_registered_seam_frac"] == pytest.approx(0.2)
        assert source == "liara|*"

    def test_a_saved_profile_reloads_into_the_controls(self, panel):
        sp.save_profile("Liara", None, {"quality_threshold": 0.15})
        panel._reload_scopes(select="liara")
        assert panel._spins["quality_threshold"].value() == pytest.approx(0.15)

    def test_saving_without_a_name_does_not_write_a_profile(self, panel, monkeypatch):
        from PyQt5.QtWidgets import QMessageBox

        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
        panel._new_scope_edit.setText("   ")
        panel._on_save()
        assert sp.list_profiles() == {}

    def test_a_per_objective_profile_saves_under_its_own_key(self, panel):
        panel._new_scope_edit.setText("Liara")
        panel._on_selection_changed()
        index = panel._objective_combo.findData("__new__")
        panel._objective_combo.setCurrentIndex(index)
        panel._new_objective_edit.setValue(25.0)
        panel._spins["quality_threshold"].setValue(0.33)
        panel._on_save()
        assert "liara|25.0x" in sp.list_profiles()

    def test_defaults_button_restores_without_saving(self, panel):
        from flamingo_stitcher.pipeline import StitchingConfig

        sp.save_profile("Liara", None, {"quality_threshold": 0.15})
        panel._reload_scopes(select="liara")
        panel._on_defaults()
        assert panel._spins["quality_threshold"].value() == pytest.approx(
            StitchingConfig().quality_threshold
        )
        # Still on disk: the button loads defaults, it does not delete.
        assert sp.load_profile("Liara", None)[0]["quality_threshold"] == 0.15

    def test_delete_is_disabled_until_a_profile_exists(self, panel):
        assert not panel._delete_btn.isEnabled()
        sp.save_profile("Liara", None, {"quality_threshold": 0.15})
        panel._reload_scopes(select="liara")
        assert panel._delete_btn.isEnabled()


class TestIsolationBetweenScopes:
    def test_switching_scope_shows_that_scopes_values(self, panel):
        sp.save_profile("Liara", None, {"quality_threshold": 0.1})
        sp.save_profile("Bruce", None, {"quality_threshold": 0.9})
        panel._reload_scopes(select="liara")
        assert panel._spins["quality_threshold"].value() == pytest.approx(0.1)
        panel._reload_scopes(select="bruce")
        assert panel._spins["quality_threshold"].value() == pytest.approx(0.9)

    def test_a_scope_with_no_profile_falls_back_to_defaults_not_the_last_one(
        self, panel
    ):
        from flamingo_stitcher.pipeline import StitchingConfig

        sp.save_profile("Liara", None, {"quality_threshold": 0.1})
        panel._reload_scopes(select="liara")
        assert panel._spins["quality_threshold"].value() == pytest.approx(0.1)
        panel._scope_combo.setCurrentIndex(panel._scope_combo.count() - 1)
        panel._new_scope_edit.setText("Brand New Rig")
        panel._on_selection_changed()
        assert panel._spins["quality_threshold"].value() == pytest.approx(
            StitchingConfig().quality_threshold
        )


class TestWheelGuard:
    def test_an_unfocused_spin_box_ignores_the_wheel(self, panel):
        # The settings sit in a scroll area. Without the guard, scrolling past a
        # spin box rewrites it — a misconfiguration nobody sees until the output
        # is wrong.
        from PyQt5.QtCore import QPoint, Qt
        from PyQt5.QtGui import QWheelEvent

        spin = panel._spins["quality_threshold"]
        spin.clearFocus()
        before = spin.value()
        event = QWheelEvent(
            QPoint(5, 5), QPoint(5, 5), QPoint(0, 0), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
        )
        from PyQt5.QtWidgets import QApplication

        QApplication.sendEvent(spin, event)
        assert spin.value() == pytest.approx(before)
