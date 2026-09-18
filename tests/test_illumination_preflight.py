"""Split/Blend must not start a run it cannot honour.

On 2026-09-18 a 49-tile run was launched with Illumination fusion set to
Split. No illumination side was configured, so every tile fell back to Max and
said so — 50+ identical warnings — while the settings echo at the top still
read `Illumination fusion: split`. The output would have been Max under a Split
label, which the operator cannot detect by looking at it.

Two guards now. The pipeline refuses outright (tested in
test_illumination_fusion_modes.py via the library fallback), and the dialog
blocks BEFORE the run and shows where the setting lives — because an error that
only says "this is wrong" is barely better than the silent fallback.

Run: python -m pytest tests/test_illumination_preflight.py -q
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flamingo_stitcher.pipeline import StitchingConfig  # noqa: E402

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp):
    from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

    d = StitchingDialog()
    yield d
    d.deleteLater()


@pytest.fixture
def silent_box(monkeypatch):
    """Answer the modal without showing it, and capture what it said."""
    said = {}

    def fake_exec(self):
        said["text"] = self.text()
        said["informative"] = self.informativeText()
        said["title"] = self.windowTitle()
        return QMessageBox.Rejected

    monkeypatch.setattr(QMessageBox, "exec_", fake_exec)
    return said


def _unconfigured(monkeypatch, scope="n7"):
    """Make every queued acquisition resolve to a scope with no low side."""
    from flamingo_stitcher import scope_profiles

    monkeypatch.setattr(
        scope_profiles,
        "resolve_for_acquisition",
        lambda p: (scope, 6.2, {}, "none"),
    )


def _configured(monkeypatch, scope="n7", side=1):
    from flamingo_stitcher import scope_profiles

    monkeypatch.setattr(
        scope_profiles,
        "resolve_for_acquisition",
        lambda p: (scope, 6.2, {"illumination_low_side": side}, f"{scope}|6.2x"),
    )


PENDING = [{"path": "/data/acq"}]


class TestItBlocksWhatItCannotHonour:
    @pytest.mark.parametrize("method", ["split", "blend"])
    def test_an_unconfigured_microscope_stops_the_run(
        self, dialog, monkeypatch, silent_box, method
    ):
        _unconfigured(monkeypatch)
        cfg = StitchingConfig(illumination_fusion=method)
        assert dialog._confirm_illumination_low_side(PENDING, cfg) is False

    @pytest.mark.parametrize("method", ["split", "blend"])
    def test_a_configured_microscope_proceeds(
        self, dialog, monkeypatch, silent_box, method
    ):
        _configured(monkeypatch)
        cfg = StitchingConfig(illumination_fusion=method)
        assert dialog._confirm_illumination_low_side(PENDING, cfg) is True

    @pytest.mark.parametrize("method", ["max", "mean", "content", "leonardo"])
    def test_modes_that_need_no_geometry_are_never_blocked(
        self, dialog, monkeypatch, silent_box, method
    ):
        _unconfigured(monkeypatch)
        cfg = StitchingConfig(illumination_fusion=method)
        assert dialog._confirm_illumination_low_side(PENDING, cfg) is True

    def test_side_zero_counts_as_configured(self, dialog, monkeypatch, silent_box):
        # 0 is a real side; only -1 means unset. An `if not side` test here
        # would reject the valid answer.
        _configured(monkeypatch, side=0)
        cfg = StitchingConfig(illumination_fusion="split")
        assert dialog._confirm_illumination_low_side(PENDING, cfg) is True


class TestItSaysWhereToFixIt:
    def test_it_names_the_microscope_that_is_missing_the_setting(
        self, dialog, monkeypatch, silent_box
    ):
        _unconfigured(monkeypatch, scope="n7")
        dialog._confirm_illumination_low_side(
            PENDING, StitchingConfig(illumination_fusion="split")
        )
        assert "n7" in silent_box["text"]

    def test_it_names_the_exact_place_to_set_it(
        self, dialog, monkeypatch, silent_box
    ):
        _unconfigured(monkeypatch)
        dialog._confirm_illumination_low_side(
            PENDING, StitchingConfig(illumination_fusion="split")
        )
        info = silent_box["informative"]
        assert "Options tab" in info
        assert "LOW end of the frame" in info

    def test_it_says_how_to_find_out_which_side(
        self, dialog, monkeypatch, silent_box
    ):
        # The part a user genuinely cannot guess. Telling them to set a value
        # without telling them how to determine it is not a fix.
        _unconfigured(monkeypatch)
        dialog._confirm_illumination_low_side(
            PENDING, StitchingConfig(illumination_fusion="split")
        )
        assert "Separate" in silent_box["informative"]

    def test_it_offers_the_mode_that_needs_no_geometry(
        self, dialog, monkeypatch, silent_box
    ):
        _unconfigured(monkeypatch)
        dialog._confirm_illumination_low_side(
            PENDING, StitchingConfig(illumination_fusion="blend")
        )
        assert "Content" in silent_box["informative"]


class TestItNeverCrashesTheDialog:
    def test_a_broken_resolve_still_blocks_rather_than_raising(
        self, dialog, monkeypatch, silent_box
    ):
        from flamingo_stitcher import scope_profiles

        def boom(_p):
            raise RuntimeError("unreadable ScopeSettings.txt")

        monkeypatch.setattr(scope_profiles, "resolve_for_acquisition", boom)
        cfg = StitchingConfig(illumination_fusion="split")
        assert dialog._confirm_illumination_low_side(PENDING, cfg) is False

    def test_an_item_with_no_path_is_skipped(self, dialog, monkeypatch, silent_box):
        _unconfigured(monkeypatch)
        cfg = StitchingConfig(illumination_fusion="split")
        assert dialog._confirm_illumination_low_side([{}], cfg) is True
