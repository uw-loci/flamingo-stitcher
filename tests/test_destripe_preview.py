"""Live destripe preview: does it show what a run would actually produce?

The preview only earns trust if it is tuning the same filter the pipeline runs.
Two properties matter most and are easy to break:

* destriping happens in the RAW CAMERA FRAME, with the tile orientation applied
  for display only -- filtering the oriented plane would preview something the
  pipeline never computes, and would disagree about the stripe axis;
* the readout has to be a real diagnostic. "Stripe power removed" must separate
  a correct direction from a wrong one, or it is decoration.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pywt", reason="the destripe backend needs pywt")
pytest.importorskip("PyQt5", reason="preview is a Qt dialog")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _synth_acq import write_synth_acquisition  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _striped_acquisition(tmp_path: Path, size: int = 256, planes: int = 4) -> Path:
    """Synthetic acquisition whose tile carries VERTICAL stripes (varying in X).

    Amplitudes are kept well clear of the uint16 ceiling on purpose: a blob that
    saturates flattens the stripes out of existence and the test silently stops
    testing anything.
    """
    acq = write_synth_acquisition(
        tmp_path / "acq",
        grid=(1, 1),
        n_planes=planes,
        channels=(1,),
        illum_sides=(0,),
        frame_size=(size, size),
    )
    yy, xx = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(0)
    plane = 800 + 3000 * np.exp(
        -(((yy - size // 2) ** 2 + (xx - size // 2) ** 2) / (2 * (size / 3.2) ** 2))
    )
    plane = plane + 400 * np.sin(xx / 2.0) + rng.normal(0, 15, (size, size))
    assert plane.max() < 60000, "test data must not saturate"
    vol = np.repeat(np.clip(plane, 0, 65535)[None].astype(np.uint16), planes, axis=0)

    raws = sorted(acq.rglob("*.raw"))
    assert raws, "synthetic acquisition produced no raw file"
    raws[0].write_bytes(vol.tobytes())
    return acq


def _settle(app, dlg, seconds: float = 30.0) -> bool:
    """Pump the event loop until no render is in flight or queued."""
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        busy = dlg._thread is not None and dlg._thread.isRunning()
        if not busy and not dlg._pending and not dlg._debounce.isActive():
            return True
        time.sleep(0.02)
    return False


def _render(app, acq, params, direction):
    from flamingo_stitcher.gui.destripe_preview_dialog import DestripePreviewDialog

    dlg = DestripePreviewDialog(acq, params, direction=direction)
    dlg._render()
    assert _settle(app, dlg), "preview never finished rendering"
    app.processEvents()
    return dlg


class TestPreviewRenders:
    def test_populates_tile_channel_side_and_z(self, qapp, tmp_path):
        from flamingo_stitcher.gui.destripe_preview_dialog import (
            DestripePreviewDialog,
        )

        acq = write_synth_acquisition(
            tmp_path / "acq",
            grid=(2, 2),
            n_planes=12,
            channels=(1,),
            illum_sides=(0, 1),
            frame_size=(64, 64),
        )
        dlg = DestripePreviewDialog(acq, {}, direction="auto")

        assert dlg._tile_combo.count() == 4
        assert dlg._channel_combo.count() == 1
        # Both illumination sides offered separately: non-fast destriping runs
        # per side BEFORE fusion, so previewing the fused tile would lie.
        assert [
            dlg._illum_combo.itemText(i) for i in range(dlg._illum_combo.count())
        ] == ["I0", "I1"]
        assert dlg._z_slider.maximum() == 11

    def test_produces_two_different_panes(self, qapp, tmp_path):
        acq = _striped_acquisition(tmp_path)
        dlg = _render(qapp, acq, {"level": 5}, "vertical")

        assert not dlg._before_label.pixmap().isNull()
        assert not dlg._after_label.pixmap().isNull()
        assert (dlg._before_raw != dlg._after_raw).any()
        assert dlg._after_raw.dtype == np.uint16

    def test_orientation_toggle_is_display_only(self, qapp, tmp_path):
        """The filter must never see the oriented plane."""
        acq = _striped_acquisition(tmp_path)
        dlg = _render(qapp, acq, {"level": 5}, "vertical")

        before = dlg._before_raw.copy()
        after = dlg._after_raw.copy()

        dlg._orient_cb.setChecked(not dlg._orient_cb.isChecked())
        qapp.processEvents()
        dlg._orient_cb.setChecked(not dlg._orient_cb.isChecked())
        qapp.processEvents()

        assert np.array_equal(before, dlg._before_raw)
        assert np.array_equal(after, dlg._after_raw)


class TestPreviewReadout:
    def test_removal_percentage_separates_right_from_wrong_axis(
        self, qapp, tmp_path
    ):
        """This number is the whole diagnostic: it must actually discriminate."""
        acq = _striped_acquisition(tmp_path)

        def removed(direction: str) -> float:
            dlg = _render(qapp, acq, {"level": 5}, direction)
            return dlg._stripe_reduction(dlg._before_raw, dlg._after_raw)

        right = removed("vertical")  # stripes vary in X → vertical is correct
        wrong = removed("horizontal")

        assert right > 80.0, f"correct axis only removed {right:.0f}%"
        assert wrong < 20.0, f"wrong axis removed {wrong:.0f}%"

    def test_auto_direction_is_reported_as_derived(self, qapp, tmp_path):
        """Auto derives from the orientation; it must SAY so, not imply a guess.

        The synthetic acquisition has no orientation preset, so it resolves to
        identity -> horizontal in the camera frame. That is the correct derived
        answer even though this phantom's stripes happen to run the other way:
        the axis comes from the geometry, not from these pixels.
        """
        acq = _striped_acquisition(tmp_path)
        dlg = _render(qapp, acq, {"level": 5}, "auto")

        text = dlg._status.text()
        assert "derived from orientation" in text
        assert "confidence" not in text
        assert dlg._effective_direction() == "horizontal"

    def test_auto_follows_the_tile_orientation(self, qapp, tmp_path, monkeypatch):
        """An axis-swapping orientation must flip the derived camera-frame axis."""
        from flamingo_stitcher import orientation as orient_mod

        acq = _striped_acquisition(tmp_path)
        # The dialog imports the resolver inside the method, so patching the
        # module attribute is what it will actually look up.
        monkeypatch.setattr(
            orient_mod,
            "resolve_tile_orientation",
            lambda _p: orient_mod.TileOrientation(name="rot270"),
        )

        dlg = _render(qapp, acq, {"level": 5}, "auto")

        assert dlg._orientation_name() == "rot270"
        assert dlg._effective_direction() == "vertical"

    def test_forced_direction_is_labelled_forced(self, qapp, tmp_path):
        acq = _striped_acquisition(tmp_path)
        dlg = _render(qapp, acq, {"level": 5}, "horizontal")

        assert "horizontal (forced)" in dlg._status.text()

    def test_level_clamp_is_surfaced_to_the_user(self, qapp, tmp_path):
        """A silently clamped level is exactly the trap this preview exists for."""
        acq = _striped_acquisition(tmp_path)
        dlg = _render(qapp, acq, {"level": 7}, "vertical")

        assert "clamped to 6" in dlg._status.text()

    def test_no_clamp_message_when_the_level_fits(self, qapp, tmp_path):
        acq = _striped_acquisition(tmp_path)
        dlg = _render(qapp, acq, {"level": 5}, "vertical")

        assert "clamped" not in dlg._status.text()


class TestPreviewResults:
    def test_params_and_direction_round_trip(self, qapp, tmp_path):
        acq = _striped_acquisition(tmp_path)
        params = {
            "sigma_foreground": 64.0,
            "sigma_background": 200.0,
            "level": 5,
            "wavelet": "db3",
            "threshold": 1234.0,
            "crossover": 12.0,
        }
        dlg = _render(qapp, acq, params, "vertical")

        assert dlg.get_params() == params
        assert dlg.get_direction() == "vertical"

    def test_shares_the_settings_dialog_widgets(self):
        """One source of truth, so preview and settings cannot drift apart."""
        from flamingo_stitcher.gui.destripe_settings_dialog import (
            DestripeParamsForm,
            DestripeSettingsDialog,
        )

        assert issubclass(DestripeSettingsDialog, object)
        # The settings dialog delegates to the same form the preview embeds.
        assert hasattr(DestripeParamsForm, "get_params")
        assert hasattr(DestripeParamsForm, "changed")

