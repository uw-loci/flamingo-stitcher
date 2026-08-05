"""Live before/after preview for the destripe filter on a single tile plane.

Tuning destriping by running a whole stitch is a very slow feedback loop, and
several of the ways it can go wrong are invisible in the finished mosaic: which
axis the filter is removing, whether the requested wavelet level is deeper than
the frame supports, and how much the per-plane Otsu split changes the result
between a dim edge tile and a bright interior one. This dialog puts one plane
on screen with the filter applied live.

Two deliberate choices about what is shown:

* **Destriping is applied in the RAW CAMERA FRAME**, exactly where the pipeline
  does it (before the per-tile rot/flip), and the orientation transform is
  applied to BOTH panes for display only. Filtering the already-oriented plane
  would preview something the pipeline never computes, and would silently
  disagree about the stripe axis.
* **Illumination sides are shown separately**, because non-fast destriping runs
  per side before fusion. Fusing first would hide a side-specific stripe
  pattern.

Cost is a memmap seek plus one ``filter_streaks`` call: ~30 ms at 512x512,
~150 ms at 1024x1024, ~850 ms at 2048x2048 — hence the debounce and the
off-thread render.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSlider,
    QVBoxLayout,
)

from flamingo_stitcher.gui._compat import PersistentDialog
from flamingo_stitcher.gui.destripe_settings_dialog import (
    DEFAULTS,
    DestripeParamsForm,
)

logger = logging.getLogger(__name__)

_PANE_PX = 420
# Long enough that dragging a spinbox doesn't queue a render per keystroke,
# short enough to still feel live at the 512/1024 frame sizes.
_DEBOUNCE_MS = 250


def _to_qpixmap(plane: np.ndarray, max_px: int = _PANE_PX) -> QPixmap:
    """Render a 2-D array to a grayscale QPixmap using a robust stretch.

    Percentile limits rather than min/max: a single hot pixel would otherwise
    crush the whole plane to black and make the comparison useless.
    """
    a = np.asarray(plane, dtype=np.float32)
    if a.size == 0:
        return QPixmap()
    lo, hi = np.percentile(a, (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    buf = np.ascontiguousarray((norm * 255).astype(np.uint8))
    h, w = buf.shape
    img = QImage(buf.data, w, h, w, QImage.Format_Grayscale8)
    # .copy() detaches from the numpy buffer, which is about to go out of scope.
    pix = QPixmap.fromImage(img.copy())
    return pix.scaled(max_px, max_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class _RenderThread(QThread):
    """Destripe one plane off the UI thread."""

    done = pyqtSignal(object, object, str)  # before|None, after|None, message

    def __init__(
        self,
        raw_path: Path,
        n_planes: int,
        frame_w: int,
        frame_h: int,
        z: int,
        direction: str,
        params: Dict[str, Any],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._args = (raw_path, n_planes, frame_w, frame_h, z, direction, params)

    def run(self) -> None:  # noqa: D401
        raw_path, n_planes, frame_w, frame_h, z, direction, params = self._args
        try:
            from flamingo_stitcher.pipeline import destripe_volume, load_tile_volume

            vol = load_tile_volume(raw_path, n_planes, frame_w, frame_h)
            z = max(0, min(int(z), vol.shape[0] - 1))
            before = np.array(vol[z])  # detach the single plane from the memmap

            # Run the real filter on a 1-plane volume so the preview goes
            # through the SAME code path as a run (level clamping included).
            after_vol = destripe_volume(
                before[None, ...],
                max_workers=1,
                direction=direction,
                params=params,
            )
            self.done.emit(before, np.asarray(after_vol)[0], "")
        except Exception as e:  # noqa: BLE001 - surfaced in the dialog
            logger.warning(f"Destripe preview failed: {e}", exc_info=True)
            self.done.emit(None, None, str(e))


class DestripePreviewDialog(PersistentDialog):
    """Pick a tile, scrub Z, and see the destripe filter applied live."""

    def __init__(self, acq_path: Path, params: Dict[str, Any] | None = None,
                 direction: str = "auto", parent=None) -> None:
        super().__init__(parent)
        self._acq_path = Path(acq_path)
        self._direction_setting = (direction or "auto").lower()
        self.setWindowTitle(f"Destripe preview — {self._acq_path.name}")
        self.setMinimumWidth(980)

        self._tiles: List[Any] = []
        self._thread: Optional[_RenderThread] = None
        self._pending = False
        self._resolved_direction = self._direction_setting

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._render)

        self._build_ui(params or {})
        self._load_tiles()

    # -- construction ----------------------------------------------------
    def _build_ui(self, params: Dict[str, Any]) -> None:
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Destriping is applied in the raw camera frame — where the run "
            "applies it — and both panes are then rotated for display, so what "
            "you see matches the stitched result without faking the filter."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#555; font-size:11px;")
        layout.addWidget(intro)

        # --- source selectors ---
        sel = QHBoxLayout()
        sel.addWidget(QLabel("Tile:"))
        self._tile_combo = QComboBox()
        self._tile_combo.setMinimumWidth(240)
        self._tile_combo.currentIndexChanged.connect(self._on_tile_changed)
        sel.addWidget(self._tile_combo, 1)

        sel.addWidget(QLabel("Channel:"))
        self._channel_combo = QComboBox()
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        sel.addWidget(self._channel_combo)

        sel.addWidget(QLabel("Illum. side:"))
        self._illum_combo = QComboBox()
        self._illum_combo.setToolTip(
            "Non-fast destriping runs per illumination side, BEFORE fusion, so "
            "each side is previewed on its own."
        )
        self._illum_combo.currentIndexChanged.connect(self._request_render)
        sel.addWidget(self._illum_combo)
        layout.addLayout(sel)

        # --- Z scrubber ---
        zrow = QHBoxLayout()
        zrow.addWidget(QLabel("Z plane:"))
        self._z_slider = QSlider(Qt.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        zrow.addWidget(self._z_slider, 1)
        self._z_label = QLabel("—")
        self._z_label.setMinimumWidth(90)
        zrow.addWidget(self._z_label)
        layout.addLayout(zrow)

        # --- image panes ---
        panes = QHBoxLayout()
        self._before_label = QLabel("Select a tile")
        self._after_label = QLabel("")
        for lbl, title in ((self._before_label, "Before"), (self._after_label, "After")):
            box = QGroupBox(title)
            inner = QVBoxLayout(box)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(_PANE_PX, _PANE_PX)
            lbl.setStyleSheet("background:#111;")
            inner.addWidget(lbl)
            panes.addWidget(box)
        layout.addLayout(panes)

        # --- status readout ---
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#333; font-size:11px;")
        layout.addWidget(self._status)

        # --- parameters (the SAME widgets as the settings dialog) ---
        pbox = QGroupBox("Destripe parameters")
        pv = QVBoxLayout(pbox)
        self._form = DestripeParamsForm(params)
        self._form.changed.connect(self._request_render)
        pv.addWidget(self._form)

        drow = QHBoxLayout()
        drow.addWidget(QLabel("Direction:"))
        self._dir_combo = QComboBox()
        self._dir_combo.addItem("Auto (detect)", "auto")
        self._dir_combo.addItem("Horizontal stripes", "horizontal")
        self._dir_combo.addItem("Vertical stripes", "vertical")
        idx = self._dir_combo.findData(self._direction_setting)
        self._dir_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._dir_combo.currentIndexChanged.connect(self._request_render)
        self._dir_combo.setToolTip(
            "Which stripe orientation to remove, in the RAW CAMERA FRAME.\n"
            "Auto detects it; a run locks one axis for the whole acquisition."
        )
        drow.addWidget(self._dir_combo)

        self._orient_cb = QCheckBox("Show oriented (as in the stitch)")
        self._orient_cb.setChecked(True)
        self._orient_cb.setToolTip(
            "Display both panes with this acquisition's tile orientation "
            "applied. Display only — the filter always runs in the camera frame."
        )
        self._orient_cb.stateChanged.connect(self._redraw_only)
        drow.addWidget(self._orient_cb)
        drow.addStretch(1)
        pv.addLayout(drow)
        layout.addWidget(pbox)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
            | QDialogButtonBox.RestoreDefaults
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            lambda: self._form.set_params(DEFAULTS)
        )
        layout.addWidget(buttons)

    # -- data ------------------------------------------------------------
    def _load_tiles(self) -> None:
        try:
            from flamingo_stitcher.pipeline import discover_tiles

            self._tiles = discover_tiles(self._acq_path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(
                self, "Destripe preview", f"Could not read that acquisition:\n{e}"
            )
            self._tiles = []
            return

        if not self._tiles:
            QMessageBox.information(
                self,
                "Destripe preview",
                f"No tiles found in:\n{self._acq_path}",
            )
            return

        self._tile_combo.blockSignals(True)
        for i, t in enumerate(self._tiles):
            if getattr(t, "tile_index", None):
                label = f"X{t.tile_index[0]:03d}_Y{t.tile_index[1]:03d}"
            else:
                label = f"X{t.x_mm:.2f}_Y{t.y_mm:.2f}"
            self._tile_combo.addItem(f"{i + 1}: {label}", i)
        self._tile_combo.blockSignals(False)
        self._on_tile_changed(0)

    def _current_tile(self):
        i = self._tile_combo.currentData()
        if i is None or not (0 <= i < len(self._tiles)):
            return None
        return self._tiles[i]

    def _on_tile_changed(self, _index: int) -> None:
        tile = self._current_tile()
        if tile is None:
            return

        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        for ch in sorted(tile.raw_files.keys()):
            self._channel_combo.addItem(f"C{ch:02d}", ch)
        self._channel_combo.blockSignals(False)

        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(max(0, int(tile.n_planes) - 1))
        self._z_slider.setValue(max(0, int(tile.n_planes) // 2))
        self._z_slider.blockSignals(False)

        self._on_channel_changed(0)

    def _on_channel_changed(self, _index: int) -> None:
        tile = self._current_tile()
        ch = self._channel_combo.currentData()
        if tile is None or ch is None:
            return
        sides = sorted(tile.raw_files.get(ch, {}).keys())
        self._illum_combo.blockSignals(True)
        self._illum_combo.clear()
        for s in sides:
            self._illum_combo.addItem(f"I{s}", s)
        self._illum_combo.blockSignals(False)
        self._request_render()

    def _on_z_changed(self, value: int) -> None:
        tile = self._current_tile()
        total = int(tile.n_planes) if tile is not None else 0
        self._z_label.setText(f"{value + 1} / {total}")
        self._request_render()

    # -- rendering -------------------------------------------------------
    def _request_render(self, *_args) -> None:
        self._debounce.start()

    def _render(self) -> None:
        tile = self._current_tile()
        ch = self._channel_combo.currentData()
        side = self._illum_combo.currentData()
        if tile is None or ch is None or side is None:
            return
        raw_path = tile.raw_files.get(ch, {}).get(side)
        if raw_path is None:
            return

        if self._thread is not None and self._thread.isRunning():
            # Coalesce: one more render after the in-flight one finishes, so a
            # burst of edits costs two renders rather than one per edit.
            self._pending = True
            return

        self._status.setText("Rendering…")
        self._thread = _RenderThread(
            Path(raw_path),
            int(tile.n_planes),
            int(tile.frame_width),
            int(tile.frame_height),
            int(self._z_slider.value()),
            str(self._dir_combo.currentData()),
            self._form.get_params(),
            parent=self,
        )
        self._thread.done.connect(self._on_rendered)
        self._thread.start()

    def _on_rendered(self, before, after, message: str) -> None:
        if before is None:
            self._status.setText(f"Preview failed: {message}")
        else:
            self._before_raw = before
            self._after_raw = after
            self._redraw_only()
            self._status.setText(self._describe(before, after))

        if self._pending:
            self._pending = False
            self._request_render()

    def _redraw_only(self, *_args) -> None:
        """Re-render the panes from the cached planes (no refiltering)."""
        before = getattr(self, "_before_raw", None)
        after = getattr(self, "_after_raw", None)
        if before is None or after is None:
            return
        before_disp, after_disp = before, after
        if self._orient_cb.isChecked():
            ori = self._tile_orientation()
            if ori is not None:
                before_disp = ori.apply2d(before)
                after_disp = ori.apply2d(after)
        self._before_label.setPixmap(_to_qpixmap(before_disp))
        self._after_label.setPixmap(_to_qpixmap(after_disp))

    def _tile_orientation(self):
        try:
            from flamingo_stitcher.orientation import (
                MosaicOrientation,
                resolve_tile_orientation,
            )

            ori = resolve_tile_orientation(self._acq_path)
            if ori is None or not ori.name:
                return None
            return MosaicOrientation(ori.name)
        except Exception as e:  # noqa: BLE001 - display nicety only
            logger.debug(f"No tile orientation for preview: {e!r}")
            return None

    def _describe(self, before: np.ndarray, after: np.ndarray) -> str:
        """Report what the filter actually did, not just that it ran."""
        parts: List[str] = []

        setting = str(self._dir_combo.currentData())
        if setting == "auto":
            try:
                from flamingo_stitcher.pipeline import _stripe_axis_vote

                axis, conf = _stripe_axis_vote(before[None, ...])
                parts.append(f"Direction: {axis} (auto, confidence {conf:.2f})")
            except Exception:  # noqa: BLE001
                parts.append("Direction: auto")
        else:
            parts.append(f"Direction: {setting} (forced)")

        # Effective wavelet level after the frame-size clamp.
        try:
            from flamingo_stitcher._pystripe_core import max_level

            requested = int(self._form.get_params().get("level") or 0)
            wavelet = str(self._form.get_params().get("wavelet") or "db2")
            usable = int(max_level(int(min(before.shape)), wavelet))
            if requested > 0 and requested > usable >= 1:
                parts.append(
                    f"Level: {requested} → clamped to {usable} "
                    f"({min(before.shape)}px frame)"
                )
            elif requested > 0:
                parts.append(f"Level: {requested}")
            else:
                parts.append(f"Level: auto ({usable})")
        except Exception:  # noqa: BLE001
            pass

        removed = self._stripe_reduction(before, after)
        if removed is not None:
            parts.append(f"Stripe power removed: {removed:.0f}%")

        parts.append(f"Frame: {before.shape[1]}×{before.shape[0]}")
        return "   ·   ".join(parts)

    @staticmethod
    def _stripe_reduction(before: np.ndarray, after: np.ndarray) -> Optional[float]:
        """Percent of high-frequency stripe power removed, in the camera frame.

        Measured along whichever axis carries more of it, so the number tracks
        the stripes rather than the display orientation.
        """
        try:
            def power(img: np.ndarray, axis: int) -> float:
                prof = img.astype(np.float32).mean(axis=axis)
                f = np.abs(np.fft.rfft(prof - prof.mean())) ** 2
                return float(f[max(1, len(f) // 8):].sum())

            axis = 0 if power(before, 0) >= power(before, 1) else 1
            b = power(before, axis)
            if b <= 0:
                return None
            return max(0.0, min(100.0, 100.0 * (1.0 - power(after, axis) / b)))
        except Exception:  # noqa: BLE001
            return None

    # -- results ---------------------------------------------------------
    def get_params(self) -> Dict[str, Any]:
        return self._form.get_params()

    def get_direction(self) -> str:
        return str(self._dir_combo.currentData())

    def closeEvent(self, event):  # noqa: N802 - Qt override
        self._debounce.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(3000)
        super().closeEvent(event)
