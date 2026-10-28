"""GUI dialog: preview all 8 whole-mosaic orientations to pick the right one.

Assembles a fast, low-res mosaic from an acquisition's per-tile MIP files and
shows the eight dihedral orientations side by side, so a user can visually pick
the one where the sample is framed correctly (hard to judge on a beads sample
otherwise). Building the mosaic reads the small ``*_MP.tif`` files, so it runs
on a background thread to keep the UI responsive.

Determination only for now: it tells the user which orientation is correct.
Applying it to the fused output (and a per-microscope preset selector) is wired
separately.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from flamingo_stitcher.gui._wheel_guard import install_wheel_guard

logger = logging.getLogger(__name__)

from flamingo_stitcher.orientation import MosaicOrientation

_CELL_PX = 260

# Z-projection presets: label -> (lo, hi) plane fractions, or None for full.
# "Bottom" = low plane indices (Z start); structure often lives there while the
# full-stack projection is washed out by beads spread across all Z.
_Z_PRESETS = [
    ("Full stack", None),
    ("Bottom 50%", (0.0, 0.5)),
    ("Bottom 33%", (0.0, 0.33)),
    ("Bottom 25%", (0.0, 0.25)),
    ("Bottom 10%", (0.0, 0.10)),
    ("Middle 50%", (0.25, 0.75)),
    ("Top 50%", (0.5, 1.0)),
    ("Top 25%", (0.75, 1.0)),
]


class OrientationZoomDialog(QDialog):
    """One orientation panel, big enough to read.

    The grid shows eight panels at _CELL_PX, but the arrays behind them are
    rendered at ``orientation_previews(target_long_px=...)`` — roughly four
    times that — with the tile labels burned in. So the detail needed to tell
    which panel has the tiles connected correctly is already there and was
    simply being thrown away by the thumbnail scale. This shows it at 1:1 and
    lets it be scaled further, which is what the MIP overview gets for free by
    living in napari.
    """

    _STEPS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)

    def __init__(self, name: str, arr: np.ndarray, parent=None):
        super().__init__(parent)
        from flamingo_stitcher.orientation import orientation_label

        self.setWindowTitle(f"Orientation: {orientation_label(name)}")
        self._arr = arr
        self._zoom_idx = self._STEPS.index(1.0)
        self.chosen = False

        outer = QVBoxLayout(self)
        hint = QLabel(
            "Ctrl + mouse wheel to zoom, drag the scrollbars to pan. "
            "Pick the panel where neighboring tiles CONNECT."
        )
        hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        outer.addWidget(hint)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background:#111;")
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._scroll, 1)

        row = QHBoxLayout()
        for text, delta, tip in (
            ("−", -1, "Zoom out (or Ctrl + mouse wheel)."),
            ("Reset", 0, "Back to 1:1."),
            ("+", +1, "Zoom in (or Ctrl + mouse wheel)."),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFixedWidth(70)
            b.clicked.connect(lambda _c, d=delta: self._zoom(d))
            row.addWidget(b)
        self._zoom_label = QLabel()
        row.addWidget(self._zoom_label)
        row.addStretch(1)
        use = QPushButton("Choose this one")
        use.setToolTip(
            "Select this orientation and go back to the preview. Press 'Use "
            "for stitching' there to apply it."
        )
        use.setDefault(True)
        use.clicked.connect(self._accept_choice)
        row.addWidget(use)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        outer.addLayout(row)

        h, w = arr.shape[:2]
        self.resize(min(1400, w + 80), min(950, h + 150))
        self._render()

    def _zoom(self, delta: int) -> None:
        self._zoom_idx = (
            self._STEPS.index(1.0)
            if delta == 0
            else max(0, min(len(self._STEPS) - 1, self._zoom_idx + delta))
        )
        self._render()

    def _render(self) -> None:
        factor = self._STEPS[self._zoom_idx]
        h, w = self._arr.shape[:2]
        pix = _to_qpixmap(self._arr, max_px=max(w, h))
        if factor != 1.0:
            # Fast transform when enlarging: smoothing a burned-in label past
            # 1:1 blurs the very text this dialog exists to make readable.
            mode = Qt.FastTransformation if factor > 1.0 else Qt.SmoothTransformation
            pix = pix.scaled(int(w * factor), int(h * factor), Qt.KeepAspectRatio, mode)
        self._label.setPixmap(pix)
        self._label.resize(pix.size())
        self._zoom_label.setText(f"{factor:g}x")

    def _accept_choice(self) -> None:
        self.chosen = True
        self.accept()

    def wheelEvent(self, event):  # noqa: N802 - Qt naming
        if event.modifiers() & Qt.ControlModifier:
            self._zoom(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)


class _ClickableLabel(QLabel):
    """A preview cell that opens the full-size view when clicked."""

    def __init__(self, on_click):
        super().__init__()
        self._on_click = on_click
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):  # noqa: N802 - Qt naming
        self._on_click()
        super().mousePressEvent(event)


def _to_qpixmap(arr01: np.ndarray, max_px: int = _CELL_PX) -> QPixmap:
    """Convert a float [0, 1] 2-D array to a grayscale QPixmap, scaled to fit."""
    a = np.ascontiguousarray((np.clip(arr01, 0.0, 1.0) * 255).astype(np.uint8))
    h, w = a.shape
    img = QImage(a.data, w, h, w, QImage.Format_Grayscale8)
    # .copy() detaches from the numpy buffer, which is about to go out of scope.
    pix = QPixmap.fromImage(img.copy())
    return pix.scaled(max_px, max_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class _PreviewBuildThread(QThread):
    """Builds the MIP mosaic + all 8 orientation previews off the UI thread."""

    done = pyqtSignal(object, object, object)  # previews|None, name, preset
    failed = pyqtSignal(str)

    def __init__(
        self, acq_path: Path, z_range=None, reverse_x=False, reverse_y=False,
        parent=None, channel=None, illum_side=None,
    ) -> None:
        super().__init__(parent)
        self._acq_path = Path(acq_path)
        self._z_range = z_range
        self._reverse_x = reverse_x
        self._reverse_y = reverse_y
        self._channel = channel
        self._illum_side = illum_side

    def run(self) -> None:  # noqa: D401
        try:
            from flamingo_stitcher.orientation import (
                orientation_previews,
                read_microscope_name,
                resolve_output_orientation,
            )

            name = read_microscope_name(self._acq_path)
            preset = resolve_output_orientation(self._acq_path)
            # One mosaic per orientation, each transforming EVERY TILE before
            # placement — so the panels differ in how tiles CONNECT.
            previews = orientation_previews(
                self._acq_path,
                z_range=self._z_range,
                reverse_x=self._reverse_x,
                reverse_y=self._reverse_y,
                channel=self._channel,
                illum_side=self._illum_side,
            )
            if not previews:
                self.failed.emit(
                    "Could not build a preview — no tiles or MIP (*_MP.tif) "
                    "files were found in this acquisition."
                )
                return
            self.done.emit(previews, name, preset)
        except Exception as e:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(f"Orientation preview failed: {e}")


class OrientationPreviewDialog(QDialog):
    """Preview the 8 per-tile orientations and choose one to stitch with.

    Emits ``orientation_chosen(tile_orientation, reverse_x, reverse_y,
    microscope_name)`` when the user clicks "Use for stitching", so the main
    dialog can apply it to the run and name the microscope it was saved for
    (empty string when the acquisition metadata has no microscope name).
    """

    orientation_chosen = pyqtSignal(str, bool, bool, str)

    def __init__(self, acq_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._acq_path = Path(acq_path)
        self.setWindowTitle(f"Orientation preview — {self._acq_path.name}")
        self.setMinimumSize(1160, 720)
        self._thread: Optional[_PreviewBuildThread] = None
        # State that must exist before _build_ui/_start_build read it.
        self._z_range = None
        self._reverse_x = False
        self._reverse_y = False
        self._selected_name: Optional[str] = None
        self._microscope_name: Optional[str] = None
        self._radio_group: Optional[QButtonGroup] = None
        self._build_ui()
        # Populate BEFORE the first build so it uses the chosen source rather
        # than silently defaulting to the lowest channel and side.
        self._populate_source_pickers()
        install_wheel_guard(self)
        self._start_build()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        self._header = QLabel(
            "Building preview from per-tile MIPs… reading *_MP.tif files."
        )
        self._header.setWordWrap(True)
        outer.addWidget(self._header)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate busy bar
        outer.addWidget(self._progress)

        # Controls live ABOVE the panels. They change WHAT is being previewed,
        # so burying them under a scroll area of eight images meant the tile
        # order flips — the control most often needed alongside the panels —
        # were the last thing anyone found.
        top = QHBoxLayout()

        top.addWidget(QLabel("Channel:"))
        self._channel_combo = QComboBox()
        self._channel_combo.setToolTip(
            "Which acquired channel to preview. Orientation is the same for "
            "all channels, but a dim one may not show whether tiles connect, "
            "so pick the channel with the clearest structure."
        )
        self._channel_combo.currentIndexChanged.connect(self._on_source_changed)
        top.addWidget(self._channel_combo)

        top.addWidget(QLabel("Illum. side:"))
        self._illum_combo = QComboBox()
        self._illum_combo.setToolTip(
            "Which light-sheet side to preview. Orientation is the same for "
            "both sides, so pick the one where the sample looks sharpest."
        )
        self._illum_combo.currentIndexChanged.connect(self._on_source_changed)
        top.addWidget(self._illum_combo)

        top.addSpacing(16)

        # Tile-ORDER reversal (stage-sign): flips the tile layout along an axis
        # WITHOUT touching the tile pixels. Independent of the per-tile
        # orientation panels — a system can need a per-tile flip in X but a
        # tile-order reversal in Y.
        self._rev_x_cb = QCheckBox("Reverse X order")
        self._rev_x_cb.setToolTip(
            "Lay the tiles out in the opposite order along X (X3 X2 X1 X0 "
            "instead of X0 X1 X2 X3) without flipping the images. Check it if "
            "the tiles look right individually but sit in the wrong order left "
            "to right."
        )
        self._rev_x_cb.toggled.connect(self._on_reverse_toggled)
        top.addWidget(self._rev_x_cb)
        self._rev_y_cb = QCheckBox("Reverse Y order")
        self._rev_y_cb.setToolTip(
            "Lay the tiles out in the opposite order along Y without flipping "
            "the images. Check it if the tiles look right individually but sit "
            "in the wrong order top to bottom."
        )
        self._rev_y_cb.toggled.connect(self._on_reverse_toggled)
        top.addWidget(self._rev_y_cb)

        top.addStretch()
        outer.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setSpacing(8)
        scroll.setWidget(self._grid_host)
        outer.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addWidget(QLabel("Z projection:"))
        self._z_combo = QComboBox()
        for label, _rng in _Z_PRESETS:
            self._z_combo.addItem(label)
        self._z_combo.setToolTip(
            "Which part of each stack to project for the preview. If 'Full "
            "stack' looks washed out (for example by beads at every depth), "
            "pick the part that holds the sample. Any choice other than 'Full "
            "stack' reads the raw data, so it takes longer."
        )
        self._z_combo.currentIndexChanged.connect(self._on_z_changed)
        btn_row.addWidget(self._z_combo)

        self._rebuild_btn = QPushButton("Rebuild")
        self._rebuild_btn.setToolTip(
            "Read the data again and rebuild all panels. Changing any setting "
            "here already does this; use it to retry after a failed build."
        )
        self._rebuild_btn.clicked.connect(self._start_build)
        self._rebuild_btn.setEnabled(False)
        btn_row.addWidget(self._rebuild_btn)
        btn_row.addStretch()

        self._apply_btn = QPushButton("Use for stitching")
        self._apply_btn.setToolTip(
            "Select the panel where the tiles connect, then press this to use "
            "that orientation and tile order for the next stitch. It is also "
            "saved for this microscope, so later runs use it automatically "
            "(not when the acquisition has no microscope name)."
        )
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(self._apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    def _populate_source_pickers(self) -> None:
        """Fill Channel / Illum. side from what the acquisition actually holds."""
        try:
            from flamingo_stitcher.orientation import available_channels_and_sides

            channels, sides = available_channels_and_sides(self._acq_path)
        except Exception as e:  # noqa: BLE001 - pickers must not block the preview
            logger.debug(f"Could not list channels/sides: {e!r}")
            channels, sides = [], []

        for combo, values, fmt in (
            (self._channel_combo, channels, lambda v: f"C{v:02d}"),
            (self._illum_combo, sides, lambda v: f"I{v}"),
        ):
            combo.blockSignals(True)
            combo.clear()
            for v in values:
                combo.addItem(fmt(v), v)
            combo.setEnabled(len(values) > 1)
            combo.blockSignals(False)

    def _on_source_changed(self, _index: int) -> None:
        """Channel or illumination side changed — rebuild from that source."""
        self._start_build()

    def _on_reverse_toggled(self, _checked: bool) -> None:
        self._reverse_x = self._rev_x_cb.isChecked()
        self._reverse_y = self._rev_y_cb.isChecked()
        self._start_build()

    def _on_z_changed(self, index: int) -> None:
        if 0 <= index < len(_Z_PRESETS):
            self._z_range = _Z_PRESETS[index][1]
            self._start_build()

    def _start_build(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._rebuild_btn.setEnabled(False)
        self._progress.setVisible(True)
        full = self._z_range is None
        self._header.setText(
            "Building preview from per-tile MIPs…"
            if full
            else "Building preview — projecting the selected Z range from the "
            "raw data (this reads image data, so it may take a moment)…"
        )
        self._thread = _PreviewBuildThread(
            self._acq_path,
            self._z_range,
            self._reverse_x,
            self._reverse_y,
            self,
            channel=self._channel_combo.currentData(),
            illum_side=self._illum_combo.currentData(),
        )
        self._thread.done.connect(self._on_done)
        self._thread.failed.connect(self._on_failed)
        self._thread.start()

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _on_failed(self, message: str) -> None:
        self._progress.setVisible(False)
        self._rebuild_btn.setEnabled(True)
        self._header.setText(message)

    def _on_done(
        self,
        previews: Dict[str, np.ndarray],
        name: Optional[str],
        preset: Optional[str],
    ) -> None:
        self._progress.setVisible(False)
        self._rebuild_btn.setEnabled(True)
        z_label = (
            _Z_PRESETS[self._z_combo.currentIndex()][0]
            if 0 <= self._z_combo.currentIndex() < len(_Z_PRESETS)
            else "Full stack"
        )
        bits = [
            f"Microscope: <b>{name or '(none found)'}</b>",
            f"Preset: <b>{preset or '(none)'}</b>",
            f"Z: <b>{z_label}</b>",
        ]
        self._microscope_name = name
        self._header.setText(
            "  •  ".join(bits)
            + "<br>Each panel re-orients <b>every tile</b> and re-tiles at the "
            "stage grid. Select the one where the tissue is <b>continuous across "
            "the seams</b> (tiles connect) — use <b>Reverse X/Y order</b> if the "
            "tiles are laid out backwards — then click <b>Use for stitching</b>. "
            "Tiles are labelled by grid index (e.g. <b>X0Y0</b>). If scattered "
            "beads hide the structure, narrow the Z projection."
        )
        self._clear_grid()
        self._radio_group = QButtonGroup(self)
        self._radio_group.buttonClicked.connect(self._on_radio_selected)
        cols = 4
        for i, key in enumerate(MosaicOrientation.NAMES):
            arr = previews.get(key)
            cell = QWidget()
            v = QVBoxLayout(cell)
            v.setContentsMargins(2, 2, 2, 2)
            img_label = _ClickableLabel(
                lambda k=key, a=arr: self._open_zoom(k, a)
            )
            img_label.setAlignment(Qt.AlignCenter)
            if arr is not None and arr.size:
                img_label.setPixmap(_to_qpixmap(arr))
            img_label.setStyleSheet("background:#111; border:1px solid #444;")
            img_label.setMinimumSize(_CELL_PX, _CELL_PX)
            img_label.setToolTip(
                f"Click to open '{MosaicOrientation.LABELS.get(key, key)}' full "
                f"size, where the tile labels are readable. You can choose it "
                f"from there too."
            )
            radio = QRadioButton(f"{i + 1}. {MosaicOrientation.LABELS.get(key, key)}")
            radio.setToolTip(
                "Select this orientation, then press 'Use for stitching'."
            )
            radio.setProperty("orientation_name", key)
            if key == self._selected_name:
                radio.setChecked(True)
            self._radio_group.addButton(radio)
            v.addWidget(img_label)
            v.addWidget(radio, alignment=Qt.AlignCenter)
            self._grid.addWidget(cell, i // cols, i % cols)
        self._apply_btn.setEnabled(self._selected_name is not None)

    def _open_zoom(self, name: str, arr) -> None:
        """Show one panel full size; selecting there selects it here too."""
        if arr is None or not getattr(arr, "size", 0):
            return
        dlg = OrientationZoomDialog(name, arr, self)
        dlg.exec_()
        if not dlg.chosen:
            return
        self._selected_name = name
        for button in self._radio_group.buttons():
            if button.property("orientation_name") == name:
                button.setChecked(True)
                break
        self._apply_btn.setEnabled(True)

    def _on_radio_selected(self, button) -> None:
        self._selected_name = button.property("orientation_name")
        self._apply_btn.setEnabled(True)

    def _on_apply(self) -> None:
        """Apply the selected orientation to the stitch and remember it."""
        if not self._selected_name:
            return
        # Persist per-microscope so future runs auto-apply (best-effort).
        try:
            from flamingo_stitcher.orientation import save_microscope_orientation

            if self._microscope_name:
                save_microscope_orientation(
                    self._microscope_name,
                    self._selected_name,
                    self._reverse_x,
                    self._reverse_y,
                )
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass
        self.orientation_chosen.emit(
            self._selected_name,
            self._reverse_x,
            self._reverse_y,
            self._microscope_name or "",
        )
        from flamingo_stitcher.orientation import describe_orientation

        self._header.setText(
            f"Applied tile orientation <b>"
            f"{describe_orientation(self._selected_name, self._reverse_x, self._reverse_y)}</b>"
            f"{' and saved it for ' + self._microscope_name if self._microscope_name else ''}"
            ". Close this window; the next stitch uses it."
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(3000)
        super().closeEvent(event)
