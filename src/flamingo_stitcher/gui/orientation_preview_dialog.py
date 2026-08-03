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
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._acq_path = Path(acq_path)
        self._z_range = z_range
        self._reverse_x = reverse_x
        self._reverse_y = reverse_y

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
            "Which Z planes to project. Structure often lives in part of the "
            "stack (e.g. the bottom), while the full-stack projection is washed "
            "out by beads across all depths. Changing this rebuilds the preview "
            "by reading those planes from the raw data."
        )
        self._z_combo.currentIndexChanged.connect(self._on_z_changed)
        btn_row.addWidget(self._z_combo)

        # Tile-ORDER reversal (stage-sign): flips the tile layout along an axis
        # WITHOUT touching the tile pixels. Independent of the per-tile
        # orientation panels — a system can need a per-tile flip in X but a tile-
        # order reversal in Y.
        self._rev_x_cb = QCheckBox("Reverse X order")
        self._rev_x_cb.setToolTip(
            "Lay tiles out X3 X2 X1 X0 instead of X0 X1 X2 X3 (stage X sign), "
            "without flipping the images."
        )
        self._rev_x_cb.toggled.connect(self._on_reverse_toggled)
        btn_row.addWidget(self._rev_x_cb)
        self._rev_y_cb = QCheckBox("Reverse Y order")
        self._rev_y_cb.setToolTip("Reverse the tile order along stage Y.")
        self._rev_y_cb.toggled.connect(self._on_reverse_toggled)
        btn_row.addWidget(self._rev_y_cb)

        self._rebuild_btn = QPushButton("Rebuild")
        self._rebuild_btn.setToolTip("Re-read the data and rebuild the preview")
        self._rebuild_btn.clicked.connect(self._start_build)
        self._rebuild_btn.setEnabled(False)
        btn_row.addWidget(self._rebuild_btn)
        btn_row.addStretch()

        self._apply_btn = QPushButton("Use for stitching")
        self._apply_btn.setToolTip(
            "Select a panel (the one where tiles CONNECT), then apply that "
            "orientation + tile order to the stitch and remember it for this "
            "microscope."
        )
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(self._apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

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
            self._acq_path, self._z_range, self._reverse_x, self._reverse_y, self
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
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            if arr is not None and arr.size:
                img_label.setPixmap(_to_qpixmap(arr))
            img_label.setStyleSheet("background:#111; border:1px solid #444;")
            img_label.setMinimumSize(_CELL_PX, _CELL_PX)
            radio = QRadioButton(f"{i + 1}. {key}")
            radio.setProperty("orientation_name", key)
            if key == self._selected_name:
                radio.setChecked(True)
            self._radio_group.addButton(radio)
            v.addWidget(img_label)
            v.addWidget(radio, alignment=Qt.AlignCenter)
            self._grid.addWidget(cell, i // cols, i % cols)
        self._apply_btn.setEnabled(self._selected_name is not None)

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
        rev = []
        if self._reverse_x:
            rev.append("reverse X")
        if self._reverse_y:
            rev.append("reverse Y")
        suffix = (" + " + ", ".join(rev)) if rev else ""
        self._header.setText(
            f"Applied tile orientation <b>{self._selected_name}{suffix}</b>"
            f"{' and saved it for ' + self._microscope_name if self._microscope_name else ''}"
            ". Close this window; the next stitch uses it."
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(3000)
        super().closeEvent(event)
