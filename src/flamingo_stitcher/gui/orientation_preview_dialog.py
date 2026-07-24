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
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from flamingo_stitcher.orientation import MosaicOrientation

_CELL_PX = 260


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

    def __init__(self, acq_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._acq_path = Path(acq_path)

    def run(self) -> None:  # noqa: D401
        try:
            from flamingo_stitcher.orientation import (
                build_mip_mosaic,
                orientation_previews,
                read_microscope_name,
                resolve_output_orientation,
            )

            name = read_microscope_name(self._acq_path)
            preset = resolve_output_orientation(self._acq_path)
            mosaic = build_mip_mosaic(self._acq_path)
            if mosaic is None:
                self.failed.emit(
                    "Could not build a preview — no tiles or MIP (*_MP.tif) "
                    "files were found in this acquisition."
                )
                return
            self.done.emit(orientation_previews(mosaic), name, preset)
        except Exception as e:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(f"Orientation preview failed: {e}")


class OrientationPreviewDialog(QDialog):
    """Shows the eight whole-mosaic orientations for one acquisition."""

    def __init__(self, acq_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._acq_path = Path(acq_path)
        self.setWindowTitle(f"Orientation preview — {self._acq_path.name}")
        self.setMinimumSize(1100, 640)
        self._thread: Optional[_PreviewBuildThread] = None
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
        self._rebuild_btn = QPushButton("Rebuild")
        self._rebuild_btn.setToolTip("Re-read the MIPs and rebuild the preview")
        self._rebuild_btn.clicked.connect(self._start_build)
        self._rebuild_btn.setEnabled(False)
        btn_row.addWidget(self._rebuild_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    def _start_build(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._rebuild_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._header.setText(
            "Building preview from per-tile MIPs… reading *_MP.tif files."
        )
        self._thread = _PreviewBuildThread(self._acq_path, self)
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
        bits = [
            f"Microscope: <b>{name or '(none found)'}</b>",
            f"Current preset: <b>{preset or '(none)'}</b>",
        ]
        self._header.setText(
            "  •  ".join(bits)
            + "<br>Pick the panel where the sample is framed correctly "
            "(e.g. low stage X on the right, low stage Y at the bottom). "
            "That orientation name is this system's setting."
        )
        self._clear_grid()
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
            cap = QLabel(f"{i + 1}. {key}")
            cap.setAlignment(Qt.AlignCenter)
            v.addWidget(img_label)
            v.addWidget(cap)
            self._grid.addWidget(cell, i // cols, i % cols)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(3000)
        super().closeEvent(event)
