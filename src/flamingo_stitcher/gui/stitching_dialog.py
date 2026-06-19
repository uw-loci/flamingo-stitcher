"""Tile Stitching Dialog.

Non-modal dialog for stitching raw acquisition tile data into a single volume.
Operates on saved acquisition data on disk — no microscope connection required.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import QProcess, QSettings, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from flamingo_stitcher.gui._compat import PersistentDialog

logger = logging.getLogger(__name__)

# QSettings keys
_SETTINGS_GROUP = "StitchingDialog"


def _napari_available() -> bool:
    """True if napari can be imported.

    napari is an *optional* dependency, needed only for the background-zero
    threshold preview. When it is absent (e.g. the frozen Windows build, which
    deliberately excludes it), background zeroing still works numerically — only
    the live 3-D preview is disabled.
    """
    import importlib.util

    return importlib.util.find_spec("napari") is not None


class BackgroundZeroPanel(QWidget):
    """Per-channel background-zeroing controls for the stitching dialog.

    Lossy preprocessing: voxels at or below the per-channel threshold
    are zeroed in the fused dask graph. Compresses the empty space
    around cleared-tissue samples to almost nothing while leaving
    tissue voxels untouched.

    The user MUST inspect the per-channel preview before running, so
    the panel exposes a "Preview..." button that emits ``preview_requested``.
    The hosting dialog runs the pipeline at the configured preview
    downsample factors and feeds the result into a napari viewer.
    """

    preview_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent=parent)
        self._channel_spinboxes: Dict[int, QSpinBox] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._toggle = QPushButton("▶ Background zeroing (lossy)")
        self._toggle.setCheckable(True)
        self._toggle.setStyleSheet(
            "QPushButton { text-align: left; padding: 4px 8px; "
            "border: none; font-weight: bold; }"
        )
        self._toggle.toggled.connect(self._on_toggle)
        outer.addWidget(self._toggle)

        self._body = QGroupBox()
        self._body.setStyleSheet(
            "QGroupBox { border: 1px solid #ccc; border-radius: 4px; "
            "margin-top: 0px; padding-top: 6px; }"
        )
        body_layout = QVBoxLayout()
        body_layout.setSpacing(6)

        self._enable_cb = QCheckBox(
            "Zero voxels at or below per-channel threshold (lossy)"
        )
        self._enable_cb.setToolTip(
            "When ON, voxels with intensity <= the per-channel threshold\n"
            "are written as 0. Compresses empty space around the sample\n"
            "to almost nothing under blosc/zstd.\n\n"
            "LOSSY for the masked region. Click Preview... first to see\n"
            "what would be zeroed at coarse resolution before committing\n"
            "to a full-resolution run."
        )
        body_layout.addWidget(self._enable_cb)

        warn = QLabel(
            "⚠ Lossy. Always preview first. The full-res write applies "
            "the same threshold; if it would zero >99% of voxels in any "
            "channel the run aborts before any output is written."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #c25e00; font-size: 11px;")
        body_layout.addWidget(warn)

        self._channels_container = QWidget()
        self._channels_layout = QGridLayout(self._channels_container)
        self._channels_layout.setSpacing(4)
        self._channels_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self._channels_container)
        self._render_no_channels_placeholder()

        btn_row = QHBoxLayout()
        self._preview_btn = QPushButton("Preview…")
        self._preview_btn.setToolTip(
            "Run the stitching pipeline at coarse downsample without\n"
            "writing any output, then open a napari viewer with sliders\n"
            "to pick per-channel thresholds. Click Apply in the preview\n"
            "to copy the chosen values back into this panel."
        )
        self._preview_btn.clicked.connect(self.preview_requested)
        btn_row.addWidget(self._preview_btn)
        btn_row.addStretch()
        body_layout.addLayout(btn_row)

        self._body.setLayout(body_layout)
        self._body.setVisible(False)
        outer.addWidget(self._body)

    def _render_no_channels_placeholder(self) -> None:
        # Placeholder shown until tiles are discovered.
        for i in reversed(range(self._channels_layout.count())):
            item = self._channels_layout.takeAt(i)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        self._channel_spinboxes.clear()
        ph = QLabel("Discover tiles first — channels appear here.")
        ph.setStyleSheet("color: #888; font-style: italic;")
        self._channels_layout.addWidget(ph, 0, 0, 1, 2)

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._toggle.setText(("▼ " if checked else "▶ ") + "Background zeroing (lossy)")

    # Public API ----------------------------------------------------------
    def set_channels(self, channel_ids: List[int]) -> None:
        """Rebuild the per-channel threshold rows. Preserves any thresholds
        already set for channels that survive the rebuild.
        """
        previous = {ch: sb.value() for ch, sb in self._channel_spinboxes.items()}
        for i in reversed(range(self._channels_layout.count())):
            item = self._channels_layout.takeAt(i)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        self._channel_spinboxes.clear()

        if not channel_ids:
            self._render_no_channels_placeholder()
            return

        header_label = QLabel("Channel")
        header_label.setStyleSheet("font-weight: bold;")
        header_thresh = QLabel("Threshold (uint16)")
        header_thresh.setStyleSheet("font-weight: bold;")
        self._channels_layout.addWidget(header_label, 0, 0)
        self._channels_layout.addWidget(header_thresh, 0, 1)

        for row, ch_id in enumerate(channel_ids, start=1):
            self._channels_layout.addWidget(QLabel(f"ch {ch_id}"), row, 0)
            sb = QSpinBox()
            sb.setRange(0, 65535)
            sb.setSingleStep(10)
            sb.setValue(previous.get(ch_id, 0))
            sb.setToolTip(
                f"Channel {ch_id}: voxels with value <= this threshold "
                f"will be zeroed in the fused output. 0 disables thresholding "
                f"for this channel."
            )
            self._channels_layout.addWidget(sb, row, 1)
            self._channel_spinboxes[ch_id] = sb

    def is_enabled(self) -> bool:
        return self._enable_cb.isChecked()

    def set_enabled_state(self, enabled: bool) -> None:
        self._enable_cb.setChecked(bool(enabled))

    def thresholds(self) -> Dict[int, int]:
        # Only include non-zero entries — a zero threshold is a no-op
        # and we don't want it counted in the safety-cap audit log.
        return {
            ch: int(sb.value())
            for ch, sb in self._channel_spinboxes.items()
            if sb.value() > 0
        }

    def set_thresholds(self, thresholds: Dict[int, int]) -> None:
        """Apply persisted or preview-Applied thresholds. Channels not in
        ``thresholds`` are left at their current value (typically 0).
        """
        for ch, val in thresholds.items():
            sb = self._channel_spinboxes.get(int(ch))
            if sb is not None:
                sb.setValue(int(val))

    def expanded(self) -> bool:
        return self._toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(bool(expanded))


class _FlatFieldSetupThread(QThread):
    """Builds the pixi flat-field environment off the UI thread.

    Emits ``progress`` lines as pixi downloads Python + basicpy, then ``done``
    with (success, message). All the heavy lifting lives in
    ``preprocessing_env.build_env`` — this just bridges it to Qt signals.
    """

    progress = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def run(self):
        try:
            from flamingo_stitcher import preprocessing_env

            preprocessing_env.build_env(self.progress.emit)
            self.done.emit(True, "")
        except Exception as e:  # surfaced to the user in _on_setup_finished
            self.done.emit(False, str(e))


class StitchingDialog(PersistentDialog):
    """Dialog for stitching raw acquisition tile data.

    Provides UI for configuring and running the stitching pipeline:
    - Acquisition/output directory selection
    - Pixel size, Z step, downsample factor, illumination fusion, destripe
    - Tile discovery, pipeline execution with progress/log, cancellation
    """

    # Emitted when user wants to load stitched output into SampleView
    load_stitched_requested = pyqtSignal(str)

    # How many directory levels to go up when restoring the acq dir path.
    # Subfolder-per-tile layout: grandparent (2 levels up).
    _acq_dir_restore_levels_up = 2

    def __init__(self, parent=None, **kwargs):
        # **kwargs forwards PersistentDialog options (geometry_manager,
        # window_id) so the host app can inject its own geometry manager when
        # embedding this dialog; standalone uses the module default.
        super().__init__(parent=parent, **kwargs)
        self._logger = logging.getLogger(__name__)
        self._worker = None

        # Whether a 3D "Sample View" exists to receive stitched output. True
        # when embedded in the Py2Flamingo control app (which connects
        # load_stitched_requested to its napari viewer); the standalone app
        # sets this False via set_sample_view_available() since it has no
        # viewer — so the "Load … into Sample View" completion button is hidden.
        self._sample_view_available = True

        # Batch queue state
        self._queue = []  # List of dicts: {path, status, tiles, error, output_path}
        self._queue_index = -1  # Index of currently processing item
        self._batch_running = False
        self._batch_config = None
        self._batch_channels = None
        self._batch_results = []  # List of (path, success, error_msg)

        self.setWindowTitle("Tile Stitching")
        # Keep the width comfortable for the settings row, but allow a short
        # window (laptops) — the upper controls live in a scroll area and the
        # log collapses, so nothing squishes; it scrolls instead.
        self.setMinimumWidth(650)
        self.setMinimumHeight(440)
        self.resize(720, 750)  # Default size before geometry restore
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._setup_ui()
        # Accept folders dropped anywhere on the dialog → add them to the queue.
        self.setAcceptDrops(True)
        self._restore_settings()
        # Initial paint of the Native → Output voxel readout using whatever
        # restore left in the spins + combos.
        self._update_voxel_readout()

    def _setup_ui(self):
        """Create and layout UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # The configuration controls (queue, settings, processing options) live
        # in a scroll area so they keep their natural size on small screens —
        # the window scrolls rather than squishing fields below readability.
        # The action buttons, log, and progress stay fixed below it.
        _scroll = QScrollArea()
        _scroll.setWidgetResizable(True)
        _scroll.setFrameShape(QScrollArea.NoFrame)
        _content = QWidget()
        content_layout = QVBoxLayout(_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        _scroll.setWidget(_content)
        layout.addWidget(_scroll, 1)

        # --- Batch queue ---
        queue_group = QGroupBox("Acquisition Queue")
        queue_layout = QVBoxLayout()
        queue_layout.setSpacing(4)

        self._queue_table = QTableWidget(0, 2)
        self._queue_table.setHorizontalHeaderLabels(["Status", "Directory"])
        self._queue_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._queue_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self._queue_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._queue_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._queue_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._queue_table.verticalHeader().setVisible(False)
        self._queue_table.setMaximumHeight(140)
        # Right-click menu + double-click to re-queue a finished item, so the
        # reset-to-Pending action is discoverable without hunting for the button.
        self._queue_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._queue_table.customContextMenuRequested.connect(
            self._on_queue_context_menu
        )
        self._queue_table.cellDoubleClicked.connect(
            lambda _row, _col: self._requeue_selected()
        )
        queue_layout.addWidget(self._queue_table)

        queue_btn_layout = QHBoxLayout()
        self._add_btn = QPushButton("Add...")
        self._add_btn.setToolTip("Add an acquisition directory to the queue")
        self._add_btn.clicked.connect(self._add_to_queue)
        queue_btn_layout.addWidget(self._add_btn)

        self._add_folder_btn = QPushButton("Add All in Folder...")
        self._add_folder_btn.setToolTip(
            "Select a parent folder and add all acquisition\n"
            "subdirectories to the queue"
        )
        self._add_folder_btn.clicked.connect(self._add_folder_to_queue)
        queue_btn_layout.addWidget(self._add_folder_btn)

        self._requeue_btn = QPushButton("Re-queue")
        self._requeue_btn.setToolTip(
            "Reset the selected Done / Cancelled / Error items back to Pending\n"
            "so they can be stitched again. This re-enables the Run button."
        )
        self._requeue_btn.clicked.connect(self._requeue_selected)
        queue_btn_layout.addWidget(self._requeue_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setToolTip("Remove selected directories from the queue")
        self._remove_btn.clicked.connect(self._remove_from_queue)
        queue_btn_layout.addWidget(self._remove_btn)

        queue_btn_layout.addStretch()
        queue_layout.addLayout(queue_btn_layout)
        queue_group.setLayout(queue_layout)
        content_layout.addWidget(queue_group)

        # --- Output directory ---
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output Directory:"))
        self._output_dir_edit = QLineEdit()
        self._output_dir_edit.setPlaceholderText(
            "Shared output folder (each acquisition gets a subfolder)..."
        )
        out_layout.addWidget(self._output_dir_edit)
        out_browse_btn = QPushButton("Browse...")
        out_browse_btn.clicked.connect(self._browse_output_dir)
        out_layout.addWidget(out_browse_btn)
        content_layout.addLayout(out_layout)

        # --- Settings group ---
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(6)

        # Load hardware-derived pixel size default
        try:
            from flamingo_stitcher.config_loader import get_hardware_config

            _hw = get_hardware_config()
            self._default_pixel_um = round(_hw.effective_pixel_size_um, 4)
        except Exception:
            self._default_pixel_um = 0.406

        # Row 0: Pixel size + Z step
        settings_layout.addWidget(QLabel("Pixel size (\u00b5m):"), 0, 0)
        self._pixel_size_spin = QDoubleSpinBox()
        self._pixel_size_spin.setRange(0.01, 100.0)
        self._pixel_size_spin.setDecimals(3)
        self._pixel_size_spin.setValue(self._default_pixel_um)
        self._pixel_size_spin.setSingleStep(0.001)
        settings_layout.addWidget(self._pixel_size_spin, 0, 1)

        settings_layout.addWidget(QLabel("Z step (\u00b5m):"), 0, 2)
        self._z_step_spin = QDoubleSpinBox()
        self._z_step_spin.setRange(0.0, 1000.0)
        self._z_step_spin.setDecimals(3)
        self._z_step_spin.setValue(0.0)
        self._z_step_spin.setSpecialValueText("Auto")
        self._z_step_spin.setToolTip(
            "0 = auto-detect from Workflow.txt Z range and plane count"
        )
        settings_layout.addWidget(self._z_step_spin, 0, 3)

        # Frame (camera AOI) size override. "Auto" derives the true frame size
        # from the on-disk file size (robust to cropped AOIs and missing/wrong
        # Workflow.txt metadata). Force a fixed crop when files may be truncated
        # or the auto-inference is ambiguous.
        settings_layout.addWidget(QLabel("Frame (AOI):"), 0, 4)
        self._frame_size_combo = QComboBox()
        for label, value in [
            ("Auto (from file)", None),
            ("1024 × 1024", 1024),
            ("2048 × 2048", 2048),
        ]:
            self._frame_size_combo.addItem(label, value)
        self._frame_size_combo.setToolTip(
            "Raw frame (camera AOI) size.\n"
            "Auto: inferred from each file's size (bytes / planes / 2 → "
            "square side) — recommended.\n"
            "Fix to 1024/2048 to force a crop when files may be partial or the "
            "AOI metadata is missing/wrong."
        )
        settings_layout.addWidget(self._frame_size_combo, 0, 5)

        # Row 1: Downsample XY/Z + Illumination fusion
        settings_layout.addWidget(QLabel("Downsample:"), 1, 0)
        ds_layout = QHBoxLayout()
        ds_layout.setSpacing(4)
        ds_layout.addWidget(QLabel("XY"))
        self._downsample_xy_combo = QComboBox()
        for label, value in [
            ("1x", 1),
            ("2x", 2),
            ("4x", 4),
            ("8x", 8),
            ("iso", -1),
        ]:
            self._downsample_xy_combo.addItem(label, value)
        self._downsample_xy_combo.setToolTip(
            "XY downsample factor.\n"
            "1x/2x/4x/8x reduces tile width and height.\n\n"
            "iso: auto-pick BOTH XY and Z factors so the output voxel\n"
            "is as close to cubic as possible, using the allowed\n"
            "factors (XY: 1/2/4/8, Z: 1/2/4). Resolved at run time\n"
            "from the acquisition's own Z step, so each item in a\n"
            "batch queue is sized independently. When iso is chosen\n"
            "the Z downsample combo is ignored and greys out.\n\n"
            "Examples at XY pixel = 0.406 \u00b5m:\n"
            "  Z 0.5 \u00b5m  \u2192 XY 1x, Z 1x  (0.41 \u00d7 0.50 \u00b5m)\n"
            "  Z 1.0 \u00b5m  \u2192 XY 2x, Z 1x  (0.81 \u00d7 1.00 \u00b5m)\n"
            "  Z 2.5 \u00b5m  \u2192 XY 8x, Z 1x  (3.25 \u00d7 2.50 \u00b5m)\n"
            "  Z 5.0 \u00b5m  \u2192 XY 8x, Z 1x  (3.25 \u00d7 5.00 \u00b5m)"
        )
        ds_layout.addWidget(self._downsample_xy_combo)
        ds_layout.addSpacing(16)  # visible gap between the XY group and Z group
        ds_layout.addWidget(QLabel("Z"))
        self._downsample_z_combo = QComboBox()
        for label, value in [("1x", 1), ("2x", 2), ("4x", 4)]:
            self._downsample_z_combo.addItem(label, value)
        self._downsample_z_combo.setToolTip(
            "Z downsample factor (independent of XY).\n"
            "Greys out when XY is set to 'iso' — iso overrides both."
        )
        ds_layout.addWidget(self._downsample_z_combo)
        # Absorb leftover width on the right so each label hugs its own combo,
        # instead of the (Expanding) combos stretching wide and opening big
        # gaps between each label and its dropdown.
        ds_layout.addStretch(1)
        # When XY=iso, Z is overridden by the iso-resolution algorithm at
        # run time, so grey out the Z combo as a visual cue.
        self._downsample_xy_combo.currentIndexChanged.connect(
            self._on_downsample_xy_changed
        )
        settings_layout.addLayout(ds_layout, 1, 1)

        settings_layout.addWidget(QLabel("Illum. fusion:"), 1, 2)
        self._fusion_combo = QComboBox()
        for label, value in [
            ("Max", "max"),
            ("Mean", "mean"),
            ("Leonardo FUSE", "leonardo"),
        ]:
            self._fusion_combo.addItem(label, value)
        self._fusion_combo.setToolTip(
            "Combines the LEFT and RIGHT light-sheet illumination sides of a\n"
            "single tile. Has no effect when only one illumination side was\n"
            "acquired. This is NOT how adjacent tiles are combined — see\n"
            "'Tile overlap'."
        )
        settings_layout.addWidget(self._fusion_combo, 1, 3)

        # Tile-overlap fusion (how adjacent TILES are combined — distinct from
        # the illumination-side fusion above). Placed alongside it to make the
        # distinction obvious.
        tile_fuse_box = QHBoxLayout()
        tile_fuse_box.setContentsMargins(0, 0, 0, 0)
        tile_fuse_box.addWidget(QLabel("Tile overlap:"))
        self._tile_fusion_combo = QComboBox()
        # Maximum first → it is the default on a fresh install (best for this
        # rig's light-sheet data; Blend available for dense/registered data).
        for label, value in [
            ("Maximum", "max"),
            ("Blend", "blend"),
        ]:
            self._tile_fusion_combo.addItem(label, value)
        self._tile_fusion_combo.setToolTip(
            "How OVERLAPPING TILES are combined (distinct from 'Illum. fusion',\n"
            "which combines left/right light-sheet sides of one tile):\n\n"
            "  Blend   — weighted cosine blend. Seamless on dense, well\n"
            "            flat-fielded samples. But in the overlap it averages\n"
            "            each tile against its neighbour, so a sparse sample on\n"
            "            a mostly-empty FOV (or any slight stage-only\n"
            "            misregistration) gets diluted against background — a\n"
            "            dark dip right at the seam.\n\n"
            "  Maximum — pixel-wise maximum across tiles. Keeps the brighter\n"
            "            tile in the overlap so signal can't be diluted. Best\n"
            "            for sparse / sub-FOV samples (e.g. a thin object that\n"
            "            doesn't fill the field of view)."
        )
        tile_fuse_box.addWidget(self._tile_fusion_combo)
        settings_layout.addLayout(tile_fuse_box, 1, 4)

        # Row 2: Output format + Compression
        settings_layout.addWidget(QLabel("Output format:"), 2, 0)
        self._format_combo = QComboBox()
        self._format_combo.addItem("OME-Zarr (Fiji compatible)", "ome-zarr-v2")
        self._format_combo.addItem("OME-Zarr Sharded", "ome-zarr-sharded")
        self._format_combo.addItem("OME-TIFF (single file)", "ome-tiff")
        self._format_combo.addItem("Imaris (.ims)", "imaris")
        self._format_combo.addItem("Both (Sharded + TIFF)", "both")
        # Disable Imaris option if PyImarisWriter not available
        try:
            from flamingo_stitcher.writers import imaris_writer

            if not imaris_writer.is_available():
                idx = self._format_combo.findData("imaris")
                if idx >= 0:
                    item = self._format_combo.model().item(idx)
                    item.setEnabled(False)
                    item.setToolTip(
                        "Imaris (.ims) unavailable:\n"
                        + imaris_writer.unavailable_reason()
                    )
        except Exception:
            pass

        self._format_combo.setToolTip(
            "OME-Zarr (Fiji compatible): opens in Fiji, QuPath, BigDataViewer, napari\n"
            "OME-Zarr Sharded: fewest files, napari only (Fiji cannot open)\n"
            "OME-TIFF: single file, universal viewer support\n"
            "Imaris (.ims): direct writer \u2014 opens correctly in Imaris (Windows only)\n"
            "Both: write Zarr Sharded + TIFF"
        )
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        settings_layout.addWidget(self._format_combo, 2, 1)

        # Format help label
        format_help = QLabel("?")
        format_help.setStyleSheet(
            "QLabel { color: #1976D2; font-weight: bold; font-size: 11px; "
            "border: 1px solid #1976D2; border-radius: 8px; "
            "padding: 0px 4px; "
            "qproperty-alignment: AlignCenter; }"
        )
        format_help.setToolTip(
            "<b>Output Format Guide</b><br><br>"
            "<b>OME-Zarr (Fiji compatible)</b> &mdash; Zarr v2 + OME-NGFF v0.4<br>"
            "Opens in <b>Fiji</b> (N5 plugin), <b>QuPath</b>, <b>BigDataViewer</b>, "
            "<b>napari</b>, and most bio-imaging tools.<br><br>"
            "<b>OME-Zarr Sharded</b> &mdash; Zarr v3 + OME-NGFF v0.5<br>"
            "Fewest files via sharding. Only readable by "
            "<b>napari</b> and zarr-python 3.x.<br><br>"
            "<b>OME-TIFF</b> &mdash; Single file per channel<br>"
            "Universally readable. Best for sharing and archiving.<br><br>"
            "<b>Imaris (.ims)</b> &mdash; Direct writer via PyImarisWriter (Apache 2.0)<br>"
            "Opens correctly in <b>Imaris</b> with auto-generated pyramids. "
            "Avoids ImarisFileConverter's SubIFD-pyramid bug that otherwise "
            "collapses all data into Z=0. Block-streamed write for TB-scale "
            "datasets. <b>Windows only</b> (<code>pip install PyImarisWriter</code>).<br><br>"
            "<b>Alternative for Imaris:</b> Export OME-TIFF with pyramids OFF, "
            "then run through ImarisFileConverter."
        )
        format_help.setFixedWidth(20)
        settings_layout.addWidget(format_help, 2, 2)

        settings_layout.addWidget(QLabel("Compression:"), 2, 3)
        self._compression_combo = QComboBox()
        self._compression_combo.setToolTip(
            "Compression codec for the output file.\n\n"
            "Options depend on the output format:\n"
            "  Zarr: zstd (recommended), lz4 (fastest), blosc, none\n"
            "  TIFF: zlib (best compatibility), lzw (balanced), "
            "zstd (best ratio), none\n\n"
            "Speed vs. storage tradeoff:\n"
            "• 'None' skips encode on write and decode on read, so it's\n"
            "  the fastest option on both ends when reading from a fast\n"
            "  local SSD / NVMe — at the cost of a much larger file.\n"
            "• Compressed codecs (zstd, lz4, zlib, lzw) produce smaller\n"
            "  files that can read FASTER than 'None' over slow disks,\n"
            "  network shares, or external HDDs, because disk I/O — not\n"
            "  CPU decode — is the bottleneck.\n"
            "• zstd / lz4 have modern fast decoders; zlib / lzw are the\n"
            "  universally-readable TIFF fallbacks for older software."
        )
        settings_layout.addWidget(self._compression_combo, 2, 4)
        self._update_compression_options()

        # Row 3: Channels + Memory mode
        settings_layout.addWidget(QLabel("Channels:"), 3, 0)
        self._channels_edit = QLineEdit()
        self._channels_edit.setPlaceholderText("All (or e.g. 0,1)")
        self._channels_edit.setToolTip(
            "Leave empty for all channels, or comma-separated list (e.g. 0,1)"
        )
        settings_layout.addWidget(self._channels_edit, 3, 1)

        settings_layout.addWidget(QLabel("Memory mode:"), 3, 3)
        self._streaming_combo = QComboBox()
        self._streaming_combo.addItem("Auto", None)
        self._streaming_combo.addItem("In-memory (fast)", False)
        self._streaming_combo.addItem("Streaming (low memory)", True)
        self._streaming_combo.setToolTip(
            "Auto: automatically chooses based on estimated data size and RAM.\n"
            "In-memory: fast, requires RAM > ~2.5x output size.\n"
            "Streaming: low RAM, required for TB-scale data."
        )
        self._streaming_combo.currentIndexChanged.connect(self._update_memory_indicator)
        settings_layout.addWidget(self._streaming_combo, 3, 4)

        # Memory safety indicator
        self._memory_indicator = QLabel("")
        self._memory_indicator.setFixedWidth(40)
        self._memory_indicator.setAlignment(Qt.AlignCenter)
        self._last_mem_estimate = None
        settings_layout.addWidget(self._memory_indicator, 3, 2)

        # Row 4: Memory estimate
        self._memory_label = QLabel("")
        # Rich text so per-term spans can carry their own colour.
        self._memory_label.setTextFormat(Qt.RichText)
        self._memory_label.setStyleSheet("font-size: 11px;")
        settings_layout.addWidget(self._memory_label, 4, 0, 1, 5)

        # Row 5: rough queue-time estimate (sum across pending queue items).
        self._time_label = QLabel("")
        self._time_label.setTextFormat(Qt.RichText)
        self._time_label.setStyleSheet("font-size: 11px;")
        self._time_label.setToolTip(
            "Approximate total wall time to stitch the Pending items in the "
            "queue.\nUses measured times from previous runs of similar settings "
            "when available;\notherwise a rough guess. Accuracy improves as you "
            "run more acquisitions."
        )
        settings_layout.addWidget(self._time_label, 5, 0, 1, 5)

        # Live "Output voxel" readout showing how Pixel size, Z step, and
        # the two downsample combos combine. Updated whenever any of those
        # four inputs change so the relationship is always visible.
        self._voxel_readout_label = QLabel("")
        self._voxel_readout_label.setStyleSheet("color: #444; font-size: 11px;")
        self._voxel_readout_label.setToolTip(
            "Native voxel (XY pixel × XY pixel × Z step) from the fields\n"
            "above, then the resulting output voxel after applying the\n"
            "chosen downsample factors. For 'iso', the factors are\n"
            "resolved from the native voxel and shown in parentheses."
        )
        settings_layout.addWidget(self._voxel_readout_label, 6, 0, 1, 5)
        # Track whether the user has manually set the XY pixel size, so the
        # ScopeSettings-derived auto-fill on discover doesn't clobber a
        # deliberate choice. A guard distinguishes programmatic setValue.
        self._pixel_size_user_set = False
        self._setting_pixel_programmatically = False
        self._pixel_size_spin.valueChanged.connect(self._on_pixel_size_changed)
        self._pixel_size_spin.valueChanged.connect(self._update_voxel_readout)
        self._z_step_spin.valueChanged.connect(self._update_voxel_readout)
        self._downsample_xy_combo.currentIndexChanged.connect(
            self._update_voxel_readout
        )
        self._downsample_z_combo.currentIndexChanged.connect(self._update_voxel_readout)

        # Any of these change the memory footprint — re-run the estimate
        # so the colored label stays in sync with current settings.
        self._pixel_size_spin.valueChanged.connect(self._refresh_memory_estimate)
        self._z_step_spin.valueChanged.connect(self._refresh_memory_estimate)
        self._downsample_xy_combo.currentIndexChanged.connect(
            self._refresh_memory_estimate
        )
        self._downsample_z_combo.currentIndexChanged.connect(
            self._refresh_memory_estimate
        )
        self._format_combo.currentIndexChanged.connect(self._refresh_memory_estimate)
        self._channels_edit.textChanged.connect(self._refresh_memory_estimate)

        settings_group.setLayout(settings_layout)
        self._settings_group = settings_group
        content_layout.addWidget(settings_group)

        # --- Collapsible processing options ---
        self._proc_toggle = QPushButton("\u25b6 Processing Options")
        self._proc_toggle.setCheckable(True)
        self._proc_toggle.setChecked(False)
        self._proc_toggle.setStyleSheet(
            "QPushButton { text-align: left; border: none; "
            "padding: 4px 2px; font-weight: bold; color: #555; }"
            "QPushButton:hover { color: #333; }"
        )
        self._proc_toggle.toggled.connect(self._on_proc_toggle)
        content_layout.addWidget(self._proc_toggle)

        self._proc_widget = QGroupBox()
        self._proc_widget.setStyleSheet(
            "QGroupBox { border: 1px solid #ccc; border-radius: 4px; "
            "margin-top: 0px; padding-top: 6px; }"
        )
        proc_layout = QGridLayout()
        proc_layout.setSpacing(6)

        # Proc Row 0: Destripe + Content-based blending
        self._destripe_cb = QCheckBox("Destripe (PyStripe) \u2731")
        self._destripe_cb.setToolTip(
            "\u2731 Processes every Z-plane at full resolution\n"
            "before downsampling.\n\n"
            "Removes horizontal stripe artifacts from light-sheet data.\n"
            "Uses multiple CPU cores automatically."
        )
        proc_layout.addWidget(self._destripe_cb, 0, 0)

        self._destripe_fast_cb = QCheckBox("Fast")
        self._destripe_fast_cb.setToolTip(
            "Destripe after downsampling instead of before.\n"
            "Much faster but slightly lower quality.\n\n"
            "Only effective when downsample factor > 1."
        )
        self._destripe_fast_cb.setEnabled(False)
        # Fast is a destripe-only variant (apply after downsample). When
        # Destripe is unchecked it must also be unchecked + disabled so
        # the UI state matches what the pipeline will actually run. The
        # plain toggled→setEnabled link only handled the enabled state.
        self._destripe_cb.toggled.connect(self._on_destripe_toggled)
        proc_layout.addWidget(self._destripe_fast_cb, 0, 1)

        self._content_fusion_cb = QCheckBox("Content-based blending \u2731")
        self._content_fusion_cb.setToolTip(
            "\u2731 Weights tile overlaps by local sharpness\n"
            "(Preibisch local-variance, inspired by BigStitcher).\n\n"
            "Improves fusion quality in overlap regions.\n\n"
            "COST: Adds two NaN Gaussian filters per output chunk on\n"
            "float32 — the fuse-store step becomes CPU-bound and can\n"
            "be 5-10× slower than the default cosine blending. On a\n"
            "66-tile TB-scale acquisition that's multiple hours of\n"
            "fusion instead of ~10 min.\n\n"
            "Leave off unless you've visually confirmed seams need it."
        )
        proc_layout.addWidget(self._content_fusion_cb, 0, 2, 1, 2)

        # Proc Row 1: Deconvolution + Flat-field
        self._deconv_cb = QCheckBox("Deconvolution \u2731")
        self._deconv_cb.setToolTip(
            "\u2731 GPU Richardson-Lucy deconvolution per tile.\n"
            "Requires pycudadecon or RedLionfish.\n\n"
            "Significantly improves resolution."
        )
        proc_layout.addWidget(self._deconv_cb, 1, 0)

        self._flat_field_cb = QCheckBox("Flat-field correction")
        self._update_preprocessing_availability()
        proc_layout.addWidget(self._flat_field_cb, 1, 1)

        self._ozx_cb = QCheckBox("Package as .ozx")
        self._ozx_cb.setToolTip(
            "Create a single .ozx ZIP file from the OME-Zarr output\n"
            "for easy sharing/copying"
        )
        proc_layout.addWidget(self._ozx_cb, 1, 2)

        self._tiff_pyramids_cb = QCheckBox("TIFF pyramids")
        self._tiff_pyramids_cb.setChecked(True)
        self._tiff_pyramids_cb.setToolTip(
            "Write multi-resolution pyramid SubIFDs in the OME-TIFF output.\n\n"
            "UNCHECK for ImarisFileConverter compatibility.\n"
            "ImarisFileConverter may misread SubIFD pyramids as extra Z\n"
            "planes, collapsing all real data into the first Z layer.\n\n"
            "Pyramids help napari and QuPath viewing but are not required\n"
            "for Fiji or Imaris (.ims has its own pyramid format)."
        )
        proc_layout.addWidget(self._tiff_pyramids_cb, 1, 3)

        # Proc Row 2: Registration
        self._skip_reg_cb = QCheckBox("Skip registration")
        self._skip_reg_cb.setToolTip(
            "Use stage positions only \u2014 skip phase-correlation registration.\n\n"
            "CHECK this when:\n"
            "  \u2022 Tiles have no overlap\n"
            "  \u2022 Stage positions are precise\n\n"
            "UNCHECK (default) when:\n"
            "  \u2022 Tiles overlap and you need sub-pixel alignment"
        )
        self._skip_reg_cb.toggled.connect(self._on_skip_reg_toggled)
        proc_layout.addWidget(self._skip_reg_cb, 2, 0)

        self._reg_binning_label = QLabel("Reg. binning:")
        proc_layout.addWidget(self._reg_binning_label, 2, 1)
        self._reg_binning_combo = QComboBox()
        self._reg_binning_combo.addItem("Fine (z1 y2 x2)", {"z": 1, "y": 2, "x": 2})
        self._reg_binning_combo.addItem("Default (z2 y4 x4)", {"z": 2, "y": 4, "x": 4})
        self._reg_binning_combo.addItem("Fast (z4 y8 x8)", {"z": 4, "y": 8, "x": 8})
        self._reg_binning_combo.setCurrentIndex(1)
        self._reg_binning_combo.setToolTip(
            "How much to downsample tiles for phase-correlation registration."
        )
        proc_layout.addWidget(self._reg_binning_combo, 2, 2, 1, 2)

        # Proc Row 3: Fusion chunk size
        proc_layout.addWidget(QLabel("Fusion chunk size:"), 3, 0)
        self._chunk_size_combo = QComboBox()
        # Dask graph granularity for the fuse step. Separate from
        # zarr_chunks (the final storage chunk size that determines
        # viewer pan/zoom cost) — this only affects fusion throughput.
        self._chunk_size_combo.addItem(
            "Small (4 MB, 32×256×256)", {"z": 32, "y": 256, "x": 256}
        )
        self._chunk_size_combo.addItem(
            "Medium (16 MB, 128×256×256)", {"z": 128, "y": 256, "x": 256}
        )
        self._chunk_size_combo.addItem(
            "Large (32 MB, 64×512×512)", {"z": 64, "y": 512, "x": 512}
        )
        self._chunk_size_combo.addItem(
            "XL (128 MB, 64×1024×1024)", {"z": 64, "y": 1024, "x": 1024}
        )
        self._chunk_size_combo.setCurrentIndex(2)  # Large
        self._chunk_size_combo.setToolTip(
            "Dask compute-chunk granularity for the fuse step.\n\n"
            "This is internal to the fusion pipeline — it does NOT\n"
            "affect the chunk size of the final OME-Zarr/Imaris output,\n"
            "so pan/zoom performance in napari / Fiji / Imaris is\n"
            "unchanged.\n\n"
            "• Smaller chunks → less RAM per worker, more dask overhead\n"
            "  (~47 graph tasks per chunk).\n"
            "• Larger chunks → fewer scheduling decisions, faster fusion\n"
            "  (7-8× fewer chunks from Small → XL).\n\n"
            "Large (default) is a safe balance. Pick XL if the fuse\n"
            "step is unexpectedly slow and you have RAM headroom.\n"
            "Small helps on tight-RAM systems at the cost of throughput."
        )
        proc_layout.addWidget(self._chunk_size_combo, 3, 1, 1, 3)

        # Proc Row 4: Legend
        legend = QLabel("\u2731 = significantly increases processing time")
        legend.setStyleSheet("color: #FF8C00; font-style: italic; font-size: 11px;")
        proc_layout.addWidget(legend, 4, 0, 1, 4)

        self._proc_widget.setLayout(proc_layout)
        self._proc_widget.setVisible(False)
        content_layout.addWidget(self._proc_widget)

        # --- Background zeroing (lossy compression aid) ---
        self._bg_zero_panel = BackgroundZeroPanel()
        self._bg_zero_panel.preview_requested.connect(self._on_preview_background_zero)
        if not _napari_available():
            # No napari (e.g. the frozen Windows build) — disable the live
            # preview but keep numeric background zeroing fully functional.
            self._bg_zero_panel._preview_btn.setEnabled(False)
            self._bg_zero_panel._preview_btn.setToolTip(
                "Live preview requires napari (not installed in this build).\n"
                "Background zeroing still works — set thresholds numerically.\n"
                'Install with: pip install "flamingo-stitcher[preview]"'
            )
        content_layout.addWidget(self._bg_zero_panel)

        # --- Action buttons ---
        btn_layout = QHBoxLayout()

        self._discover_btn = QPushButton("Discover Tiles")
        self._discover_btn.setToolTip(
            "Scan all queued directories for tile data\n"
            "(optional — Run will auto-discover if needed)"
        )
        self._discover_btn.clicked.connect(self._on_discover)
        btn_layout.addWidget(self._discover_btn)

        self._run_btn = QPushButton("Run All")
        self._run_btn.setToolTip(
            "Process all pending directories in the queue sequentially"
        )
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        btn_layout.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)

        btn_layout.addStretch()

        self._setup_env_btn = QPushButton("Set up flat-field…")
        self._setup_env_btn.setToolTip(
            "One-click install of flat-field correction (basicpy).\n\n"
            "Downloads a self-contained environment (~1–2 GB) — no Python\n"
            "or technical setup required, just internet. CPU only; needed once."
        )
        self._setup_env_btn.clicked.connect(self._on_setup_env)
        # The earlier _update_preprocessing_availability() (when the flat-field
        # checkbox was built) ran before this button existed, so reflect the
        # install state now. is_built() is a cheap file check (no backend probe).
        try:
            from flamingo_stitcher import preprocessing_env as _pe

            if _pe.is_built():
                self._setup_env_btn.setText("Reinstall flat-field…")
        except Exception:
            pass
        btn_layout.addWidget(self._setup_env_btn)

        self._help_btn = QPushButton("Help / Troubleshooting")
        self._help_btn.setToolTip(
            "Hardware requirements, disk/RAM planning, and a diagnostic\n"
            "matrix for stuck or slow runs. Opens\n"
            "docs/stitching_hardware_troubleshooting.md in your default\n"
            "markdown/text viewer."
        )
        self._help_btn.clicked.connect(self._on_open_help_doc)
        btn_layout.addWidget(self._help_btn)
        layout.addLayout(btn_layout)

        # --- Log area (collapsible, collapsed by default to save space) ---
        self._log_toggle = QPushButton("▶ Log")
        self._log_toggle.setCheckable(True)
        self._log_toggle.setChecked(False)
        self._log_toggle.setStyleSheet(
            "QPushButton { text-align: left; border: none; "
            "padding: 4px 2px; font-weight: bold; color: #555; }"
            "QPushButton:hover { color: #333; }"
        )
        self._log_toggle.toggled.connect(self._on_log_toggle)
        layout.addWidget(self._log_toggle)

        self._log_group = QGroupBox()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        self._verbose_log_cb = QCheckBox("Verbose log (include Python output)")
        self._verbose_log_cb.setToolTip(
            "Include behind-the-scenes Python output (flat-field, Imaris/Zarr\n"
            "writers, isolated environment, etc.) in this log. Off by default\n"
            "for a concise run log; turn on to troubleshoot a failed/odd run.\n"
            "Takes effect on the next run."
        )
        log_layout.addWidget(self._verbose_log_cb)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMinimumHeight(120)
        self._log_text.setStyleSheet(
            "QTextEdit { font-family: monospace; font-size: 11px; }"
        )
        log_layout.addWidget(self._log_text)
        self._log_group.setLayout(log_layout)
        self._log_group.setVisible(False)  # collapsed initially
        layout.addWidget(self._log_group)

        # --- Progress: step list + detail ---
        # A single 0–100% bar is misleading here because step costs are
        # wildly uneven (preprocess ≪ fuse ≪ write for large runs). We
        # instead show a list of pipeline phases with state pills:
        #   Yellow = to-do
        #   Orange = in progress
        #   Blue = done
        #   Red = error
        # The raw status message (includes per-step % and ETA from the
        # pipeline's dask Callback) is shown below the pills.
        progress_group = QGroupBox("Progress")
        progress_v = QVBoxLayout()

        self._step_row = QHBoxLayout()
        self._step_row.setSpacing(4)
        self._step_labels: Dict[str, QLabel] = {}
        self._step_order = [
            ("discover", "Discover"),
            ("register", "Register"),
            ("preprocess", "Preprocess"),
            ("fuse", "Fuse"),
            ("write", "Write output"),
            ("metadata", "Metadata"),
        ]
        for key, text in self._step_order:
            pill = QLabel(text)
            pill.setAlignment(Qt.AlignCenter)
            pill.setProperty("_stitch_step", key)
            self._step_labels[key] = pill
            self._step_row.addWidget(pill)
        self._step_row.addStretch()
        progress_v.addLayout(self._step_row)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #555; font-size: 11px;")
        # Wrap so the ETA tail "M:SS remaining (done at ~HH:MM)" stays visible
        # even when the phase prefix is long (e.g. multi-channel fuse).
        self._status_label.setWordWrap(True)
        progress_v.addWidget(self._status_label)

        progress_group.setLayout(progress_v)
        layout.addWidget(progress_group)

        self._reset_step_progress()

        # Sync .ozx checkbox enabled state with initial format
        self._on_format_changed()

    # --- Queue management ---

    def _add_to_queue(self):
        """Add an acquisition directory to the batch queue."""
        start = self._queue_browse_start()
        folder = QFileDialog.getExistingDirectory(
            self, "Select Acquisition Directory", start
        )
        if folder:
            self._add_path_to_queue(Path(folder))

    def _add_folder_to_queue(self):
        """Add all acquisition subdirectories from a parent folder."""
        start = self._queue_browse_start()
        parent = QFileDialog.getExistingDirectory(
            self, "Select Parent Folder (contains acquisition folders)", start
        )
        if not parent:
            return

        parent_path = Path(parent)
        added = 0
        for child in sorted(parent_path.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                if self._looks_like_acquisition(child):
                    self._add_path_to_queue(child)
                    added += 1

        if added == 0:
            self._log(f"No acquisition directories found in: {parent}")
        else:
            self._log(f"Added {added} directories from: {parent}")

    def _queue_browse_start(self) -> str:
        """Determine the starting path for the file browser."""
        if self._queue:
            last = self._queue[-1]["path"]
            start = last.parent
            for _ in range(self._acq_dir_restore_levels_up - 1):
                if start.parent != start:
                    start = start.parent
            return str(start)
        output = self._output_dir_edit.text()
        if output and Path(output).parent.exists():
            return str(Path(output).parent)
        return str(Path.home())

    def _looks_like_acquisition(self, path: Path) -> bool:
        """Check if a directory looks like an acquisition folder.

        Subclasses can override for different layout detection.
        """
        if (path / "Workflow.txt").exists():
            return True
        # Check first few children for tile indicators
        checked = 0
        for child in path.iterdir():
            if child.is_dir() and (child / "Workflow.txt").exists():
                return True
            if child.suffix == ".raw":
                return True
            checked += 1
            if checked >= 5:
                break
        return False

    def _add_path_to_queue(self, path: Path):
        """Add a single path to the queue (with dedup)."""
        path = Path(path)
        for item in self._queue:
            if item["path"] == path:
                return  # Already in queue

        self._queue.append(
            {
                "path": path,
                "status": "pending",
                "tiles": None,
                "error": None,
                "output_path": None,
            }
        )
        self._update_queue_table()

        # Auto-set output directory from first item
        if len(self._queue) == 1 and not self._output_dir_edit.text().strip():
            self._output_dir_edit.setText(str(path.parent))

        self._update_action_buttons()

    def _remove_from_queue(self):
        """Remove selected items from the queue."""
        rows = sorted(
            set(idx.row() for idx in self._queue_table.selectedIndexes()),
            reverse=True,
        )
        for row in rows:
            if 0 <= row < len(self._queue):
                # Don't remove the currently running item
                if self._batch_running and row == self._queue_index:
                    continue
                del self._queue[row]
                if self._batch_running and row < self._queue_index:
                    self._queue_index -= 1
        self._update_queue_table()
        self._update_action_buttons()

    def _requeue_selected(self):
        """Reset selected finished items (Done / Error / Cancelled) to Pending.

        After a completed or cancelled run every item is in a terminal state, so
        the Run button stays disabled (it needs at least one Pending item). This
        flips the selected terminal items back to Pending — clearing any prior
        error and keeping discovered tiles — so they can be stitched again and
        the Run button re-enables. Items currently discovering/stitching are
        left untouched.
        """
        rows = sorted(set(idx.row() for idx in self._queue_table.selectedIndexes()))
        if not rows:
            return
        n = 0
        for row in rows:
            if 0 <= row < len(self._queue):
                item = self._queue[row]
                if item["status"] in ("done", "error", "cancelled"):
                    item["status"] = "pending"
                    item["error"] = None
                    n += 1
        if n:
            self._update_queue_table()
            self._update_action_buttons()
            self._log(f"Re-queued {n} item(s) to Pending — ready to run again.")

    def _on_queue_context_menu(self, pos):
        """Right-click menu on the queue table: re-queue or remove rows."""
        from PyQt5.QtWidgets import QMenu

        # Right-click doesn't change selection by default — if the clicked row
        # isn't part of the current selection, select it so the action targets
        # what the user clicked on.
        index = self._queue_table.indexAt(pos)
        if index.isValid() and index.row() not in {
            i.row() for i in self._queue_table.selectedIndexes()
        }:
            self._queue_table.selectRow(index.row())

        menu = QMenu(self)
        act_requeue = menu.addAction("Re-queue (reset to Pending)")
        act_remove = menu.addAction("Remove from queue")
        chosen = menu.exec_(self._queue_table.viewport().mapToGlobal(pos))
        if chosen == act_requeue:
            self._requeue_selected()
        elif chosen == act_remove:
            self._remove_from_queue()

    # --- Drag-and-drop of acquisition folders into the queue ---

    @staticmethod
    def _dropped_dirs(mime) -> list:
        """Local directory paths from a drag-and-drop mime payload."""
        if not mime.hasUrls():
            return []
        dirs = []
        for url in mime.urls():
            local = url.toLocalFile()
            if local and Path(local).is_dir():
                dirs.append(Path(local))
        return dirs

    def dragEnterEvent(self, event):
        if self._dropped_dirs(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._dropped_dirs(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        dirs = self._dropped_dirs(event.mimeData())
        if not dirs:
            event.ignore()
            return
        before = len(self._queue)
        for d in dirs:
            self._add_path_to_queue(d)
        added = len(self._queue) - before
        event.acceptProposedAction()
        if added:
            self._log(
                f"Added {added} folder(s) via drag-and-drop. "
                f"Click Discover or Run."
            )

    def _update_queue_table(self):
        """Refresh the queue table from self._queue."""
        self._queue_table.setRowCount(len(self._queue))
        status_styles = {
            "pending": ("\u25cb Pending", "#888888"),
            "discovering": ("\u25c9 Discovering", "#1976D2"),
            "stitching": ("\u25b6 Stitching", "#1976D2"),
            "done": ("\u2713 Done", "#388E3C"),
            "error": ("\u2717 Error", "#D32F2F"),
            "cancelled": ("\u2014 Cancelled", "#888888"),
        }
        for i, item in enumerate(self._queue):
            text, color = status_styles.get(item["status"], (item["status"], "#888888"))
            status_item = QTableWidgetItem(text)
            status_item.setForeground(QColor(color))
            if item["status"] == "stitching":
                font = status_item.font()
                font.setBold(True)
                status_item.setFont(font)
            self._queue_table.setItem(i, 0, status_item)

            path_item = QTableWidgetItem(str(item["path"]))
            if item.get("error"):
                path_item.setToolTip(f"Error: {item['error']}")
            self._queue_table.setItem(i, 1, path_item)

        # Status changes (item started/finished, re-queue) shift what's still
        # Pending, so refresh the remaining-queue time estimate.
        self._update_time_estimate()

    def _update_action_buttons(self):
        """Update Discover/Run button states based on queue."""
        has_pending = any(item["status"] == "pending" for item in self._queue)
        if self._batch_running:
            self._discover_btn.setEnabled(False)
            self._run_btn.setEnabled(False)
            self._cancel_btn.setEnabled(True)
        elif has_pending:
            self._discover_btn.setEnabled(True)
            self._run_btn.setEnabled(True)
            self._set_btn_green(self._run_btn)
            self._set_btn_default(self._discover_btn)
        elif self._queue:
            # Queue exists but nothing pending (all done/error)
            self._discover_btn.setEnabled(False)
            self._run_btn.setEnabled(False)
            self._set_btn_default(self._discover_btn)
            self._set_btn_default(self._run_btn)
        else:
            # Empty queue
            self._discover_btn.setEnabled(False)
            self._run_btn.setEnabled(False)
            self._set_btn_default(self._discover_btn)
            self._set_btn_default(self._run_btn)

    def _browse_output_dir(self):
        start = self._output_dir_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start
        )
        if folder:
            self._output_dir_edit.setText(str(Path(folder)))

    # --- Tile discovery ---

    def _on_discover(self):
        """Discover tiles for all pending items in the queue."""
        pending = [item for item in self._queue if item["status"] == "pending"]
        if not pending:
            QMessageBox.warning(
                self,
                "Nothing to Discover",
                "No pending directories in the queue.\n"
                "Add directories with 'Add...' first.",
            )
            return

        self._log_text.clear()
        self._log(f"Discovering tiles for {len(pending)} directories...\n")

        for item in pending:
            self._log(f"Scanning: {item['path'].name}")
            try:
                tiles = self._discover_tiles_for_path(item["path"])
                if tiles:
                    item["tiles"] = tiles
                    self._log(f"  Found {len(tiles)} tiles")
                else:
                    self._log("  No tiles found")
                    item["status"] = "error"
                    item["error"] = "No tiles found"
            except Exception as e:
                self._log(f"  Error: {e}")
                self._logger.exception("Tile discovery error")
                item["status"] = "error"
                item["error"] = str(e)

        self._update_queue_table()
        total = sum(len(it["tiles"]) for it in self._queue if it["tiles"])
        ok = sum(1 for it in pending if it["tiles"])
        self._log(f"\nDiscovered {total} tiles across {ok}/{len(pending)} directories")
        self._update_action_buttons()

        # Auto-fill Z step from first discovered tile if still "Auto".
        # Each tile carries z_step_mm derived from its raw filenames +
        # Workflow Z range, so we don't need to re-parse Workflow.txt.
        if self._z_step_spin.value() == 0.0:
            first_tiles = next((it["tiles"] for it in self._queue if it["tiles"]), None)
            if first_tiles and first_tiles[0].z_step_mm:
                z_um = first_tiles[0].z_step_mm * 1000.0
                self._z_step_spin.setValue(z_um)
                self._log(f"Auto-detected Z step: {z_um:.3f} µm (from tile metadata)")

        # Auto-fill XY pixel size from the recorded objective (ScopeSettings.txt).
        self._autofill_pixel_size()

        # Surface the detected frame (AOI) size, especially when it differs from
        # the hardware-config default (cropped/binned acquisition).
        self._log_detected_frame_size()

        # Show memory estimate across ALL discovered queue items (worst case).
        if any(it["tiles"] for it in self._queue):
            self._update_memory_estimate()

        # Refresh the output-voxel readout now that Z may have auto-filled.
        self._update_voxel_readout()

        # Populate per-channel rows in the background-zero panel once
        # tiles are known. Use the union of channels across queue items.
        all_channels: List[int] = sorted(
            {
                ch
                for it in self._queue
                if it.get("tiles")
                for t in it["tiles"]
                for ch in t.channels
            }
        )
        self._bg_zero_panel.set_channels(all_channels)
        # Replay any thresholds restored before channels were known.
        pending = getattr(self, "_pending_bg_zero_thresholds", None)
        if pending:
            self._bg_zero_panel.set_thresholds(pending)
            self._pending_bg_zero_thresholds = {}

    def _on_pixel_size_changed(self, _value):
        """Mark the XY pixel size as user-set unless we set it ourselves."""
        if not self._setting_pixel_programmatically:
            self._pixel_size_user_set = True

    def _autofill_pixel_size(self):
        """Auto-fill XY pixel size from the acquisition's recorded objective.

        Reads `Objective lens magnification` from ScopeSettings.txt so an
        objective swap doesn't silently keep a stale pixel size (which places
        tiles by stage spacing but renders them at the wrong scale → gaps).
        Respects a manual override; warns instead of overwriting in that case.
        """
        from flamingo_stitcher.pipeline import (
            read_objective_magnification,
            suggested_pixel_size_um,
        )

        first_item = next((it for it in self._queue if it.get("tiles")), None)
        if first_item is None:
            return
        acq = Path(first_item["path"])
        try:
            suggested = suggested_pixel_size_um(acq)
        except Exception as e:
            self._logger.debug(f"pixel-size auto-fill skipped: {e}")
            return
        if not suggested:
            return
        mag = read_objective_magnification(acq)
        mag_str = f"{mag:.2f}×" if mag else "?"
        cur = self._pixel_size_spin.value()
        if not self._pixel_size_user_set:
            self._setting_pixel_programmatically = True
            self._pixel_size_spin.setValue(round(suggested, 4))
            self._setting_pixel_programmatically = False
            self._log(
                f"Auto-detected XY pixel size: {suggested:.3f} µm "
                f"(objective {mag_str} from ScopeSettings.txt)"
            )
        elif cur > 0 and abs(cur - suggested) / suggested > 0.15:
            self._log(
                f"⚠ XY pixel size {cur:.3f} µm differs from the objective-derived "
                f"~{suggested:.3f} µm ({mag_str}). Verify before stitching — a wrong "
                f"pixel size causes gaps/overlap between tiles."
            )

    def _log_detected_frame_size(self):
        """Report the resolved raw frame (AOI) size across discovered tiles."""
        from flamingo_stitcher.pipeline import (
            FRAME_HEIGHT,
            FRAME_WIDTH,
            _resolve_frame_size,
        )

        all_tiles = [t for it in self._queue if it.get("tiles") for t in it["tiles"]]
        if not all_tiles:
            return
        distinct = sorted({(t.frame_width, t.frame_height) for t in all_tiles})
        fw, fh = _resolve_frame_size(all_tiles)
        if len(distinct) > 1:
            self._log(
                f"⚠ Frame (AOI) size varies across tiles: {distinct}. "
                f"Using {fw}×{fh}. Check for mixed acquisitions."
            )
        elif (fw, fh) != (FRAME_WIDTH, FRAME_HEIGHT):
            self._log(
                f"Detected frame (AOI) size: {fw}×{fh} px "
                f"(cropped/binned — differs from default {FRAME_WIDTH}×{FRAME_HEIGHT})"
            )

    def _discover_tiles_for_path(self, acq_path: Path):
        """Discover tiles in an acquisition directory.

        Override in subclasses for different tile layouts.
        """
        from flamingo_stitcher.pipeline import discover_tiles

        return discover_tiles(acq_path)

    def _log_tile_summary(self, tiles):
        """Display a summary of discovered tiles."""
        xs = sorted(set(t.x_mm for t in tiles))
        ys = sorted(set(t.y_mm for t in tiles))
        all_ch = sorted(set(ch for t in tiles for ch in t.channels))
        all_illum = sorted(set(il for t in tiles for il in t.illumination_sides))

        self._log(f"Found {len(tiles)} tiles in ~{len(xs)}x{len(ys)} grid")
        self._log(
            f"  X range: {min(xs):.2f} \u2013 {max(xs):.2f} mm  "
            f"Y range: {min(ys):.2f} \u2013 {max(ys):.2f} mm"
        )
        self._log(f"  Channels: {all_ch}")
        self._log(f"  Illumination sides: {all_illum}")
        self._log(
            f"  Planes per tile: {tiles[0].n_planes} "
            f"(Z: {tiles[0].z_min_mm:.3f} \u2013 {tiles[0].z_max_mm:.3f} mm)"
        )
        self._log("")
        self._log("Ready to stitch. Click 'Run Stitching' to begin.")

    def _on_log_toggle(self, checked: bool):
        """Show/hide the log panel."""
        self._log_group.setVisible(checked)
        self._log_toggle.setText(("▼ " if checked else "▶ ") + "Log")

    def _expand_log(self):
        """Ensure the log is visible (called when a run starts)."""
        if not self._log_toggle.isChecked():
            self._log_toggle.setChecked(True)

    def _set_config_controls_enabled(self, enabled: bool):
        """Enable/disable every config control as a unit for run locking.

        Covers the settings grid, output dir, processing-options panel (and its
        toggle), and the background-zeroing panel so none can be changed while a
        stitch is in progress.
        """
        self._settings_group.setEnabled(enabled)
        self._output_dir_edit.setEnabled(enabled)
        self._proc_toggle.setEnabled(enabled)
        self._proc_widget.setEnabled(enabled)
        self._bg_zero_panel.setEnabled(enabled)

    def _on_proc_toggle(self, checked: bool):
        """Show/hide the processing options panel and resize the dialog.

        Without resizing, showing the panel squeezes the queue/log widgets
        and hiding it leaves an empty gap. We recompute the dialog's
        preferred height and resize to it, preserving the user's current
        width.
        """
        self._proc_widget.setVisible(checked)
        self._proc_toggle.setText(
            ("\u25bc " if checked else "\u25b6 ") + "Processing Options"
        )
        # Let Qt recompute size hints for the new visibility state, then
        # resize height to match. Width is preserved so the user's
        # horizontal layout isn't disturbed.
        self.updateGeometry()
        if self.layout() is not None:
            self.layout().activate()
        self.resize(self.width(), self.sizeHint().height())

    def _on_skip_reg_toggled(self, checked: bool):
        """Enable/disable registration controls based on skip state."""
        self._reg_binning_combo.setEnabled(not checked)
        self._reg_binning_label.setEnabled(not checked)

    def _on_downsample_xy_changed(self, _index: int):
        """Grey out Z combo when XY is iso (iso overrides both factors)."""
        is_iso = self._downsample_xy_combo.currentData() == -1
        self._downsample_z_combo.setEnabled(not is_iso)

    def _on_destripe_toggled(self, checked: bool):
        """Keep the Fast sub-option in sync with the Destripe master.

        Fast is a pystripe variant (destripe after downsample), so if
        Destripe is off Fast must also be off — both the enabled state
        and the checked state. Previously only enabled state was linked,
        which let a persisted Fast=True survive Destripe=False and get
        submitted to the pipeline.
        """
        self._destripe_fast_cb.setEnabled(checked)
        if not checked:
            self._destripe_fast_cb.setChecked(False)

    def _update_voxel_readout(self, *_args):
        """Refresh the Native → Output voxel line below the downsample row."""
        if not hasattr(self, "_voxel_readout_label"):
            # Signals can fire during _build_ui before the label exists.
            return
        xy_pixel = self._pixel_size_spin.value()
        z_step = self._z_step_spin.value()
        ds_xy = self._downsample_xy_combo.currentData()
        ds_z = self._downsample_z_combo.currentData()

        if z_step <= 0.0:
            native_str = f"Native: {xy_pixel:.3f} × {xy_pixel:.3f} × (Z auto) µm"
            output_str = "Output: discover tiles to preview"
            self._voxel_readout_label.setText(f"{native_str}   →   {output_str}")
            return

        iso_note = ""
        if ds_xy == -1 or ds_z == -1:
            try:
                from flamingo_stitcher.pipeline import compute_iso_downsample

                ds_xy, ds_z = compute_iso_downsample(xy_pixel, z_step)
                iso_note = f"  (iso \u2192 XY {ds_xy}x, Z {ds_z}x)"
            except Exception:
                self._voxel_readout_label.setText("Output: iso preview unavailable")
                return

        out_xy = xy_pixel * ds_xy
        out_z = z_step * ds_z
        self._voxel_readout_label.setText(
            f"Native: {xy_pixel:.3f} × {xy_pixel:.3f} × {z_step:.3f} \u00b5m"
            f"   \u2192   "
            f"Output: {out_xy:.3f} × {out_xy:.3f} × {out_z:.3f} \u00b5m"
            f"{iso_note}"
        )

    def _refresh_memory_estimate(self, *_args):
        """Debounced re-run of the memory estimate after a settings change.

        Swallows the Qt signal arg so this can be connected directly to
        ``valueChanged``/``currentIndexChanged``/``textChanged``.
        """
        if any(it.get("tiles") for it in self._queue):
            self._update_memory_estimate()

    def _update_memory_estimate(self, tiles=None):
        """Compute and display memory estimates for in-memory vs streaming modes.

        When ``tiles`` is None and the queue has multiple discovered items,
        the estimate reports the worst case across all queued items (picked
        by largest in-memory peak) so the user sees the ceiling the run
        must fit under, not just the first acquisition.
        """
        # Build the list of (label, tiles) pairs to estimate against. When
        # called with an explicit tiles list, honour it as a single item.
        if tiles is not None:
            tile_sets = [(None, tiles)]
        else:
            tile_sets = [
                (it.get("path"), it["tiles"]) for it in self._queue if it["tiles"]
            ]

        if not tile_sets:
            self._memory_label.setText("")
            self._last_mem_estimate = None
            self._update_memory_indicator()
            return

        try:
            from flamingo_stitcher.pipeline import estimate_memory_usage

            config = self._build_config()
            channels = self._parse_channels()

            worst_est = None
            worst_label = None
            for label, tset in tile_sets:
                all_ch = sorted(set(ch for t in tset for ch in t.channels))
                process_ch = channels if channels else all_ch
                est = estimate_memory_usage(tset, process_ch, config)
                if worst_est is None or est["in_memory_gb"] > worst_est["in_memory_gb"]:
                    worst_est = est
                    worst_label = label

            est = worst_est
            self._last_mem_estimate = est

            mode_hint = ""
            if est["auto_streaming"]:
                mode_hint = " \u2192 auto will use streaming"
            else:
                mode_hint = " \u2192 auto will use in-memory"

            queue_hint = ""
            if len(tile_sets) > 1:
                queue_hint = f"  (worst of {len(tile_sets)} queued)"

            # Colour each peak term against system RAM so the user can see
            # at a glance which mode fits. green <70%, orange 70–95%, red >95%.
            try:
                import psutil as _psutil

                total_ram_gb = _psutil.virtual_memory().total / (1024**3)
                avail_ram_gb = _psutil.virtual_memory().available / (1024**3)
            except ImportError:
                total_ram_gb = 0.0
                avail_ram_gb = 0.0

            def _colour(peak_gb: float) -> str:
                if avail_ram_gb <= 0:
                    return "#444"
                ratio = peak_gb / avail_ram_gb
                if ratio > 0.95:
                    return "#D32F2F"  # red
                if ratio > 0.70:
                    return "#F57C00"  # orange
                return "#388E3C"  # green

            in_mem_gb = est["in_memory_gb"]
            stream_gb = est["streaming_gb"]
            ram_str = (
                f" &nbsp;vs RAM {total_ram_gb:.0f} GB total, "
                f"{avail_ram_gb:.0f} GB free"
                if total_ram_gb > 0
                else ""
            )
            self._memory_label.setText(
                f"<span style='color:{_colour(in_mem_gb)};font-weight:bold;'>"
                f"In-memory: ~{in_mem_gb:.0f} GB</span> &nbsp;|&nbsp; "
                f"<span style='color:{_colour(stream_gb)};font-weight:bold;'>"
                f"Streaming: ~{stream_gb:.1f} GB</span> &nbsp;|&nbsp; "
                f"Output: ~{est['output_gb']:.0f} GB"
                f"{ram_str}"
                f"{queue_hint}"
                f"{mode_hint}"
            )

            # Update the indicator badge
            self._update_memory_indicator()

            # Also log to the log area for visibility
            try:
                import psutil

                sys_ram = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                sys_ram = 0
            worst_suffix = ""
            if len(tile_sets) > 1 and worst_label is not None:
                try:
                    worst_suffix = (
                        f" (worst of {len(tile_sets)} queued: "
                        f"{Path(str(worst_label)).name})"
                    )
                except Exception:
                    worst_suffix = f" (worst of {len(tile_sets)} queued)"
            self._log(
                f"Memory estimate{worst_suffix} (system RAM: {sys_ram:.0f} GB):\n"
                f"  In-memory mode: ~{est['in_memory_gb']:.0f} GB peak\n"
                f"  Streaming mode: ~{est['streaming_gb']:.1f} GB peak\n"
                f"  Output size:    ~{est['output_gb']:.0f} GB\n"
                f"  Recommendation: {'Streaming (low memory)' if est['auto_streaming'] else 'In-memory (fast)'}"
            )
        except Exception as e:
            self._logger.debug(f"Memory estimate failed: {e}")
            self._memory_label.setText("")
            self._last_mem_estimate = None
            self._update_memory_indicator()
        # Time estimate shares the same triggers (config + discovery changes).
        self._update_time_estimate()

    def _update_time_estimate(self):
        """Show a rough total wall-time estimate for the Pending queue items.

        Sums per-acquisition estimates: a measured total from the timing cache
        when a similar config has run before, otherwise a clearly-labelled rough
        guess. Excludes already-Done/Cancelled items, so during a run it reflects
        the remaining queue. Best-effort — never raises into the UI.
        """
        if not hasattr(self, "_time_label"):
            return
        items = [
            it
            for it in self._queue
            if it.get("tiles") and it.get("status") == "pending"
        ]
        if not items:
            self._time_label.setText("")
            return
        try:
            from flamingo_stitcher.multi_phase_estimator import _format_duration
            from flamingo_stitcher.pipeline import (
                build_timing_key,
                rough_run_seconds,
            )
            from flamingo_stitcher.timing_cache import StitchingTimingCache

            config = self._build_config()
            output_dir = self._output_dir_edit.text().strip() or None
            cache = StitchingTimingCache()
            total_s = 0.0
            measured = 0
            for it in items:
                key = build_timing_key(
                    it["tiles"],
                    config,
                    acquisition_dir=it.get("path"),
                    output_dir=output_dir,
                )
                cached = cache.get_total_s(key)
                if cached:
                    total_s += cached
                    measured += 1
                else:
                    total_s += rough_run_seconds(it["tiles"], config)

            n = len(items)
            if measured == n:
                note = f"from timings of {n} similar run{'s' if n != 1 else ''}"
            elif measured == 0:
                note = "rough guess — no timing history yet"
            else:
                note = f"{measured}/{n} measured, rest estimated"
            self._time_label.setText(
                f"Estimated queue time: <b>~{_format_duration(total_s)}</b> "
                f"<span style='color:#888;'>({note})</span>"
            )
        except Exception as e:
            self._logger.debug(f"Time estimate failed: {e}")
            self._time_label.setText("")

    def _update_memory_indicator(self, _index=None):
        """Update the memory safety indicator next to the Memory Mode combo.

        Shows a colored badge:
          Green "OK"     — selected mode fits comfortably in RAM
          Orange "Warn"  — selected mode is tight (>80% RAM) or auto would differ
          Red "OOM!"     — selected mode will likely exceed RAM
          Empty          — no tile data yet (nothing to estimate)
        """
        est = self._last_mem_estimate
        if est is None:
            self._memory_indicator.setText("")
            self._memory_indicator.setToolTip("")
            return

        # Get system RAM
        try:
            import psutil

            sys_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            sys_ram = 192.0

        # Determine which mode will actually be used
        selected = self._streaming_combo.currentData()
        if selected is None:  # Auto
            will_stream = est["auto_streaming"]
            peak_gb = est["streaming_gb"] if will_stream else est["in_memory_gb"]
            mode_name = "streaming" if will_stream else "in-memory"
        elif selected:  # Streaming forced
            will_stream = True
            peak_gb = est["streaming_gb"]
            mode_name = "streaming"
        else:  # In-memory forced
            will_stream = False
            peak_gb = est["in_memory_gb"]
            mode_name = "in-memory"

        ratio = peak_gb / sys_ram if sys_ram > 0 else 1.0

        if ratio > 0.95:
            # Red — will almost certainly OOM
            self._memory_indicator.setText("OOM!")
            self._memory_indicator.setStyleSheet(
                "QLabel { color: white; background-color: #D32F2F; "
                "font-weight: bold; font-size: 10px; "
                "border-radius: 8px; padding: 1px 3px; }"
            )
            self._memory_indicator.setToolTip(
                f"<b>Out of memory risk!</b><br>"
                f"Estimated peak: ~{peak_gb:.0f} GB ({mode_name})<br>"
                f"System RAM: {sys_ram:.0f} GB<br><br>"
                f"Switch to <b>Streaming</b> mode (~{est['streaming_gb']:.1f} GB peak) "
                f"or increase downsample factor."
            )
        elif ratio > 0.70:
            # Orange — tight, might work but risky
            self._memory_indicator.setText("Tight")
            self._memory_indicator.setStyleSheet(
                "QLabel { color: white; background-color: #F57C00; "
                "font-weight: bold; font-size: 10px; "
                "border-radius: 8px; padding: 1px 3px; }"
            )
            self._memory_indicator.setToolTip(
                f"<b>Memory is tight</b><br>"
                f"Estimated peak: ~{peak_gb:.0f} GB ({mode_name})<br>"
                f"System RAM: {sys_ram:.0f} GB ({ratio*100:.0f}% usage)<br><br>"
                f"Should work but leave little room for other applications.<br>"
                f"Streaming mode would use ~{est['streaming_gb']:.1f} GB."
            )
        else:
            # Green — comfortable
            self._memory_indicator.setText("OK")
            self._memory_indicator.setStyleSheet(
                "QLabel { color: white; background-color: #388E3C; "
                "font-weight: bold; font-size: 10px; "
                "border-radius: 8px; padding: 1px 3px; }"
            )
            self._memory_indicator.setToolTip(
                f"Estimated peak: ~{peak_gb:.0f} GB ({mode_name})<br>"
                f"System RAM: {sys_ram:.0f} GB ({ratio*100:.0f}% usage)"
            )

    # --- Run / Cancel ---

    def _build_config(self):
        """Build a StitchingConfig from YAML defaults + current UI settings."""
        from flamingo_stitcher.pipeline import StitchingConfig

        # Start from YAML defaults (fills in all non-UI-exposed fields)
        config = StitchingConfig.with_yaml_defaults()

        # Overlay UI settings
        z_step = self._z_step_spin.value()
        config.pixel_size_um = self._pixel_size_spin.value()
        config.z_step_um = z_step if z_step > 0 else None
        # Frame (AOI) override: None = auto-detect from file size.
        _frame = self._frame_size_combo.currentData()
        config.frame_width = _frame
        config.frame_height = _frame
        config.illumination_fusion = self._fusion_combo.currentData()
        config.tile_overlap_fusion = self._tile_fusion_combo.currentData()
        config.output_format = self._format_combo.currentData()
        config.flat_field_correction = self._flat_field_cb.isChecked()
        config.destripe = self._destripe_cb.isChecked()
        config.destripe_fast = self._destripe_fast_cb.isChecked()
        config.downsample_xy = self._downsample_xy_combo.currentData()
        config.downsample_z = self._downsample_z_combo.currentData()
        config.deconvolution_enabled = self._deconv_cb.isChecked()
        config.content_based_fusion = self._content_fusion_cb.isChecked()
        config.skip_registration = self._skip_reg_cb.isChecked()
        config.registration_binning = self._reg_binning_combo.currentData()
        config.package_ozx = self._ozx_cb.isChecked()
        config.tiff_pyramids = self._tiff_pyramids_cb.isChecked()
        config.streaming_mode = self._streaming_combo.currentData()
        chunk = self._chunk_size_combo.currentData()
        if chunk:
            config.output_chunksize = dict(chunk)

        # Background zeroing: only forward thresholds when the master
        # checkbox is on. The pipeline guards on background_zero_enabled
        # but keep the dialog's intent explicit.
        config.background_zero_enabled = self._bg_zero_panel.is_enabled()
        if config.background_zero_enabled:
            config.background_zero_thresholds = dict(self._bg_zero_panel.thresholds())
        else:
            config.background_zero_thresholds = {}

        # Set compression based on format
        compression = self._compression_combo.currentData()
        if compression:
            fmt = config.output_format
            if fmt == "ome-tiff":
                config.tiff_compression = compression
            elif fmt in ("ome-zarr-sharded", "ome-zarr-v2"):
                config.zarr_compression = compression
            elif fmt == "both":
                config.zarr_compression = compression
                config.tiff_compression = compression
        return config

    def _parse_channels(self) -> Optional[List[int]]:
        """Parse channels from the channels line edit. Returns None for 'all'."""
        text = self._channels_edit.text().strip()
        if not text:
            return None
        try:
            return [int(ch.strip()) for ch in text.split(",") if ch.strip()]
        except ValueError:
            return None

    def _confirm_pixel_size(self, pending, config) -> bool:
        """Block the run if the XY pixel size badly mismatches the objective.

        Reads each pending item's `Objective lens magnification` from
        ScopeSettings.txt and compares the implied pixel size to the configured
        one. On a large divergence (>25%), forces an explicit choice — use the
        objective-derived value (recommended), keep the current value, or
        cancel — so a stale pixel size can't silently produce a gappy stitch.
        Returns True to proceed, False to abort.
        """
        from flamingo_stitcher.pipeline import (
            read_objective_magnification,
            suggested_pixel_size_um,
        )

        cur = config.pixel_size_um
        suggestions = []  # (pixel_um, magnification)
        for it in pending:
            if not it.get("tiles"):
                continue
            try:
                s = suggested_pixel_size_um(Path(it["path"]))
            except Exception:
                s = None
            if s and s > 0:
                suggestions.append((s, read_objective_magnification(Path(it["path"]))))
        if not suggestions or cur <= 0:
            return True

        # Worst (largest) divergence across the queue.
        s, mag = max(suggestions, key=lambda sm: abs(cur - sm[0]) / sm[0])
        if abs(cur - s) / s <= 0.25:
            return True

        mixed = len({round(v[0], 3) for v in suggestions}) > 1
        mag_str = f"{mag:.2f}×" if mag else "?"
        pct = abs(cur - s) / s * 100
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Pixel size doesn't match the objective")
        text = (
            f"XY pixel size is set to {cur:.3f} µm, but the objective recorded in "
            f"ScopeSettings.txt ({mag_str}) implies ~{s:.3f} µm — a {pct:.0f}% "
            f"difference.\n\n"
            f"Stitching at {cur:.3f} µm places tiles by stage spacing but renders "
            f"them at the wrong scale, producing gaps (or excessive overlap) "
            f"between tiles."
        )
        if mixed:
            text += (
                "\n\n⚠ Queue items report different objectives; one pixel size is "
                "used for the whole batch."
            )
        box.setText(text)
        use_btn = box.addButton(f"Use {s:.3f} µm (recommended)", QMessageBox.AcceptRole)
        box.addButton(f"Keep {cur:.3f} µm", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(use_btn)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is cancel_btn:
            return False
        if clicked is use_btn:
            config.pixel_size_um = round(s, 4)
            self._setting_pixel_programmatically = True
            self._pixel_size_spin.setValue(round(s, 4))
            self._setting_pixel_programmatically = False
            self._log(
                f"Pixel size set to {s:.3f} µm from objective {mag_str} "
                f"(was {cur:.3f} µm)."
            )
        else:
            self._log(
                f"Proceeding with pixel size {cur:.3f} µm despite objective "
                f"mismatch (user override)."
            )
        return True

    def _on_run(self):
        """Start batch stitching of all pending queue items."""
        pending = [item for item in self._queue if item["status"] == "pending"]
        if not pending:
            QMessageBox.warning(
                self, "Nothing to Run", "No pending directories in the queue."
            )
            return

        output_dir = self._output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(
                self, "Invalid Input", "Please specify an output directory."
            )
            return

        config = self._build_config()

        # Pre-flight: block if the XY pixel size clashes with the objective
        # recorded in ScopeSettings.txt (the #1 cause of gappy/overlapping
        # stitches). A log warning alone was too easy to run past.
        if not self._confirm_pixel_size(pending, config):
            return

        # Pre-flight: warn if flat-field is requested but basicpy missing
        if config.flat_field_correction:
            from flamingo_stitcher.flat_field import is_available

            if not is_available():
                reply = QMessageBox.question(
                    self,
                    "basicpy Not Installed",
                    "Flat-field correction requires basicpy which is not installed.\n\n"
                    "Install with:  pip install basicpy\n\n"
                    "Continue without flat-field correction?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return
                config.flat_field_correction = False

        # Pre-flight: warn (popup) if projected peak RAM or disk free looks
        # risky for any queue item. User must accept to proceed.
        if not self._confirm_resource_headroom(pending, config):
            return

        self._log_text.clear()
        self._expand_log()
        self._reset_step_progress()
        n_pending = len(pending)
        self._log(f"Starting batch stitching: {n_pending} directories\n")

        # Store batch state
        self._batch_running = True
        self._batch_config = config
        self._batch_channels = self._parse_channels()
        self._batch_results = []

        # Lock all config controls during the run (settings, processing
        # options, background zeroing) so they can't be toggled mid-stitch —
        # the config is already captured and changing them would mislead.
        self._set_config_controls_enabled(False)
        self._update_action_buttons()

        # Start processing
        self._advance_queue()

    def _confirm_resource_headroom(self, pending, config) -> bool:
        """Show a popup if any pending item is likely to OOM or run out of
        disk, and require user acceptance before continuing. Returns True
        if the user confirmed (or nothing is tight), False to abort.
        """
        try:
            import shutil as _shutil

            import psutil as _psutil

            from flamingo_stitcher.pipeline import estimate_memory_usage
        except ImportError:
            return True

        try:
            avail_ram_gb = _psutil.virtual_memory().available / (1024**3)
            total_ram_gb = _psutil.virtual_memory().total / (1024**3)
        except Exception:
            return True

        output_dir = Path(self._output_dir_edit.text().strip())
        try:
            out_free_gb = _shutil.disk_usage(output_dir).free / (1024**3)
        except OSError:
            out_free_gb = None

        ram_warnings = []  # per-item lines
        disk_warnings = []
        for item in pending:
            tiles = item.get("tiles")
            if not tiles:
                # Not discovered yet — skip, the pipeline will check again.
                continue
            channels = self._parse_channels() or sorted(
                set(ch for t in tiles for ch in t.channels)
            )
            try:
                est = estimate_memory_usage(tiles, channels, config)
            except Exception:
                continue

            use_streaming = config.streaming_mode
            if use_streaming is None:
                use_streaming = est["auto_streaming"]
            peak_gb = est["streaming_gb" if use_streaming else "in_memory_gb"]

            # Format-specific writer overhead (Imaris keeps a big block cache
            # on top of our pipeline peak; Zarr v2 materializes pyramid levels).
            fmt_overhead_gb = 0.0
            fmt_note = ""
            fmt = getattr(config, "output_format", "")
            if fmt == "imaris":
                fmt_overhead_gb = est["output_gb"] * 0.25
                fmt_note = f" + ~{fmt_overhead_gb:.0f} GB Imaris writer overhead"
            elif fmt == "ome-zarr-v2":
                fmt_overhead_gb = est["output_gb"] * 0.30
                fmt_note = f" + ~{fmt_overhead_gb:.0f} GB OME-Zarr v2 pyramid overhead"

            combined_gb = peak_gb + fmt_overhead_gb
            if combined_gb > avail_ram_gb * 0.9:
                ram_warnings.append(
                    f"  • {item['path'].name}: "
                    f"peak ~{peak_gb:.0f} GB{fmt_note} "
                    f"(available RAM {avail_ram_gb:.0f} GB)"
                )

            if out_free_gb is not None:
                # Streaming mode holds three things on disk: per-channel
                # tile memmaps (one channel at a time), the fused
                # (C,Z,Y,X) memmap that feeds the writer, and the final
                # output file.
                bpv = 2
                ds_xy = max(config.downsample_xy, 1)
                ds_z = max(config.downsample_z, 1)
                n_planes = max(t.n_planes for t in tiles)
                tile_bytes = (
                    (n_planes // ds_z if ds_z > 1 else n_planes)
                    * (2048 // ds_xy if ds_xy > 1 else 2048)
                    * (2048 // ds_xy if ds_xy > 1 else 2048)
                    * bpv
                )
                spill_gb = len(tiles) * tile_bytes / (1024**3) if use_streaming else 0.0
                fused_memmap_gb = est["output_gb"] if use_streaming else 0.0
                needed_gb = est["output_gb"] + spill_gb + fused_memmap_gb
                if out_free_gb < needed_gb * 1.1:
                    parts = [f"~{est['output_gb']:.0f} GB output"]
                    if spill_gb:
                        parts.append(f"~{spill_gb:.0f} GB tile spill")
                    if fused_memmap_gb:
                        parts.append(f"~{fused_memmap_gb:.0f} GB fused memmap")
                    disk_warnings.append(
                        f"  • {item['path'].name}: "
                        f"need {' + '.join(parts)}, "
                        f"{out_free_gb:.0f} GB free on {output_dir}"
                    )

        if not ram_warnings and not disk_warnings:
            return True

        lines = [
            f"System RAM: {total_ram_gb:.0f} GB total, "
            f"{avail_ram_gb:.0f} GB available."
        ]
        if ram_warnings:
            lines.append("\nMemory is tight for:")
            lines.extend(ram_warnings)
            lines.append(
                "\nIf the write hits swap it will slow to a crawl or be killed "
                "by the OS. Consider Streaming mode or a higher downsample."
            )
        if disk_warnings:
            lines.append("\nDisk space is tight for:")
            lines.extend(disk_warnings)
            lines.append("\nFree up space on the output drive or point it elsewhere.")
        lines.append("\nContinue anyway?")

        reply = QMessageBox.warning(
            self,
            "Resource Headroom Warning",
            "\n".join(lines),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _advance_queue(self):
        """Process the next pending item in the queue."""
        # Find next pending
        next_idx = None
        for i, item in enumerate(self._queue):
            if item["status"] == "pending":
                next_idx = i
                break

        if next_idx is None:
            self._on_batch_complete()
            return

        self._queue_index = next_idx
        item = self._queue[next_idx]
        n_total = sum(1 for it in self._queue if it["status"] not in ("cancelled",))
        n_done = sum(1 for it in self._queue if it["status"] in ("done", "error"))

        self._log(f"\n{'=' * 60}")
        self._log(f"Processing {n_done + 1}/{n_total}: {item['path'].name}")
        self._log(f"{'=' * 60}\n")

        # Discover tiles if not already discovered
        if item["tiles"] is None:
            item["status"] = "discovering"
            self._update_queue_table()
            try:
                tiles = self._discover_tiles_for_path(item["path"])
                if not tiles:
                    item["status"] = "error"
                    item["error"] = "No tiles found"
                    self._log("  No tiles found \u2014 skipping")
                    self._update_queue_table()
                    self._batch_results.append((item["path"], False, "No tiles found"))
                    self._advance_queue()
                    return
                item["tiles"] = tiles
                self._log_tile_summary(tiles)
            except Exception as e:
                item["status"] = "error"
                item["error"] = str(e)
                self._log(f"  Discovery error: {e}")
                self._logger.exception("Batch tile discovery error")
                self._update_queue_table()
                self._batch_results.append((item["path"], False, str(e)))
                self._advance_queue()
                return

        # Compute output path
        acq_name = item["path"].name
        output_dir = Path(self._output_dir_edit.text()) / f"{acq_name}_stitched"
        item["output_path"] = str(output_dir)

        # Update status
        item["status"] = "stitching"
        self._update_queue_table()

        # Start worker
        from flamingo_stitcher.worker import StitchingWorker

        self._worker = StitchingWorker(
            config=self._batch_config,
            acq_dir=item["path"],
            output_dir=output_dir,
            channels=self._batch_channels,
            tiles=item["tiles"],
            parent=self,
            verbose=self._verbose_log_cb.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._on_log_message)
        self._worker.completed.connect(self._on_item_completed)
        self._worker.error.connect(self._on_item_error)
        self._worker.finished.connect(self._on_item_finished)
        self._worker.start()

    def _on_cancel(self):
        """Cancel the running pipeline and stop the batch."""
        if self._worker:
            self._worker.cancel()
            self._status_label.setText("Cancelling...")
            self._log("Cancellation requested...")
        # Mark remaining pending items as cancelled
        for item in self._queue:
            if item["status"] == "pending":
                item["status"] = "cancelled"
        self._update_queue_table()

        from flamingo_stitcher.gui._compat import get_notification_service

        svc = get_notification_service(self)
        if svc is not None:
            svc.notify(
                "errors",
                title="Flamingo: stitching cancelled",
                message="Stitching was cancelled before finishing.",
                tags="no_entry_sign",
            )

    # Step-list colours. Kept here so a11y changes are one place.
    _STEP_COLORS = {
        "todo": ("#FFC107", "#5D4200"),  # yellow pill, dark text
        "running": ("#FF8C00", "#FFFFFF"),  # orange pill, white text
        "done": ("#1976D2", "#FFFFFF"),  # blue pill, white text
        "error": ("#D32F2F", "#FFFFFF"),  # red pill, white text
        "skipped": ("#B0B0B0", "#404040"),  # grey pill, dark text
    }

    def _reset_step_progress(self):
        """Mark every pipeline step as TODO and clear the status line."""
        for key, _ in self._step_order:
            self._set_step_state(key, "todo")
        self._status_label.setText("Ready")

    def _set_step_state(self, key: str, state: str):
        """Set the colour of a step pill to TODO / RUNNING / DONE /
        ERROR / SKIPPED. Unknown ``key`` is a no-op (safer than a
        KeyError into the worker thread)."""
        pill = self._step_labels.get(key)
        if pill is None:
            return
        bg, fg = self._STEP_COLORS.get(state, self._STEP_COLORS["todo"])
        pill.setStyleSheet(
            f"QLabel {{ background-color: {bg}; color: {fg}; "
            f"font-weight: bold; padding: 3px 10px; border-radius: 10px; }}"
        )

    # Map status substrings → step key. Ordered because several
    # substrings legitimately match multiple keys (e.g. "channel" is
    # in both preprocess and fuse status messages) — first match wins
    # by design, and we order it to follow the actual pipeline flow.
    _STATUS_TO_STEP = [
        ("discover", "discover"),
        ("loading reference", "register"),
        ("registering", "register"),
        ("skip registration", "register"),  # will be marked "skipped" separately
        ("preprocess", "preprocess"),
        ("materializing", "preprocess"),
        ("fusing", "fuse"),
        ("computing channel", "fuse"),
        ("storing channel", "fuse"),
        ("channel ", "fuse"),  # "channel 2 store progress"
        ("writing ome", "write"),
        ("writing imaris", "write"),
        ("writing sharded", "write"),
        ("writing pyramidal", "write"),
        ("writing multi-channel", "write"),
        ("finalizing", "write"),
        (".ims write complete", "write"),
        ("metadata", "metadata"),
    ]

    def _classify_step(self, status: str):
        """Return the step key for the current status string (or None
        if nothing obvious matches)."""
        s = status.lower()
        for needle, key in self._STATUS_TO_STEP:
            if needle in s:
                return key
        return None

    def _on_progress(self, percentage: int, status: str):
        """Handle progress updates from worker.

        ``percentage`` is ignored here — it was badly calibrated across
        phases and jumped to ~50 % on the first tile. We derive the
        current phase from the status string instead and light the
        pills accordingly. ``percentage`` still shows up in the raw
        status line on its own when the pipeline sends it.
        """
        del percentage  # intentionally unused; see docstring
        # Add batch context if multiple items
        if self._batch_running and len(self._queue) > 1:
            n_total = sum(1 for it in self._queue if it["status"] != "cancelled")
            n_done = sum(1 for it in self._queue if it["status"] in ("done", "error"))
            self._status_label.setText(f"[{n_done + 1}/{n_total}] {status}")
        else:
            self._status_label.setText(status)

        key = self._classify_step(status)
        if key is None:
            return

        # Skip-registration special case: the status says "Skipping
        # registration..." which we want to render as skipped (grey),
        # not running (orange).
        if key == "register" and "skip" in status.lower():
            self._set_step_state("register", "skipped")
            return

        # Mark everything up to and including this step as done/running,
        # everything after as todo. Keeps earlier pills from re-flipping
        # when a later step emits a status line.
        keys = [k for k, _ in self._step_order]
        try:
            idx = keys.index(key)
        except ValueError:
            return
        for i, k in enumerate(keys):
            if i < idx:
                # Don't overwrite 'skipped'; leave it grey.
                existing = self._step_labels[k].styleSheet()
                if "#B0B0B0" not in existing:
                    self._set_step_state(k, "done")
            elif i == idx:
                self._set_step_state(k, "running")
            else:
                # Only reset future steps if they haven't already been
                # marked done — keeps the list stable when a later
                # phase briefly echoes a string from an earlier one.
                existing = self._step_labels[k].styleSheet()
                if "#1976D2" not in existing:
                    self._set_step_state(k, "todo")

    def _on_log_message(self, message: str):
        """Handle log messages from worker."""
        self._log_text.append(message)
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_item_completed(self, output_path: str):
        """Handle successful completion of one queue item."""
        if 0 <= self._queue_index < len(self._queue):
            item = self._queue[self._queue_index]
            item["status"] = "done"
            item["output_path"] = output_path
            self._batch_results.append((item["path"], True, None))
            self._update_queue_table()
        self._log(f"\n\u2713 Completed: {Path(output_path).parent.name}")

        from flamingo_stitcher.gui._compat import get_notification_service

        svc = get_notification_service(self)
        if svc is not None:
            acq_name = Path(output_path).parent.name
            svc.notify(
                "stitching_item_completed",
                title="Flamingo: stitched acquisition done",
                message=f"Stitched output written: {acq_name}",
                tags="card_file_box",
            )

    def _on_item_error(self, error_msg: str):
        """Handle error in one queue item.

        ntfy: no manual notify here \u2014 the stitching worker calls
        `logger.exception("Stitching pipeline error")` first, which the
        NtfyLogHandler attached to the root logger captures.
        """
        if 0 <= self._queue_index < len(self._queue):
            item = self._queue[self._queue_index]
            item["status"] = "error"
            item["error"] = error_msg
            self._batch_results.append((item["path"], False, error_msg))
            self._update_queue_table()
        self._log(f"\n\u2717 Error: {error_msg}")

    def _on_item_finished(self):
        """Handle worker thread completion for a queue item."""
        # Release the finished worker and reclaim its memory BEFORE the next
        # item's resource check reads available RAM. Critical on the failure
        # path: an in-memory fusion OOM raises a MemoryError whose traceback
        # holds the giant tile/output arrays in a reference cycle, so without
        # an explicit gc the next items see ~0 GB free and falsely abort
        # (cascading one failure into all remaining items).
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        import gc

        gc.collect()
        # If the item is still 'stitching', it was cancelled mid-run
        if 0 <= self._queue_index < len(self._queue):
            item = self._queue[self._queue_index]
            if item["status"] in ("stitching", "discovering"):
                item["status"] = "cancelled"
                self._batch_results.append((item["path"], False, "Cancelled"))
                self._update_queue_table()

        if self._batch_running:
            self._advance_queue()

    def _on_batch_complete(self):
        """Handle completion of all queue items."""
        from flamingo_stitcher.gui._compat import get_notification_service

        svc = get_notification_service(self)
        if svc is not None:
            n_success = sum(1 for _, ok, _ in self._batch_results if ok)
            n_error = sum(1 for _, ok, _ in self._batch_results if not ok)
            total = len(self._batch_results)
            if n_error:
                svc.notify(
                    "stitching_batch_completed",
                    title="Flamingo: stitching batch finished (with errors)",
                    message=f"{n_success}/{total} succeeded, {n_error} failed.",
                    priority="high",
                    tags="warning",
                )
            else:
                svc.notify(
                    "stitching_batch_completed",
                    title="Flamingo: stitching batch done",
                    message=f"All {total} acquisition(s) stitched successfully.",
                    tags="white_check_mark",
                )

        self._batch_running = False
        self._queue_index = -1
        self._batch_config = None
        self._batch_channels = None

        # Re-enable UI
        self._set_config_controls_enabled(True)
        self._cancel_btn.setEnabled(False)
        self._update_action_buttons()

        n_success = sum(1 for _, ok, _ in self._batch_results if ok)
        n_error = sum(1 for _, ok, _ in self._batch_results if not ok)
        total = len(self._batch_results)

        self._log(f"\n{'=' * 60}")
        self._log(f"Batch complete: {n_success}/{total} succeeded")
        if n_error:
            self._log(f"  {n_error} failed:")
            for path, ok, err in self._batch_results:
                if not ok:
                    self._log(f"    \u2717 {path.name}: {err}")
        self._log(f"{'=' * 60}")

        # Final state: everything done, or mark the pill for whichever
        # step was current as errored if the run failed.
        if n_error:
            # Mark whichever step is currently "running" as error; any
            # earlier steps stay done.
            for k, _ in self._step_order:
                existing = self._step_labels[k].styleSheet()
                if "#FF8C00" in existing:  # orange = running
                    self._set_step_state(k, "error")
                    break
        elif n_success > 0:
            for k, _ in self._step_order:
                existing = self._step_labels[k].styleSheet()
                # Leave 'skipped' pills grey.
                if "#B0B0B0" not in existing:
                    self._set_step_state(k, "done")

        self._status_label.setText(f"Done: {n_success}/{total} succeeded")

        # Show summary dialog
        msg = QMessageBox(self)
        msg.setWindowTitle("Batch Stitching Complete")
        if n_error:
            msg.setIcon(QMessageBox.Warning)
            msg.setText(
                f"Batch stitching complete.\n\n"
                f"Succeeded: {n_success}/{total}\n"
                f"Failed: {n_error}/{total}"
            )
        else:
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"All {total} acquisition(s) stitched successfully!")

        # Find last successful output for "Load" option
        last_success_path = None
        for path, ok, _ in reversed(self._batch_results):
            if ok:
                acq_name = path.name
                last_success_path = str(
                    Path(self._output_dir_edit.text()) / f"{acq_name}_stitched"
                )
                break

        if last_success_path:
            # "Load … into Sample View" only makes sense when there's a viewer
            # to receive it (embedded in Py2Flamingo). In the standalone app
            # there's no Sample View, so offer just "Open Output Folder".
            load_btn = None
            if self._sample_view_available:
                load_btn = msg.addButton(
                    "Load Latest into Sample View", QMessageBox.AcceptRole
                )
            open_btn = msg.addButton("Open Output Folder", QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Close)
            msg.exec_()
            clicked = msg.clickedButton()
            if load_btn is not None and clicked == load_btn:
                self.load_stitched_requested.emit(last_success_path)
            elif clicked == open_btn:
                import subprocess
                import sys

                folder = self._output_dir_edit.text()
                if sys.platform == "win32":
                    subprocess.Popen(["explorer", folder])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
        else:
            msg.addButton(QMessageBox.Close)
            msg.exec_()

    # --- Format-dependent UI ---

    def _on_format_changed(self, _index=None):
        """Update compression options and .ozx/pyramid toggles for selected format."""
        fmt = self._format_combo.currentData()
        has_zarr = fmt in ("ome-zarr-sharded", "ome-zarr-v2", "both")
        has_tiff = fmt in ("ome-tiff", "both")
        self._ozx_cb.setEnabled(has_zarr)
        if not has_zarr:
            self._ozx_cb.setChecked(False)
        # TIFF pyramids toggle only relevant for TIFF output
        if hasattr(self, "_tiff_pyramids_cb"):
            self._tiff_pyramids_cb.setEnabled(has_tiff)
        self._update_compression_options()

    def _update_compression_options(self):
        """Populate compression combo based on selected output format.

        Only offers codecs that are actually available in the current
        environment. zstd for TIFF requires the imagecodecs package.
        """
        fmt = self._format_combo.currentData()
        prev = self._compression_combo.currentData()
        self._compression_combo.blockSignals(True)
        self._compression_combo.clear()

        if fmt == "imaris":
            # PyImarisWriter handles compression internally (Gzip Level 2).
            # Lock the combo to a single informational entry.
            self._compression_combo.addItem("(internal: Gzip L2)", "gzip")
            self._compression_combo.setEnabled(False)
            self._compression_combo.blockSignals(False)
            return

        self._compression_combo.setEnabled(True)

        if fmt == "ome-tiff":
            # zlib and lzw are always available (built into tifffile/Python)
            self._compression_combo.addItem("zlib (best compatibility)", "zlib")
            self._compression_combo.addItem("lzw (balanced)", "lzw")
            # zstd for TIFF requires imagecodecs
            if self._tiff_zstd_available():
                self._compression_combo.addItem("zstd (best ratio)", "zstd")
            self._compression_combo.addItem("None (no compression)", "none")
            default = "zlib"
        elif fmt in ("ome-zarr-sharded", "ome-zarr-v2"):
            # Zarr codecs are handled by numcodecs, always available
            self._compression_combo.addItem("zstd (recommended)", "zstd")
            self._compression_combo.addItem("lz4 (fastest codec)", "lz4")
            self._compression_combo.addItem("blosc (compatible)", "blosc")
            self._compression_combo.addItem("None (no compression)", "none")
            default = "zstd"
        else:
            # "both" — show zarr options (tiff will use zlib internally)
            self._compression_combo.addItem("zstd (recommended)", "zstd")
            self._compression_combo.addItem("lz4 (fastest codec)", "lz4")
            self._compression_combo.addItem("blosc (compatible)", "blosc")
            self._compression_combo.addItem("None (no compression)", "none")
            default = "zstd"

        # Restore previous selection if still valid for this format
        idx = self._compression_combo.findData(prev)
        if idx >= 0:
            self._compression_combo.setCurrentIndex(idx)
        else:
            idx = self._compression_combo.findData(default)
            if idx >= 0:
                self._compression_combo.setCurrentIndex(idx)
        self._compression_combo.blockSignals(False)

    @staticmethod
    def _tiff_zstd_available() -> bool:
        """Check if zstd compression is available for TIFF output."""
        try:
            import imagecodecs  # noqa: F401

            return True
        except ImportError:
            pass
        try:
            from compression import zstd  # noqa: F401

            return True
        except ImportError:
            pass
        return False

    # --- Button styling helpers ---

    def _set_btn_green(self, btn):
        """Style a button with a green 'call to action' appearance."""
        btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 6px 14px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #888; color: #ccc; }"
        )

    def _set_btn_default(self, btn):
        """Reset a button to the default (platform) appearance."""
        btn.setStyleSheet("")

    # --- Logging helper ---

    def _log(self, message: str):
        """Append a message to the log area."""
        self._log_text.append(message)

    # --- Preprocessing environment ---

    def showEvent(self, event):
        """Refresh flat-field availability when this tab/dialog becomes visible.

        Lets an env built from the *other* tab light up here without a restart.
        Re-probes only while flat-field still looks unavailable, so it doesn't
        spawn a probe on every show once it's known good.
        """
        super().showEvent(event)
        if getattr(self, "_flat_field_cb", None) is not None and (
            not self._flat_field_cb.isEnabled()
        ):
            try:
                self._update_preprocessing_availability()
            except Exception:
                pass

    def _update_preprocessing_availability(self):
        """Grey out processing options whose backend isn't importable.

        Covers flat-field (basicpy), Leonardo FUSE (leonardo-toolset),
        destriping (pystripe), and deconvolution (pycudadecon / RedLionfish).
        Each disabled control gets a tooltip explaining what's missing and
        how to get it.
        """
        # --- Flat-field correction (basicpy, direct or isolated env) ---
        from flamingo_stitcher import preprocessing_env
        from flamingo_stitcher.flat_field import is_available as _ff_available

        if _ff_available():
            self._flat_field_cb.setEnabled(True)
            self._flat_field_cb.setToolTip(
                "Estimate and correct illumination non-uniformity\n"
                "from tile data (BaSiC algorithm, no calibration needed).\n"
                "Improves tile intensity consistency and reduces seams."
            )
        else:
            self._flat_field_cb.setChecked(False)
            self._flat_field_cb.setEnabled(False)
            self._flat_field_cb.setToolTip(
                "Flat-field correction requires basicpy.\n"
                "Click 'Set up flat-field…' to install it\n"
                "in an isolated environment."
            )

        # Reflect install state in the setup button so it's clear when it's done.
        # Guarded: this method is first called while the flat-field checkbox is
        # built (to set its state), which is *before* the setup button exists.
        if hasattr(self, "_setup_env_btn"):
            if preprocessing_env.is_built():
                self._setup_env_btn.setText("Reinstall flat-field…")
                self._setup_env_btn.setToolTip(
                    f"Flat-field environment is installed at:\n"
                    f"{preprocessing_env.env_dir()}\n\n"
                    "Click only to repair it or move it to another drive."
                )
            else:
                self._setup_env_btn.setText("Set up flat-field…")

        # --- Destriping (pystripe) ---
        try:
            import pystripe  # noqa: F401

            pystripe_ok = True
        except Exception:
            pystripe_ok = False

        if pystripe_ok:
            self._destripe_cb.setEnabled(True)
            self._destripe_cb.setToolTip(
                "PyStripe wavelet destriping per Z-plane.\n"
                "Removes stripe artifacts from illumination-side scanning."
            )
        else:
            self._destripe_cb.setChecked(False)
            self._destripe_cb.setEnabled(False)
            self._destripe_fast_cb.setChecked(False)
            self._destripe_fast_cb.setEnabled(False)
            self._destripe_cb.setToolTip(
                "Destriping requires pystripe.\n" "Install with: pip install pystripe"
            )

        # --- Deconvolution (pycudadecon or RedLionfish) ---
        from flamingo_stitcher import deconvolution as _decon

        if _decon.is_available():
            self._deconv_cb.setEnabled(True)
            self._deconv_cb.setToolTip(
                "Per-tile GPU deconvolution before stitching.\n"
                "Uses pycudadecon (NVIDIA) or RedLionfish (any GPU)."
            )
        else:
            self._deconv_cb.setChecked(False)
            self._deconv_cb.setEnabled(False)
            self._deconv_cb.setToolTip(_decon.unavailable_reason())

        # --- Leonardo FUSE illumination fusion combo item ---
        try:
            from flamingo_stitcher.isolated_service import (
                IsolatedPreprocessingService,
            )

            leo_ok = IsolatedPreprocessingService().has_leonardo()
        except Exception:
            leo_ok = False

        leo_idx = self._fusion_combo.findData("leonardo")
        if leo_idx >= 0:
            item = self._fusion_combo.model().item(leo_idx)
            if item is not None:
                item.setEnabled(leo_ok)
                if leo_ok:
                    item.setToolTip(
                        "Leonardo FUSE content-based illumination fusion\n"
                        "(leonardo-toolset in the isolated preprocessing env)."
                    )
                else:
                    item.setToolTip(
                        "Leonardo FUSE requires leonardo-toolset in the\n"
                        "isolated preprocessing environment.\n"
                        "Click 'Set up flat-field…' to install it."
                    )
            # If the current selection is Leonardo but it's now disabled,
            # fall back to Max so the run doesn't pick an unavailable mode.
            if not leo_ok and self._fusion_combo.currentData() == "leonardo":
                max_idx = self._fusion_combo.findData("max")
                if max_idx >= 0:
                    self._fusion_combo.setCurrentIndex(max_idx)

    def set_sample_view_available(self, available: bool) -> None:
        """Tell the dialog whether a 3D Sample View exists to receive output.

        The standalone app calls this with False (it has no viewer), which
        hides the "Load … into Sample View" button on the batch-complete
        dialog. Embedded in Py2Flamingo it stays True (the default).
        """
        self._sample_view_available = bool(available)

    # GitHub blob URL for the bundled troubleshooting doc — the always-works
    # fallback when a local copy can't be found (and the most current source).
    _TROUBLESHOOTING_URL = (
        "https://github.com/uw-loci/flamingo-stitcher/blob/main/"
        "src/flamingo_stitcher/docs/stitching_hardware_troubleshooting.md"
    )

    def _find_troubleshooting_doc(self) -> Optional[Path]:
        """Locate the bundled troubleshooting doc, or None.

        The doc is package data at ``flamingo_stitcher/docs/``. That path works
        for source/pip runs directly, and for the frozen build via PyInstaller's
        ``_MEIPASS`` unpack dir. Returns the first that exists.
        """
        import sys

        here = Path(__file__).resolve()
        candidates = [
            # Bundled in the package: src/flamingo_stitcher/docs/...
            here.parent.parent / "docs" / "stitching_hardware_troubleshooting.md",
        ]
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            candidates.append(
                Path(sys._MEIPASS)
                / "flamingo_stitcher"
                / "docs"
                / "stitching_hardware_troubleshooting.md"
            )
        return next((p for p in candidates if p.exists()), None)

    def _on_open_help_doc(self):
        """Open the stitching hardware / troubleshooting doc.

        Opens a local bundled copy in the user's default markdown/text viewer
        when one is present; otherwise falls back to opening the doc on GitHub
        in the browser (always available, always current). No more dead-end
        "couldn't be found" message.
        """
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices

        doc_path = self._find_troubleshooting_doc()
        if doc_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(doc_path)))
        else:
            QDesktopServices.openUrl(QUrl(self._TROUBLESHOOTING_URL))

    def _choose_install_location(self) -> bool:
        """Confirm or change where the flat-field env installs. Persists the
        choice. Returns True to proceed, False to cancel.
        """
        from flamingo_stitcher import preprocessing_env

        current = preprocessing_env.install_root()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Flat-field install location")
        box.setText(
            f"The flat-field environment (~2–4 GB incl. package cache) will "
            f"install to:\n\n{current}\n\nInstall here, or choose another "
            f"drive/folder if space is limited?"
        )
        here = box.addButton("Install here", QMessageBox.AcceptRole)
        box.addButton("Choose folder…", QMessageBox.ActionRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(here)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is cancel:
            return False
        if clicked is not here:  # "Choose folder…"
            start = str(current if current.exists() else current.parent)
            chosen = QFileDialog.getExistingDirectory(
                self, "Choose flat-field install folder", start
            )
            if not chosen:
                return False
            target = Path(chosen) / "FlamingoFlatField"
            preprocessing_env.set_install_root(target)
            self._log(f"Flat-field will install to: {target}")
        return True

    def _on_setup_env(self):
        """Build the flat-field environment with pixi (no system Python needed).

        Downloads pixi + Python + basicpy in a background thread, streaming
        progress to the log. The user does nothing technical — one click.
        """
        from flamingo_stitcher import preprocessing_env

        if preprocessing_env.is_built():
            reply = QMessageBox.question(
                self,
                "Flat-field environment exists",
                "The flat-field environment is already installed.\n\n"
                "Reinstall it (e.g. to move it to another drive or repair it)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # Let the user choose where it installs (the default may be on a
        # space-limited C: drive). Persisted for the runtime to find it.
        if not self._choose_install_location():
            return

        self._expand_log()
        self._log("\n=== Setting up flat-field environment (pixi) ===")
        self._setup_env_btn.setEnabled(False)
        self._setup_env_btn.setText("Setting up…")

        self._env_thread = _FlatFieldSetupThread(self)
        self._env_thread.progress.connect(self._log)
        self._env_thread.done.connect(self._on_setup_finished)
        self._env_thread.start()

    def _on_setup_finished(self, success: bool, error: str):
        """Handle completion of the pixi flat-field setup thread."""
        self._setup_env_btn.setEnabled(True)
        self._setup_env_btn.setText("Set up flat-field…")

        if success:
            self._log("\n=== Flat-field environment ready ===")
            from flamingo_stitcher.isolated_service import (
                IsolatedPreprocessingService,
            )

            service = IsolatedPreprocessingService()
            service.clear_cache()
            self._update_preprocessing_availability()
            self._log(
                f"  basicpy: {'available' if service.has_basicpy() else 'not found'}"
            )
        else:
            self._log("\n=== Setup failed ===")
            self._log(error)
            QMessageBox.warning(
                self,
                "Flat-field setup failed",
                f"Could not build the flat-field environment:\n\n{error}",
            )

    # --- Background-zero preview ---

    def _on_preview_background_zero(self) -> None:
        """Run the stitching pipeline at preview downsample on the first
        discovered queue item, then open a napari viewer for threshold
        picking. The chosen thresholds are written back into the panel.
        """
        from PyQt5.QtCore import QThread

        from flamingo_stitcher.pipeline import StitchingPipeline

        if not _napari_available():
            QMessageBox.information(
                self,
                "napari not installed",
                "The background-zero <b>preview</b> needs napari, which is not "
                "installed in this build.<br><br>"
                "Background zeroing still works — set per-channel thresholds "
                "numerically and run as normal. To enable the live 3-D preview, "
                'install napari (<code>pip install "flamingo-stitcher[preview]"</code>).',
            )
            return

        first = next(
            (it for it in self._queue if it.get("tiles")),
            None,
        )
        if first is None:
            QMessageBox.warning(
                self,
                "No tiles",
                "Discover tiles first — the preview needs at least one "
                "discovered acquisition to run on.",
            )
            return

        # Build a config snapshot. Background zeroing itself is forced
        # off inside run_preview(); the rest of the preprocessing chain
        # runs as configured so the preview matches the real pipeline.
        config = self._build_config()

        channels = self._parse_channels()
        tiles = first["tiles"]
        acquisition_dir = first["path"]

        self._log(
            f"\n=== Background-zero preview for {acquisition_dir.name} "
            f"({len(tiles)} tiles) ==="
        )

        class _PreviewWorker(QThread):
            done = pyqtSignal(object, object)  # (result_dict | None, error_str | None)

            def __init__(self, _config, _acq_dir, _channels, _tiles):
                super().__init__()
                self._config = _config
                self._acq_dir = _acq_dir
                self._channels = _channels
                self._tiles = _tiles

            def run(self):
                try:
                    pipeline = StitchingPipeline(self._config)
                    result = pipeline.run_preview(
                        self._acq_dir,
                        channels=self._channels,
                        tiles=self._tiles,
                    )
                    self.done.emit(result, None)
                except Exception as exc:  # surface to UI thread
                    logger.exception("Background-zero preview failed")
                    self.done.emit(None, str(exc))

        # Lock the panel while the preview is running.
        self._bg_zero_panel.setEnabled(False)
        self._discover_btn.setEnabled(False)
        self._run_btn.setEnabled(False)

        worker = _PreviewWorker(config, acquisition_dir, channels, tiles)
        # Hold a reference so the QThread is not garbage-collected mid-run.
        self._bg_preview_worker = worker
        worker.done.connect(
            lambda r, e: self._on_preview_done(
                r, e, current_thresholds=self._bg_zero_panel.thresholds()
            )
        )
        worker.start()

    def _on_preview_done(
        self,
        result: Optional[Dict[int, "object"]],
        error: Optional[str],
        current_thresholds: Dict[int, int],
    ) -> None:
        self._bg_zero_panel.setEnabled(True)
        self._discover_btn.setEnabled(True)
        self._update_action_buttons()

        worker = getattr(self, "_bg_preview_worker", None)
        if worker is not None:
            worker.deleteLater()
            self._bg_preview_worker = None

        if error is not None:
            self._log(f"Preview failed: {error}")
            QMessageBox.critical(
                self,
                "Preview failed",
                f"Background-zero preview failed:\n\n{error}",
            )
            return
        if not result:
            self._log("Preview produced no channel data.")
            return

        for ch_id, vol in result.items():
            self._log(
                f"  ch {ch_id}: shape={tuple(vol.shape)} "
                f"min={int(vol.min())} max={int(vol.max())}"
            )

        from flamingo_stitcher.gui.background_zero_preview_dialog import (
            BackgroundZeroPreviewDialog,
        )

        dlg = BackgroundZeroPreviewDialog(
            preview_volumes=result,
            initial_thresholds=current_thresholds,
            parent=self,
        )
        if dlg.exec_() == dlg.Accepted:
            chosen = dlg.thresholds()
            self._log(
                "Applied thresholds from preview: "
                + ", ".join(f"ch{ch}={t}" for ch, t in sorted(chosen.items()))
            )
            self._bg_zero_panel.set_thresholds(chosen)
            self._bg_zero_panel.set_enabled_state(True)
        else:
            self._log("Preview cancelled — thresholds unchanged.")

    # --- Settings persistence ---

    def _save_settings(self):
        """Save dialog settings to QSettings."""
        s = QSettings()
        s.beginGroup(_SETTINGS_GROUP)
        # Save queue paths (only pending/done items, not transient states)
        paths = [str(item["path"]) for item in self._queue]
        s.setValue("queue_paths", paths)
        s.setValue("output_dir", self._output_dir_edit.text())
        s.setValue("pixel_size", self._pixel_size_spin.value())
        s.setValue("z_step", self._z_step_spin.value())
        s.setValue("downsample_xy", self._downsample_xy_combo.currentData())
        s.setValue("downsample_z", self._downsample_z_combo.currentData())
        s.setValue("fusion", self._fusion_combo.currentData())
        s.setValue("tile_overlap_fusion", self._tile_fusion_combo.currentData())
        s.setValue("frame_size_idx", self._frame_size_combo.currentIndex())
        s.setValue("verbose_log", self._verbose_log_cb.isChecked())
        s.setValue("flat_field", self._flat_field_cb.isChecked())
        s.setValue("destripe", self._destripe_cb.isChecked())
        s.setValue("destripe_fast", self._destripe_fast_cb.isChecked())
        s.setValue("deconvolution", self._deconv_cb.isChecked())
        s.setValue("content_based_fusion", self._content_fusion_cb.isChecked())
        s.setValue("chunk_size_idx", self._chunk_size_combo.currentIndex())
        s.setValue("package_ozx", self._ozx_cb.isChecked())
        s.setValue("tiff_pyramids", self._tiff_pyramids_cb.isChecked())
        s.setValue("output_format", self._format_combo.currentData())
        s.setValue("compression", self._compression_combo.currentData())
        s.setValue("channels", self._channels_edit.text())
        s.setValue("streaming_mode", self._streaming_combo.currentIndex())
        s.setValue("skip_registration", self._skip_reg_cb.isChecked())
        s.setValue("reg_binning", self._reg_binning_combo.currentIndex())
        s.setValue("proc_options_expanded", self._proc_toggle.isChecked())
        s.setValue("log_expanded", self._log_toggle.isChecked())
        s.setValue("bg_zero_enabled", self._bg_zero_panel.is_enabled())
        s.setValue("bg_zero_expanded", self._bg_zero_panel.expanded())
        s.setValue(
            "bg_zero_thresholds_json",
            json.dumps(
                {str(k): int(v) for k, v in self._bg_zero_panel.thresholds().items()}
            ),
        )
        s.endGroup()

    def _restore_settings(self):
        """Restore dialog settings from QSettings."""
        s = QSettings()
        s.beginGroup(_SETTINGS_GROUP)

        # Restore queue paths
        paths = s.value("queue_paths", [], type=list)
        if paths:
            if isinstance(paths, str):
                paths = [paths]
            for p in paths:
                if p and Path(p).is_dir():
                    self._add_path_to_queue(Path(p))
        # Legacy: also try old single acq_dir key
        if not self._queue:
            acq_dir = s.value("acq_dir", "", type=str)
            if acq_dir and Path(acq_dir).is_dir():
                self._add_path_to_queue(Path(acq_dir))

        output_dir = s.value("output_dir", "", type=str)
        if output_dir:
            self._output_dir_edit.setText(output_dir)

        pixel_size = s.value("pixel_size", self._default_pixel_um, type=float)
        # Restoring a persisted value must not count as a manual override —
        # auto-fill from the acquisition's objective should still win on discover.
        self._setting_pixel_programmatically = True
        self._pixel_size_spin.setValue(pixel_size)
        self._setting_pixel_programmatically = False
        self._pixel_size_user_set = False

        z_step = s.value("z_step", 0.0, type=float)
        self._z_step_spin.setValue(z_step)

        ds_xy = s.value("downsample_xy", 0, type=int)
        if ds_xy:
            idx = self._downsample_xy_combo.findData(ds_xy)
            if idx >= 0:
                self._downsample_xy_combo.setCurrentIndex(idx)
        else:
            # Legacy: try old single "downsample" key as XY
            ds_old = s.value("downsample", 0, type=int)
            if ds_old:
                idx = self._downsample_xy_combo.findData(ds_old)
                if idx >= 0:
                    self._downsample_xy_combo.setCurrentIndex(idx)
        ds_z = s.value("downsample_z", 0, type=int)
        if ds_z:
            idx = self._downsample_z_combo.findData(ds_z)
            if idx >= 0:
                self._downsample_z_combo.setCurrentIndex(idx)

        fusion = s.value("fusion", "", type=str)
        if fusion:
            idx = self._fusion_combo.findData(fusion)
            if idx >= 0:
                self._fusion_combo.setCurrentIndex(idx)

        tile_fusion = s.value("tile_overlap_fusion", "", type=str)
        if tile_fusion:
            idx = self._tile_fusion_combo.findData(tile_fusion)
            if idx >= 0:
                self._tile_fusion_combo.setCurrentIndex(idx)

        frame_idx = s.value("frame_size_idx", 0, type=int)
        if 0 <= frame_idx < self._frame_size_combo.count():
            self._frame_size_combo.setCurrentIndex(frame_idx)

        self._verbose_log_cb.setChecked(s.value("verbose_log", False, type=bool))

        flat_field = s.value("flat_field", False, type=bool)
        self._flat_field_cb.setChecked(flat_field)

        destripe = s.value("destripe", False, type=bool)
        self._destripe_cb.setChecked(destripe)

        destripe_fast = s.value("destripe_fast", False, type=bool)
        self._destripe_fast_cb.setChecked(destripe_fast)

        deconv = s.value("deconvolution", False, type=bool)
        self._deconv_cb.setChecked(deconv)

        content_fusion = s.value("content_based_fusion", False, type=bool)
        self._content_fusion_cb.setChecked(content_fusion)

        chunk_idx = s.value("chunk_size_idx", 2, type=int)
        if 0 <= chunk_idx < self._chunk_size_combo.count():
            self._chunk_size_combo.setCurrentIndex(chunk_idx)

        ozx = s.value("package_ozx", False, type=bool)
        self._ozx_cb.setChecked(ozx)

        tiff_pyramids = s.value("tiff_pyramids", True, type=bool)
        self._tiff_pyramids_cb.setChecked(tiff_pyramids)

        output_format = s.value("output_format", "", type=str)
        if output_format:
            idx = self._format_combo.findData(output_format)
            if idx >= 0:
                self._format_combo.setCurrentIndex(idx)

        # Restore compression after format (format determines available options)
        compression = s.value("compression", "", type=str)
        if compression:
            idx = self._compression_combo.findData(compression)
            if idx >= 0:
                self._compression_combo.setCurrentIndex(idx)

        channels = s.value("channels", "", type=str)
        if channels:
            self._channels_edit.setText(channels)

        streaming_idx = s.value("streaming_mode", 0, type=int)
        if 0 <= streaming_idx < self._streaming_combo.count():
            self._streaming_combo.setCurrentIndex(streaming_idx)

        skip_reg = s.value("skip_registration", False, type=bool)
        self._skip_reg_cb.setChecked(skip_reg)

        reg_binning_idx = s.value("reg_binning", 1, type=int)  # default = index 1
        if 0 <= reg_binning_idx < self._reg_binning_combo.count():
            self._reg_binning_combo.setCurrentIndex(reg_binning_idx)

        proc_expanded = s.value("proc_options_expanded", False, type=bool)
        self._proc_toggle.setChecked(proc_expanded)
        self._log_toggle.setChecked(s.value("log_expanded", False, type=bool))

        bg_zero_enabled = s.value("bg_zero_enabled", False, type=bool)
        self._bg_zero_panel.set_enabled_state(bg_zero_enabled)
        bg_zero_expanded = s.value("bg_zero_expanded", False, type=bool)
        self._bg_zero_panel.set_expanded(bg_zero_expanded)
        # Per-channel thresholds are persisted as JSON {ch_id: threshold}.
        # They cannot be applied until set_channels() runs (after Discover),
        # so stash them and replay after discovery.
        bg_zero_json = s.value("bg_zero_thresholds_json", "", type=str)
        self._pending_bg_zero_thresholds: Dict[int, int] = {}
        if bg_zero_json:
            try:
                parsed = json.loads(bg_zero_json)
                self._pending_bg_zero_thresholds = {
                    int(k): int(v) for k, v in parsed.items()
                }
            except (ValueError, TypeError):
                pass

        s.endGroup()

        # Persisted values can re-check options whose backend is missing
        # (e.g. Destripe True but pystripe not installed on this box). Re-run
        # the availability gate so unavailable options end up unchecked+disabled.
        self._update_preprocessing_availability()

    def closeEvent(self, event):
        """Save settings and cancel worker on close."""
        self._save_settings()
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        self._batch_running = False
        super().closeEvent(event)

    def hideEvent(self, event):
        """Save settings on hide."""
        self._save_settings()
        super().hideEvent(event)


# QSettings keys for native dialog (independent persistence)
_NATIVE_SETTINGS_GROUP = "NativeStitchingDialog"


class NativeStitchingDialog(StitchingDialog):
    """Stitching dialog for C++ server native flat-layout acquisitions.

    Overrides tile discovery to use discover_flat_tiles() which scans for
    .raw files with integer tile indices (X000_Y000) in a flat directory,
    rather than the subfolder-per-tile layout.
    """

    # Flat layout: parent only (1 level up).
    _acq_dir_restore_levels_up = 1

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.setWindowTitle("Tile Stitching (Single Workflow)")

    def _discover_tiles_for_path(self, acq_path: Path):
        """Discover tiles using flat-layout scanner."""
        from flamingo_stitcher.pipeline import (
            _read_plane_spacing,
            discover_flat_tiles,
        )

        tiles = discover_flat_tiles(acq_path)

        # Auto-set Z step from Workflow.txt if currently "Auto"
        if tiles and self._z_step_spin.value() == 0.0:
            for wf_candidate in [
                acq_path / "Workflow.txt",
                tiles[0].folder / "Workflow.txt",
            ]:
                if wf_candidate.exists():
                    spacing = _read_plane_spacing(wf_candidate)
                    if spacing:
                        self._z_step_spin.setValue(spacing)
                        self._log(f"Auto-detected Z step: {spacing} \u00b5m")
                    break

        return tiles

    def _looks_like_acquisition(self, path: Path) -> bool:
        """Check if a directory looks like a flat-layout acquisition."""
        if (path / "Workflow.txt").exists():
            return True
        return any(path.glob("*.raw"))

    def _save_settings(self):
        """Save dialog settings to QSettings (independent group)."""
        s = QSettings()
        s.beginGroup(_NATIVE_SETTINGS_GROUP)
        paths = [str(item["path"]) for item in self._queue]
        s.setValue("queue_paths", paths)
        s.setValue("output_dir", self._output_dir_edit.text())
        s.setValue("pixel_size", self._pixel_size_spin.value())
        s.setValue("z_step", self._z_step_spin.value())
        s.setValue("downsample_xy", self._downsample_xy_combo.currentData())
        s.setValue("downsample_z", self._downsample_z_combo.currentData())
        s.setValue("fusion", self._fusion_combo.currentData())
        s.setValue("tile_overlap_fusion", self._tile_fusion_combo.currentData())
        s.setValue("frame_size_idx", self._frame_size_combo.currentIndex())
        s.setValue("verbose_log", self._verbose_log_cb.isChecked())
        s.setValue("flat_field", self._flat_field_cb.isChecked())
        s.setValue("destripe", self._destripe_cb.isChecked())
        s.setValue("destripe_fast", self._destripe_fast_cb.isChecked())
        s.setValue("deconvolution", self._deconv_cb.isChecked())
        s.setValue("content_based_fusion", self._content_fusion_cb.isChecked())
        s.setValue("chunk_size_idx", self._chunk_size_combo.currentIndex())
        s.setValue("package_ozx", self._ozx_cb.isChecked())
        s.setValue("tiff_pyramids", self._tiff_pyramids_cb.isChecked())
        s.setValue("output_format", self._format_combo.currentData())
        s.setValue("compression", self._compression_combo.currentData())
        s.setValue("channels", self._channels_edit.text())
        s.setValue("streaming_mode", self._streaming_combo.currentIndex())
        s.setValue("skip_registration", self._skip_reg_cb.isChecked())
        s.setValue("reg_binning", self._reg_binning_combo.currentIndex())
        s.setValue("proc_options_expanded", self._proc_toggle.isChecked())
        s.setValue("log_expanded", self._log_toggle.isChecked())
        s.setValue("bg_zero_enabled", self._bg_zero_panel.is_enabled())
        s.setValue("bg_zero_expanded", self._bg_zero_panel.expanded())
        s.setValue(
            "bg_zero_thresholds_json",
            json.dumps(
                {str(k): int(v) for k, v in self._bg_zero_panel.thresholds().items()}
            ),
        )
        s.endGroup()

    def _restore_settings(self):
        """Restore dialog settings from QSettings (independent group)."""
        s = QSettings()
        s.beginGroup(_NATIVE_SETTINGS_GROUP)

        # Restore queue paths
        paths = s.value("queue_paths", [], type=list)
        if paths:
            if isinstance(paths, str):
                paths = [paths]
            for p in paths:
                if p and Path(p).is_dir():
                    self._add_path_to_queue(Path(p))
        # Legacy: also try old single acq_dir key
        if not self._queue:
            acq_dir = s.value("acq_dir", "", type=str)
            if acq_dir and Path(acq_dir).is_dir():
                self._add_path_to_queue(Path(acq_dir))

        output_dir = s.value("output_dir", "", type=str)
        if output_dir:
            self._output_dir_edit.setText(output_dir)

        pixel_size = s.value("pixel_size", self._default_pixel_um, type=float)
        # Restoring a persisted value must not count as a manual override —
        # auto-fill from the acquisition's objective should still win on discover.
        self._setting_pixel_programmatically = True
        self._pixel_size_spin.setValue(pixel_size)
        self._setting_pixel_programmatically = False
        self._pixel_size_user_set = False

        z_step = s.value("z_step", 0.0, type=float)
        self._z_step_spin.setValue(z_step)

        ds_xy = s.value("downsample_xy", 0, type=int)
        if ds_xy:
            idx = self._downsample_xy_combo.findData(ds_xy)
            if idx >= 0:
                self._downsample_xy_combo.setCurrentIndex(idx)
        else:
            # Legacy: try old single "downsample" key as XY
            ds_old = s.value("downsample", 0, type=int)
            if ds_old:
                idx = self._downsample_xy_combo.findData(ds_old)
                if idx >= 0:
                    self._downsample_xy_combo.setCurrentIndex(idx)
        ds_z = s.value("downsample_z", 0, type=int)
        if ds_z:
            idx = self._downsample_z_combo.findData(ds_z)
            if idx >= 0:
                self._downsample_z_combo.setCurrentIndex(idx)

        fusion = s.value("fusion", "", type=str)
        if fusion:
            idx = self._fusion_combo.findData(fusion)
            if idx >= 0:
                self._fusion_combo.setCurrentIndex(idx)

        tile_fusion = s.value("tile_overlap_fusion", "", type=str)
        if tile_fusion:
            idx = self._tile_fusion_combo.findData(tile_fusion)
            if idx >= 0:
                self._tile_fusion_combo.setCurrentIndex(idx)

        frame_idx = s.value("frame_size_idx", 0, type=int)
        if 0 <= frame_idx < self._frame_size_combo.count():
            self._frame_size_combo.setCurrentIndex(frame_idx)

        self._verbose_log_cb.setChecked(s.value("verbose_log", False, type=bool))

        flat_field = s.value("flat_field", False, type=bool)
        self._flat_field_cb.setChecked(flat_field)

        destripe = s.value("destripe", False, type=bool)
        self._destripe_cb.setChecked(destripe)

        destripe_fast = s.value("destripe_fast", False, type=bool)
        self._destripe_fast_cb.setChecked(destripe_fast)

        deconv = s.value("deconvolution", False, type=bool)
        self._deconv_cb.setChecked(deconv)

        content_fusion = s.value("content_based_fusion", False, type=bool)
        self._content_fusion_cb.setChecked(content_fusion)

        chunk_idx = s.value("chunk_size_idx", 2, type=int)
        if 0 <= chunk_idx < self._chunk_size_combo.count():
            self._chunk_size_combo.setCurrentIndex(chunk_idx)

        ozx = s.value("package_ozx", False, type=bool)
        self._ozx_cb.setChecked(ozx)

        tiff_pyramids = s.value("tiff_pyramids", True, type=bool)
        self._tiff_pyramids_cb.setChecked(tiff_pyramids)

        output_format = s.value("output_format", "", type=str)
        if output_format:
            idx = self._format_combo.findData(output_format)
            if idx >= 0:
                self._format_combo.setCurrentIndex(idx)

        compression = s.value("compression", "", type=str)
        if compression:
            idx = self._compression_combo.findData(compression)
            if idx >= 0:
                self._compression_combo.setCurrentIndex(idx)

        channels = s.value("channels", "", type=str)
        if channels:
            self._channels_edit.setText(channels)

        streaming_idx = s.value("streaming_mode", 0, type=int)
        if 0 <= streaming_idx < self._streaming_combo.count():
            self._streaming_combo.setCurrentIndex(streaming_idx)

        skip_reg = s.value("skip_registration", False, type=bool)
        self._skip_reg_cb.setChecked(skip_reg)

        reg_binning_idx = s.value("reg_binning", 1, type=int)
        if 0 <= reg_binning_idx < self._reg_binning_combo.count():
            self._reg_binning_combo.setCurrentIndex(reg_binning_idx)

        proc_expanded = s.value("proc_options_expanded", False, type=bool)
        self._proc_toggle.setChecked(proc_expanded)
        self._log_toggle.setChecked(s.value("log_expanded", False, type=bool))

        bg_zero_enabled = s.value("bg_zero_enabled", False, type=bool)
        self._bg_zero_panel.set_enabled_state(bg_zero_enabled)
        bg_zero_expanded = s.value("bg_zero_expanded", False, type=bool)
        self._bg_zero_panel.set_expanded(bg_zero_expanded)
        bg_zero_json = s.value("bg_zero_thresholds_json", "", type=str)
        self._pending_bg_zero_thresholds: Dict[int, int] = {}
        if bg_zero_json:
            try:
                parsed = json.loads(bg_zero_json)
                self._pending_bg_zero_thresholds = {
                    int(k): int(v) for k, v in parsed.items()
                }
            except (ValueError, TypeError):
                pass

        s.endGroup()

        # Persisted values can re-check options whose backend is missing
        # (e.g. Destripe True but pystripe not installed on this box). Re-run
        # the availability gate so unavailable options end up unchecked+disabled.
        self._update_preprocessing_availability()
