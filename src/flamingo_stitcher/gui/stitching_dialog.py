"""Tile Stitching Dialog.

Non-modal dialog for stitching raw acquisition tile data into a single volume.
Operates on saved acquisition data on disk — no microscope connection required.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QProcess, QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
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


class _NoScrollDoubleSpinBox(QDoubleSpinBox):
    """A spin box that ignores the mouse wheel unless it already has focus.

    The default QDoubleSpinBox grabs wheel events on hover, so brushing the
    scroll wheel over the form silently nudges a value — most damagingly the
    XY pixel size, where a stray 0.001 µm change quietly disables Discover's
    auto-fill. Click or tab into the box first to scroll-adjust it.
    """

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()
from flamingo_stitcher.gui._wheel_guard import install_wheel_guard


# QSettings keys
_SETTINGS_GROUP = "StitchingDialog"
# Shared (cross-dialog) key for the last folder browsed to via "Add…", so the
# file picker reopens there even when the queue is empty.
_LAST_BROWSE_KEY = "StitchingShared/last_browse_dir"

# Shown in the pinned memory-estimate label before tiles are discovered (the
# estimate needs the tile geometry), so the always-visible row explains why it's
# empty rather than just sitting blank.
_ESTIMATES_PLACEHOLDER = (
    "<span style='color:#888;'>Discover tiles to see in-memory &amp; streaming "
    "memory estimates and output size.</span>"
)


def _fmt_gb(gb: float) -> str:
    """Human-readable data size from a value in GiB, scaling MB / GB / TB.

    The estimate code works in GiB (bytes / 1024**3). Printing that with a
    fixed ``{:.0f} GB`` collapses any sub-GB size to a misleading "0 GB"
    (a 0.4 GB output read as nothing). Scale the unit instead so small jobs
    show MB and huge ones show TB. Binary units throughout (matches the rest
    of the estimate math, which is 1024-based).
    """
    try:
        gb = float(gb)
    except (TypeError, ValueError):
        return "? GB"
    if gb < 0:
        gb = 0.0
    if gb >= 1024.0:
        return f"{gb / 1024.0:.1f} TB"
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = gb * 1024.0
    if mb >= 1.0:
        return f"{mb:.0f} MB"
    kb = mb * 1024.0
    return f"{kb:.0f} KB" if kb >= 1.0 else "0 MB"


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

    # QSettings group for this dialog's persisted state. Overridable so sibling
    # tabs (Single Workflow, Multi-View) keep independent settings and don't
    # clobber each other's queue/options.
    _settings_group = _SETTINGS_GROUP

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

        # Tile orientation chosen via the Orientation Preview this session (a
        # TileOrientation, or None to fall back to the saved per-microscope
        # preset / default). Applied in _build_config.
        self._session_orientation = None

        # "Apply to all remaining" decision for existing outputs this batch:
        # None (ask each time), "overwrite", or "rename".
        self._overwrite_policy = None
        self._batch_channels = None
        self._batch_results = []  # List of (path, success, error_msg)
        # Last classified pipeline step; drives step-aware OOM advice on failure.
        self._current_step_key = None

        self.setWindowTitle("Tile Stitching")
        # Keep the width comfortable for the settings row, but allow a short
        # window (laptops) — the upper controls live in a scroll area and the
        # log collapses, so nothing squishes; it scrolls instead.
        self.setMinimumWidth(650)
        self.setMinimumHeight(440)
        self.resize(720, 750)  # Default size before geometry restore
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._setup_ui()
        # Stray wheel scrolls must not rewrite settings on the way past.
        install_wheel_guard(self)
        # Accept folders dropped anywhere on the dialog → add them to the queue.
        self.setAcceptDrops(True)
        self._restore_settings()
        # Initial paint of the Native → Output voxel readout using whatever
        # restore left in the spins + combos.
        self._update_voxel_readout()
        # And the initial button state. Without this the row opened with
        # Discover Tiles enabled over an empty queue, and nothing pointing at
        # Add... — the one thing that can actually be done from here.
        self._update_action_buttons()

    def _setup_ui(self):
        """Create and layout UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Frozen Output row — pinned at the very top of the tab, ABOVE the
        # (scrolling) queue + settings, so the output target / size / free space
        # stays visible no matter what the user scrolls to. Empty until set.
        self._build_frozen_output_row(layout)

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

        # Subclass hook: inject extra controls at the top of the scroll area
        # (e.g. the Multi-View tab's rotation controls + warning banner).
        self._add_dialog_extras(content_layout)

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

        # Left to right in the order the work is done. Add... comes first
        # because it is the first thing anyone must click: with an empty queue
        # there is nothing to discover and nothing to run, so leading with
        # Discover Tiles pointed the eye at a disabled button. Add All in Folder
        # sits beside it as the bulk variant of the same step, then Discover.
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

        # Accent-coloured to draw the eye — it's the easy-to-forget step between
        # adding directories and running. It flashes while the queue has items
        # added since the last Discover (see _discover_needed and the timer).
        self._discover_btn = QPushButton("Discover Tiles")
        self._discover_btn.setToolTip(
            "Scan all queued directories for tile data so channels, frame size\n"
            "and Z step are detected before you Run.\n"
            "Flashes when directories were added since the last Discover.\n"
            "(Optional — Run will auto-discover if needed.)"
        )
        self._discover_btn.clicked.connect(self._on_discover)
        queue_btn_layout.addWidget(self._discover_btn)

        # One flashing call-to-action at a time, pointing at whichever step is
        # next: Add... while the queue is empty, Discover Tiles once there is
        # something to discover. Two buttons flashing at once is noise, not
        # guidance.
        self._discover_needed = False
        self._cta_flash_on = False
        self._cta_button = None
        self._cta_flash_timer = QTimer(self)
        self._cta_flash_timer.setInterval(550)
        self._cta_flash_timer.timeout.connect(self._on_cta_flash_tick)

        self._orientation_btn = QPushButton("Orientation Preview…")
        self._orientation_btn.setToolTip(
            "Preview all 8 whole-mosaic orientations for the selected\n"
            "acquisition (built from its per-tile MIP files) so you can pick\n"
            "the one where the sample is framed correctly. Useful when a new\n"
            "microscope frames the mosaic differently. Does not run a stitch."
        )
        self._orientation_btn.clicked.connect(self._on_orientation_preview)
        queue_btn_layout.addWidget(self._orientation_btn)

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

        self._load_config_btn = QPushButton("Load Configuration…")
        self._load_config_btn.setToolTip(
            "Load stitching settings from another run's stitch_metadata.json\n"
            "(or a saved configuration file) to reuse a setup that worked —\n"
            "e.g. one shared by another user.\n\n"
            "Applies processing options only. Pixel size, Z spacing, frame AOI\n"
            "and the output location are left as they are and re-detected by\n"
            "Discover, so the settings transfer cleanly to your own data."
        )
        self._load_config_btn.clicked.connect(self._on_load_configuration)
        queue_btn_layout.addWidget(self._load_config_btn)

        queue_layout.addLayout(queue_btn_layout)
        queue_group.setLayout(queue_layout)
        content_layout.addWidget(queue_group)

        # (The Output Directory row moved to the frozen top row — see
        # _build_frozen_output_row, pinned above the scroll area.)

        # --- Settings, split into three plainly-named groups so a new user
        # isn't met with one dense wall of controls. Advanced toggles stay in
        # the collapsible "Processing Options" panel below. The three boxes
        # live in a container assigned to self._config_container, so the single
        # setEnabled() that locks settings during a run still disables them all.
        settings_container = QWidget()
        settings_vbox = QVBoxLayout(settings_container)
        settings_vbox.setContentsMargins(0, 0, 0, 0)
        settings_vbox.setSpacing(10)

        def _group_hint(text):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
            return lbl

        # ===== Group 1: "Tell me about your image" (facts about the data) =====
        image_group = QGroupBox("Tell me about your image")
        image_outer = QVBoxLayout()
        self._image_hint = _group_hint(
            "Detected automatically when you add an acquisition and press "
            "Discover — usually you can leave these as they are."
        )
        image_outer.addWidget(self._image_hint)
        settings_layout = QGridLayout()
        settings_layout.setSpacing(6)
        image_outer.addLayout(settings_layout)

        # Load hardware-derived pixel size default
        try:
            from flamingo_stitcher.config_loader import get_hardware_config

            _hw = get_hardware_config()
            self._default_pixel_um = round(_hw.effective_pixel_size_um, 4)
        except Exception:
            self._default_pixel_um = 0.406

        # Row 0: Pixel size + Z step
        settings_layout.addWidget(QLabel("Pixel size (\u00b5m):"), 0, 0)
        self._pixel_size_spin = _NoScrollDoubleSpinBox()
        self._pixel_size_spin.setFocusPolicy(Qt.StrongFocus)
        self._pixel_size_spin.setRange(0.01, 100.0)
        self._pixel_size_spin.setDecimals(3)
        self._pixel_size_spin.setValue(self._default_pixel_um)
        self._pixel_size_spin.setSingleStep(0.001)
        settings_layout.addWidget(self._pixel_size_spin, 0, 1)

        settings_layout.addWidget(QLabel("Z step (\u00b5m):"), 0, 2)
        self._z_step_spin = _NoScrollDoubleSpinBox()
        self._z_step_spin.setFocusPolicy(Qt.StrongFocus)
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

        # Which channels to stitch — a fact about the acquisition, so it lives
        # here with the other "about your image" inputs rather than with output.
        settings_layout.addWidget(QLabel("Channels:"), 1, 0)
        self._channels_edit = QLineEdit()
        self._channels_edit.setPlaceholderText("All (or e.g. 0,1)")
        self._channels_edit.setToolTip(
            "Leave empty for all channels, or comma-separated list (e.g. 0,1)"
        )
        settings_layout.addWidget(self._channels_edit, 1, 1)

        image_group.setLayout(image_outer)
        settings_vbox.addWidget(image_group)

        # ===== Group 2: "What kind of processing should we do?" =====
        proc_basic_group = QGroupBox("What kind of processing should we do?")
        proc_outer = QVBoxLayout()
        proc_outer.addWidget(
            _group_hint(
                "The defaults are a good starting point for most samples. "
                "Finer controls live under “Processing Options” below."
            )
        )
        settings_layout = QGridLayout()
        settings_layout.setSpacing(6)
        proc_outer.addLayout(settings_layout)

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
            ("16x", 16),
            ("32x", 32),
            ("iso", -1),
        ]:
            self._downsample_xy_combo.addItem(label, value)
        self._downsample_xy_combo.setToolTip(
            "XY downsample factor.\n"
            "1x…32x reduces tile width and height. The heavy factors\n"
            "(16x/32x) are for fast overviews and test runs.\n\n"
            "iso: auto-pick BOTH XY and Z factors so the output voxel\n"
            "is as close to cubic as possible, using the allowed\n"
            "factors (XY: 1/2/4/8/16/32, Z: 1/2/4/8/16). Resolved at run time\n"
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
        for label, value in [
            ("1x", 1),
            ("2x", 2),
            ("4x", 4),
            ("8x", 8),
            ("16x", 16),
        ]:
            self._downsample_z_combo.addItem(label, value)
        self._downsample_z_combo.setToolTip(
            "Z downsample factor (independent of XY).\n"
            "8x/16x are for fast overviews and test runs.\n\n"
            "Averages each group of N planes (a block mean), streamed one\n"
            "slab at a time — so it shrinks the output, speeds up fusion,\n"
            "and lowers the memory peak.\n\n"
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
            ("Separate (keep light paths)", "separate"),
        ]:
            self._fusion_combo.addItem(label, value)
        self._fusion_combo.setToolTip(
            "Combines the LEFT and RIGHT light-sheet illumination sides of a\n"
            "single tile. Has no effect when only one illumination side was\n"
            "acquired. This is NOT how adjacent tiles are combined — see\n"
            "'Tile overlap'.\n\n"
            "  Max / Mean / Leonardo FUSE — fuse the two sides into one channel.\n\n"
            "  Separate (keep light paths) — DIAGNOSTIC: do NOT fuse. Stitch each\n"
            "    side independently and write it as its own output channel\n"
            "    (Channel_<ch>_I0, Channel_<ch>_I1) in the same file, so you can\n"
            "    flip between them in one viewer and tell a per-side artifact\n"
            "    apart from one introduced by fusing. Doubles the output channel\n"
            "    count and runs in streaming mode."
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
            ("Brightest tile", "brightest"),
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
            "            doesn't fill the field of view).\n\n"
            "  Brightest tile — winner-take-all. Ranks tiles by overall mean\n"
            "            intensity and, in every overlap, takes ALL pixels from\n"
            "            the brighter tile (not a per-pixel max). Avoids the\n"
            "            per-pixel noise-picking of Maximum and the seam dip of\n"
            "            Blend — a clean, self-consistent patchwork of whole\n"
            "            tiles. Best when tiles differ in exposure/illumination."
        )
        tile_fuse_box.addWidget(self._tile_fusion_combo)
        settings_layout.addLayout(tile_fuse_box, 1, 4)

        proc_basic_group.setLayout(proc_outer)
        settings_vbox.addWidget(proc_basic_group)

        # ===== Group 3: "How should we save it?" =====
        output_group = QGroupBox("How should we save it?")
        output_outer = QVBoxLayout()
        output_outer.addWidget(
            _group_hint(
                "OME-Zarr (Fiji compatible) is a safe default. The size, time, "
                "and memory estimates below update as you change these."
            )
        )
        settings_layout = QGridLayout()
        settings_layout.setSpacing(6)
        output_outer.addLayout(settings_layout)

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

        # Row 3: Memory mode (Channels now lives in the "image" group above)
        settings_layout.addWidget(QLabel("Memory mode:"), 3, 0)
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
        settings_layout.addWidget(self._streaming_combo, 3, 1)

        # Memory safety indicator
        self._memory_indicator = QLabel("")
        self._memory_indicator.setFixedWidth(40)
        self._memory_indicator.setAlignment(Qt.AlignCenter)
        self._last_mem_estimate = None
        settings_layout.addWidget(self._memory_indicator, 3, 2)

        # Scratch (temp) directory for streaming spill files.
        settings_layout.addWidget(QLabel("Scratch dir:"), 3, 3)
        self._scratch_dir_edit = QLineEdit()
        self._scratch_dir_edit.setPlaceholderText("(default: alongside output)")
        self._scratch_dir_edit.setToolTip(
            "Where streaming's temporary spill files (.stitch_tmp: per-tile\n"
            "memmaps + the fused memmap) are written.\n"
            "\n"
            "DEFAULT (blank): alongside the output — <output_dir>/.stitch_tmp.\n"
            "\n"
            "Point this at a FAST LOCAL disk (NVMe/SSD) when the input/output are\n"
            "on a slow or network drive: streaming is I/O-bound on this spill, so\n"
            "moving it off the busy drive can cut wall time a lot.\n"
            "Cautions:\n"
            "• No benefit if it is the SAME physical drive as the input/output\n"
            "  (the run log warns in that case).\n"
            "• It must have room for the tile spill + fused memmap (see the disk\n"
            "  estimate); the run aborts up front if it lacks the space.\n"
            "• The finished output still writes to the output folder, so the\n"
            "  fused data crosses drives once (the normal fuse→write pass — NOT\n"
            "  an extra copy of the result). .stitch_tmp is deleted at the end."
        )
        scratch_row = QHBoxLayout()
        scratch_row.setContentsMargins(0, 0, 0, 0)
        scratch_row.addWidget(self._scratch_dir_edit)
        scratch_browse = QPushButton("Browse...")
        scratch_browse.clicked.connect(self._browse_scratch_dir)
        scratch_row.addWidget(scratch_browse)
        scratch_wrap = QWidget()
        scratch_wrap.setLayout(scratch_row)
        settings_layout.addWidget(scratch_wrap, 3, 4)

        # (The "Output information" live estimates — native→output voxel,
        # in-memory/streaming memory, and queue time — are pinned in the frozen
        # Output row at the top of the tab, see _build_frozen_output_row, so
        # they stay visible while the user changes settings here.)

        # Track whether the user has manually set the XY pixel size, so the
        # ScopeSettings-derived auto-fill on discover doesn't clobber a
        # deliberate choice. A guard distinguishes programmatic setValue.
        self._pixel_size_user_set = False
        self._setting_pixel_programmatically = False
        self._pixel_size_spin.valueChanged.connect(self._on_pixel_size_changed)
        # Same guard for Z step: distinguish a deliberate in-session edit from a
        # programmatic/restored value, so Discover re-detects Z from the data
        # (overriding a stale persisted value) unless the user set it by hand.
        self._z_step_user_set = False
        self._setting_z_step_programmatically = False
        self._z_step_spin.valueChanged.connect(self._on_z_step_changed)
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

        output_group.setLayout(output_outer)
        settings_vbox.addWidget(output_group)

        # The group boxes share one container so the existing
        # self._config_container.setEnabled(False) during a run disables them
        # all. NOTE: keep this distinct from the class attr _settings_group,
        # which is the QSettings group-name string used by _save/_restore.
        self._config_container = settings_container
        content_layout.addWidget(settings_container)

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
            "Removes stripe/shadow artifacts from light-sheet data\n"
            "(orientation set by the Dir: control).\n"
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

        # Stripe orientation. The filter is axis-fixed AND destriping runs in the
        # raw camera frame (before the per-tile rot/flip), so the wrong axis
        # removes nothing. "Auto" detects the stripe direction per tile.
        self._destripe_dir_combo = QComboBox()
        for _lbl, _val in (
            ("Dir: Auto", "auto"),
            ("Dir: Horizontal", "horizontal"),
            ("Dir: Vertical", "vertical"),
        ):
            self._destripe_dir_combo.addItem(_lbl, _val)
        self._destripe_dir_combo.setToolTip(
            "Which stripe orientation to remove.\n"
            "Auto (default): detect per tile.\n"
            "Horizontal / Vertical: force it if Auto guesses wrong.\n\n"
            "Note: orientation is judged in the raw CAMERA frame, before the\n"
            "tile is rotated to stage — so it may look 90° off vs the final image."
        )
        self._destripe_dir_combo.setEnabled(False)

        # Both destripe sub-options share grid cell (0, 1) via an HBox. Row 0's
        # remaining cells (0,2)-(0,3) belong to the content-based-blending
        # checkbox — putting the combo there made the two overlap on screen.
        # Filter tuning (sigma / level / wavelet / threshold / crossover) lives in
        # a small dialog rather than cluttering this grid; values persist via
        # QSettings like every other option.
        self._destripe_params: Dict[str, Any] = {}
        self._destripe_settings_btn = QPushButton("Settings…")
        self._destripe_settings_btn.setToolTip(
            "Tune the destripe filter: sigma (strength), wavelet level/type,\n"
            "and the foreground/background split."
        )
        self._destripe_settings_btn.setEnabled(False)
        self._destripe_settings_btn.clicked.connect(self._on_destripe_settings)

        # Live preview: tuning by re-running a whole stitch is far too slow a
        # loop, and the things that go wrong (which axis is filtered, a wavelet
        # level deeper than the frame supports, how much the per-plane Otsu
        # split shifts the result between tiles) are invisible in the mosaic.
        self._destripe_preview_btn = QPushButton("Preview…")
        self._destripe_preview_btn.setToolTip(
            "See the filter applied to one tile plane, before and after,\n"
            "updating live as you change the settings. Pick a tile, scrub Z."
        )
        self._destripe_preview_btn.setEnabled(False)
        self._destripe_preview_btn.clicked.connect(self._on_destripe_preview)

        _destripe_opts = QHBoxLayout()
        _destripe_opts.setContentsMargins(0, 0, 0, 0)
        _destripe_opts.addWidget(self._destripe_fast_cb)
        _destripe_opts.addWidget(self._destripe_dir_combo)
        _destripe_opts.addWidget(self._destripe_settings_btn)
        _destripe_opts.addWidget(self._destripe_preview_btn)
        _destripe_opts.addStretch()
        proc_layout.addLayout(_destripe_opts, 0, 1)

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

        # Registration gets its own titled box. Previously its controls were
        # spread across the shared grid with each combo stretched to the right
        # edge, so "Reg. binning:" sat a hand's width from the dropdown it
        # names and nothing indicated which settings belonged together. Labels
        # now sit immediately left of the control they label.
        reg_group = QGroupBox("Registration")
        reg_layout = QGridLayout()
        reg_layout.setContentsMargins(8, 4, 8, 6)
        reg_layout.setHorizontalSpacing(6)
        reg_layout.setVerticalSpacing(4)
        # Only the control columns stretch; the label columns stay tight to
        # their widget instead of being pushed apart by the free space.
        reg_layout.setColumnStretch(0, 0)
        reg_layout.setColumnStretch(1, 1)
        reg_layout.setColumnStretch(2, 0)
        reg_layout.setColumnStretch(3, 1)

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
        reg_layout.addWidget(self._skip_reg_cb, 0, 0, 1, 2)

        # XY and Z chosen separately, laid out like the Downsample row above.
        # They were one three-way preset (Fine/Default/Fast) that moved both at
        # once, which is the wrong coupling: Z binning sets the floor on how
        # precisely a Z shift can be resolved -- at the default z=2 that is one
        # raw plane, and the 3-6 frame offsets this registration exists to fix
        # are only a few times bigger. Halving Z costs one axis of correlation
        # work; being forced to halve XY as well costs four times that for no
        # gain in Z precision.
        self._reg_binning_label = QLabel("Reg. binning:")
        reg_layout.addWidget(self._reg_binning_label, 0, 2)
        rb_layout = QHBoxLayout()
        rb_layout.setSpacing(4)
        self._reg_binning_xy_label = QLabel("XY")
        rb_layout.addWidget(self._reg_binning_xy_label)
        self._reg_binning_xy_combo = QComboBox()
        for _label, _value in [("1x", 1), ("2x", 2), ("4x", 4), ("8x", 8)]:
            self._reg_binning_xy_combo.addItem(_label, _value)
        self._reg_binning_xy_combo.setToolTip(
            "How much to bin tiles laterally for phase correlation.\n"
            "Applied to BOTH X and Y — a lateral peak is one joint\n"
            "correlation, so splitting them further would not buy anything.\n\n"
            "Cost scales with the square of this: 2x is a quarter of the work\n"
            "of 1x. 4x (default) is enough for the lateral shifts a stage\n"
            "makes; drop to 2x or 1x only if the seam report shows XY\n"
            "disagreement."
        )
        rb_layout.addWidget(self._reg_binning_xy_combo)
        rb_layout.addSpacing(16)  # visible gap between the XY group and Z group
        self._reg_binning_z_label = QLabel("Z")
        rb_layout.addWidget(self._reg_binning_z_label)
        self._reg_binning_z_combo = QComboBox()
        for _label, _value in [("1x", 1), ("2x", 2), ("4x", 4), ("8x", 8)]:
            self._reg_binning_z_combo.addItem(_label, _value)
        self._reg_binning_z_combo.setToolTip(
            "How much to bin tiles axially for phase correlation, independent\n"
            "of XY.\n\n"
            "This sets the FLOOR on Z precision: a shift can only be resolved\n"
            "to about one binned voxel, so at the default 2x that is one raw\n"
            "plane. The tile offsets this registration exists to fix are 3-6\n"
            "frames, so there is not much headroom.\n\n"
            "1x doubles the correlation volume but is the cheapest way to\n"
            "improve Z alignment — it costs one axis, where lowering XY costs\n"
            "two. Use it before reaching for 'Refine Z alignment', which runs\n"
            "a whole second pass."
        )
        rb_layout.addWidget(self._reg_binning_z_combo)
        rb_layout.addStretch(1)
        reg_layout.addLayout(rb_layout, 0, 3)
        self._set_registration_binning({"z": 2, "y": 4, "x": 4})

        # Proc Row 3: Max registration shift (registration sub-option; only
        # meaningful when registration runs, so it greys out with Skip reg).
        self._max_reg_shift_label = QLabel("Max reg. shift:")
        reg_layout.addWidget(self._max_reg_shift_label, 1, 0)
        self._max_reg_shift_spin = _NoScrollDoubleSpinBox()
        self._max_reg_shift_spin.setRange(0.0, 100000.0)
        self._max_reg_shift_spin.setDecimals(1)
        self._max_reg_shift_spin.setSingleStep(5.0)
        self._max_reg_shift_spin.setSuffix(" µm")
        # 0 shows "Auto" — the pipeline then uses one overlap width as the cap.
        self._max_reg_shift_spin.setSpecialValueText("Auto (one overlap)")
        self._max_reg_shift_spin.setValue(0.0)
        self._max_reg_shift_spin.setToolTip(
            "Cap how far registration may move a tile from its stage position.\n\n"
            "multiview-stitcher bounds a phase-correlation shift to the tile SIZE,\n"
            "not the overlap, so a low-content tile (background / featureless blur)\n"
            "can be flung ~a full tile away and open a gap. Tiles whose correction\n"
            "exceeds this cap are reset to their stage position.\n\n"
            "Auto (0) = the smaller of the X/Y overlap widths, so a tile can never\n"
            "move more than one overlap (gaps impossible). Set a smaller value to\n"
            "allow only fine refinement. Ignored when Skip registration is on."
        )
        reg_layout.addWidget(self._max_reg_shift_spin, 1, 1)

        # Proc Row 4: axial registration controls. Z is separated from the
        # lateral cap above because the two are not the same measurement: the
        # lateral bound is a tile overlap width, and a mosaic tiled only in X/Y
        # has no Z overlap to derive a bound from.
        self._max_reg_shift_z_label = QLabel("Max Z shift:")
        reg_layout.addWidget(self._max_reg_shift_z_label, 1, 2)
        self._max_reg_shift_z_spin = _NoScrollDoubleSpinBox()
        self._max_reg_shift_z_spin.setRange(0.0, 100000.0)
        self._max_reg_shift_z_spin.setDecimals(1)
        self._max_reg_shift_z_spin.setSingleStep(5.0)
        self._max_reg_shift_z_spin.setSuffix(" µm")
        self._max_reg_shift_z_spin.setSpecialValueText("Auto")
        self._max_reg_shift_z_spin.setValue(0.0)
        self._max_reg_shift_z_spin.setToolTip(
            "Cap how far registration may move a tile along Z.\n\n"
            "Separate from 'Max reg. shift', which bounds X/Y. They cannot share\n"
            "a number: the lateral cap is one tile OVERLAP width, and tiles that\n"
            "only tile in X/Y all span the same depth — there is no Z overlap to\n"
            "derive a bound from, so the lateral value would be arbitrary in Z.\n\n"
            "Auto (0) = at least 8 Z steps and never under 25 µm (enough for the\n"
            "few-frame stage/focus drift seen in practice), capped at a quarter of\n"
            "the stack so a peak found halfway down the volume is still rejected.\n\n"
            "A tile clamped in Z was NOT measured — it kept its stage position.\n"
            "The registration report says which tiles those were."
        )
        reg_layout.addWidget(self._max_reg_shift_z_spin, 1, 3)

        self._z_refine_cb = QCheckBox("Refine Z alignment ✱")
        self._z_refine_cb.setToolTip(
            "Run a second registration pass whose only job is Z.\n\n"
            "The main pass registers at Z binning 2 and multiview-stitcher\n"
            "upsamples 3-D phase correlation by only 2, so Z resolves to about one\n"
            "raw plane while X/Y resolve a fraction of a pixel — the coarsest\n"
            "treatment on the axis with the coarsest voxel. This pass re-registers\n"
            "from the first pass's result at full Z resolution and contributes ONLY\n"
            "its Z component; X/Y are left exactly as the first pass set them.\n\n"
            "Costs roughly 2-3x the registration time. Turn it on when\n"
            "registration_seams.csv shows pairs disagreeing in Z while X/Y is clean."
        )
        reg_layout.addWidget(self._z_refine_cb, 2, 0, 1, 2)

        self._z_refine_range_label = QLabel("Z search:")
        self._z_refine_range_spin = _NoScrollDoubleSpinBox()
        self._z_refine_range_spin.setRange(1.0, 100000.0)
        self._z_refine_range_spin.setDecimals(0)
        self._z_refine_range_spin.setSingleStep(10.0)
        self._z_refine_range_spin.setPrefix("± ")
        self._z_refine_range_spin.setSuffix(" µm")
        self._z_refine_range_spin.setValue(40.0)
        self._z_refine_range_spin.setToolTip(
            "Half-width of the Z-refinement search.\n\n"
            "A correction that comes back AT this limit is a lower bound on the\n"
            "error, not the error, so it is rejected — the tile keeps its\n"
            "first-pass Z and the report counts it separately. Widening the search\n"
            "is the fix, not trusting the number."
        )
        reg_layout.addWidget(self._z_refine_range_label, 2, 2)
        reg_layout.addWidget(self._z_refine_range_spin, 2, 3)

        reg_group.setLayout(reg_layout)
        proc_layout.addWidget(reg_group, 2, 0, 3, 4)

        # Proc Row 5: Fusion chunk size
        proc_layout.addWidget(QLabel("Fusion chunk size:"), 5, 0)
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
        proc_layout.addWidget(self._chunk_size_combo, 5, 1, 1, 3)

        # Proc Row 5: Tile-border artifact QC (diagnostic)
        self._border_qc_cb = QCheckBox("Detect border artifacts (QC)")
        self._border_qc_cb.setToolTip(
            "After preprocessing, scan neighboring-tile seams for sharp\n"
            "intensity steps (stitching artifacts). Writes a plain-text report\n"
            "next to the run log listing the offending tile pairs. Reference\n"
            "channel only; cheap (reads only the thin border strips).\n\n"
            "Most sensitive at downsample_xy \u2264 2 \u2014 heavy downsampling softens\n"
            "single-pixel steps."
        )
        proc_layout.addWidget(self._border_qc_cb, 6, 0)
        self._border_qc_label = QLabel("QC detail:")
        proc_layout.addWidget(self._border_qc_label, 6, 1)
        self._border_qc_mode_combo = QComboBox()
        self._border_qc_mode_combo.addItem("MIP length (fast)", "mip")
        self._border_qc_mode_combo.addItem("Full (area + Z-range)", "full")
        self._border_qc_mode_combo.addItem("Pairs only", "pairs")
        self._border_qc_mode_combo.setToolTip(
            "MIP length: Z max-project, report affected border length (fast).\n"
            "Full: per-Z area + Z-range (richer; slower at native resolution).\n"
            "Pairs only: just the list of offending tile pairs."
        )
        proc_layout.addWidget(self._border_qc_mode_combo, 6, 2)

        self._reg_report_cb = QCheckBox("Registration report")
        self._reg_report_cb.setChecked(True)
        self._reg_report_cb.setToolTip(
            "Write registration_report.csv, registration_seams.csv and\n"
            "registration_report.txt into the output folder, beside\n"
            "stitch_metadata.json.\n\n"
            "Registration is the one step whose effect is invisible in the output:\n"
            "it either silently helped or silently hurt. These say how far each\n"
            "tile moved, which seams were used, and which corrections were\n"
            "rejected as implausible.\n\n"
            "Stays on even with Skip registration — the report then says the tiles\n"
            "were never registered, which is the thing worth knowing."
        )
        proc_layout.addWidget(self._reg_report_cb, 6, 3)

        # Proc Row 6: Legend
        legend = QLabel("\u2731 = significantly increases processing time")
        legend.setStyleSheet("color: #FF8C00; font-style: italic; font-size: 11px;")
        proc_layout.addWidget(legend, 7, 0, 1, 4)

        self._proc_widget.setLayout(proc_layout)
        self._proc_widget.setVisible(False)
        content_layout.addWidget(self._proc_widget)

        # Processing options change the memory footprint too (content-based
        # blending adds a per-block halo, deconvolution adds a float32 working
        # set, chunk/frame size change the fusion block, etc.). Wire every one
        # of them to the estimate so the readout updates live as they toggle.
        # These widgets are created above line-by-line, so this block runs
        # after they all exist (unlike the output-group wiring earlier).
        for _cb in (
            self._destripe_cb,
            self._destripe_fast_cb,
            self._content_fusion_cb,
            self._deconv_cb,
            self._flat_field_cb,
            self._skip_reg_cb,
        ):
            _cb.toggled.connect(self._refresh_memory_estimate)
        for _combo in (
            self._frame_size_combo,
            self._fusion_combo,
            self._tile_fusion_combo,
            self._reg_binning_xy_combo,
            self._reg_binning_z_combo,
            self._chunk_size_combo,
        ):
            _combo.currentIndexChanged.connect(self._refresh_memory_estimate)

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
        # (Discover Tiles is pinned in the frozen Output row at the top; the
        # run controls stay here at the bottom.)
        btn_layout = QHBoxLayout()

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
        # A clear header bar (background + border) so it's obvious this is a
        # collapsible pane, not just a label — the log otherwise blends into the
        # controls above it when open.
        self._log_toggle.setStyleSheet(
            "QPushButton { text-align: left; padding: 5px 8px; font-weight: bold; "
            "color: #333; background-color: #ececec; "
            "border: 1px solid #b0b0b0; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e0e0e0; }"
            "QPushButton:checked { background-color: #dbe7f3; "
            "border-color: #7aa7d0; }"
        )
        self._log_toggle.toggled.connect(self._on_log_toggle)
        layout.addWidget(self._log_toggle)

        self._log_group = QGroupBox()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_opts_row = QHBoxLayout()
        log_opts_row.setContentsMargins(0, 0, 0, 0)
        self._verbose_log_cb = QCheckBox("Verbose log (include Python output)")
        self._verbose_log_cb.setToolTip(
            "Include behind-the-scenes Python output (flat-field, Imaris/Zarr\n"
            "writers, isolated environment, etc.) in this log. Off by default\n"
            "for a concise run log; turn on to troubleshoot a failed/odd run.\n"
            "Takes effect on the next run."
        )
        log_opts_row.addWidget(self._verbose_log_cb)
        self._timestamp_log_cb = QCheckBox("Timestamps")
        self._timestamp_log_cb.setToolTip(
            "Prefix each new log line with a compact date + time to the minute\n"
            "(MM-DD HH:MM), so you can see when a line was printed — handy on\n"
            "long runs where progress lines repeat with only the ETA changing.\n"
            "Applies to lines printed from now on; takes effect immediately."
        )
        log_opts_row.addWidget(self._timestamp_log_cb)
        log_opts_row.addStretch()
        log_layout.addLayout(log_opts_row)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMinimumHeight(120)
        self._log_text.setStyleSheet(
            "QTextEdit { font-family: monospace; font-size: 11px; }"
        )

        # A cheerful "working" indicator: marching flamingos animate to the
        # right of the log, but only while a run is in progress. Purely
        # cosmetic — degrades to just the log if the GIF/QMovie can't load.
        self._flamingo_movie = None
        self._flamingo_label = self._build_flamingo_indicator()
        if self._flamingo_label is None:
            log_layout.addWidget(self._log_text)
        else:
            log_body_row = QHBoxLayout()
            log_body_row.setContentsMargins(0, 0, 0, 0)
            log_body_row.addWidget(self._log_text, 1)
            log_body_row.addWidget(
                self._flamingo_label, 0, Qt.AlignTop | Qt.AlignHCenter
            )
            log_layout.addLayout(log_body_row)
        self._log_group.setLayout(log_layout)
        # A visible frame around the open log so the pane's extent is obvious
        # (it previously blended into the surrounding controls).
        self._log_group.setStyleSheet(
            "QGroupBox { border: 1px solid #7aa7d0; border-radius: 4px; "
            "margin-top: 2px; padding: 6px; }"
        )
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
        # Wrap so the ETA tail "M:SS remaining (Done at ~HH:MM)" stays visible
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
            self._remember_browse_dir(Path(folder).parent)
            self._add_path_to_queue(Path(folder))

    def _add_folder_to_queue(self):
        """Add all acquisition subdirectories from a parent folder."""
        start = self._queue_browse_start()
        parent = QFileDialog.getExistingDirectory(
            self, "Select Parent Folder (contains acquisition folders)", start
        )
        if not parent:
            return

        self._remember_browse_dir(parent)
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
        # Fall back to the last folder the user browsed to — survives an empty
        # queue (e.g. after processing + clearing) so "Add…" reopens there.
        last_browse = QSettings().value(_LAST_BROWSE_KEY, "", type=str)
        if last_browse and Path(last_browse).exists():
            return last_browse
        output = self._output_dir_edit.text()
        if output and Path(output).parent.exists():
            return str(Path(output).parent)
        return str(Path.home())

    def _remember_browse_dir(self, folder: str) -> None:
        """Persist the folder the user just browsed to (shared across dialogs)."""
        try:
            if folder:
                QSettings().setValue(_LAST_BROWSE_KEY, str(folder))
        except Exception:
            pass

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
            if child.suffix.lower() in (".raw", ".tif", ".tiff", ".btf"):
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
        # A directory was added, so a Discover is now due — flag it so the
        # Discover Tiles button flashes until the user runs it.
        self._discover_needed = True
        self._update_queue_table()

        # Auto-set output directory from first item. Also re-set if the field
        # holds a stale path on a now-disconnected drive (otherwise the old
        # value sticks and the user keeps writing to a dead location).
        if len(self._queue) == 1:
            out = self._output_dir_edit.text().strip()
            out_unreachable = bool(out) and not Path(out).parent.exists()
            if not out or out_unreachable:
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

    def _on_orientation_preview(self):
        """Open the whole-mosaic orientation preview for the chosen acquisition.

        Uses the selected queue row, or the only/first queued item when nothing
        is selected. Builds an 8-orientation MIP-mosaic preview so the user can
        pick the orientation that frames the sample correctly.
        """
        acq_path = self._selected_queue_path("Orientation preview")
        if acq_path is None:
            return
        self._open_orientation_preview_for(acq_path)

    def _open_orientation_preview_for(self, acq_path) -> None:
        """Open the 8-orientation preview for a specific acquisition path."""
        try:
            from flamingo_stitcher.gui.orientation_preview_dialog import (
                OrientationPreviewDialog,
            )

            dlg = OrientationPreviewDialog(acq_path, parent=self)
            dlg.orientation_chosen.connect(self._on_orientation_chosen)
            dlg.exec_()
        except Exception as e:  # noqa: BLE001 - never let a preview crash the app
            self._logger.exception("Orientation preview failed to open")
            QMessageBox.critical(
                self, "Orientation preview", f"Could not open preview:\n{e}"
            )

    def _on_orientation_chosen(
        self, name: str, reverse_x: bool, reverse_y: bool, microscope_name: str = ""
    ):
        """Feedback for an orientation picked in the preview.

        The preview persists the choice per-microscope ("Use for stitching"), so
        it auto-applies to matching acquisitions via per-entry resolution at run
        time — this handler only logs it. It is deliberately NOT pushed onto the
        whole batch (that used to mis-orient datasets from other systems).
        """
        from flamingo_stitcher.orientation import TileOrientation

        self._session_orientation = TileOrientation(
            name=name, reverse_x=reverse_x, reverse_y=reverse_y
        )
        rev = []
        if reverse_x:
            rev.append("reverse X")
        if reverse_y:
            rev.append("reverse Y")
        suffix = (" + " + ", ".join(rev)) if rev else ""
        # Name the microscope so the log is unambiguous about what it applies to.
        scope = (
            f"microscope '{microscope_name}'" if microscope_name else "this microscope"
        )
        self._log(
            f"Tile orientation '{name}{suffix}' saved for {scope}; "
            f"it will auto-apply to its acquisitions."
        )

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
                f"Added {added} folder(s) via drag-and-drop. " f"Click Discover or Run."
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
        elif self._queue:
            # Queue exists but nothing pending (all done/error)
            self._discover_btn.setEnabled(False)
            self._run_btn.setEnabled(False)
            self._set_btn_default(self._run_btn)
        else:
            # Empty queue
            self._discover_btn.setEnabled(False)
            self._run_btn.setEnabled(False)
            self._set_btn_default(self._run_btn)
        # Discover button accent/flash reflects whether a Discover is due.
        self._refresh_discover_style()

    def _build_frozen_output_row(self, parent_layout):
        """Pinned Output row at the very top of the tab.

        Stays visible above the (scrolling) queue + settings so the output
        target, projected size, and free space are always in view. Holds the
        Output Directory field (moved out of the scroll area) plus a one-line
        ``dir | size | free`` summary that starts empty.
        """
        group = QGroupBox("Output")
        v = QVBoxLayout(group)
        v.setContentsMargins(8, 4, 8, 6)
        v.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output Directory:"))
        self._output_dir_edit = QLineEdit()
        self._output_dir_edit.setPlaceholderText(
            "Shared output folder (each acquisition gets a subfolder)..."
        )
        self._output_dir_edit.textChanged.connect(self._update_output_info)
        row.addWidget(self._output_dir_edit)
        out_browse_btn = QPushButton("Browse...")
        out_browse_btn.clicked.connect(self._browse_output_dir)
        row.addWidget(out_browse_btn)
        v.addLayout(row)

        # One-line summary: dir | ~size | free space. Empty until a dir is set.
        self._output_info_label = QLabel("")
        self._output_info_label.setTextFormat(Qt.PlainText)
        self._output_info_label.setWordWrap(True)
        self._output_info_label.setStyleSheet("color: #555;")
        v.addWidget(self._output_info_label)

        # ===== Live estimates (pinned here so they stay visible while the user
        # changes downsample/format/etc. instead of scrolling away). Laid out
        # to use the horizontal space rather than stack tall. =====
        # Memory: in-memory vs streaming peak, colour-coded — the headline the
        # user watches while tuning options. Rich text for per-term colour.
        self._memory_label = QLabel(_ESTIMATES_PLACEHOLDER)
        self._memory_label.setTextFormat(Qt.RichText)
        self._memory_label.setWordWrap(True)
        self._memory_label.setStyleSheet("font-size: 11px;")
        v.addWidget(self._memory_label)

        # Native → output voxel size (left) and rough queue time (right) share
        # one row to keep the block short.
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        self._voxel_readout_label = QLabel("")
        self._voxel_readout_label.setStyleSheet("color: #444; font-size: 11px;")
        self._voxel_readout_label.setToolTip(
            "Native voxel (XY pixel × XY pixel × Z step) from the fields\n"
            "above, then the resulting output voxel after applying the\n"
            "chosen downsample factors. For 'iso', the factors are\n"
            "resolved from the native voxel and shown in parentheses."
        )
        info_row.addWidget(self._voxel_readout_label)
        info_row.addStretch()
        self._time_label = QLabel("")
        self._time_label.setTextFormat(Qt.RichText)
        self._time_label.setStyleSheet("font-size: 11px;")
        self._time_label.setToolTip(
            "Approximate total wall time to stitch the Pending items in the "
            "queue.\nUses measured times from previous runs of similar settings "
            "when available;\notherwise a rough guess. Accuracy improves as you "
            "run more acquisitions."
        )
        info_row.addWidget(self._time_label)
        v.addLayout(info_row)

        parent_layout.addWidget(group)

    def _update_output_info(self, *_):
        """Refresh the frozen Output summary: ``dir | ~size | free space``.

        Size comes from the last memory estimate (once tiles are discovered);
        free space is probed on the nearest existing parent of the output dir
        (the dir itself may not exist yet). Defensive — cosmetic only.
        """
        label = getattr(self, "_output_info_label", None)
        if label is None:
            return
        edit = getattr(self, "_output_dir_edit", None)
        out = edit.text().strip() if edit is not None else ""
        if not out:
            label.setText("")
            return
        parts = []
        est = getattr(self, "_last_mem_estimate", None)
        if est and est.get("output_gb"):
            parts.append(f"~{_fmt_gb(est['output_gb'])} output")
        try:
            import shutil

            probe = Path(out)
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            free_gb = shutil.disk_usage(str(probe)).free / (1024**3)
            drive = Path(out).drive or str(probe)
            parts.append(f"{free_gb:.0f} GB free on {drive}")
        except Exception:
            pass
        suffix = ("   |   " + "   |   ".join(parts)) if parts else ""
        label.setText(f"Output: {out}{suffix}")

    def _browse_output_dir(self):
        start = self._output_dir_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start
        )
        if folder:
            self._output_dir_edit.setText(str(Path(folder)))

    def _browse_scratch_dir(self):
        start = (
            self._scratch_dir_edit.text()
            or self._output_dir_edit.text()
            or str(Path.home())
        )
        folder = QFileDialog.getExistingDirectory(
            self, "Select Scratch (Temp) Directory — use a fast local disk", start
        )
        if folder:
            self._scratch_dir_edit.setText(str(Path(folder)))

    # --- Tile discovery ---


    def _describe_discovered(self, tiles) -> str:
        """One line naming the channels and illumination sides discovered.

        Channel numbers are zero-based HARDWARE indices — the 4th laser is
        channel 3 — so a bare "3" reads as three channels. Leads with the
        count and names the laser.
        """
        channels = sorted({ch for t in tiles for ch in getattr(t, "channels", [])})
        try:
            from flamingo_stitcher.pipeline import describe_channel_set

            ch_text = describe_channel_set(channels)
        except Exception:  # noqa: BLE001 - never break discovery over a label
            ch_text = ", ".join(str(c) for c in channels) or "none"

        sides = sorted(
            {s for t in tiles for m in getattr(t, "raw_files", {}).values() for s in m}
        )
        if len(sides) > 1:
            side_text = (
                f"{len(sides)} illumination sides ({', '.join(f'I{s}' for s in sides)})"
            )
        elif sides:
            side_text = f"1 illumination side (I{sides[0]})"
        else:
            side_text = "illumination sides unknown"
        return f"Channels: {ch_text} · {side_text}"

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
            item["warnings"] = []
            try:
                tiles = self._discover_tiles_for_path(item["path"])
                if tiles:
                    item["tiles"] = tiles
                    self._log(f"  Found {len(tiles)} tiles")
                    # Say what is IN them. The channel set decides what gets
                    # written and how long the run takes, and until now
                    # discovery never showed it — the user found out at run
                    # time, or from the stitched result.
                    self._log(f"  {self._describe_discovered(tiles)}")
                    # Surface per-tile data-quality warnings (corrupt / unreadable
                    # metadata used a grid-estimated position; truncated image
                    # files) prominently — a run can continue but the user must
                    # know the data may be off.
                    warned = [
                        t for t in tiles if getattr(t, "metadata_warning", None)
                    ]
                    if warned:
                        item["warnings"] = [t.metadata_warning for t in warned]
                        self._log(
                            f"  ⚠ {len(warned)} of {len(tiles)} tiles have "
                            f"data-quality warnings:"
                        )
                        for t in warned[:12]:
                            self._log(f"      • {t.metadata_warning}")
                        if len(warned) > 12:
                            self._log(
                                f"      • …and {len(warned) - 12} more"
                            )
                else:
                    self._log("  No tiles found")
                    item["status"] = "error"
                    item["error"] = "No tiles found"
            except Exception as e:
                self._log(f"  Error: {e}")
                self._logger.exception("Tile discovery error")
                item["status"] = "error"
                item["error"] = str(e)

        # Discover has now run for everything pending, so clear the "due" flag
        # and stop the button flashing.
        self._discover_needed = False
        self._update_queue_table()
        total = sum(len(it["tiles"]) for it in self._queue if it["tiles"])
        ok = sum(1 for it in pending if it["tiles"])
        self._log(f"\nDiscovered {total} tiles across {ok}/{len(pending)} directories")

        # A visible, can't-miss warning if any tile had corrupt/degraded data —
        # the Log pane is collapsed by default, so a log line alone is not enough.
        self._warn_on_discovery_issues(pending)
        self._update_action_buttons()

        # Auto-detect the Z step from the data on every Discover, overriding a
        # stale persisted value. A non-zero restored Z step used to block this
        # (the check was `== 0.0`), so a previous session's value silently rode
        # through and squished/stretched the volume along Z. We now re-detect
        # unless the user deliberately set Z by hand this session.
        detected_z = None
        first_item = next((it for it in self._queue if it.get("tiles")), None)
        if first_item:
            tiles0 = first_item["tiles"]
            # Prefer the authoritative Workflow.txt "Plane spacing (um)" field;
            # fall back to the z-range / plane-count value carried on the tile.
            try:
                from flamingo_stitcher.pipeline import _read_plane_spacing

                for wf in (
                    Path(first_item["path"]) / "Workflow.txt",
                    tiles0[0].folder / "Workflow.txt",
                ):
                    if wf.exists():
                        sp = _read_plane_spacing(wf)
                        if sp:
                            detected_z = float(sp)
                        break
            except Exception as e:
                self._logger.debug(f"plane-spacing read skipped: {e}")
            if detected_z is None and tiles0 and tiles0[0].z_step_mm:
                detected_z = tiles0[0].z_step_mm * 1000.0

        if detected_z:
            if not self._z_step_user_set:
                self._setting_z_step_programmatically = True
                self._z_step_spin.setValue(detected_z)
                self._setting_z_step_programmatically = False
                self._log(f"Auto-detected Z step: {detected_z:.3f} µm")
            elif abs(self._z_step_spin.value() - detected_z) > 0.05 * detected_z:
                self._log(
                    f"⚠ Z step {self._z_step_spin.value():.3f} µm differs from the "
                    f"detected {detected_z:.3f} µm. Verify before stitching — a wrong "
                    f"Z step stretches/squishes the volume along Z."
                )

        # Auto-fill XY pixel size from the recorded objective (ScopeSettings.txt).
        self._autofill_pixel_size()

        # Always record the effective XY pixel size in the LOG (the GUI spinbox
        # shows it, but the log is what gets shared for analysis). This is the
        # value that WILL be used, whatever its source (auto / manual / default).
        self._log(
            f"Effective XY pixel size: {self._pixel_size_spin.value():.4f} µm/px "
            f"(the value stitching will use)"
        )

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

        # Surface the auto-detected facts in the always-visible "image" group.
        # The full detail is logged above, but the log panel is collapsed by
        # default, so users miss what was detected (frame/AOI, pixel, Z, etc.).
        self._update_image_detected_hint()

    def _warn_on_discovery_issues(self, pending):
        """Pop a visible warning if any discovered tile had corrupt/degraded data.

        Discovery is resilient — a tile with a corrupt/unreadable _Settings.txt
        falls back to a grid-estimated position, and a truncated raw is flagged
        rather than aborting the run. But those only reach the (collapsed) Log
        pane, so this raises a modal summary the user can't miss. The stitch can
        still proceed; the point is informed consent that the data may be off.
        """
        issues = [
            (str(it["path"].name), w)
            for it in pending
            for w in it.get("warnings", [])
        ]
        if not issues:
            return
        n = len(issues)
        dirs = sorted({name for name, _ in issues})
        lines = [
            f"{n} tile{'s' if n != 1 else ''} across "
            f"{len(dirs)} acquisition{'s' if len(dirs) != 1 else ''} had "
            "corrupt or degraded data during discovery.",
            "",
            "Stitching can still run, but these tiles may be mis-placed or "
            "incomplete. Review before trusting the output:",
            "",
        ]
        for name, w in issues[:15]:
            lines.append(f"  • [{name}] {w}")
        if n > 15:
            lines.append(f"  • …and {n - 15} more (see the Log for the full list)")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Data-quality warnings")
        box.setText("\n".join(lines))
        box.setStandardButtons(QMessageBox.Ok)
        box.exec_()

    def _update_image_detected_hint(self):
        """Refresh the always-visible "Tell me about your image" hint with the
        values resolved on Discover: frame (camera AOI / ROI) size, XY pixel
        size, Z step, channels, and tile count. Keeps the auto-detected facts
        in front of the user without needing to expand the log.
        """
        if not hasattr(self, "_image_hint"):
            return
        all_tiles = [t for it in self._queue if it.get("tiles") for t in it["tiles"]]
        if not all_tiles:
            self._image_hint.setText(
                "Detected automatically when you add an acquisition and press "
                "Discover — usually you can leave these as they are."
            )
            return

        try:
            from flamingo_stitcher.pipeline import _resolve_frame_size

            fw, fh = _resolve_frame_size(all_tiles)
            frame_str = f"{fw}×{fh} px"
            if len({(t.frame_width, t.frame_height) for t in all_tiles}) > 1:
                frame_str += " (varies — check for mixed acquisitions!)"
        except Exception:
            frame_str = "auto"

        channels = sorted({ch for t in all_tiles for ch in t.channels})
        # "channel(s) 3" reads as three channels when it means the 4th laser.
        try:
            from flamingo_stitcher.pipeline import describe_channel_set

            ch_str = describe_channel_set(channels) if channels else "?"
        except Exception:  # noqa: BLE001 - a hint must never break the dialog
            ch_str = ", ".join(str(c) for c in channels) if channels else "?"
        px = self._pixel_size_spin.value()
        z = self._z_step_spin.value()
        z_str = f"{z:.3f} µm" if z > 0 else "auto"

        self._image_hint.setTextFormat(Qt.RichText)
        self._image_hint.setText(
            f"<b>Detected:</b> frame/ROI <b>{frame_str}</b> · "
            f"pixel <b>{px:.3f} µm</b> · Z step <b>{z_str}</b> · "
            f"channels <b>{ch_str}</b> · <b>{len(all_tiles)}</b> tiles. "
            "Filled in from the acquisition — change only if you know a value is wrong."
        )

    def _on_pixel_size_changed(self, _value):
        """Mark the XY pixel size as user-set unless we set it ourselves."""
        if not self._setting_pixel_programmatically:
            self._pixel_size_user_set = True

    def _on_z_step_changed(self, _value):
        """Mark the Z step as user-set unless we set it ourselves (restore or
        auto-detect), so Discover's auto-detection keeps overriding a stale
        persisted value but never clobbers a deliberate manual entry."""
        if not self._setting_z_step_programmatically:
            self._z_step_user_set = True

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

        # Only consider PENDING items — an old completed ("Done") run still
        # parked in the queue must not drive the pixel size for a new run.
        first_item = next(
            (
                it
                for it in self._queue
                if it.get("tiles") and it.get("status") == "pending"
            ),
            None,
        )
        if first_item is None:
            return
        acq = Path(first_item["path"])
        try:
            suggested = suggested_pixel_size_um(acq)
        except Exception as e:
            self._logger.debug(f"pixel-size auto-fill skipped: {e}")
            return
        if not suggested:
            # Objective/pixel size could not be derived. Don't fall back
            # SILENTLY — an unreported wrong pixel size renders tiles at the
            # wrong scale (spaced-out "dice") with no clue why. Report the
            # value that WILL be used and how to fix it.
            cur = self._pixel_size_spin.value()
            mag = read_objective_magnification(acq)
            if mag:
                self._log(
                    f"⚠ Read objective {mag:.2f}× from ScopeSettings.txt but "
                    f"could not derive a pixel size; using {cur:.4f} µm/px. "
                    f"Verify the scale."
                )
            else:
                self._log(
                    f"⚠ Could not read objective magnification from "
                    f"ScopeSettings.txt for '{acq.name}' — using XY pixel size "
                    f"{cur:.4f} µm/px (default/manual, NOT scope-derived). If "
                    f"tiles render spaced apart, this is likely the cause: check "
                    f"the acquisition has a ScopeSettings.txt containing "
                    f"'Objective lens magnification'."
                )
            return
        mag = read_objective_magnification(acq)
        mag_str = f"{mag:.2f}×" if mag else "?"
        detected = round(suggested, 4)
        cur = self._pixel_size_spin.value()

        def _apply_detected():
            self._setting_pixel_programmatically = True
            self._pixel_size_spin.setValue(detected)
            self._setting_pixel_programmatically = False
            # Applying the detected value clears the manual-override flag so a
            # later Discover keeps tracking the acquisition automatically.
            self._pixel_size_user_set = False

        # Untouched, or effectively equal to the detected value (e.g. a single
        # stray wheel notch within rounding) — just sync silently.
        if not self._pixel_size_user_set or abs(cur - detected) <= max(
            0.0005, detected * 0.005
        ):
            _apply_detected()
            self._log(
                f"Auto-detected XY pixel size: {detected:.3f} µm "
                f"(objective {mag_str} from ScopeSettings.txt)"
            )
            return

        # The field was changed and now differs from the detected value. This
        # is often an accidental scroll-wheel nudge, so offer to restore the
        # detected value rather than silently keeping a possibly-wrong one —
        # while still letting a deliberate override stand (choose No).
        reply = QMessageBox.question(
            self,
            "Overwrite XY pixel size?",
            f"The XY pixel size is set to {cur:.4f} µm, but the acquisition's "
            f"objective ({mag_str}, from ScopeSettings.txt) gives "
            f"{detected:.4f} µm.\n\n"
            f"Overwrite with the detected {detected:.4f} µm?\n\n"
            f"Choose No to keep your current value — e.g. a measured "
            f"calibration you set on purpose.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            _apply_detected()
            self._log(
                f"XY pixel size overwritten with detected {detected:.3f} µm "
                f"(objective {mag_str})."
            )
        else:
            self._log(
                f"Kept manual XY pixel size {cur:.3f} µm "
                f"(detected {detected:.3f} µm from {mag_str} not applied)."
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
        self._log(f"  Channel list: {all_ch}")
        self._log(f"  Illumination side list: {all_illum}")
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

    # The Log pane is entirely user-controlled: its open/closed state is
    # persisted (QSettings "log_expanded", default closed) and restored on
    # launch. A run no longer force-opens it — auto-opening pushed the always-
    # visible Progress section off small screens and overrode the remembered
    # preference. Progress is shown by the phase pills below regardless.

    def _set_config_controls_enabled(self, enabled: bool):
        """Enable/disable every config control as a unit for run locking.

        Covers the settings grid, output dir, processing-options panel (and its
        toggle), and the background-zeroing panel so none can be changed while a
        stitch is in progress.

        Verbose log is locked too: it's a run input snapshotted at start
        (passed to the worker), so toggling it mid-run can't affect the current
        run and would only mislead. The Timestamps checkbox is deliberately
        LEFT enabled — it's read live in ``_append_log`` and is meant to be
        flipped while a run streams output.
        """
        self._config_container.setEnabled(enabled)
        self._output_dir_edit.setEnabled(enabled)
        self._proc_toggle.setEnabled(enabled)
        self._proc_widget.setEnabled(enabled)
        self._bg_zero_panel.setEnabled(enabled)
        self._verbose_log_cb.setEnabled(enabled)

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

    def _registration_binning(self) -> dict:
        """The per-axis binning dict multiview-stitcher wants, from XY and Z.

        The dict stays the one canonical form -- it is what the pipeline, the
        YAML and the CLI all carry, and what MVS is handed. XY and Z are how
        it is CHOSEN, not a second place it is stored.
        """
        xy = int(self._reg_binning_xy_combo.currentData() or 1)
        z = int(self._reg_binning_z_combo.currentData() or 1)
        return {"z": z, "y": xy, "x": xy}

    def _set_registration_binning(self, binning) -> None:
        """Drive the XY and Z combos from a per-axis dict.

        A dict whose X and Y differ cannot be shown exactly, so the larger of
        the two is selected and the mismatch logged rather than silently
        halving one axis's binning -- the run would then do less work than the
        configuration asked for, which is the direction that costs time, not
        correctness.
        """
        if not isinstance(binning, dict):
            return
        try:
            z = int(binning.get("z", 2))
            y = int(binning.get("y", 4))
            x = int(binning.get("x", y))
        except (TypeError, ValueError):
            return
        if x != y:
            logger.warning(
                f"Registration binning has x={x} and y={y}; the dialog offers "
                f"one lateral factor, so showing {max(x, y)}x. Set them "
                f"separately in the config file if they must differ."
            )
        self._set_combo_by_data(self._reg_binning_xy_combo, max(x, y))
        self._set_combo_by_data(self._reg_binning_z_combo, z)

    def _on_skip_reg_toggled(self, checked: bool):
        """Enable/disable registration controls based on skip state."""
        for _widget in (
            self._reg_binning_xy_combo,
            self._reg_binning_z_combo,
            self._reg_binning_xy_label,
            self._reg_binning_z_label,
            self._reg_binning_label,
        ):
            _widget.setEnabled(not checked)
        # Max reg. shift only applies when registration runs.
        self._max_reg_shift_spin.setEnabled(not checked)
        self._max_reg_shift_label.setEnabled(not checked)
        # Same for the axial controls. The report checkbox deliberately stays
        # live: with registration skipped it writes a file saying the tiles were
        # never registered, which is exactly the fact that went unnoticed for
        # months.
        self._max_reg_shift_z_spin.setEnabled(not checked)
        self._max_reg_shift_z_label.setEnabled(not checked)
        self._z_refine_cb.setEnabled(not checked)
        self._z_refine_range_label.setEnabled(not checked)
        self._z_refine_range_spin.setEnabled(not checked)

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
        self._destripe_dir_combo.setEnabled(checked)
        self._destripe_settings_btn.setEnabled(checked)
        self._destripe_preview_btn.setEnabled(checked)
        if not checked:
            self._destripe_fast_cb.setChecked(False)

    def _on_destripe_settings(self):
        """Open the destripe filter-tuning dialog and keep the chosen values."""
        from flamingo_stitcher.gui.destripe_settings_dialog import (
            DestripeSettingsDialog,
        )

        dlg = DestripeSettingsDialog(self._destripe_params, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._destripe_params = dlg.get_params()
            self._log(f"Destripe settings updated: {self._destripe_params}")
            self._save_destripe_preset_for_selection()

    def _on_destripe_preview(self):
        """Open the live before/after destripe preview for the chosen queue row.

        Accepting the preview adopts BOTH the tuned parameters and the stripe
        direction, so a setting that was proven on a real plane is the one the
        run uses -- otherwise the preview would just be a nice picture.
        """
        item = self._selected_queue_item("Destripe preview")
        if item is None:
            return
        acq_path = item["path"]

        from flamingo_stitcher.gui.destripe_preview_dialog import (
            DestripePreviewDialog,
        )

        # Hand over the tiles the queue already discovered. Each dialog
        # subclass discovers its own LAYOUT (_discover_tiles_for_path: the
        # subfolder-per-tile scanner here, discover_flat_tiles for the C++
        # server's flat X000_Y000 files), so letting the preview re-scan with
        # one hardcoded scanner reported "no tiles" on the Single Workflow tab.
        # Reusing the queue's result is both correct by construction and skips
        # a redundant scan of every tile.
        tiles = item.get("tiles")
        if not tiles:
            try:
                tiles = self._discover_tiles_for_path(acq_path)
            except Exception as e:  # noqa: BLE001 - preview falls back itself
                self._logger.debug(f"Preview tile discovery failed: {e!r}")
                tiles = None

        dlg = DestripePreviewDialog(
            acq_path,
            self._destripe_params,
            direction=str(self._destripe_dir_combo.currentData() or "auto"),
            tiles=tiles,
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._destripe_params = dlg.get_params()
            idx = self._destripe_dir_combo.findData(dlg.get_direction())
            if idx >= 0:
                self._destripe_dir_combo.setCurrentIndex(idx)
            self._log(
                f"Destripe settings from preview: {self._destripe_params} "
                f"(direction {dlg.get_direction()})"
            )
            self._save_destripe_preset_for_selection()

    def _selected_queue_path(self, title: str):
        """Path of the selected queue row, or the first when none is selected."""
        item = self._selected_queue_item(title)
        return None if item is None else item["path"]

    def _selected_queue_item(self, title: str):
        """The selected queue row (or the first), validated to have a path."""
        if not self._queue:
            QMessageBox.information(
                self,
                title,
                "Add an acquisition to the queue first, then select it.",
            )
            return None
        rows = sorted(set(idx.row() for idx in self._queue_table.selectedIndexes()))
        row = rows[0] if rows else 0
        if not (0 <= row < len(self._queue)):
            row = 0
        item = self._queue[row]
        if not item.get("path"):
            QMessageBox.warning(self, title, "That queue item has no path.")
            return None
        return item

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
            self._memory_label.setText(_ESTIMATES_PLACEHOLDER)
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
                f"In-memory: ~{_fmt_gb(in_mem_gb)}</span> &nbsp;|&nbsp; "
                f"<span style='color:{_colour(stream_gb)};font-weight:bold;'>"
                f"Streaming: ~{_fmt_gb(stream_gb)}</span> &nbsp;|&nbsp; "
                f"Output: ~{_fmt_gb(est['output_gb'])}"
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
            fusion_note = ""
            if est.get("fusion_gb") is not None:
                fusion_note = (
                    f"  Fusion working set: ~{_fmt_gb(est['fusion_gb'])} "
                    f"(~{est.get('views_per_block', '?')} tiles/block)\n"
                )
            if est.get("preprocess_gb") is not None:
                fusion_note += (
                    f"  Preprocess working set: ~{_fmt_gb(est['preprocess_gb'])} "
                    f"({est.get('preprocess_workers', '?')} tiles at NATIVE "
                    f"resolution)\n"
                )
            limit_note = ""
            if est.get("limited_by") == "preprocess":
                limit_note = (
                    "  Peak is pinned by PREPROCESSING, which runs at native\n"
                    "  resolution before downsampling — raising the downsample\n"
                    "  factor shrinks the output but not this floor, and both\n"
                    "  modes report the same number. To lower it: fewer\n"
                    "  preprocess workers, fewer planes per tile, or turn off\n"
                    "  destripe / flat-field / deconvolution / Z downsample.\n"
                )
            self._log(
                f"Memory estimate{worst_suffix} (system RAM: {sys_ram:.0f} GB):\n"
                f"  In-memory mode: ~{_fmt_gb(est['in_memory_gb'])} peak\n"
                f"  Streaming mode: ~{_fmt_gb(est['streaming_gb'])} peak\n"
                f"{fusion_note}"
                f"{limit_note}"
                f"  Output size:    ~{_fmt_gb(est['output_gb'])}\n"
                f"  Recommendation: {'Streaming (low memory)' if est['auto_streaming'] else 'In-memory (fast)'}"
            )
        except Exception as e:
            self._logger.debug(f"Memory estimate failed: {e}")
            self._memory_label.setText(_ESTIMATES_PLACEHOLDER)
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
        # Keep the frozen Output summary in step with the estimate (size term)
        # whenever the indicator refreshes; it also updates on dir change.
        self._update_output_info()
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

    def _add_dialog_extras(self, content_layout) -> None:
        """Hook for subclasses to inject extra controls at the top of the scroll
        area. No-op in the base dialog."""
        return

    def _apply_tile_orientation(self, config) -> None:
        """Enable PER-ACQUISITION tile-orientation resolution for the run.

        Orientation is resolved per entry by the pipeline from each
        acquisition's OWN microscope name (user preset > bundled YAML > the
        camera_x_inverted default). A choice made in the Orientation Preview is
        persisted per-microscope by "Use for stitching", so it is picked up here
        automatically for matching acquisitions — and, crucially, is NOT applied
        to datasets from other systems. (Previously a single session choice, or
        the first queued item's preset, was pushed onto the WHOLE batch, which
        silently mis-oriented unrelated acquisitions such as N7.)
        """
        config.auto_tile_orientation = True

    def _build_config(self):
        """Build a StitchingConfig from YAML defaults + current UI settings."""
        from flamingo_stitcher.pipeline import StitchingConfig

        # Start from YAML defaults (fills in all non-UI-exposed fields)
        config = StitchingConfig.with_yaml_defaults()

        # Overlay UI settings
        z_step = self._z_step_spin.value()
        config.pixel_size_um = self._pixel_size_spin.value()
        # Per-entry pixel size: unless the user deliberately set a value, each
        # queued acquisition derives its own XY pixel size from its own
        # objective at run time (so a batch mixing objectives is correct). The
        # spin value is kept only as a fallback when an objective can't be read.
        config.auto_pixel_size = not self._pixel_size_user_set
        config.z_step_um = z_step if z_step > 0 else None
        # Frame (AOI) override: None = auto-detect from file size.
        _frame = self._frame_size_combo.currentData()
        config.frame_width = _frame
        config.frame_height = _frame
        # "Separate" is a value of the illumination-fusion combo, not a real
        # fusion method: it means "don't fuse — keep each light path as its own
        # output channel". Map it to the pipeline's split_illumination flag and
        # leave illumination_fusion at its default (unused when splitting).
        _illum = self._fusion_combo.currentData()
        if _illum == "separate":
            config.split_illumination = True
        else:
            config.split_illumination = False
            config.illumination_fusion = _illum
        config.tile_overlap_fusion = self._tile_fusion_combo.currentData()
        config.output_format = self._format_combo.currentData()
        config.flat_field_correction = self._flat_field_cb.isChecked()
        config.destripe = self._destripe_cb.isChecked()
        config.destripe_fast = self._destripe_fast_cb.isChecked()
        config.destripe_direction = self._destripe_dir_combo.currentData()
        config.destripe_params = dict(self._destripe_params)
        config.downsample_xy = self._downsample_xy_combo.currentData()
        config.downsample_z = self._downsample_z_combo.currentData()
        config.deconvolution_enabled = self._deconv_cb.isChecked()
        config.content_based_fusion = self._content_fusion_cb.isChecked()
        config.skip_registration = self._skip_reg_cb.isChecked()
        config.max_registration_shift_um = float(self._max_reg_shift_spin.value())
        config.max_registration_shift_z_um = float(self._max_reg_shift_z_spin.value())
        config.registration_z_refine = self._z_refine_cb.isChecked()
        config.registration_z_refine_range_um = float(self._z_refine_range_spin.value())
        config.registration_report_enabled = self._reg_report_cb.isChecked()
        config.border_qc_enabled = self._border_qc_cb.isChecked()
        config.border_qc_mode = self._border_qc_mode_combo.currentData() or "mip"
        config.registration_binning = self._registration_binning()
        config.package_ozx = self._ozx_cb.isChecked()
        config.tiff_pyramids = self._tiff_pyramids_cb.isChecked()
        config.streaming_mode = self._streaming_combo.currentData()
        config.scratch_dir = self._scratch_dir_edit.text().strip() or None
        chunk = self._chunk_size_combo.currentData()
        if chunk:
            config.output_chunksize = dict(chunk)

        # Tile orientation is resolved PER acquisition at run time (by each
        # item's own microscope name), not one choice for the whole batch.
        self._apply_tile_orientation(config)

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

    # StitchingConfig fields that are specific to a particular acquisition
    # (physical geometry) or to this machine, so a *shared* configuration must
    # not overwrite them. Discover re-derives the file-specific ones from the
    # actual data; the output/scratch locations are environment-specific.
    _NONSHAREABLE_CONFIG_FIELDS = frozenset(
        {"pixel_size_um", "z_step_um", "frame_width", "frame_height", "scratch_dir"}
    )

    @staticmethod
    def _set_combo_by_data(combo, value) -> bool:
        """Select the combo entry whose stored data equals ``value``.

        Uses Python equality (not Qt ``findData``) so it matches ``None``,
        bools and dict-valued combo data correctly. Returns False if no
        entry matches (option unavailable in this build/format).
        """
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return True
        return False

    def _on_load_configuration(self):
        """Load processing settings from a run's stitch_metadata.json (or a
        saved configuration file) into the current tab, so a setup that worked
        can be reused / shared. File-specific and environment-specific fields
        are intentionally left alone (Discover re-detects them)."""
        start_dir = (
            self._output_dir_edit.text().strip()
            or QSettings().value(_LAST_BROWSE_KEY, "", type=str)
            or ""
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Stitching Configuration",
            start_dir,
            "Stitching metadata / config (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:
            QMessageBox.warning(
                self,
                "Couldn't read configuration",
                f"Could not read a stitching configuration from:\n{path}\n\n{e}",
            )
            return

        cfg = None
        if isinstance(data, dict):
            block = data.get("stitching_config")
            if isinstance(block, dict) and block:
                cfg = block
            elif any(
                k in data
                for k in ("illumination_fusion", "output_format", "downsample_xy")
            ):
                cfg = data  # a bare configuration file
        if not cfg:
            QMessageBox.warning(
                self,
                "Not a stitching configuration",
                "This file doesn't contain stitching settings.\n\n"
                "Choose a stitch_metadata.json from a completed run (it carries "
                "the settings used) or a saved configuration file.",
            )
            return

        applied, skipped = self._apply_stitching_config(cfg)
        self._log(f"Loaded stitching configuration from {Path(path).name}")
        self._log(f"  Applied {applied} setting(s) from the shared configuration.")
        if skipped:
            labels = {
                "pixel_size_um": "pixel size",
                "z_step_um": "Z spacing",
                "frame_width": "frame AOI",
                "frame_height": "frame AOI",
                "scratch_dir": "scratch location",
            }
            pretty = sorted({labels.get(s, s) for s in skipped})
            self._log(
                "  Left unchanged (acquisition/environment specific, re-detected "
                "by Discover): " + ", ".join(pretty) + "."
            )
        QMessageBox.information(
            self,
            "Configuration loaded",
            f"Applied {applied} processing setting(s) from:\n{Path(path).name}\n\n"
            "Pixel size, Z spacing, frame AOI and the output location were left "
            "as they are — run Discover to detect those from your acquisition.",
        )

    def _apply_stitching_config(self, cfg: dict):
        """Apply a serialized StitchingConfig dict to the current widgets.

        Returns ``(applied_count, skipped_field_names)``. Only processing
        settings are applied; :attr:`_NONSHAREABLE_CONFIG_FIELDS` are skipped
        so a shared config never clobbers this acquisition's geometry or the
        local output location. Mirrors :meth:`_build_config` in reverse.
        """
        applied = 0
        skipped = set()

        def has(name):
            return name in cfg and name not in self._NONSHAREABLE_CONFIG_FIELDS

        # Note which non-shareable fields the file carried, for the summary.
        for name in self._NONSHAREABLE_CONFIG_FIELDS:
            if name in cfg:
                skipped.add(name)

        # Output format FIRST — its handler repopulates the compression combo,
        # so compression must be applied after (matching _restore_settings).
        if has("output_format"):
            applied += self._set_combo_by_data(
                self._format_combo, cfg["output_format"]
            )
        # Compression is stored per-format on the config; pick the one that
        # matches the (now-applied) format.
        fmt = self._format_combo.currentData()
        comp = None
        if fmt in ("ome-tiff", "both") and has("tiff_compression"):
            comp = cfg["tiff_compression"]
        elif fmt in ("ome-zarr-sharded", "ome-zarr-v2", "both") and has(
            "zarr_compression"
        ):
            comp = cfg["zarr_compression"]
        if comp is not None:
            applied += self._set_combo_by_data(self._compression_combo, comp)

        combo_fields = [
            ("illumination_fusion", self._fusion_combo),
            ("tile_overlap_fusion", self._tile_fusion_combo),
            ("downsample_xy", self._downsample_xy_combo),
            ("downsample_z", self._downsample_z_combo),
            ("streaming_mode", self._streaming_combo),
            ("output_chunksize", self._chunk_size_combo),
            ("border_qc_mode", self._border_qc_mode_combo),
        ]
        for name, combo in combo_fields:
            if has(name):
                applied += self._set_combo_by_data(combo, cfg[name])

        # Not a combo any more: one per-axis dict drives two controls.
        if has("registration_binning"):
            self._set_registration_binning(cfg["registration_binning"])
            applied += 1

        check_fields = [
            ("flat_field_correction", self._flat_field_cb),
            ("destripe", self._destripe_cb),
            ("destripe_fast", self._destripe_fast_cb),
            ("deconvolution_enabled", self._deconv_cb),
            ("content_based_fusion", self._content_fusion_cb),
            ("skip_registration", self._skip_reg_cb),
            ("package_ozx", self._ozx_cb),
            ("tiff_pyramids", self._tiff_pyramids_cb),
            ("border_qc_enabled", self._border_qc_cb),
            ("registration_z_refine", self._z_refine_cb),
            ("registration_report_enabled", self._reg_report_cb),
        ]
        for name, cb in check_fields:
            if has(name):
                cb.setChecked(bool(cfg[name]))
                applied += 1

        # split_illumination is surfaced as the "separate" entry of the
        # illumination-fusion combo, not a checkbox — a loaded config that split
        # the light paths should select it (overriding the illumination_fusion
        # method applied above, which is unused/ignored when splitting).
        if has("split_illumination") and bool(cfg["split_illumination"]):
            applied += self._set_combo_by_data(self._fusion_combo, "separate")

        # Spin-box value fields.
        if has("max_registration_shift_um"):
            self._max_reg_shift_spin.setValue(float(cfg["max_registration_shift_um"]))
        if has("max_registration_shift_z_um"):
            self._max_reg_shift_z_spin.setValue(
                float(cfg["max_registration_shift_z_um"])
            )
        if has("registration_z_refine_range_um"):
            self._z_refine_range_spin.setValue(
                float(cfg["registration_z_refine_range_um"])
            )
            applied += 1

        # Background zeroing: enable state now; per-channel thresholds can only
        # be applied once channels are known, so stash them for replay after
        # Discover (the same mechanism _restore_settings uses).
        if has("background_zero_enabled"):
            self._bg_zero_panel.set_enabled_state(bool(cfg["background_zero_enabled"]))
            applied += 1
        if has("background_zero_thresholds"):
            try:
                thr = {
                    int(k): int(v)
                    for k, v in dict(cfg["background_zero_thresholds"]).items()
                }
            except (ValueError, TypeError):
                thr = {}
            self._pending_bg_zero_thresholds = thr
            # Apply immediately too, in case channels are already populated.
            self._bg_zero_panel.set_thresholds(thr)
            applied += 1

        # A shared config can re-check options whose backend is missing on this
        # box (e.g. Destripe with pystripe absent); re-gate so those clear.
        self._update_preprocessing_availability()
        try:
            self._update_memory_indicator()
        except Exception:
            pass
        return applied, skipped

    def _parse_channels(self) -> Optional[List[int]]:
        """Parse channels from the channels line edit. Returns None for 'all'."""
        text = self._channels_edit.text().strip()
        if not text:
            return None
        try:
            return [int(ch.strip()) for ch in text.split(",") if ch.strip()]
        except ValueError:
            return None

    def _confirm_orientation_known(self, pending) -> bool:
        """Force a chosen tile orientation for every pending microscope.

        Returns True to proceed. If any pending acquisition's microscope has no
        orientation preset, blocks the run and prompts — offering to open the
        Orientation Preview when the data can build it, or warning that a dataset
        with MIPs is needed when it can't.
        """
        from flamingo_stitcher.orientation import (
            has_orientation_preview_data,
            read_microscope_name,
            resolve_tile_orientation,
        )

        unknown = []  # (item, scope_name, has_data)
        for it in pending:
            p = it.get("path")
            if not p:
                continue
            try:
                if resolve_tile_orientation(p) is not None:
                    continue  # this microscope's orientation is known
            except Exception:  # noqa: BLE001 - resolve is best-effort
                pass
            try:
                scope = read_microscope_name(p)
            except Exception:  # noqa: BLE001
                scope = None
            try:
                has_data = has_orientation_preview_data(p)
            except Exception:  # noqa: BLE001
                has_data = False
            unknown.append((it, scope or "(unnamed)", has_data))

        if not unknown:
            return True

        scopes = ", ".join(sorted({u[1] for u in unknown}))
        any_data = any(u[2] for u in unknown)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Tile orientation not set")
        msg = (
            f"No tile orientation has been chosen for: {scopes}.\n\n"
            "Each microscope needs its orientation chosen once so tiles connect "
            "— stitching without it would silently mis-place tiles."
        )
        if any_data:
            msg += (
                "\n\nOpen the Orientation Preview, pick the panel where the "
                "tissue is continuous across seams, and click 'Use for "
                "stitching'. Then run again."
            )
            box.setText(msg)
            open_btn = box.addButton(
                "Open Orientation Preview…", QMessageBox.AcceptRole
            )
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.exec_()
            if box.clickedButton() is open_btn:
                target = next((u[0] for u in unknown if u[2]), None)
                if target is not None:
                    self._open_orientation_preview_for(target.get("path"))
        else:
            msg += (
                "\n\n⚠ These datasets have no MIPs to determine the orientation "
                "from. Use a dataset that includes per-tile MIPs (*_MP.tif) or "
                "readable raw stacks for this microscope to choose it."
            )
            box.setText(msg)
            box.addButton("OK", QMessageBox.RejectRole)
            box.exec_()
        return False

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

        # In auto mode each acquisition derives its own pixel size at run time,
        # so a single-value-vs-objective prompt doesn't apply — the pipeline
        # handles per-entry scaling. Just note it and proceed.
        if getattr(config, "auto_pixel_size", False):
            self._log(
                "XY pixel size: each acquisition will use its own "
                "objective-derived value (per-entry)."
            )
            return True

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

        # Pre-flight: every acquisition's microscope must have a chosen tile
        # orientation. A new microscope has none — force the selection (via the
        # preview) instead of stitching with a guessed default.
        if not self._confirm_orientation_known(pending):
            return

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
        self._reset_step_progress()
        n_pending = len(pending)
        self._log(f"Starting batch stitching: {n_pending} directories\n")

        # Store batch state
        self._batch_running = True
        self._mem_wait_popup_shown = False
        self._set_flamingos_marching(True)
        self._batch_config = config
        self._batch_channels = self._parse_channels()
        self._batch_results = []
        # Fresh existing-output decision each batch (ask again per Run).
        self._overwrite_policy = None

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
        any_in_memory = False  # any RAM-warned item running in in-memory mode
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
                if not use_streaming:
                    any_in_memory = True

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
                    parts = [f"~{_fmt_gb(est['output_gb'])} output"]
                    if spill_gb:
                        parts.append(f"~{_fmt_gb(spill_gb)} tile spill")
                    if fused_memmap_gb:
                        parts.append(f"~{_fmt_gb(fused_memmap_gb)} fused memmap")
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
            if any_in_memory:
                # In-memory mode is the biggest lever and isn't in force yet.
                lines.append(
                    "\nIf the write hits swap it will slow to a crawl or be "
                    "killed by the OS. Switch Memory mode to Streaming (keeps "
                    "full resolution) or raise the downsample factor."
                )
            else:
                # Already streaming — don't suggest it. Point at the levers
                # that reduce the streaming working set instead.
                lines.append(
                    "\nAlready in Streaming mode. To lower the peak at full "
                    "resolution: reduce preprocess/fuse workers, turn off "
                    "content-based blending / deconvolution / depth "
                    "attenuation if enabled, or switch OME-TIFF/Imaris output "
                    "to OME-Zarr. Last resort: raise the downsample factor."
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
                # Run may skip the explicit Discover step, so warn here too if
                # any tile came back with corrupt/degraded data.
                warned = [
                    t for t in tiles if getattr(t, "metadata_warning", None)
                ]
                if warned:
                    item["warnings"] = [t.metadata_warning for t in warned]
                    for t in warned:
                        self._log(f"  ⚠ {t.metadata_warning}")
                    self._warn_on_discovery_issues([item])
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

        # Existing-output guard: the store name encodes the acquisition +
        # settings, so an existing one means this exact stitch was run before.
        # Ask before overwriting (Overwrite / Rename / Cancel) instead of
        # silently replacing a prior result.
        output_dir = self._resolve_existing_output(item, output_dir)
        if output_dir is None:  # user cancelled → skip this item
            item["status"] = "cancelled"
            item["error"] = "Skipped — output already exists"
            self._update_queue_table()
            self._batch_results.append(
                (item["path"], False, "Skipped — output already exists")
            )
            self._advance_queue()
            return
        item["output_path"] = str(output_dir)

        # Update status
        item["status"] = "stitching"
        self._update_queue_table()

        # Give RAM a chance to recover before launching, so a transient dip
        # left by the previous item doesn't trip the pipeline's resource guard
        # into a false abort. gc already ran in _on_item_finished; this
        # (non-blocking) gate additionally waits out slower OS-level
        # reclamation. Instant/no-op for the first item and whenever RAM is
        # already ample.
        self._launch_item_when_memory_ready(item, output_dir)

    def _resolve_existing_output(self, item, output_dir):
        """Decide what to do if this item's output already exists.

        Returns the (possibly renamed) output dir to write to, or None to skip
        this item. Honours a "apply to all remaining" decision made earlier in
        the batch; otherwise prompts Overwrite / New folder / Cancel.
        """
        from flamingo_stitcher.pipeline import StitchingPipeline

        try:
            pipe = StitchingPipeline(self._build_config())
            expected = pipe.expected_output_path(item["path"], output_dir)
        except Exception:  # noqa: BLE001 - can't determine → old behaviour
            return output_dir
        if not expected.exists():
            return output_dir
        if self._overwrite_policy == "overwrite":
            return output_dir
        if self._overwrite_policy == "rename":
            return self._unique_output_dir(pipe, item["path"], output_dir)

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Output already exists")
        box.setText(
            f"A stitched output already exists for '{item['path'].name}' with "
            f"these settings:\n\n{expected}\n\nOverwrite it, write to a new "
            f"numbered folder (keeping the old one), or skip this item?"
        )
        overwrite_btn = box.addButton("Overwrite", QMessageBox.DestructiveRole)
        rename_btn = box.addButton("New folder", QMessageBox.AcceptRole)
        box.addButton("Skip", QMessageBox.RejectRole)
        apply_all = QCheckBox("Apply to all remaining")
        box.setCheckBox(apply_all)
        box.exec_()
        clicked = box.clickedButton()

        if clicked is rename_btn:
            if apply_all.isChecked():
                self._overwrite_policy = "rename"
            return self._unique_output_dir(pipe, item["path"], output_dir)
        if clicked is overwrite_btn:
            if apply_all.isChecked():
                self._overwrite_policy = "overwrite"
            return output_dir
        return None  # Skip

    def _unique_output_dir(self, pipe, acq_path, output_dir):
        """A sibling output dir whose store doesn't collide (…_stitched_2, _3)."""
        base = Path(output_dir)
        candidate = base
        n = 2
        while pipe.expected_output_path(acq_path, candidate).exists():
            candidate = Path(str(base) + f"_{n}")
            n += 1
        return candidate

    # ---- Between-item memory-recovery gate -------------------------------
    _MEM_WAIT_TIMEOUT_S = 120.0  # stop waiting and proceed after this long
    _MEM_WAIT_POLL_S = 5.0  # re-check cadence while waiting
    _MEM_WAIT_LOG_S = 30.0  # re-log cadence (poll is noisier than useful)

    def _memory_gate_gb(self, item):
        """(need_gb, avail_gb) for *item*, or (None, None) if unknowable.

        ``need`` is the projected peak of the mode the pipeline would auto-pick
        (streaming vs in-memory), matching how the resource guard decides.
        """
        try:
            import psutil

            from flamingo_stitcher.pipeline import estimate_memory_usage

            all_ch = sorted({ch for t in item["tiles"] for ch in t.channels})
            process_ch = self._batch_channels or all_ch
            est = estimate_memory_usage(item["tiles"], process_ch, self._batch_config)
            need = (
                est["streaming_gb"]
                if est.get("auto_streaming")
                else est["in_memory_gb"]
            )
            avail = psutil.virtual_memory().available / (1024**3)
            return float(need), float(avail)
        except Exception:
            return None, None

    def _memory_hog_hint(self) -> str:
        """"  Largest memory users: …" block, or "" when nothing notable."""
        try:
            from flamingo_stitcher.memory_monitor import (
                format_memory_consumers,
                top_memory_consumers,
            )

            rows = top_memory_consumers(limit=5, min_gb=1.0)
        except Exception:
            return ""
        if not rows:
            return ""
        return (
            "\n  Largest memory users right now (closing one frees RAM):\n"
            + format_memory_consumers(rows)
        )

    def _show_memory_wait_popup(self, need_gb, avail_gb) -> None:
        """Non-blocking heads-up naming what is holding the RAM.

        Warn-only and never modal: the run continues either way (the gate times
        out and the resource guard decides), so this must not block the batch.
        """
        try:
            hint = self._memory_hog_hint()
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle("Waiting for memory")
            box.setText(
                f"Stitching is waiting for RAM to free up.\n\n"
                f"Free: ~{avail_gb:.0f} GB    Needed: ~{need_gb:.0f} GB\n\n"
                + (
                    hint.strip()
                    if hint
                    else "No single large memory user stands out."
                )
                + f"\n\nClosing one of these now would let the run start sooner. "
                f"It will proceed anyway after "
                f"{int(self._MEM_WAIT_TIMEOUT_S)}s regardless."
            )
            box.setStandardButtons(QMessageBox.Ok)
            box.setModal(False)  # never block the batch
            box.show()
        except Exception as exc:  # a UI hiccup must never break the run
            self._logger.debug(f"memory-wait popup failed: {exc}")

    def _launch_item_when_memory_ready(self, item, output_dir, elapsed=0.0):
        # The batch may have been cancelled while this timer was pending.
        if not self._batch_running or item.get("status") != "stitching":
            return
        need_gb, avail_gb = self._memory_gate_gb(item)
        ready = (
            need_gb is None
            or avail_gb is None
            or avail_gb >= need_gb
            or elapsed >= self._MEM_WAIT_TIMEOUT_S
        )
        if ready:
            if need_gb is not None and avail_gb is not None and avail_gb < need_gb:
                self._log(
                    f"  Proceeding after {int(elapsed)}s — RAM still low "
                    f"({avail_gb:.0f} GB free vs ~{need_gb:.0f} GB projected); "
                    f"the resource guard will make the final call."
                    + self._memory_hog_hint()
                )
            self._start_item_worker(item, output_dir)
            return

        # Log on the FIRST trip, then only every _MEM_WAIT_LOG_S — the 5 s poll
        # produced ~24 identical lines per item, which buried the actionable part.
        first = elapsed <= 0
        if first or (int(elapsed) % int(self._MEM_WAIT_LOG_S) == 0):
            self._log(
                f"  Waiting for memory to recover before this item: "
                f"{avail_gb:.0f} GB free, ~{need_gb:.0f} GB projected "
                f"(re-checking every {int(self._MEM_WAIT_POLL_S)}s, up to "
                f"{int(self._MEM_WAIT_TIMEOUT_S)}s)…"
                + (self._memory_hog_hint() if first else "")
            )
        # Surface it once per batch: the log is easy to miss when a run is
        # kicked off and left alone, and closing a memory hog is only useful
        # while we're still waiting.
        if first and not getattr(self, "_mem_wait_popup_shown", False):
            self._mem_wait_popup_shown = True
            self._show_memory_wait_popup(need_gb, avail_gb)
        import gc

        gc.collect()
        from PyQt5.QtCore import QTimer

        QTimer.singleShot(
            int(self._MEM_WAIT_POLL_S * 1000),
            lambda: self._launch_item_when_memory_ready(
                item, output_dir, elapsed + self._MEM_WAIT_POLL_S
            ),
        )

    # ---- "Working" flamingo animation (cosmetic) -------------------------
    def _build_flamingo_indicator(self):
        """A small looping 'marching flamingos' QMovie for the log's right
        edge, shown only while a run is active. Returns the QLabel, or None if
        the movie can't be created (missing file / no gif image plugin) — this
        is optional chrome and must never block a stitch.

        Re-enabled once the run-start crash was traced to a missing
        OVERALL_ETA_SEP import in _colorize_status (not this movie); v0.5.3
        had disabled it while the flamingo was still on the suspect list.
        """
        try:
            from PyQt5.QtCore import QSize
            from PyQt5.QtGui import QMovie

            gif = Path(__file__).parent / "working_flamingos.gif"
            if not gif.exists():
                return None
            movie = QMovie(str(gif))
            if not movie.isValid():
                return None
            movie.setCacheMode(QMovie.CacheAll)
            movie.setScaledSize(QSize(120, 120))
            label = QLabel()
            label.setMovie(movie)
            label.setFixedSize(120, 120)
            label.setToolTip("Stitching in progress…")
            label.setVisible(False)  # revealed only while processing
            self._flamingo_movie = movie
            return label
        except Exception:
            return None

    def _set_flamingos_marching(self, marching: bool) -> None:
        """Start/stop + show/hide the 'working' flamingo animation."""
        label = getattr(self, "_flamingo_label", None)
        movie = getattr(self, "_flamingo_movie", None)
        if label is None or movie is None:
            return
        if marching:
            label.setVisible(True)
            movie.start()
        else:
            movie.stop()
            label.setVisible(False)

    def _apply_destripe_preset(self, item_config, acq_path):
        """Swap in this acquisition's microscope-specific destripe settings.

        An unrecognised scope is not blocked — the run continues with whatever
        was last used — but it says so loudly, because the failure mode is
        invisible: threshold and crossover are absolute intensities, so another
        scope's values can quietly treat the whole image as background (or all
        foreground) instead of merely being mistuned.
        """
        from dataclasses import replace

        from flamingo_stitcher.destripe_presets import (
            describe_resolution,
            resolve_for_acquisition,
        )

        try:
            scope, preset = resolve_for_acquisition(acq_path)
        except Exception as e:  # noqa: BLE001 - never block a run on this
            self._log(f"⚠ Could not resolve destripe settings by microscope: {e}")
            return item_config

        applied, message = describe_resolution(scope, preset)
        if applied:
            item_config = replace(
                item_config,
                destripe_params=dict(preset["params"]),
                destripe_direction=preset["direction"],
            )
            self._log(f"{message} ({preset['params']})")
        else:
            self._log(f"⚠ {message}")
        return item_config

    def _save_destripe_preset_for_selection(self) -> None:
        """Persist the current destripe settings under the selected scope."""
        if not self._queue:
            return
        rows = sorted(set(idx.row() for idx in self._queue_table.selectedIndexes()))
        row = rows[0] if rows else 0
        if not (0 <= row < len(self._queue)):
            return
        acq_path = self._queue[row].get("path")
        if not acq_path:
            return

        from flamingo_stitcher.destripe_presets import (
            microscope_for_acquisition,
            save_destripe_preset,
        )

        scope = microscope_for_acquisition(acq_path)
        direction = str(self._destripe_dir_combo.currentData() or "auto")
        if save_destripe_preset(scope, self._destripe_params, direction):
            self._log(f"Destripe settings saved for microscope '{scope}'.")
        elif not scope:
            self._log(
                "⚠ Destripe settings not saved per microscope — this "
                "acquisition has no readable microscope name."
            )

    def _start_item_worker(self, item, output_dir):
        """Launch the stitching worker for *item* (memory gate already passed)."""
        from dataclasses import replace

        from flamingo_stitcher.worker import StitchingWorker

        # Give each item its OWN config copy: in auto mode the pipeline resolves
        # this acquisition's pixel size into the config, and a fresh copy keeps
        # one item's derived value from leaking into the next as a fallback.
        item_config = replace(self._batch_config)

        # Destripe settings are per-MICROSCOPE, resolved here (per item) rather
        # than once for the batch, so a queue mixing instruments gets each one's
        # own tuning instead of whichever was last touched in the GUI.
        if item_config.destripe:
            item_config = self._apply_destripe_preset(item_config, item["path"])

        self._worker = StitchingWorker(
            config=item_config,
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
        self._worker.memory_warning.connect(self._on_memory_warning)
        self._worker.finished.connect(self._on_item_finished)
        self._worker.start()

    def _on_cancel(self):
        """Cancel the running pipeline and stop the batch."""
        if self._worker:
            self._worker.cancel()
            self._status_label.setText("Cancelling...")
            self._log(
                "Cancellation requested — aborting the current compute "
                "(stops within about one chunk)..."
            )
            # Immediate feedback + prevent repeat clicks; re-enabled on next run.
            self._cancel_btn.setEnabled(False)
        # Mark remaining pending items as cancelled
        for item in self._queue:
            if item["status"] == "pending":
                item["status"] = "cancelled"
        self._update_queue_table()

        # Cancel arriving *between* items — i.e. during the memory-recovery
        # wait, when no worker is live: no worker-finished callback will come
        # to advance/complete the queue, so finalise the batch here. (When a
        # worker IS running the normal cancel→finish→advance path handles it.)
        if self._worker is None and getattr(self, "_batch_running", False):
            if 0 <= self._queue_index < len(self._queue):
                cur = self._queue[self._queue_index]
                if cur["status"] in ("stitching", "discovering"):
                    cur["status"] = "cancelled"
                    self._batch_results.append((cur["path"], False, "Cancelled"))
            self._update_queue_table()
            self._on_batch_complete()
            return

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
        self._current_step_key = None

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
            status = f"[{n_done + 1}/{n_total}] {status}"
        self._status_label.setText(self._colorize_status(status))

        key = self._classify_step(status)
        if key is None:
            return

        # Remember the last classified step so a failure handler can give
        # step-aware OOM advice (see _on_item_error).
        self._current_step_key = key

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

    def _colorize_status(self, status: str) -> str:
        """Render the status line as rich text so the per-step ETA and the
        whole-run ETA are told apart at a glance.

        Both segments read ``"... remaining (Done at ~...)"``, so with no cue
        the two "Done at" times are easy to confuse. The pipeline now labels
        them ``"this step:"`` and ``"overall:"`` in plain text; here we also
        colour them — blue for the current step (matches the in-progress pill),
        orange for the whole run. Non-ETA messages are returned unchanged.
        """
        # Never let status-line colouring crash the progress slot. This runs
        # inside a queued Qt slot (_on_progress), and an unhandled exception
        # there makes PyQt abort the process (qFatal → 0xc0000409) — which is
        # exactly how a missing OVERALL_ETA_SEP import took down early builds.
        # Any failure now just falls back to the plain, uncoloured status.
        try:
            from html import escape

            from flamingo_stitcher.pipeline import OVERALL_ETA_SEP
        except Exception:
            return status

        STEP_COLOR = "#1976D2"  # blue — matches the in-progress step pill
        OVERALL_COLOR = "#E65100"  # orange — distinct from the step colour

        head, sep, overall = status.partition(OVERALL_ETA_SEP)
        marker = "this step:"
        i = head.find(marker)
        if i == -1 and not sep:
            return status  # plain message (no ETA tails) — leave as-is

        if i != -1:
            head_html = (
                escape(head[:i])
                + f'<span style="color:{STEP_COLOR};">{escape(head[i:])}</span>'
            )
        else:
            head_html = escape(head)

        if sep:
            return (
                head_html
                + escape(sep)
                + f'<span style="color:{OVERALL_COLOR}; font-weight:600;">'
                + escape(overall)
                + "</span>"
            )
        return head_html

    def _append_log(self, message: str):
        """Append one line to the log, optionally timestamped, and autoscroll.

        Single choke point for both worker log lines and the dialog's own
        messages so the "Timestamps" toggle applies uniformly. The stamp is
        compact (``MM-DD HH:MM``, to the minute) and prepended per line so a
        multi-line message stays aligned.
        """
        if getattr(self, "_timestamp_log_cb", None) is not None and (
            self._timestamp_log_cb.isChecked()
        ):
            from datetime import datetime

            stamp = datetime.now().strftime("%m-%d %H:%M")
            message = "\n".join(f"[{stamp}] {ln}" for ln in message.split("\n"))
        self._log_text.append(message)
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_log_message(self, message: str):
        """Handle log messages from worker."""
        self._append_log(message)

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
        self._maybe_log_oom_advice(error_msg)

    def _maybe_log_oom_advice(self, error_msg: str):
        """If the failure was an out-of-memory error, log step-aware advice.

        Most software's only answer to an OOM is "use less data". The pipeline
        has several levers that keep the chosen resolution \u2014 which one to reach
        for depends on the step that failed. We surface those, most-effective
        first, so the user can retry at full resolution instead of downsampling.
        """
        try:
            from flamingo_stitcher.oom_advice import (
                format_oom_advice,
                is_memory_error,
            )

            if not is_memory_error(error_msg):
                return
            self._log("\n" + format_oom_advice(
                self._current_step_key,
                self._oom_settings_snapshot(),
                use_streaming=self._resolved_streaming_mode(),
            ))
        except Exception as e:  # advice must never mask the real error
            self._logger.debug(f"OOM advice unavailable: {e}")

    def _oom_settings_snapshot(self) -> dict:
        """Collect the memory-relevant config flags for OOM advice."""
        cfg = self._batch_config
        if cfg is None:
            return {}
        keys = (
            "output_format",
            "content_based_fusion",
            "deconvolution_enabled",
            "depth_attenuation",
            "destripe",
            "destripe_fast",
            "flat_field_correction",
            "illumination_fusion",
            "preprocess_workers",
            "fuse_workers",
            "skip_registration",
        )
        return {k: getattr(cfg, k, None) for k in keys}

    def _resolved_streaming_mode(self):
        """The memory mode actually in force (True=streaming, False=in-memory,
        None=unknown). ``streaming_mode`` is a tri-state on the config
        (None=auto); we can only report a definite in-memory when it was
        explicitly forced off."""
        cfg = self._batch_config
        if cfg is None:
            return None
        return getattr(cfg, "streaming_mode", None)

    def _on_memory_warning(self, info: dict):
        """Show a non-blocking popup when the memory watchdog trips.

        Warn-only: the run keeps going. We surface it visually (not just in the
        log) so a long unattended run that's climbing toward an OOM gets noticed,
        while never interrupting it. Shown once per run (the watchdog fires its
        callback only on the first crossing).
        """
        used = info.get("used_gb", "?")
        proj = info.get("projected_gb", "?")
        phase = info.get("phase", "?")
        mode = info.get("mode", "?")
        line = (
            f"⚠ Memory watchdog: using ~{used} GB (projected ~{proj} GB) "
            f"during '{phase}' [{mode}]."
        )
        self._log(line)

        # Build step/mode-aware advice so we never suggest something already
        # in force (e.g. "switch to Streaming" while the run IS streaming).
        # ``phase`` uses the same step keys as oom_advice; ``mode`` is the
        # authoritative resolved memory mode.
        tips = []
        try:
            from flamingo_stitcher.oom_advice import oom_advice

            step_key = phase if phase in (
                "discover", "register", "preprocess", "fuse", "write", "metadata"
            ) else None
            tips = oom_advice(
                step_key,
                self._oom_settings_snapshot(),
                use_streaming=(mode == "streaming"),
            )
        except Exception as e:  # advice must never break the warning popup
            self._logger.debug(f"watchdog advice unavailable: {e}")

        if tips:
            self._log(
                "If it runs out of memory, these keep the chosen resolution "
                "(most effective first):"
            )
            for i, t in enumerate(tips, 1):
                self._log(f"  {i}. {t}")
            # Popup stays terse: the top two levers; the log has the full list.
            advice_block = "\n".join(f"• {t}" for t in tips[:2])
        else:
            advice_block = (
                "The run is continuing. If it runs out of memory, cancel and "
                "reduce the per-tile memory (fewer workers / lighter processing) "
                "or raise the downsample factor."
            )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("High memory use")
        box.setText(
            f"Stitching is using more memory than projected.\n\n"
            f"Live: ~{used} GB   Projected: ~{proj} GB\n"
            f"Phase: {phase}    Mode: {mode}\n\n"
            f"The run is continuing. If it runs out of memory, try (full list "
            f"in the log):\n\n{advice_block}"
        )
        box.setStandardButtons(QMessageBox.Ok)
        box.setModal(False)  # non-blocking — don't freeze the run/UI
        box.show()

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
        self._set_flamingos_marching(False)
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

                # Open the acquisition's OWN output folder, not the parent
                # directory that holds every stitch ever written here. The
                # per-run path is already resolved above for the Sample View
                # button; opening the parent left the user hunting for the one
                # folder they just made. Fall back to the parent only if that
                # folder somehow isn't there.
                folder = last_success_path
                if not Path(folder).is_dir():
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

    def _set_btn_discover(self, btn, highlight=False):
        """Accent style for the Discover Tiles button.

        A calm blue accent (steady) so it stands out from the neutral
        Add/Remove buttons without competing with the green Run call-to-action;
        an amber highlight is the bright frame used while flashing.
        """
        if highlight:
            bg, hover = "#FFB300", "#FFA000"  # amber flash frame
        else:
            bg, hover = "#1E88E5", "#1976D2"  # steady blue accent
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {bg}; color: white; "
            f"font-weight: bold; padding: 6px 14px; }}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            f"QPushButton:disabled {{ background-color: #888; color: #ccc; }}"
        )

    def _on_cta_flash_tick(self):
        """Toggle the current call-to-action between accent and bright highlight."""
        if self._cta_button is None:
            return
        self._cta_flash_on = not self._cta_flash_on
        self._set_btn_discover(self._cta_button, highlight=self._cta_flash_on)

    def _stop_cta_flash(self):
        """Stop flashing and return the flashing button to its plain state."""
        self._cta_flash_timer.stop()
        self._cta_flash_on = False
        if self._cta_button is not None:
            self._set_btn_default(self._cta_button)
            self._cta_button = None

    def _flash_cta(self, button):
        """Flash exactly this button, stopping any other that was flashing."""
        if self._cta_button is not button:
            self._stop_cta_flash()
            self._cta_button = button
        if not self._cta_flash_timer.isActive():
            self._cta_flash_on = True
            self._set_btn_discover(button, highlight=True)
            self._cta_flash_timer.start()

    def _next_step_button(self):
        """The button the user has to press next, or None if nothing is owed.

        Empty queue -> Add..., because nothing else can be done until something
        is queued; a flashing Discover Tiles over an empty queue pointed at a
        button that was disabled anyway. Otherwise Discover Tiles while one is
        due. Never both: two things flashing is noise, not guidance.
        """
        if self._batch_running:
            return None
        if not self._queue:
            return self._add_btn
        if self._discover_needed and self._discover_btn.isEnabled():
            return self._discover_btn
        return None

    def _refresh_discover_style(self):
        """Point the flashing call-to-action at the next step, and style Discover.

        Discover keeps its steady blue accent whenever it is enabled and not
        itself the flashing one, so it still reads as the notable action in the
        row rather than dropping to a plain button.
        """
        target = self._next_step_button()
        if target is None:
            self._stop_cta_flash()
        else:
            self._flash_cta(target)

        if self._cta_button is not self._discover_btn:
            if self._discover_btn.isEnabled():
                self._set_btn_discover(self._discover_btn, highlight=False)
            else:
                self._set_btn_default(self._discover_btn)

    # --- Logging helper ---

    def _log(self, message: str):
        """Append a message to the log area."""
        self._append_log(message)

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
        # Capture WHY the import fails (previously swallowed silently), so a
        # frozen build that can't destripe is self-diagnosing: the exact missing
        # module lands in the per-launch log AND the checkbox tooltip. pystripe.core
        # imports dcimg/tqdm/pywt/imageio at module top, so a single missing one
        # here disables destriping with no other clue.
        pystripe_err = None
        try:
            # Vendored stripe filter — needs only pywt/scipy/scikit-image, which
            # the app already bundles (unlike the full pystripe package, whose
            # dcimg/imageio/tqdm imports kept failing to bundle → checkbox greyed).
            from flamingo_stitcher import _pystripe_core  # noqa: F401

            pystripe_ok = True
        except Exception as exc:
            pystripe_ok = False
            pystripe_err = exc
            logger.warning(
                "Destriping unavailable — destripe backend import failed: %r", exc
            )

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
            self._destripe_dir_combo.setEnabled(False)
            self._destripe_settings_btn.setEnabled(False)
            self._destripe_preview_btn.setEnabled(False)
            self._destripe_cb.setToolTip(
                "Destriping is unavailable — the destripe backend failed to load:\n"
                f"    {type(pystripe_err).__name__}: {pystripe_err}\n"
                "(This build should bundle it; the error above names the missing "
                "piece — likely pywt / scipy / scikit-image.)"
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
                        "Leonardo FUSE requires leonardo-toolset (GPU/CUDA) in\n"
                        "the isolated preprocessing environment. The flat-field\n"
                        "setup does NOT install it — Leonardo needs a separate\n"
                        "GPU env (see the preprocessing-env setup scripts)."
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
            here.parent.parent
            / "docs"
            / "stitching_hardware_troubleshooting.md",
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
        self._refresh_discover_style()  # stop flashing while disabled
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
        s.beginGroup(self._settings_group)
        # Save queue paths (only pending/done items, not transient states)
        paths = [str(item["path"]) for item in self._queue]
        s.setValue("queue_paths", paths)
        s.setValue("output_dir", self._output_dir_edit.text())
        s.setValue("scratch_dir", self._scratch_dir_edit.text())
        s.setValue("pixel_size", self._pixel_size_spin.value())
        s.setValue("z_step", self._z_step_spin.value())
        s.setValue("downsample_xy", self._downsample_xy_combo.currentData())
        s.setValue("downsample_z", self._downsample_z_combo.currentData())
        s.setValue("fusion", self._fusion_combo.currentData())
        s.setValue("tile_overlap_fusion", self._tile_fusion_combo.currentData())
        s.setValue("frame_size_idx", self._frame_size_combo.currentIndex())
        s.setValue("verbose_log", self._verbose_log_cb.isChecked())
        s.setValue("timestamp_log", self._timestamp_log_cb.isChecked())
        s.setValue("flat_field", self._flat_field_cb.isChecked())
        s.setValue("destripe", self._destripe_cb.isChecked())
        s.setValue("destripe_fast", self._destripe_fast_cb.isChecked())
        s.setValue("destripe_direction", self._destripe_dir_combo.currentData())
        s.setValue("destripe_params_json", json.dumps(self._destripe_params))
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
        s.setValue("border_qc", self._border_qc_cb.isChecked())
        s.setValue("border_qc_mode", self._border_qc_mode_combo.currentData())
        # New keys, and deliberately the FACTORS rather than a combo index.
        # The old "reg_binning" key held an index into a three-way preset, so
        # reading it now would turn a stale 0/1/2 into 1x/2x/4x -- a different
        # binning than was ever chosen, applied silently. Leaving that key
        # unread lets it age out instead.
        s.setValue("reg_binning_xy", self._reg_binning_xy_combo.currentData())
        s.setValue("reg_binning_z", self._reg_binning_z_combo.currentData())
        s.setValue("max_reg_shift", self._max_reg_shift_spin.value())
        s.setValue("max_reg_shift_z", self._max_reg_shift_z_spin.value())
        s.setValue("z_refine", self._z_refine_cb.isChecked())
        s.setValue("z_refine_range_um", self._z_refine_range_spin.value())
        s.setValue("registration_report", self._reg_report_cb.isChecked())
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
        s.beginGroup(self._settings_group)

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
        # Only restore if the location is still reachable. A persisted path on a
        # disconnected drive would otherwise stick (and block the auto-set).
        if output_dir and Path(output_dir).parent.exists():
            self._output_dir_edit.setText(output_dir)

        # Scratch dir: only restore if still reachable; a stale path on a
        # disconnected drive would otherwise stick. Blank => default (alongside
        # output).
        scratch_dir = s.value("scratch_dir", "", type=str)
        if scratch_dir and Path(scratch_dir).exists():
            self._scratch_dir_edit.setText(scratch_dir)

        pixel_size = s.value("pixel_size", self._default_pixel_um, type=float)
        # Restoring a persisted value must not count as a manual override —
        # auto-fill from the acquisition's objective should still win on discover.
        self._setting_pixel_programmatically = True
        self._pixel_size_spin.setValue(pixel_size)
        self._setting_pixel_programmatically = False
        self._pixel_size_user_set = False

        z_step = s.value("z_step", 0.0, type=float)
        # Restoring a persisted value must not count as a manual override —
        # Discover's auto-detection from the data should still win.
        self._setting_z_step_programmatically = True
        self._z_step_spin.setValue(z_step)
        self._setting_z_step_programmatically = False
        self._z_step_user_set = False

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
        self._timestamp_log_cb.setChecked(s.value("timestamp_log", False, type=bool))

        flat_field = s.value("flat_field", False, type=bool)
        self._flat_field_cb.setChecked(flat_field)

        destripe = s.value("destripe", False, type=bool)
        self._destripe_cb.setChecked(destripe)

        destripe_fast = s.value("destripe_fast", False, type=bool)
        self._destripe_fast_cb.setChecked(destripe_fast)
        _dir = s.value("destripe_direction", "auto", type=str)
        _di = self._destripe_dir_combo.findData(_dir)
        if _di >= 0:
            self._destripe_dir_combo.setCurrentIndex(_di)
        _dp = s.value("destripe_params_json", "", type=str)
        if _dp:
            try:
                self._destripe_params = json.loads(_dp)
            except (ValueError, TypeError):
                self._destripe_params = {}

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

        self._border_qc_cb.setChecked(s.value("border_qc", False, type=bool))
        qc_mode = s.value("border_qc_mode", "mip", type=str)
        _qi = self._border_qc_mode_combo.findData(qc_mode)
        if _qi >= 0:
            self._border_qc_mode_combo.setCurrentIndex(_qi)

        _rb_xy = s.value("reg_binning_xy", 4, type=int)
        self._set_registration_binning(
            {
                "z": s.value("reg_binning_z", 2, type=int),
                "y": _rb_xy,
                "x": _rb_xy,
            }
        )
        self._max_reg_shift_spin.setValue(s.value("max_reg_shift", 0.0, type=float))
        self._max_reg_shift_z_spin.setValue(
            s.value("max_reg_shift_z", 0.0, type=float)
        )
        self._z_refine_cb.setChecked(s.value("z_refine", False, type=bool))
        self._z_refine_range_spin.setValue(
            s.value("z_refine_range_um", 40.0, type=float)
        )
        self._reg_report_cb.setChecked(
            s.value("registration_report", True, type=bool)
        )

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
        """Discover tiles using flat-layout scanner.

        Z-step auto-detection is handled centrally in `_on_discover` (which
        prefers the Workflow.txt plane spacing and respects the user-set
        guard), so both dialogs share one path and can't disagree.
        """
        from flamingo_stitcher.pipeline import discover_flat_tiles

        return discover_flat_tiles(acq_path)

    def _looks_like_acquisition(self, path: Path) -> bool:
        """Check if a directory looks like a flat-layout acquisition."""
        if (path / "Workflow.txt").exists():
            return True
        from flamingo_stitcher.pipeline import _glob_tile_files

        return bool(_glob_tile_files(path))

    def _save_settings(self):
        """Save dialog settings to QSettings (independent group)."""
        s = QSettings()
        s.beginGroup(_NATIVE_SETTINGS_GROUP)
        paths = [str(item["path"]) for item in self._queue]
        s.setValue("queue_paths", paths)
        s.setValue("output_dir", self._output_dir_edit.text())
        s.setValue("scratch_dir", self._scratch_dir_edit.text())
        s.setValue("pixel_size", self._pixel_size_spin.value())
        s.setValue("z_step", self._z_step_spin.value())
        s.setValue("downsample_xy", self._downsample_xy_combo.currentData())
        s.setValue("downsample_z", self._downsample_z_combo.currentData())
        s.setValue("fusion", self._fusion_combo.currentData())
        s.setValue("tile_overlap_fusion", self._tile_fusion_combo.currentData())
        s.setValue("frame_size_idx", self._frame_size_combo.currentIndex())
        s.setValue("verbose_log", self._verbose_log_cb.isChecked())
        s.setValue("timestamp_log", self._timestamp_log_cb.isChecked())
        s.setValue("flat_field", self._flat_field_cb.isChecked())
        s.setValue("destripe", self._destripe_cb.isChecked())
        s.setValue("destripe_fast", self._destripe_fast_cb.isChecked())
        s.setValue("destripe_direction", self._destripe_dir_combo.currentData())
        s.setValue("destripe_params_json", json.dumps(self._destripe_params))
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
        s.setValue("border_qc", self._border_qc_cb.isChecked())
        s.setValue("border_qc_mode", self._border_qc_mode_combo.currentData())
        # New keys, and deliberately the FACTORS rather than a combo index.
        # The old "reg_binning" key held an index into a three-way preset, so
        # reading it now would turn a stale 0/1/2 into 1x/2x/4x -- a different
        # binning than was ever chosen, applied silently. Leaving that key
        # unread lets it age out instead.
        s.setValue("reg_binning_xy", self._reg_binning_xy_combo.currentData())
        s.setValue("reg_binning_z", self._reg_binning_z_combo.currentData())
        s.setValue("max_reg_shift", self._max_reg_shift_spin.value())
        s.setValue("max_reg_shift_z", self._max_reg_shift_z_spin.value())
        s.setValue("z_refine", self._z_refine_cb.isChecked())
        s.setValue("z_refine_range_um", self._z_refine_range_spin.value())
        s.setValue("registration_report", self._reg_report_cb.isChecked())
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
        # Only restore if the location is still reachable. A persisted path on a
        # disconnected drive would otherwise stick (and block the auto-set).
        if output_dir and Path(output_dir).parent.exists():
            self._output_dir_edit.setText(output_dir)

        # Scratch dir: only restore if still reachable; a stale path on a
        # disconnected drive would otherwise stick. Blank => default (alongside
        # output).
        scratch_dir = s.value("scratch_dir", "", type=str)
        if scratch_dir and Path(scratch_dir).exists():
            self._scratch_dir_edit.setText(scratch_dir)

        pixel_size = s.value("pixel_size", self._default_pixel_um, type=float)
        # Restoring a persisted value must not count as a manual override —
        # auto-fill from the acquisition's objective should still win on discover.
        self._setting_pixel_programmatically = True
        self._pixel_size_spin.setValue(pixel_size)
        self._setting_pixel_programmatically = False
        self._pixel_size_user_set = False

        z_step = s.value("z_step", 0.0, type=float)
        # Restoring a persisted value must not count as a manual override —
        # Discover's auto-detection from the data should still win.
        self._setting_z_step_programmatically = True
        self._z_step_spin.setValue(z_step)
        self._setting_z_step_programmatically = False
        self._z_step_user_set = False

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
        self._timestamp_log_cb.setChecked(s.value("timestamp_log", False, type=bool))

        flat_field = s.value("flat_field", False, type=bool)
        self._flat_field_cb.setChecked(flat_field)

        destripe = s.value("destripe", False, type=bool)
        self._destripe_cb.setChecked(destripe)

        destripe_fast = s.value("destripe_fast", False, type=bool)
        self._destripe_fast_cb.setChecked(destripe_fast)
        _dir = s.value("destripe_direction", "auto", type=str)
        _di = self._destripe_dir_combo.findData(_dir)
        if _di >= 0:
            self._destripe_dir_combo.setCurrentIndex(_di)
        _dp = s.value("destripe_params_json", "", type=str)
        if _dp:
            try:
                self._destripe_params = json.loads(_dp)
            except (ValueError, TypeError):
                self._destripe_params = {}

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

        self._border_qc_cb.setChecked(s.value("border_qc", False, type=bool))
        qc_mode = s.value("border_qc_mode", "mip", type=str)
        _qi = self._border_qc_mode_combo.findData(qc_mode)
        if _qi >= 0:
            self._border_qc_mode_combo.setCurrentIndex(_qi)

        _rb_xy = s.value("reg_binning_xy", 4, type=int)
        self._set_registration_binning(
            {
                "z": s.value("reg_binning_z", 2, type=int),
                "y": _rb_xy,
                "x": _rb_xy,
            }
        )
        self._max_reg_shift_spin.setValue(s.value("max_reg_shift", 0.0, type=float))
        self._max_reg_shift_z_spin.setValue(
            s.value("max_reg_shift_z", 0.0, type=float)
        )
        self._z_refine_cb.setChecked(s.value("z_refine", False, type=bool))
        self._z_refine_range_spin.setValue(
            s.value("z_refine_range_um", 40.0, type=float)
        )
        self._reg_report_cb.setChecked(
            s.value("registration_report", True, type=bool)
        )

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


_MULTIVIEW_SETTINGS_GROUP = "MultiViewStitchingDialog"


class MultiViewStitchingDialog(StitchingDialog):
    """Stitching for multi-angle (rotation) acquisitions.

    A separate tab so multi-view fusion stays out of the normal single-angle
    flow. Same folder-per-tile discovery and pipeline as the base dialog, but
    multi-view fusion is forced on and the rotation controls are exposed. The
    rotation sign / center conventions are validated on synthetic data but NOT
    yet confirmed on the instrument — verify with a two-angle test acquisition
    (flip the sign if the views come out mirrored).
    """

    # Distinct QSettings group so this tab's state doesn't clobber the base
    # tab's. MUST stay a STRING — the base uses ``s.beginGroup(self._settings_group)``.
    # (A past crash came from shadowing this name with a QWidget; the widget
    # container is ``self._config_container``, kept separate.)
    _settings_group = _MULTIVIEW_SETTINGS_GROUP

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.setWindowTitle("Tile Stitching (Multi-View)")

    def _add_dialog_extras(self, content_layout) -> None:
        """Inject the rotation banner + controls at the top of the scroll area.

        Called from the base ``_setup_ui`` hook, so the widgets exist before
        ``_restore_settings`` runs. No QMovie / animation here — that was a
        suspect in the historical run-start crash.
        """
        banner = QLabel(
            "⚠ Multi-view (rotation) stitching — fuses several rotation "
            "angles into one volume. For normal single-angle acquisitions use the "
            "‘Tile Stitching’ tab instead."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "background:#fff3cd; color:#664d03; border:1px solid #ffe69c;"
            "border-radius:4px; padding:6px;"
        )
        content_layout.addWidget(banner)

        group = QGroupBox("Multi-View (rotation)")
        v = QVBoxLayout()

        note = QLabel(
            "Each view is placed by a rotation about the vertical (Y) axis; "
            "overlaps are resolved by registration and uncovered regions are "
            "zero-filled."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#555; font-size:11px;")
        v.addWidget(note)

        center_row = QHBoxLayout()
        center_row.addWidget(QLabel("Rotation center:"))
        self._mv_center_combo = QComboBox()
        self._mv_center_combo.addItem("Auto (tile centroid)", "auto")
        self._mv_center_combo.addItem("Manual (X, Z)", "manual")
        self._mv_center_combo.currentIndexChanged.connect(
            self._mv_update_center_enabled
        )
        center_row.addWidget(self._mv_center_combo)
        center_row.addWidget(QLabel("X"))
        self._mv_cx_spin = QDoubleSpinBox()
        self._mv_cx_spin.setRange(-1000.0, 1000.0)
        self._mv_cx_spin.setDecimals(3)
        self._mv_cx_spin.setSuffix(" mm")
        center_row.addWidget(self._mv_cx_spin)
        center_row.addWidget(QLabel("Z"))
        self._mv_cz_spin = QDoubleSpinBox()
        self._mv_cz_spin.setRange(-1000.0, 1000.0)
        self._mv_cz_spin.setDecimals(3)
        self._mv_cz_spin.setSuffix(" mm")
        center_row.addWidget(self._mv_cz_spin)
        center_row.addStretch()
        v.addLayout(center_row)

        sign_row = QHBoxLayout()
        sign_row.addWidget(QLabel("Rotation sign:"))
        self._mv_sign_combo = QComboBox()
        self._mv_sign_combo.addItem("+1", 1.0)
        self._mv_sign_combo.addItem("−1", -1.0)
        self._mv_sign_combo.setToolTip(
            "Handedness relating the stage angle to the fusion rotation.\n"
            "Confirm on the rig with a two-angle test if views don't align."
        )
        sign_row.addWidget(self._mv_sign_combo)
        sign_row.addStretch()
        v.addLayout(sign_row)

        group.setLayout(v)
        content_layout.addWidget(group)
        self._mv_update_center_enabled()

    def _mv_update_center_enabled(self) -> None:
        manual = self._mv_center_combo.currentData() == "manual"
        self._mv_cx_spin.setEnabled(manual)
        self._mv_cz_spin.setEnabled(manual)

    def _build_config(self):
        config = super()._build_config()
        config.multiview_fusion = True
        config.rotation_sign = float(self._mv_sign_combo.currentData())
        if self._mv_center_combo.currentData() == "manual":
            # UI is in mm; the pipeline's rotation center is in µm.
            config.rotation_center_um = (
                self._mv_cx_spin.value() * 1000.0,
                self._mv_cz_spin.value() * 1000.0,
            )
        else:
            config.rotation_center_um = None
        return config

    def _set_config_controls_enabled(self, enabled: bool):
        """Lock the rotation controls during a run, like the base config inputs."""
        super()._set_config_controls_enabled(enabled)
        # May be called before _add_dialog_extras has built the controls.
        if not hasattr(self, "_mv_sign_combo"):
            return
        self._mv_center_combo.setEnabled(enabled)
        self._mv_sign_combo.setEnabled(enabled)
        if enabled:
            self._mv_update_center_enabled()  # spins follow Auto/Manual mode
        else:
            self._mv_cx_spin.setEnabled(False)
            self._mv_cz_spin.setEnabled(False)

    def _save_settings(self):
        super()._save_settings()
        s = QSettings()
        s.beginGroup(self._settings_group)
        s.setValue("mv_center_mode", self._mv_center_combo.currentData())
        s.setValue("mv_cx_mm", self._mv_cx_spin.value())
        s.setValue("mv_cz_mm", self._mv_cz_spin.value())
        s.setValue("mv_rotation_sign", self._mv_sign_combo.currentData())
        s.endGroup()

    def _restore_settings(self):
        super()._restore_settings()
        s = QSettings()
        s.beginGroup(self._settings_group)
        mode = s.value("mv_center_mode", "auto")
        idx = self._mv_center_combo.findData(mode)
        if idx >= 0:
            self._mv_center_combo.setCurrentIndex(idx)
        self._mv_cx_spin.setValue(float(s.value("mv_cx_mm", 0.0)))
        self._mv_cz_spin.setValue(float(s.value("mv_cz_mm", 0.0)))
        sidx = self._mv_sign_combo.findData(float(s.value("mv_rotation_sign", 1.0)))
        if sidx >= 0:
            self._mv_sign_combo.setCurrentIndex(sidx)
        s.endGroup()
        self._mv_update_center_enabled()
