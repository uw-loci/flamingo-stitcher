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
# Lines the log panel keeps. A full 42-tile Ket_Trial1 run writes ~670 lines
# (verbose alignment on), so this holds the last ~30 such runs.
LOG_PANEL_MAX_LINES = 20000

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
            "Sets every voxel at or below its channel's threshold to 0\n"
            "in the output, so empty space around the sample compresses\n"
            "to almost nothing ('Compression' must not be None).\n"
            "Use it on cleared samples with a lot of empty space when\n"
            "disk space matters. Lossy: zeroed voxels cannot be recovered.\n"
            "Click 'Preview…' first to choose the thresholds."
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
            "Stitches the first discovered acquisition at low resolution\n"
            "(8x XY, 4x Z) without saving anything, then opens a viewer\n"
            "with a threshold slider per channel. Click 'Apply thresholds'\n"
            "there to copy the values into this panel.\n"
            "Run 'Discover Tiles' first."
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
                f"Channel {ch_id}: voxels at or below this threshold "
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
    # True while a batch is running; the Updates tab grays out 'Install update'.
    run_state_changed = pyqtSignal(bool)

    # How many directory levels to go up when restoring the acq dir path.
    # Subfolder-per-tile layout: grandparent (2 levels up).
    _acq_dir_restore_levels_up = 2

    # QSettings group for this dialog's persisted state. Overridable so sibling
    # tabs (Single Workflow, Multi-View) keep independent settings and don't
    # clobber each other's queue/options.
    _settings_group = _SETTINGS_GROUP

    # Which acquisition types a configuration saved HERE applies to, used to
    # name the saved file and to warn on load.
    #
    # "all-workflows" is the honest answer for this tab and Single Workflow:
    # they differ only in how tiles are DISCOVERED (folder-per-tile vs flat),
    # which is not a setting a configuration carries, so every processing value
    # transfers between them unchanged. Multi-View is the real exception — see
    # MultiViewStitchingDialog.
    _workflow_kind = "all-workflows"

    # Settings that only mean anything on a multi-angle acquisition. Loading
    # them into a single-angle run would silently switch on multi-view fusion,
    # so they are refused rather than applied — by FIELD, not by the file's own
    # label, because a stitch_metadata.json from a multi-view run carries no
    # label at all.
    _MULTIVIEW_ONLY_FIELDS = frozenset(
        {"multiview_fusion", "rotation_sign", "rotation_center_um"}
    )

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
        # The store the last successful item ACTUALLY wrote, as reported by the
        # worker. Not reconstructible from the acquisition name: the user may
        # have answered "New folder" to a duplicate, so the run wrote somewhere
        # the default naming does not point at.
        self._last_success_output = None
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
        self._queue_table.setToolTip(
            "Acquisitions to stitch. Drag folders here or use 'Add...'.\n"
            "Select a row before 'Orientation Preview…' or the destripe\n"
            "'Preview…'. Double-click a finished row to run it again;\n"
            "right-click for more options."
        )
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
        self._add_btn.setToolTip(
            "Add an acquisition folder to the queue.\n"
            "You can also drag acquisition folders onto this window."
        )
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
            "Reads the queued acquisitions before you start: fills in\n"
            "channels, frame size, pixel size and Z step, and shows any\n"
            "data problems. Press it after adding acquisitions to check\n"
            "what was found. It is optional, because 'Run All' discovers\n"
            "anything not yet discovered."
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
            "Shows the selected acquisition's mosaic in all 8 possible\n"
            "orientations. Pick the one where the sample is continuous\n"
            "across tile seams and click 'Use for stitching'; it is\n"
            "remembered for that microscope. You need it once per\n"
            "microscope, and again whenever a stitch comes out mirrored\n"
            "or rotated. It does not stitch anything."
        )
        self._orientation_btn.clicked.connect(self._on_orientation_preview)
        queue_btn_layout.addWidget(self._orientation_btn)

        self._requeue_btn = QPushButton("Re-queue")
        self._requeue_btn.setToolTip(
            "Sets the selected Done, Cancelled or Error rows back to\n"
            "Pending so 'Run All' stitches them again, for example after\n"
            "changing settings. Double-clicking a row does the same."
        )
        self._requeue_btn.clicked.connect(self._requeue_selected)
        queue_btn_layout.addWidget(self._requeue_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setToolTip(
            "Remove the selected acquisitions from the queue.\n"
            "Files on disk are not touched."
        )
        self._remove_btn.clicked.connect(self._remove_from_queue)
        queue_btn_layout.addWidget(self._remove_btn)

        queue_btn_layout.addStretch()

        self._load_config_btn = QPushButton("Load Configuration…")
        self._load_config_btn.setToolTip(
            "Loads processing settings from a saved configuration file\n"
            "or from a finished run's stitch_metadata.json, for example\n"
            "a setup that worked on the microscope computer.\n"
            "'Pixel size (µm)', 'Z step (µm)', 'Frame (AOI)', 'Channels',\n"
            "the output and scratch folders, and this computer's memory\n"
            "limits are not changed."
        )
        self._load_config_btn.clicked.connect(self._on_load_configuration)
        queue_btn_layout.addWidget(self._load_config_btn)

        self._save_config_btn = QPushButton("Save Configuration…")
        self._save_config_btn.setToolTip(
            "Saves the current settings to a .json file that you or a\n"
            "colleague can open with 'Load Configuration…' to reuse them\n"
            "on other data. You do not need to finish a run first."
        )
        self._save_config_btn.clicked.connect(self._on_save_configuration)
        queue_btn_layout.addWidget(self._save_config_btn)

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
        self._pixel_size_spin.setToolTip(
            "Filled in from each acquisition's objective when you press\n"
            "'Discover Tiles', and each queued acquisition uses its own\n"
            "value. Type a value only if the objective is not recorded or\n"
            "the scale is known to be wrong: a typed value applies to\n"
            "every queued acquisition."
        )
        settings_layout.addWidget(self._pixel_size_spin, 0, 1)

        settings_layout.addWidget(QLabel("Z step (\u00b5m):"), 0, 2)
        self._z_step_spin = _NoScrollDoubleSpinBox()
        self._z_step_spin.setFocusPolicy(Qt.StrongFocus)
        self._z_step_spin.setRange(0.0, 1000.0)
        self._z_step_spin.setDecimals(3)
        self._z_step_spin.setValue(0.0)
        self._z_step_spin.setSpecialValueText("Auto")
        self._z_step_spin.setToolTip(
            "Each queued acquisition uses its own Z step from its data;\n"
            "'Discover Tiles' shows it here. Type a value only to correct\n"
            "a wrong Z step: it then applies to every queued acquisition.\n"
            "Set it to 0 (shows Auto) to go back to per-acquisition values."
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
            "Size of the camera frame. 'Auto (from file)' reads it from\n"
            "each file and is right in almost all cases. Choose\n"
            "'1024 × 1024' or '2048 × 2048' only if the detected frame\n"
            "size is wrong, for example when the acquisition metadata is\n"
            "missing or files are incomplete."
        )
        settings_layout.addWidget(self._frame_size_combo, 0, 5)

        # Which channels to stitch — a fact about the acquisition, so it lives
        # here with the other "about your image" inputs rather than with output.
        settings_layout.addWidget(QLabel("Channels:"), 1, 0)
        self._channels_edit = QLineEdit()
        self._channels_edit.setPlaceholderText("All (or e.g. 0,1)")
        self._channels_edit.setToolTip(
            "Leave empty to stitch all channels, or list channel numbers\n"
            "separated by commas (for example 0,1) to stitch only those.\n"
            "'Discover Tiles' lists the channels found. Text that is not\n"
            "a list of numbers stitches all channels."
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
            "Shrinks the output in X and Y by this factor. Use 1x for\n"
            "full-resolution data, and 4x-32x for quick test runs and\n"
            "overviews. 'iso' picks both the XY and Z factors for each\n"
            "acquisition so voxels come out as close to cubic as possible\n"
            "(useful for 3-D analysis); the 'Z' list is then grayed out.\n"
            "Example at 0.406 µm pixels and a 2.5 µm Z step:\n"
            "XY 8x, Z 1x (3.25 × 2.50 µm)."
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
            "Shrinks the output in Z by averaging each group of this many\n"
            "planes. Leave at 1x for final data; 2x-16x make test runs\n"
            "faster and lighter on memory. Grayed out when 'Downsample' XY\n"
            "is 'iso', which picks Z for you."
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
            ("Split (each sheet owns its half)", "split"),
            ("Blend (split with a soft handover)", "blend"),
            ("Content (weight by local detail)", "content"),
            ("Leonardo FUSE", "leonardo"),
            ("Separate (keep light paths)", "separate"),
        ]:
            self._fusion_combo.addItem(label, value)
        self._fusion_combo.setToolTip(
            "How the left and right light sheets of each tile are combined\n"
            "(no effect with one sheet; tiles are combined by 'Tile overlap').\n"
            "'Max' (default) keeps the brighter value; on scattering samples\n"
            "it can keep blur, so try 'Content (weight by local detail)' or\n"
            "'Split (each sheet owns its half)'.\n"
            "'Split (each sheet owns its half)' and 'Blend (split with a soft\n"
            "handover)' give each sheet half the frame and need 'Illumination\n"
            "side lighting the LOW end of the frame' set on the 'Options' tab.\n"
            "'Separate (keep light paths)' keeps each sheet as its own\n"
            "channel, to check which one causes an artifact."
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
            "How neighboring tiles are combined where they overlap.\n"
            "'Maximum' (default) keeps the brighter pixel; best for sparse\n"
            "samples that do not fill the field of view.\n"
            "'Blend' fades one tile into the next; best for dense, evenly\n"
            "lit samples, but can leave a dark band at seams on sparse ones.\n"
            "'Brightest tile' takes each whole overlap from the brighter\n"
            "tile; try it when tiles differ in brightness."
        )
        tile_fuse_box.addWidget(self._tile_fusion_combo)
        # How wide Blend's cross-fade is. Only Blend uses it.
        self._blend_width_combo = QComboBox()
        for _label, _value in [
            ("Fade: whole overlap", 1.0),
            ("Fade: half overlap", 0.5),
            ("Fade: fixed 100 µm", 0.0),
        ]:
            self._blend_width_combo.addItem(_label, _value)
        self._blend_width_combo.setToolTip(
            "Width of Blend's fade between tiles. 'Fade: whole overlap'\n"
            "(default) uses the full overlap between tiles; 'Fade: half\n"
            "overlap' gives a sharper hand-over; 'Fade: fixed 100 µm' uses\n"
            "100 µm regardless of the overlap. Only used when 'Tile overlap'\n"
            "is 'Blend'. A wider fade does not hide misaligned tiles."
        )
        tile_fuse_box.addWidget(self._blend_width_combo)
        self._tile_fusion_combo.currentIndexChanged.connect(
            lambda _i: self._blend_width_combo.setEnabled(
                self._tile_fusion_combo.currentData() == "blend"
            )
        )
        self._blend_width_combo.setEnabled(
            self._tile_fusion_combo.currentData() == "blend"
        )
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
            "'OME-Zarr (Fiji compatible)': the safe default; opens in Fiji,\n"
            "QuPath, BigDataViewer and napari.\n"
            "'OME-Zarr Sharded': fewest files, but napari only.\n"
            "'OME-TIFF (single file)': one file any viewer can read; good for\n"
            "sharing.\n"
            "'Imaris (.ims)': for Imaris users, Windows only.\n"
            "'Both (Sharded + TIFF)': writes both."
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
            "<b>Which format?</b><br>"
            "<b>OME-Zarr (Fiji compatible)</b>: opens in Fiji (N5 plugin), "
            "QuPath, BigDataViewer and napari. Use this if unsure.<br>"
            "<b>OME-Zarr Sharded</b>: fewest files; opens in napari only.<br>"
            "<b>OME-TIFF (single file)</b>: one file, readable almost "
            "everywhere; best for sharing and archiving.<br>"
            "<b>Imaris (.ims)</b>: opens directly in Imaris. Windows only.<br>"
            "To convert with ImarisFileConverter instead, choose "
            "'OME-TIFF (single file)' and uncheck 'TIFF pyramids'."
        )
        format_help.setFixedWidth(20)
        settings_layout.addWidget(format_help, 2, 2)

        settings_layout.addWidget(QLabel("Compression:"), 2, 3)
        self._compression_combo = QComboBox()
        self._compression_combo.setToolTip(
            "How the output is compressed. Keep the default ('zstd\n"
            "(recommended)' for Zarr, 'zlib (best compatibility)' for TIFF)\n"
            "unless you have a reason. 'None (no compression)' makes much\n"
            "larger files that load fastest from a local SSD; compressed\n"
            "files load faster from network or USB drives. Imaris always\n"
            "uses its own compression."
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
            "'Auto' (recommended) streams when the in-memory estimate is\n"
            "more than 60% of this computer's RAM.\n"
            "'In-memory (fast)' is quicker but holds everything in RAM.\n"
            "'Streaming (low memory)' keeps temporary files on disk\n"
            "('Scratch dir') and handles any size.\n"
            "Compare with the In-memory and Streaming estimates in the\n"
            "'Output' box at the top. Choose 'Streaming (low memory)' if a\n"
            "run fails for lack of memory."
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
            "Folder for the temporary files written while streaming; they\n"
            "are deleted when the run ends. Leave blank to keep them next\n"
            "to the output. Point it at a fast local SSD when the output is\n"
            "on a network or USB drive; on the same drive as the output it\n"
            "gives no speed-up. The run checks free space here before it\n"
            "starts."
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
        # Its tooltip is set by _update_preprocessing_availability.
        self._destripe_cb = QCheckBox("Destripe (PyStripe) \u2731")
        proc_layout.addWidget(self._destripe_cb, 0, 0)

        self._destripe_fast_cb = QCheckBox("Fast")
        self._destripe_fast_cb.setToolTip(
            "Destripes the downsampled tile instead of every full-resolution\n"
            "plane: much faster when 'Downsample' is above 1x. The destripe\n"
            "settings are converted for the smaller frame, so it removes the\n"
            "same stripes the preview shows, but both illumination sides are\n"
            "filtered together after they are merged. Good for test runs;\n"
            "leave it off for final data."
        )
        self._destripe_fast_cb.setEnabled(False)
        # Fast is a destripe-only variant (apply after downsample). When
        # Destripe is unchecked it must also be unchecked + disabled so
        # the UI state matches what the pipeline will actually run. The
        # plain toggled→setEnabled link only handled the enabled state.
        self._destripe_cb.toggled.connect(self._on_destripe_toggled)

        # Stripe orientation. The filter is axis-fixed AND destriping runs in the
        # raw camera frame (before the per-tile rot/flip), so the wrong axis
        # removes nothing. "Auto" derives it from the tile orientation.
        self._destripe_dir_combo = QComboBox()
        for _lbl, _val in (
            ("Dir: Auto", "auto"),
            ("Dir: Horizontal", "horizontal"),
            ("Dir: Vertical", "vertical"),
        ):
            self._destripe_dir_combo.addItem(_lbl, _val)
        self._destripe_dir_combo.setToolTip(
            "Direction of the stripes to remove. 'Dir: Auto' (default)\n"
            "works it out from the tile orientation chosen for this\n"
            "microscope. Choose 'Dir: Horizontal' or 'Dir: Vertical' only\n"
            "if 'Preview…' shows stripes left behind. Directions refer to\n"
            "the raw camera image, which can look rotated 90° from the\n"
            "stitched result."
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
            "Adjust the destripe filter strength and its other settings.\n"
            "Change them only if stripes remain or real structure is being\n"
            "smoothed away; 'Preview…' lets you try them on a real tile.\n"
            "Saved for the selected acquisition's microscope."
        )
        self._destripe_settings_btn.setEnabled(False)
        self._destripe_settings_btn.clicked.connect(self._on_destripe_settings)

        # Live preview: tuning by re-running a whole stitch is far too slow a
        # loop, and the things that go wrong (which axis is filtered, a wavelet
        # level deeper than the frame supports, how much the per-plane Otsu
        # split shifts the result between tiles) are invisible in the mosaic.
        self._destripe_preview_btn = QPushButton("Preview…")
        self._destripe_preview_btn.setToolTip(
            "Shows one plane of a tile from the selected acquisition before\n"
            "and after destriping, updating as you change the settings.\n"
            "Pick a tile with visible detail and move the 'Z plane' slider.\n"
            "Click OK to use those settings for the run."
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
            "Gives the sharper tile more weight in each overlap. Only used\n"
            "when 'Tile overlap' is 'Blend' (grayed out otherwise). Can make\n"
            "fusion many times slower. Leave it off unless Blend seams still\n"
            "look soft or uneven."
        )
        proc_layout.addWidget(self._content_fusion_cb, 0, 2, 1, 2)
        # Content-based weights only shape a BLEND; with Maximum or Brightest
        # tile the pipeline ignores them (pipeline.py, content_based).
        self._tile_fusion_combo.currentIndexChanged.connect(
            lambda _i: self._content_fusion_cb.setEnabled(
                self._tile_fusion_combo.currentData() == "blend"
            )
        )
        self._content_fusion_cb.setEnabled(
            self._tile_fusion_combo.currentData() == "blend"
        )

        # Proc Row 1: Deconvolution + Flat-field
        # Its tooltip is set by _update_preprocessing_availability.
        self._deconv_cb = QCheckBox("Deconvolution \u2731")
        proc_layout.addWidget(self._deconv_cb, 1, 0)

        self._flat_field_cb = QCheckBox("Flat-field correction")
        self._update_preprocessing_availability()
        proc_layout.addWidget(self._flat_field_cb, 1, 1)

        self._ozx_cb = QCheckBox("Package as .ozx")
        self._ozx_cb.setToolTip(
            "Also saves the result as one .ozx file, which is easier to\n"
            "copy or send than a folder of many files. Available only when\n"
            "'Output format' is 'OME-Zarr Sharded' or 'Both (Sharded + TIFF)'.\n"
            "The .ome.zarr folder is kept, so this needs the same disk space\n"
            "again."
        )
        proc_layout.addWidget(self._ozx_cb, 1, 2)

        self._tiff_pyramids_cb = QCheckBox("TIFF pyramids")
        self._tiff_pyramids_cb.setChecked(True)
        self._tiff_pyramids_cb.setToolTip(
            "Adds lower-resolution copies inside the OME-TIFF so napari and\n"
            "QuPath can zoom quickly. Uncheck it if you will convert the file\n"
            "with ImarisFileConverter, which can misread them as extra Z\n"
            "planes. Available only when 'Output format' is 'OME-TIFF\n"
            "(single file)' or 'Both (Sharded + TIFF)'."
        )
        proc_layout.addWidget(self._tiff_pyramids_cb, 1, 3)

        # Proc Row 7 (added below, with the fusion rows): per-tile intensity
        # adjustment (BigStitcher's method).
        self._intensity_adjust_combo = QComboBox()
        for label, value in (
            ("Off", "off"),
            ("Measure only", "measure"),
            ("Apply", "apply"),
        ):
            self._intensity_adjust_combo.addItem(label, value)
        self._intensity_adjust_combo.setToolTip(
            "Evens out brightness differences between tiles (laser drift,\n"
            "bleaching over a long run) so tiles don't meet at a brightness\n"
            "step.\n"
            "'Measure only': logs each tile's correction and fuses unchanged;\n"
            "use it to see whether correction is needed.\n"
            "'Apply': fuses the corrected tiles.\n"
            "Fine-tune it per microscope on the 'Options' tab."
        )
        _ia_box = QHBoxLayout()
        _ia_box.setContentsMargins(0, 0, 0, 0)
        _ia_box.addWidget(QLabel("Intensity adjustment:"))
        _ia_box.addWidget(self._intensity_adjust_combo)
        _ia_box.addStretch()
        proc_layout.addLayout(_ia_box, 7, 0, 1, 4)

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
            "Place tiles by stage position only, without measuring their\n"
            "overlaps. Check it when tiles don't overlap, or for a fast test\n"
            "run (registration can take hours). Leave unchecked (default)\n"
            "for normal runs. The registration settings below are grayed\n"
            "out while it is checked."
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
            "How much tiles are shrunk in X and Y before registration\n"
            "compares them. Higher is faster (2x is about a quarter of the\n"
            "work of 1x); lower measures X/Y more finely. 4x (default) suits\n"
            "most runs; try 2x or 1x if the registration report shows seams\n"
            "disagreeing in X/Y. It is a maximum: the run uses less when the\n"
            "tile overlap would get too thin (the log shows the value used)."
        )
        rb_layout.addWidget(self._reg_binning_xy_combo)
        rb_layout.addSpacing(16)  # visible gap between the XY group and Z group
        self._reg_z_spacing_label = QLabel("Z spacing")
        rb_layout.addWidget(self._reg_z_spacing_label)
        # Spacings in µm, built from the selected acquisition's own Z step (and
        # the Z downsample) by _rebuild_reg_z_choices: "Every plane", then
        # doubling. Data = µm, 0 = every plane. It replaced a "1x/2x/4x/8x"
        # combo where 1x silently meant 10 µm, so at a 2.5 µm step nobody could
        # register on every plane, and the log blamed the XY overlap rule.
        self._reg_z_spacing_combo = QComboBox()
        self._reg_z_voxel_um = None
        self._rebuild_reg_z_choices(None, keep=20.0)
        self._reg_z_spacing_combo.setToolTip(
            "Registration compares tiles at this Z spacing, averaging planes\n"
            "to reach it. Default 20 µm. Finer spacing measures Z more\n"
            "precisely but costs more time and memory: the 'Registration Z'\n"
            "line at the bottom of this box shows the cost for the queued\n"
            "acquisitions. Each seam's Z is measured in steps of this\n"
            "spacing times 'Peak refinement' (half the spacing at the\n"
            "default)."
        )
        rb_layout.addWidget(self._reg_z_spacing_combo)
        rb_layout.addStretch(1)
        reg_layout.addLayout(rb_layout, 0, 3)
        self._set_registration_binning({"y": 4, "x": 4})

        # Shift limits are a fact about the instrument, so they are set per
        # microscope and objective on the Options tab only. Two places for one
        # setting meant the saved profile silently replaced what was typed here.
        self._shift_limits_note = QLabel(
            "Shift limits: set on the Options tab ('Maximum lateral correction', "
            "'Maximum axial correction')."
        )
        self._shift_limits_note.setWordWrap(True)
        self._shift_limits_note.setToolTip(
            "How far registration may move a tile is set per microscope and "
            "objective on the Options tab. Leave both at Auto unless the log's "
            "'Pairwise shift bounds' line shows real seams being dropped."
        )
        reg_layout.addWidget(self._shift_limits_note, 1, 0, 1, 4)

        self._z_refine_cb = QCheckBox("Refine Z alignment ✱")
        self._z_refine_cb.setToolTip(
            "Adds a second registration pass, on every Z plane, that\n"
            "corrects only Z; X/Y stay as the first pass set them. Turn it\n"
            "on when the registration report (registration_seams.csv) shows\n"
            "neighboring tiles disagreeing in Z while X/Y line up. The\n"
            "'Registration Z' line and the queue time include its cost."
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
            "How far (± µm) 'Refine Z alignment' searches in Z; used only\n"
            "when that box is checked. Default ±40 µm. It never searches\n"
            "past the Z shift limit ('Maximum axial correction' on the\n"
            "'Options' tab). A tile whose correction lands at the limit\n"
            "keeps its first-pass Z (the log counts these). If many do,\n"
            "raise this and, if needed, 'Maximum axial correction'."
        )
        reg_layout.addWidget(self._z_refine_range_label, 2, 2)
        reg_layout.addWidget(self._z_refine_range_spin, 2, 3)

        reg_layout.addWidget(QLabel("Approach:"), 4, 0)
        self._approach_combo = QComboBox()
        self._approach_combo.addItem("Default", "default")
        self._approach_combo.addItem("Center XY Start", "center_xy")
        self._approach_combo.setToolTip(
            "How tiles are placed from the measured seams.\n"
            "'Default': tiles with no usable seam move with the main\n"
            "registered group. If fewer seams register than 'Minimum share\n"
            "of seams that must register' ('Options' tab), registration is\n"
            "not applied and all tiles use stage positions.\n"
            "'Center XY Start': starts from the center tile and places each\n"
            "unregistered tile from its placed neighbors, without that\n"
            "minimum-share check. Use it for a round or partial sample on a\n"
            "rectangular grid, where the edge tiles are empty."
        )
        reg_layout.addWidget(self._approach_combo, 4, 1, 1, 3)

        self._verbose_align_cb = QCheckBox("Verbose alignment log")
        self._verbose_align_cb.setChecked(True)
        self._verbose_align_cb.setToolTip(
            "Writes every tile's placement and every seam's measurement to\n"
            "the log, not just a summary. Leave on (default) so a log you\n"
            "share has the full detail if a run needs diagnosing; turn off\n"
            "for a shorter log."
        )
        reg_layout.addWidget(self._verbose_align_cb, 5, 0, 1, 4)

        # Global solve: how much disagreement a seam may have with the placement
        # before it is dropped, and whether tiles may rotate as well as shift.
        self._solve_tol_label = QLabel("Solve tolerance:")
        reg_layout.addWidget(self._solve_tol_label, 6, 0)
        self._solve_tol_spin = _NoScrollDoubleSpinBox()
        self._solve_tol_spin.setRange(0.5, 50.0)
        self._solve_tol_spin.setSingleStep(0.5)
        self._solve_tol_spin.setDecimals(1)
        self._solve_tol_spin.setValue(3.5)
        self._solve_tol_spin.setSuffix(" reg. px")
        self._solve_tol_spin.setToolTip(
            "How far a seam may disagree with the solved tile positions\n"
            "before it is dropped from the solve, in registration pixels\n"
            "(default 3.5). A dropped seam no longer holds its two tiles\n"
            "together. Raise it if the registration report shows many seams\n"
            "pruned and a row held by only one seam; lower it if one bad\n"
            "seam pulls tiles out of line.\n"
            "The 'Registration Z' line shows the value in µm. It grows with\n"
            "'Downsample' XY and 'Reg. binning' XY, and never goes below one\n"
            "registration voxel (so at a coarse 'Z spacing', small changes\n"
            "may have no effect). The same µm applies to Z, so a high value at\n"
            "4x or more can hide Z errors: check the seams the registration\n"
            "map marks as left visibly off."
        )
        reg_layout.addWidget(self._solve_tol_spin, 6, 1)
        self._solve_model_label = QLabel("Solve model:")
        reg_layout.addWidget(self._solve_model_label, 6, 2)
        self._solve_model_combo = QComboBox()
        self._solve_model_combo.addItem("Shift only", "translation")
        self._solve_model_combo.addItem("Shift + rotation (rigid)", "rigid")
        self._solve_model_combo.setToolTip(
            "What the solve may do to each tile.\n"
            "'Shift only' (default): tiles move but do not rotate.\n"
            "'Shift + rotation (rigid)': tiles may also tilt slightly; try it\n"
            "when rows sit at a Z offset that grows across the mosaic.\n"
            "Experimental. Rotated tiles are resampled between Z planes even\n"
            "with 'Snap Z shifts to whole planes' on. Check the rotations\n"
            "listed in the log."
        )
        reg_layout.addWidget(self._solve_model_combo, 6, 3)

        # What the Z spacing will cost on the queued acquisitions. Filled by
        # _update_reg_z_cost from the same functions the run and the memory
        # estimate use, so the warning cannot disagree with them.
        self._reg_z_cost_label = QLabel("")
        self._reg_z_cost_label.setWordWrap(True)
        self._reg_z_cost_label.setTextFormat(Qt.RichText)
        reg_layout.addWidget(self._reg_z_cost_label, 7, 0, 1, 4)

        self._z_snap_cb = QCheckBox("Snap Z shifts to whole planes")
        self._z_snap_cb.setChecked(True)
        self._z_snap_cb.setToolTip(
            "Keeps every tile on whole Z planes, so no plane is blended from\n"
            "two acquired planes. On (default) costs at most half a Z step\n"
            "of placement accuracy. Turn it off only if placing tiles finer\n"
            "than one Z step matters more than keeping planes intact."
        )
        reg_layout.addWidget(self._z_snap_cb, 3, 0, 1, 2)

        # How finely each seam's correlation peak is located, in fractions of a
        # registration voxel (registration_upsample_factor). multiview-stitcher
        # uses 1/2 for 3-D unless told otherwise, so a Z spacing of 10 µm moved
        # each seam's Z in 5 µm steps -- every seam dz on the 2026-10-18 run was
        # a multiple of 5.00 µm.
        self._reg_upsample_label = QLabel("Peak refinement:")
        reg_layout.addWidget(self._reg_upsample_label, 3, 2)
        self._reg_upsample_combo = QComboBox()
        for _label, _value in [
            ("1/2 voxel (default)", 0),
            ("1/4 voxel", 4),
            ("1/10 voxel", 10),
            ("1/20 voxel", 20),
        ]:
            self._reg_upsample_combo.addItem(_label, _value)
        self._reg_upsample_combo.setToolTip(
            "How finely each seam's shift is measured, as a fraction of a\n"
            "registration voxel ('Z spacing' in Z, the binned pixel in X/Y).\n"
            "'1/2 voxel (default)' is the coarsest: at a 20 µm 'Z spacing' a\n"
            "seam's Z moves in 10 µm steps, which can get good seams dropped\n"
            "by 'Solve tolerance'. Choose '1/10 voxel' for finer shifts.\n"
            "Adds up to about 20% registration time and no memory; the\n"
            "'Registration Z' line shows the step size."
        )
        reg_layout.addWidget(self._reg_upsample_combo, 3, 3)

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
            "Size of the blocks fusion works on. It does not change the\n"
            "saved file's chunking or how fast viewers pan and zoom.\n"
            "'Large (32 MB, 64×512×512)' (default) suits most runs. Choose\n"
            "'XL (128 MB, 64×1024×1024)' if fusion is slow and you have\n"
            "spare RAM; 'Small (4 MB, 32×256×256)' or 'Medium (16 MB,\n"
            "128×256×256)' if memory runs short."
        )
        proc_layout.addWidget(self._chunk_size_combo, 5, 1, 1, 3)

        # Proc Row 6: how much output is fused at a time when streaming.
        proc_layout.addWidget(QLabel("Fusion region size:"), 6, 0)
        self._fusion_region_combo = QComboBox()
        for _label, _gb in (
            ("Auto \u2014 ~4 GB regions (default)", 4.0),
            ("Auto \u2014 ~8 GB regions", 8.0),
            ("Auto \u2014 ~16 GB regions", 16.0),
            ("Auto \u2014 ~32 GB regions", 32.0),
            ("Whole output (no regions)", 0.0),
        ):
            self._fusion_region_combo.addItem(_label, _gb)
        self._fusion_region_combo.setToolTip(
            "When the run streams ('Memory mode' set to 'Streaming (low\n"
            "memory)', or 'Auto' choosing it), fuse the output this many GB\n"
            "at a time to keep memory in check. The ~4 GB default is\n"
            "safest. On a large-RAM machine a bigger region may be faster\n"
            "(not yet measured). 'Whole output (no regions)' can use far\n"
            "more RAM than the memory estimate shows. Outputs under twice\n"
            "the region size are always fused in one piece."
        )
        proc_layout.addWidget(self._fusion_region_combo, 6, 1, 1, 3)

        # Proc Row 7: Tile-border artifact QC (diagnostic)
        self._border_qc_cb = QCheckBox("Detect border artifacts (QC)")
        self._border_qc_cb.setToolTip(
            "After preprocessing, checks the seams between neighboring\n"
            "tiles for sharp brightness steps and writes a text report, next\n"
            "to the run log file, listing the tile pairs affected. Turn it\n"
            "on when the stitched image shows visible seams. Checks the\n"
            "registration channel only; quick. Most sensitive at\n"
            "'Downsample' XY 1x or 2x."
        )
        proc_layout.addWidget(self._border_qc_cb, 8, 0)
        self._border_qc_label = QLabel("QC detail:")
        proc_layout.addWidget(self._border_qc_label, 8, 1)
        self._border_qc_mode_combo = QComboBox()
        self._border_qc_mode_combo.addItem("MIP length (fast)", "mip")
        self._border_qc_mode_combo.addItem("Full (area + Z-range)", "full")
        self._border_qc_mode_combo.addItem("Pairs only", "pairs")
        self._border_qc_mode_combo.setToolTip(
            "How much the border QC report says.\n"
            "'MIP length (fast)': length of each bad seam, from a Z max\n"
            "projection. Enough to find which seams are bad.\n"
            "'Full (area + Z-range)': also which Z planes are affected;\n"
            "slower at full resolution.\n"
            "'Pairs only': just the list of tile pairs.\n"
            "Used only with 'Detect border artifacts (QC)' checked."
        )
        proc_layout.addWidget(self._border_qc_mode_combo, 8, 2)

        self._reg_report_cb = QCheckBox("Registration report")
        self._reg_report_cb.setChecked(True)
        self._reg_report_cb.setToolTip(
            "Writes registration_report.txt, registration_report.csv and\n"
            "registration_seams.csv into the output folder: how far each\n"
            "tile moved, which seams were used and which were rejected.\n"
            "Leave on (default); the files are small. With 'Skip\n"
            "registration' checked, the report records that tiles were\n"
            "placed by stage position."
        )
        proc_layout.addWidget(self._reg_report_cb, 8, 3)

        # Proc Row 6: Legend
        legend = QLabel("\u2731 = significantly increases processing time")
        legend.setStyleSheet("color: #FF8C00; font-style: italic; font-size: 11px;")
        proc_layout.addWidget(legend, 9, 0, 1, 4)

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
            self._z_refine_cb,
        ):
            _cb.toggled.connect(self._refresh_memory_estimate)
        self._solve_tol_spin.valueChanged.connect(self._refresh_memory_estimate)
        for _combo in (
            self._frame_size_combo,
            self._fusion_combo,
            self._tile_fusion_combo,
            self._reg_binning_xy_combo,
            self._reg_z_spacing_combo,
            self._reg_upsample_combo,
            self._solve_model_combo,
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
                "Live preview needs napari, which this build does not include.\n"
                "Background zeroing still works: type each channel's value in\n"
                "the 'Threshold (uint16)' column."
            )
        content_layout.addWidget(self._bg_zero_panel)

        # --- Action buttons ---
        # (Discover Tiles is pinned in the frozen Output row at the top; the
        # run controls stay here at the bottom.)
        btn_layout = QHBoxLayout()

        self._run_btn = QPushButton("Run All")
        self._run_btn.setToolTip(
            "Stitch every Pending item in the queue, one after another."
        )
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        btn_layout.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip(
            "Stop the run in progress and mark the remaining Pending items\n"
            "Cancelled. Use 'Re-queue' to run them again later."
        )
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)

        btn_layout.addStretch()

        self._setup_env_btn = QPushButton("Set up flat-field…")
        self._setup_env_btn.setToolTip(
            "Installs what 'Flat-field correction' needs, once. Needs\n"
            "internet and about 2-4 GB of disk; you'll be asked where to\n"
            "install it. No GPU needed. Click this if 'Flat-field\n"
            "correction' is grayed out."
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
            "Opens the hardware and troubleshooting guide: disk and RAM\n"
            "planning, and what to check when a run is stuck or slow. Opens\n"
            "in your Markdown or text viewer, or on GitHub in your browser\n"
            "if no local copy is found."
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
        self._log_toggle.setToolTip(
            "Show or hide the run log. The full log of each session is\n"
            "also saved in the app's logs folder."
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
            "Prefix each new log line with a compact date and time to the\n"
            "minute (MM-DD HH:MM), so you can see when a line was printed.\n"
            "Handy on long runs where progress lines repeat with only the\n"
            "ETA changing. Applies to lines printed from now on."
        )
        log_opts_row.addWidget(self._timestamp_log_cb)
        log_opts_row.addStretch()
        log_layout.addLayout(log_opts_row)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        # Keep only the newest lines. The panel lived for the whole app session
        # and held every run since launch; the full log is in the file under
        # <config dir>/logs either way.
        self._log_text.document().setMaximumBlockCount(LOG_PANEL_MAX_LINES)
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
        from flamingo_stitcher.orientation import describe_orientation

        # Name the microscope so the log is unambiguous about what it applies to.
        scope = (
            f"microscope '{microscope_name}'" if microscope_name else "this microscope"
        )
        self._log(
            f"Tile orientation '{describe_orientation(name, reverse_x, reverse_y)}' saved for {scope}; "
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
        self._output_dir_edit.setToolTip(
            "Folder for the stitched results; each acquisition gets its own\n"
            "subfolder. Required before 'Run All'. The line below shows the\n"
            "expected output size and the free space on that drive."
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
            "Native voxel size (from 'Pixel size (µm)' and 'Z step (µm)')\n"
            "and the output voxel size after 'Downsample'. With 'iso', the\n"
            "XY and Z factors it picks are shown in parentheses."
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
        # The box only DISPLAYS a detected value now (each entry resolves its
        # own at run time, see _build_config), but it should still show the
        # acquisition about to run rather than one already stitched.
        first_item = next(
            (it for it in self._queue
             if it.get("tiles") and it.get("status") == "pending"),
            None,
        ) or next((it for it in self._queue if it.get("tiles")), None)
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

    def _focus_log_for_run(self):
        """Hand the window over to the log once a run starts.

        Before a run the screen belongs to the settings; once the run is going
        those settings are locked anyway and what matters is the output. So the
        log opens and the Processing Options pane closes.

        Closing Processing Options is not decoration — it is what makes the
        auto-open safe. Opening the log used to push the always-visible
        Progress section off small screens (dbe3d43 removed the auto-open for
        exactly that reason, and because it overrode a remembered preference).
        The remembered preference is gone now: the pane always starts closed.
        And taking the log's height back from the settings pane instead of from
        the bottom of the window keeps Progress where it was.

        Both panes stay user-controlled — this sets a starting point for the
        run, it does not hold them there. Re-opening Processing Options to
        queue more work works exactly as before.
        """
        self._proc_toggle.setChecked(False)
        self._log_toggle.setChecked(True)
        # One resize after both, so the window is not sized against a
        # half-applied state.
        self.updateGeometry()
        if self.layout() is not None:
            self.layout().activate()
        self.resize(self.width(), self.sizeHint().height())


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

    def _restore_reg_z_spacing(self, s) -> None:
        """Saved Z spacing, or an older saved "reg_binning_z" read as what it
        meant (N x REFERENCE_Z_STEP_UM). Default 20 µm, the old z=2."""
        if s.contains("reg_z_spacing_um"):
            self._set_registration_z_spacing(s.value("reg_z_spacing_um", 20.0, type=float))
        elif s.contains("reg_binning_z"):
            from flamingo_stitcher.pipeline import REFERENCE_Z_STEP_UM

            self._set_registration_z_spacing(
                max(1, s.value("reg_binning_z", 2, type=int)) * REFERENCE_Z_STEP_UM
            )
        else:
            self._set_registration_z_spacing(20.0)

    def _registration_binning(self) -> dict:
        """The per-axis binning dict multiview-stitcher wants.

        Lateral only, from the XY combo. Z is chosen as a spacing
        (``_registration_z_spacing``) and resolved per acquisition by the
        pipeline; "z" stays in the dict at its old default only so files read
        by older versions keep a valid shape -- the run ignores it whenever
        ``registration_z_spacing_um`` is set, which the dialog always does.
        """
        xy = int(self._reg_binning_xy_combo.currentData() or 1)
        return {"z": 2, "y": xy, "x": xy}

    def _set_registration_binning(self, binning) -> None:
        """Drive the XY combo from a per-axis dict, and the Z spacing from an
        older dict's "z" (N x REFERENCE_Z_STEP_UM, what it meant then).

        A dict whose X and Y differ cannot be shown exactly, so the larger of
        the two is selected and the mismatch logged rather than silently
        halving one axis's binning -- the run would then do less work than the
        configuration asked for, which is the direction that costs time, not
        correctness.
        """
        if not isinstance(binning, dict):
            return
        try:
            y = int(binning.get("y", 4))
            x = int(binning.get("x", y))
            z = int(binning["z"]) if "z" in binning else None
        except (TypeError, ValueError):
            return
        if x != y:
            logger.warning(
                f"Registration binning has x={x} and y={y}; the dialog offers "
                f"one lateral factor, so showing {max(x, y)}x. Set them "
                f"separately in the config file if they must differ."
            )
        self._set_combo_by_data(self._reg_binning_xy_combo, max(x, y))
        if z is not None:
            from flamingo_stitcher.pipeline import REFERENCE_Z_STEP_UM

            self._set_registration_z_spacing(max(1, z) * REFERENCE_Z_STEP_UM)

    def _registration_z_spacing(self) -> float:
        """Selected registration Z spacing in µm; 0 = every plane."""
        v = self._reg_z_spacing_combo.currentData()
        try:
            return max(0.0, float(v))
        except (TypeError, ValueError):
            return 20.0

    def _set_registration_z_spacing(self, um) -> None:
        try:
            um = max(0.0, float(um))
        except (TypeError, ValueError):
            return
        if not self._select_reg_z(um):
            self._rebuild_reg_z_choices(self._reg_z_voxel_um, keep=um)

    def _select_reg_z(self, um) -> bool:
        combo = self._reg_z_spacing_combo
        for i in range(combo.count()):
            if abs(float(combo.itemData(i)) - um) < 1e-6:
                combo.setCurrentIndex(i)
                return True
        return False

    def _reg_z_voxel_from_ui(self):
        """This acquisition's Z voxel after the Z downsample, µm, or None.

        From the Z step box when set, else the first queued acquisition's own
        step. The pipeline resolves the spacing per acquisition at run time;
        this only labels the choices.
        """
        z_step = float(self._z_step_spin.value()) if hasattr(self, "_z_step_spin") else 0.0
        if z_step <= 0:
            for it in getattr(self, "_queue", []) or []:
                tiles = it.get("tiles")
                if tiles:
                    try:
                        z_step = float(tiles[0].z_step_mm) * 1000.0
                    except Exception:
                        z_step = 0.0
                    break
        if z_step <= 0:
            return None
        ds_z = self._downsample_z_combo.currentData() if hasattr(self, "_downsample_z_combo") else 1
        if ds_z == -1:
            try:
                from flamingo_stitcher.pipeline import compute_iso_downsample

                _, ds_z = compute_iso_downsample(self._pixel_size_spin.value(), z_step)
            except Exception:
                ds_z = 1
        return z_step * max(1, int(ds_z or 1))

    def _rebuild_reg_z_choices(self, voxel_um, keep=None) -> None:
        """Offer every plane plus doubling spacings for ``voxel_um``.

        ``keep`` (µm) stays selected; a value that is not one of the offered
        multiples (a saved 20 µm on a 3 µm step) is kept as its own entry and
        the pipeline rounds it to the nearest whole number of planes.
        """
        from flamingo_stitcher.pipeline import registration_z_choices

        combo = self._reg_z_spacing_combo
        if keep is None and combo.count():
            keep = self._registration_z_spacing()
        self._reg_z_voxel_um = voxel_um
        combo.blockSignals(True)
        try:
            combo.clear()
            if voxel_um:
                for um, planes in registration_z_choices(voxel_um):
                    if planes == 1:
                        combo.addItem(f"Every plane ({voxel_um:g} µm)", 0.0)
                    else:
                        combo.addItem(f"{um:g} µm ({planes} planes)", float(um))
            else:
                combo.addItem("Every plane", 0.0)
                for um in (5.0, 10.0, 20.0, 40.0, 80.0):
                    combo.addItem(f"{um:g} µm", um)
            if keep is not None and not self._select_reg_z(float(keep)):
                if voxel_um:
                    n = max(1, int(round(float(keep) / voxel_um)))
                    label = f"{float(keep):g} µm (~{n} planes)"
                else:
                    label = f"{float(keep):g} µm"
                combo.addItem(label, float(keep))
                combo.setCurrentIndex(combo.count() - 1)
        finally:
            combo.blockSignals(False)

    def _sync_reg_z_choices(self) -> None:
        """Relabel the Z spacing choices when the acquisition's Z voxel changes
        (new queue item, Z step typed, Z downsample changed)."""
        if not hasattr(self, "_reg_z_spacing_combo"):
            return
        v = self._reg_z_voxel_from_ui()
        if v != self._reg_z_voxel_um:
            self._rebuild_reg_z_choices(v)

    def _update_reg_z_cost(self, tile_sets, config) -> None:
        """Say what the chosen Z spacing costs, for the worst queued item.

        Time from the rough registration model and memory from the measured
        bytes-per-voxel model -- the same functions behind the queue time and
        the memory estimate -- next to the same at the default 20 µm, so the
        price of a finer choice is visible before Start.
        """
        label = getattr(self, "_reg_z_cost_label", None)
        if label is None:
            return
        if config.skip_registration or not tile_sets:
            label.setText("")
            return
        try:
            import copy

            from flamingo_stitcher.pipeline import (
                _registration_geometry,
                _rough_register_seconds,
                estimate_registration_peak_bytes,
            )

            ref = copy.copy(config)
            ref.registration_z_spacing_um = 20.0
            worst = None
            for _label, tset in tile_sets:
                geo = _registration_geometry(tset, config)
                if geo is None:
                    continue
                row = (
                    _rough_register_seconds(tset, config),
                    estimate_registration_peak_bytes(tset, config) / 1024**3,
                    _rough_register_seconds(tset, ref),
                    estimate_registration_peak_bytes(tset, ref) / 1024**3,
                    geo,
                )
                if worst is None or row[1] > worst[1]:
                    worst = row
            if worst is None:
                label.setText("")
                return
            t, gb, t_ref, gb_ref, geo = worst

            def _format_duration(sec):
                m = int(round(sec / 60.0))
                return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{max(1, m)} min"

            rz = geo["z"]
            try:
                import psutil

                avail = psutil.virtual_memory().available / 1024**3
            except Exception:
                avail = 0.0
            if avail and gb > 0.95 * avail:
                colour = "#D32F2F"
            elif (avail and gb > 0.7 * avail) or t > 2.0 * max(t_ref, 1.0):
                colour = "#F57C00"
            else:
                colour = "#666"
            what = (
                "every plane" if rz.bin_z == 1
                else f"{rz.bin_z} planes averaged"
            )
            cmp = ""
            if abs(rz.spacing_um - 20.0) > 0.05 and rz.voxel_um:
                cmp = (
                    f" At 20 µm: ~{_format_duration(t_ref)}, "
                    f"~{gb_ref:.1f} GB."
                )
            ram = f" of {avail:.0f} GB free" if avail else ""
            refine = (
                " Includes the Z refinement pass (every plane)."
                if getattr(config, "registration_z_refine", False) else ""
            )
            up = int(getattr(config, "registration_upsample_factor", 0) or 0)
            step = rz.spacing_um / (up if up > 0 else 2)
            from flamingo_stitcher.pipeline import global_opt_abs_tol_um

            lat = float(config.pixel_size_um) * max(1, int(config.downsample_xy or 1))
            tol_um = global_opt_abs_tol_um(
                config.global_opt_abs_tol,
                {"z": rz.voxel_um, "y": lat, "x": lat},
                {"z": geo["bin_z"], "y": geo["bin_xy"], "x": geo["bin_xy"]},
            )
            label.setText(
                f"<span style='color:{colour};'>Registration Z: {what} = "
                f"{rz.spacing_um:g} µm, {int(round(geo['reg_planes']))} planes "
                f"compared per seam; each seam's Z moves in {step:.2g} µm steps; "
                f"solve tolerance ~{tol_um:.1f} µm. "
                f"Rough cost: ~{_format_duration(t)} and "
                f"~{gb:.1f} GB{ram} for one seam.{refine}{cmp}</span>"
            )
        except Exception as e:
            logger.debug(f"Registration Z cost skipped: {e}")
            label.setText("")

    def _on_skip_reg_toggled(self, checked: bool):
        """Enable/disable registration controls based on skip state."""
        for _widget in (
            self._reg_binning_xy_combo,
            self._reg_z_spacing_combo,
            self._reg_binning_xy_label,
            self._reg_z_spacing_label,
            self._reg_upsample_combo,
            self._reg_upsample_label,
            self._solve_tol_spin,
            self._solve_tol_label,
            self._solve_model_combo,
            self._solve_model_label,
            self._reg_binning_label,
        ):
            _widget.setEnabled(not checked)
        self._shift_limits_note.setEnabled(not checked)
        # Same for the axial controls. The report checkbox deliberately stays
        # live: with registration skipped it writes a file saying the tiles were
        # never registered, which is exactly the fact that went unnoticed for
        # months.
        self._z_refine_cb.setEnabled(not checked)
        self._z_refine_range_label.setEnabled(not checked)
        self._z_refine_range_spin.setEnabled(not checked)
        self._z_snap_cb.setEnabled(not checked)
        self._approach_combo.setEnabled(not checked)

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

        params, direction, note = self._destripe_settings_for_selection()
        dlg = DestripeSettingsDialog(params, source_note=note, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._destripe_params = dlg.get_params()
            # Save under the microscope's own direction, not whatever this
            # window last showed.
            idx = self._destripe_dir_combo.findData(direction)
            if idx >= 0:
                self._destripe_dir_combo.setCurrentIndex(idx)
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

        params, direction, _note = self._destripe_settings_for_selection()
        dlg = DestripePreviewDialog(
            acq_path,
            params,
            direction=direction,
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
        self._sync_reg_z_choices()
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

        self._sync_reg_z_choices()
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
            self._update_reg_z_cost(tile_sets, config)

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
            if est.get("registration_gb"):
                fusion_note += (
                    f"  Registration working set: ~{_fmt_gb(est['registration_gb'])} "
                    f"(one seam at the chosen Z spacing)\n"
                )
            limit_note = ""
            if est.get("limited_by") == "registration":
                limit_note = (
                    "  Peak is set by REGISTRATION: one seam's overlap at the\n"
                    "  chosen Z spacing. A coarser 'Z spacing', a larger XY\n"
                    "  downsample, or 'Refine Z alignment' instead of every plane\n"
                    "  lowers it.\n"
                )
            elif est.get("limited_by") == "preprocess":
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
            if will_stream:
                _advice = "Raise 'Downsample' XY to reduce memory."
            else:
                _advice = (
                    "Set 'Memory mode' to 'Streaming (low memory)' "
                    f"(~{est['streaming_gb']:.1f} GB peak) or raise 'Downsample' XY."
                )
            self._memory_indicator.setToolTip(
                f"<b>Out of memory risk!</b><br>"
                f"Estimated peak: ~{peak_gb:.0f} GB ({mode_name})<br>"
                f"System RAM: {sys_ram:.0f} GB<br><br>"
                f"{_advice}"
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
                f"Should work but leaves little room for other programs.<br>"
                f"'Streaming (low memory)' would use ~{est['streaming_gb']:.1f} GB."
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
        # Per entry, like the pixel size above: unless the user TYPED a Z step
        # this session, every queued acquisition takes its own from its data.
        # The box is one value for the whole dialog, filled by auto-detect from
        # whichever queue item it looked at and restored from the last session,
        # so passing it on applied one acquisition's Z step to another: the
        # 2026-10-14 run of a 2.5 um acquisition went out at 1.0 um (output
        # squashed 2.5x in Z, registration binned Z twice as hard) with nobody
        # having touched the box -- most likely carried over from an ASLM
        # dataset stitched earlier.
        config.z_step_um = z_step if (z_step > 0 and self._z_step_user_set) else None
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
        config.blending_width_overlap_fraction = float(
            self._blend_width_combo.currentData()
        )
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
        config.intensity_adjustment = self._intensity_adjust_combo.currentData()
        config.skip_registration = self._skip_reg_cb.isChecked()
        config.registration_z_refine = self._z_refine_cb.isChecked()
        config.registration_z_snap_to_plane = self._z_snap_cb.isChecked()
        config.stitching_approach = self._approach_combo.currentData()
        config.verbose_alignment_log = self._verbose_align_cb.isChecked()
        config.registration_z_refine_range_um = float(self._z_refine_range_spin.value())
        config.registration_report_enabled = self._reg_report_cb.isChecked()
        config.border_qc_enabled = self._border_qc_cb.isChecked()
        config.border_qc_mode = self._border_qc_mode_combo.currentData() or "mip"
        config.registration_binning = self._registration_binning()
        config.registration_z_spacing_um = self._registration_z_spacing()
        config.registration_upsample_factor = int(
            self._reg_upsample_combo.currentData() or 0
        )
        config.global_opt_abs_tol = float(self._solve_tol_spin.value())
        config.registration_solve_transform = (
            self._solve_model_combo.currentData() or "translation"
        )
        config.package_ozx = self._ozx_cb.isChecked()
        config.tiff_pyramids = self._tiff_pyramids_cb.isChecked()
        config.streaming_mode = self._streaming_combo.currentData()
        config.scratch_dir = self._scratch_dir_edit.text().strip() or None
        chunk = self._chunk_size_combo.currentData()
        if chunk:
            config.output_chunksize = dict(chunk)
        # 0.0 is a real choice here ("whole output"), so `is not None` rather
        # than truthiness — `if region:` would silently drop it.
        region_gb = self._fusion_region_combo.currentData()
        if region_gb is not None:
            config.fusion_superblock_target_gb = float(region_gb)

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
    # Recorded in a configuration file but never applied here: geometry that
    # Discover re-measures, machine limits that do not transfer between boxes,
    # and pure provenance. Defined ONCE, in the pipeline, so this cannot drift
    # from what the writer actually emits.
    @property
    def _NONSHAREABLE_CONFIG_FIELDS(self):
        from flamingo_stitcher.pipeline import NON_APPLICABLE_CONFIG_FIELDS

        return NON_APPLICABLE_CONFIG_FIELDS

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

    def _on_save_configuration(self):
        """Write the current settings to a shareable configuration file.

        The same block a completed run records in stitch_metadata.json, so
        `Load Configuration` reads either without knowing the difference --
        but available before a run has finished, which is what makes a setup
        shareable while it is still being decided.
        """
        from flamingo_stitcher.pipeline import serialize_stitching_config

        try:
            config = self._build_config()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Couldn't build the configuration",
                f"Could not read the current settings:\n\n{e}",
            )
            return

        start_dir = (
            self._output_dir_edit.text().strip()
            or QSettings().value(_LAST_BROWSE_KEY, "", type=str)
            or ""
        )
        # Name the file after what it applies to, so a folder of saved setups
        # says which is which and an overwrite prompt is readable.
        default_name = f"stitching_configuration_{self._workflow_kind}.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Stitching Configuration",
            str(Path(start_dir) / default_name) if start_dir else default_name,
            "Stitching configuration (*.json);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        from datetime import datetime

        from flamingo_stitcher import __version__

        payload = {
            # Named so a human opening the file knows what it is, and so Load
            # can tell a configuration from a stitch_metadata.json at a glance.
            "kind": "flamingo-stitcher-configuration",
            "version": __version__,
            # What these settings apply to. "all-workflows" means every
            # processing value transfers as-is between the Multi-Acquisition and
            # Single Workflow tabs; "multi-view" carries rotation settings that
            # belong only to a multi-angle acquisition.
            "workflow": self._workflow_kind,
            "saved": datetime.now().isoformat(timespec="seconds"),
            "stitching_config": serialize_stitching_config(config),
        }
        try:
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Couldn't save the configuration",
                f"Could not write:\n{path}\n\n{e}",
            )
            return

        n = len(payload["stitching_config"])
        kind = self._workflow_kind
        self._log(
            f"Saved stitching configuration to {Path(path).name} "
            f"({n} settings, applies to: {kind})"
        )
        scope = (
            "These settings apply to any acquisition type — the Multi-"
            "Acquisition and Single Workflow tabs differ only in how tiles are "
            "found, not in how they are processed."
            if kind == "all-workflows"
            else "These settings include multi-angle rotation values, so they "
            "belong to a Multi-View acquisition. Loading them into another tab "
            "will apply everything EXCEPT the rotation settings."
        )
        QMessageBox.information(
            self,
            "Configuration saved",
            f"Wrote {n} setting(s) to:\n{Path(path).name}\n\n{scope}\n\n"
            "Load it on another machine with Load Configuration to reproduce "
            "this setup on different data.",
        )

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
        source_kind = data.get("workflow") if isinstance(data, dict) else None
        self._log(
            f"Loaded stitching configuration from {Path(path).name}"
            + (f" (saved for: {source_kind})" if source_kind else "")
        )
        self._log(f"  Applied {applied} setting(s) from the shared configuration.")
        if skipped:
            labels = {
                "pixel_size_um": "pixel size",
                "z_step_um": "Z spacing",
                "frame_width": "frame AOI",
                "frame_height": "frame AOI",
                "scratch_dir": "scratch location",
                "max_memory_gb": "memory ceiling",
                "preprocess_workers": "worker counts",
                "fuse_workers": "worker counts",
                "destripe_workers": "worker counts",
                "zarr_use_tensorstore": "zarr writer backend",
                "scope_profile_source": "which scope profile the run used",
            }
            pretty = sorted({labels.get(s, s) for s in skipped})
            self._log(
                "  Left unchanged (this acquisition's geometry and this "
                "machine's limits): " + ", ".join(pretty) + "."
            )
        psf = getattr(self, "_psf_path_unavailable", "")
        if psf:
            self._log(
                f"  WARNING: the configuration names a PSF file that is not on "
                f"this machine, so deconvolution keeps its current PSF: {psf}"
            )
        refused = getattr(self, "_refused_multiview_fields", [])
        if refused:
            self._log(
                "  Skipped multi-angle settings from a Multi-View "
                f"configuration ({', '.join(sorted(refused))}) — this tab "
                "stitches single-angle acquisitions. Everything else was "
                "applied."
            )

        detail = (
            f"Applied {applied} setting(s) from:\n{Path(path).name}\n\n"
            "This includes the processing options, destripe tuning, "
            "deconvolution parameters and registration thresholds — everything "
            "that shaped the original run.\n\n"
            "Left as they are: pixel size, Z spacing and frame AOI (Discover "
            "measures those from your own acquisition), the output location, "
            "and this machine's memory and worker limits."
        )
        if psf:
            detail += (
                f"\n\nThe PSF file it names is not on this machine, so "
                f"deconvolution kept its current PSF:\n{psf}"
            )
        if refused:
            detail += (
                "\n\nThis file came from a Multi-View acquisition. Its "
                "multi-angle settings were skipped because this tab stitches "
                "single-angle data; everything else was applied."
            )
        QMessageBox.information(self, "Configuration loaded", detail)

    def _apply_stitching_config(self, cfg: dict):
        """Apply a serialized StitchingConfig dict to the current widgets.

        Returns ``(applied_count, skipped_field_names)``. Only processing
        settings are applied; :attr:`_NONSHAREABLE_CONFIG_FIELDS` are skipped
        so a shared config never clobbers this acquisition's geometry or the
        local output location. Mirrors :meth:`_build_config` in reverse.
        """
        applied = 0
        skipped = set()
        non_shareable = self._NONSHAREABLE_CONFIG_FIELDS
        # Every field the WIDGET pass below consumes. Recorded as it goes rather
        # than kept as a second list: whatever is left over is what no control
        # owns, and that is exactly the set that has to be carried into the run
        # as an override. A hand-maintained list here would drift the same way
        # the shareable whitelist did.
        handled = set()

        def has(name):
            if name in cfg and name not in non_shareable:
                handled.add(name)
                return True
            return False

        # Note which non-applicable fields the file carried, for the summary.
        for name in non_shareable:
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
        handled.update({"tiff_compression", "zarr_compression"} & set(cfg))
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
            ("blending_width_overlap_fraction", self._blend_width_combo),
            ("intensity_adjustment", self._intensity_adjust_combo),
            ("downsample_xy", self._downsample_xy_combo),
            ("downsample_z", self._downsample_z_combo),
            ("streaming_mode", self._streaming_combo),
            ("output_chunksize", self._chunk_size_combo),
            ("fusion_superblock_target_gb", self._fusion_region_combo),
            ("border_qc_mode", self._border_qc_mode_combo),
            ("registration_upsample_factor", self._reg_upsample_combo),
            ("registration_solve_transform", self._solve_model_combo),
        ]
        for name, combo in combo_fields:
            if has(name):
                applied += self._set_combo_by_data(combo, cfg[name])

        # Not a combo any more: one per-axis dict drives two controls.
        if has("registration_binning"):
            self._set_registration_binning(cfg["registration_binning"])
            applied += 1
        if cfg.get("global_opt_abs_tol") is not None:
            try:
                self._solve_tol_spin.setValue(float(cfg["global_opt_abs_tol"]))
                applied += 1
            except (TypeError, ValueError):
                pass
        # After the dict, so a file that has both keeps its explicit spacing
        # rather than the dict's older "z" reading.
        if cfg.get("registration_z_spacing_um") is not None:
            self._set_registration_z_spacing(cfg["registration_z_spacing_um"])
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
            ("registration_z_snap_to_plane", self._z_snap_cb),
            ("verbose_alignment_log", self._verbose_align_cb),
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
        if has("stitching_approach"):
            applied += self._set_combo_by_data(
                self._approach_combo, str(cfg["stitching_approach"])
            )

        if has("split_illumination") and bool(cfg["split_illumination"]):
            applied += self._set_combo_by_data(self._fusion_combo, "separate")

        # Spin-box value fields.
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

        # Whatever is left has no control in this dialog -- destripe tuning
        # (sigma, wavelet, level), the deconvolution parameters, the border-QC
        # numbers, the registration thresholds that live in the Options tab.
        # Those are most of a real processing setup, and dropping them is why
        # "Load Configuration" only ever reproduced half a run. Carry them and
        # apply them to the config the worker actually gets.
        overrides = {}
        self._refused_multiview_fields = []
        multiview_here = self._workflow_kind == "multi-view"
        for name, value in cfg.items():
            if name in handled or name in non_shareable:
                continue
            if name == "deconvolution_psf_path":
                continue  # handled below: a path is only worth taking if it exists
            if not multiview_here and name in self._MULTIVIEW_ONLY_FIELDS:
                # Multi-angle rotation settings on a single-angle tab. Applying
                # multiview_fusion here would quietly change what the run
                # PRODUCES, not merely how it is tuned, so it is refused and
                # said out loud. Checked by field rather than by the file's
                # label: a stitch_metadata.json from a multi-view run carries no
                # label at all.
                if value not in (None, False):
                    self._refused_multiview_fields.append(name)
                continue
            overrides[name] = value

        # A PSF path from another machine is meaningless unless the PSF came
        # with it. Take it when the file is really there, say so when it is not
        # -- silently keeping a dead path would fail the run much later.
        psf = cfg.get("deconvolution_psf_path")
        self._psf_path_unavailable = ""
        if psf:
            try:
                exists = Path(str(psf)).is_file()
            except Exception:
                exists = False
            if exists:
                overrides["deconvolution_psf_path"] = psf
                applied += 1
            else:
                self._psf_path_unavailable = str(psf)

        self._loaded_config_overrides = overrides
        applied += len(overrides)

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

    def _confirm_illumination_low_side(self, pending, config) -> bool:
        """Block the run if Split/Blend has no low-end side to work from.

        Without it these fall back to Max, so the run would spend its whole
        length producing Max output under a Split label — the operator cannot
        tell from the result that they did not get what they asked for. The
        pipeline refuses too, but refusing here costs seconds instead of the
        minutes it takes to reach the first tile, and this is the only place
        that can actually SHOW the user where the setting lives.

        Answering "which side?" is the part a person genuinely cannot guess, so
        the message names the way to find out rather than only the way to set
        it.
        """
        method = getattr(config, "illumination_fusion", "max")
        if method not in ("split", "blend"):
            return True

        from flamingo_stitcher import scope_profiles

        unset = {}  # scope name -> objective, for the ones still missing it
        for it in pending or []:
            path = it.get("path")
            if not path:
                continue
            try:
                # (microscope, objective, values, source_key)
                scope, _obj, values, _src = scope_profiles.resolve_for_acquisition(
                    Path(path)
                )
            except Exception:  # noqa: BLE001 - resolve is best-effort
                scope, values = None, {}
            if int((values or {}).get("illumination_low_side", -1)) >= 0:
                continue
            unset[scope or "(unnamed microscope)"] = None

        if not unset:
            return True

        label = dict(
            split="Split (each sheet owns its half)",
            blend="Blend (split with a soft handover)",
        )[method]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Which illumination side lights the left?")
        box.setText(
            f"Illumination fusion is set to “{label}”, but no illumination "
            f"side has been chosen for: {', '.join(sorted(unset))}.\n\n"
            "Split and Blend give each light sheet its own half of the frame, "
            "so they need to know which side lights the LEFT half. Without it "
            "the run would quietly fall back to Max."
        )
        box.setInformativeText(
            "To set it: the Options tab → pick this microscope → "
            "“Illumination side lighting the LOW end of the frame” → Save. "
            "It is remembered per microscope and objective, so this is once "
            "per instrument.\n\n"
            "To find out which side it is: run once with Illumination fusion "
            "set to “Separate (keep light paths)”. That writes each sheet as "
            "its own channel, so you can flip between them and see which one "
            "is sharp on the left. That side's I number is the value.\n\n"
            "Or choose “Content (weight by local detail)” instead — it needs "
            "no geometry at all, and on a scattering sample it is the mode "
            "that rejects out-of-focus blur."
        )
        open_btn = box.addButton("Open Options…", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is open_btn:
            self._open_options_tab()
        return False

    def _open_options_tab(self) -> None:
        """Bring the Options tab forward, when this dialog lives in the tabbed
        standalone app. A no-op inside Py2Flamingo, where the dialog is not in
        that tab widget — the message has already named the path either way."""
        try:
            from PyQt5.QtWidgets import QTabWidget

            w = self.parentWidget()
            while w is not None:
                if isinstance(w, QTabWidget):
                    for i in range(w.count()):
                        if w.tabText(i) == "Options":
                            w.setCurrentIndex(i)
                            return
                w = w.parentWidget()
            # Standalone: the tab widget is a sibling, reachable from the window.
            win = self.window()
            for tabs in win.findChildren(QTabWidget):
                for i in range(tabs.count()):
                    if tabs.tabText(i) == "Options":
                        tabs.setCurrentIndex(i)
                        return
        except Exception:  # noqa: BLE001 - convenience only
            pass

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

        # Pre-flight: Split/Blend cannot run without knowing which sheet
        # lights the left of the frame, and silently becoming Max is the one
        # outcome the operator cannot see in the output.
        if not self._confirm_illumination_low_side(pending, config):
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

        # Past every pre-flight: this run is definitely starting.
        self._focus_log_for_run()

        self._log_text.clear()
        self._reset_step_progress()
        n_pending = len(pending)
        self._log(f"Starting batch stitching: {n_pending} directories\n")

        # Store batch state
        self._batch_running = True
        self._mem_wait_popup_shown = False
        self._set_flamingos_marching(True)
        self.run_state_changed.emit(True)
        self._batch_config = config
        self._batch_channels = self._parse_channels()
        self._batch_results = []
        self._last_success_output = None
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
                from flamingo_stitcher.pipeline import IMARIS_WRITER_RAM_FRACTION_ASSUMED

                fmt_overhead_gb = est["output_gb"] * IMARIS_WRITER_RAM_FRACTION_ASSUMED
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

    @staticmethod
    def _coerce_config_value(current, value):
        """JSON round-trips lose types the dataclass cares about.

        `serialize_stitching_config` stringifies dict keys so json.dumps will
        accept them, and JSON has no tuples. Rebuild both shapes from whatever
        the field currently holds -- getting this wrong is silent: an int-keyed
        per-channel threshold dict with string keys simply never matches a
        channel, and nothing raises.
        """
        if isinstance(current, dict) and isinstance(value, dict):
            int_keyed = bool(current) and all(
                isinstance(k, int) for k in current
            )
            if int_keyed:
                out = {}
                for k, v in value.items():
                    try:
                        out[int(k)] = v
                    except (TypeError, ValueError):
                        out[k] = v
                return out
            return dict(value)
        if isinstance(current, tuple) and isinstance(value, list):
            return tuple(value)
        return value

    def _apply_loaded_overrides(self, item_config):
        """Apply loaded settings that no control in this dialog owns.

        Destripe tuning (sigma, wavelet, level, direction), the deconvolution
        parameters, depth attenuation, the border-QC numbers, the registration
        thresholds that live in the Options tab -- none of these has a widget
        here, so the widget pass in `_apply_stitching_config` cannot carry them.
        They are most of a real processing setup, and until now a loaded
        configuration silently dropped every one.
        """
        overrides = getattr(self, "_loaded_config_overrides", None)
        if not overrides:
            return item_config

        applied, unknown = [], []
        for name, value in overrides.items():
            if not hasattr(item_config, name):
                unknown.append(name)  # a file from a newer build
                continue
            try:
                setattr(
                    item_config,
                    name,
                    self._coerce_config_value(getattr(item_config, name), value),
                )
                applied.append(name)
            except Exception:
                unknown.append(name)
        if applied:
            self._log(
                f"  Loaded configuration also set {len(applied)} setting(s) with "
                f"no control in this dialog: {', '.join(sorted(applied))}"
            )
        if unknown:
            self._log(
                f"  Loaded configuration named {len(unknown)} setting(s) this "
                f"build does not have (ignored): {', '.join(sorted(unknown))}"
            )
        return item_config

    def _apply_scope_profile(self, item_config, acq_path):
        """Swap in this acquisition's microscope + objective stitching options.

        Unlike the destripe preset this is never fatal to get wrong — every
        value has a working default — so an unknown scope is a note, not a
        warning. What it must not do is stay silent: a run whose thresholds came
        from a saved profile and a run using the defaults can behave very
        differently, and the log is where that gets answered later.
        """
        from dataclasses import replace

        from flamingo_stitcher import scope_profiles

        try:
            scope, objective, values, source = (
                scope_profiles.resolve_for_acquisition(acq_path)
            )
        except Exception as e:  # noqa: BLE001 - never block a run on this
            self._log(f"⚠ Could not resolve stitching options by microscope: {e}")
            return item_config

        # Remember the instrument so the Options tab can offer it. The name has
        # to match the acquisition's own "Microscope name" exactly or a saved
        # profile silently never applies, so having it typed FOR the user is
        # the point. Best-effort: a dropdown convenience must never affect a run.
        if scope_profiles.remember_microscope(scope):
            self._log(
                f"  First run from microscope '{scope}' — it is now listed in "
                f"the Options tab, where its stitching settings can be saved."
            )

        applied, message = scope_profiles.describe_resolution(
            scope, objective, values, source
        )
        if not applied:
            self._log(message)
            return item_config

        item_config = replace(item_config, scope_profile_source=source)
        changed = scope_profiles.apply_profile(item_config, values)
        self._log(message)
        for line in changed:
            self._log(f"   {line}")
        return item_config

    def _destripe_settings_for_selection(self):
        """``(params, direction, note)`` the selected queue row will run with.

        A run swaps in the microscope's saved destripe settings
        (``_apply_destripe_preset``), so the tuning dialogs must open on THOSE,
        not on the last values edited in this window. Showing the window's own
        values let run 32 (2026-10-27) run with Crossover 900 while the dialog
        read 10.
        """
        params = dict(self._destripe_params)
        direction = str(self._destripe_dir_combo.currentData() or "auto")
        if not self._queue:
            return params, direction, (
                "No acquisition in the queue. These values are used for a "
                "microscope with no saved destripe settings; add an "
                "acquisition first to see and change its microscope's settings."
            )
        rows = sorted(set(idx.row() for idx in self._queue_table.selectedIndexes()))
        row = rows[0] if rows else 0
        acq_path = self._queue[row].get("path") if 0 <= row < len(self._queue) else None
        if not acq_path:
            return params, direction, ""
        from flamingo_stitcher.destripe_presets import resolve_for_acquisition

        try:
            scope, preset = resolve_for_acquisition(acq_path)
        except Exception as e:  # noqa: BLE001 - fall back to this window's values
            self._logger.debug(f"Destripe preset lookup failed: {e!r}")
            return params, direction, ""
        if preset:
            return dict(preset["params"]), preset["direction"], (
                f"Saved for microscope '{scope}'. Every run of an acquisition "
                "from this microscope uses these values; OK saves your changes "
                "for it."
            )
        if scope:
            return params, direction, (
                f"Nothing saved yet for microscope '{scope}'. OK saves these "
                "values for it."
            )
        return params, direction, (
            "This acquisition has no readable microscope name, so these values "
            "cannot be saved for its microscope."
        )

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

        # Registration thresholds are per-MICROSCOPE and per-OBJECTIVE for the
        # same reason: how much of a mosaic can register, and how far the stage
        # can plausibly be wrong, are facts about an instrument. Resolved per
        # item so a queue mixing rigs gets each one's own tuning.
        item_config = self._apply_scope_profile(item_config, item["path"])

        # A configuration the user loaded is an explicit choice for THIS run, so
        # it wins over the per-microscope destripe preset and scope profile
        # resolved above. That is the precedence scope_profiles documents:
        # defaults -> YAML -> profile -> an explicit control for this run.
        item_config = self._apply_loaded_overrides(item_config)

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
        # Outside the queue-index guard: this is what the run wrote, and the
        # summary needs it whether or not the item is still addressable.
        self._last_success_output = output_path
        # Name the run's own folder, not the parent that holds every stitch
        # written here -- that read "Completed: ScratchTest" on every run.
        done_name = self._run_output_folder(output_path).name
        self._log(f"\n\u2713 Completed: {done_name}")

        from flamingo_stitcher.gui._compat import get_notification_service

        svc = get_notification_service(self)
        if svc is not None:
            acq_name = done_name
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

    @staticmethod
    def _run_output_folder(reported):
        """The run's own folder. Delegates to the Qt-free helper so the rule is
        testable without PyQt5 installed."""
        from flamingo_stitcher.pipeline import run_output_folder

        return run_output_folder(reported)

    def _last_output_folder(self):
        """Folder the last successful item wrote to, or None if none succeeded.

        Both summary buttons key off this, and it is the path the worker
        REPORTED -- never rebuilt from the acquisition name.

        Rebuilding gave ``<output dir>/<acq>_stitched``, which is only the
        DEFAULT location. When the user answers "New folder" to a duplicate, the
        run writes to a uniquified folder beside it, and the default name still
        names the older stitch that caused the prompt. "Open Output Folder" then
        opened the previous result, and an existence check cannot catch that --
        the default folder is there, it is just the wrong one. "Load Latest into
        Sample View" was worse: it loaded the older stitch as if it were the run
        that had just finished, and nothing on screen said so.

        Lives in its own method so a test can exercise the choice. Inline in the
        summary it sat behind a modal, and a test could only re-implement it --
        which agrees with whatever the shipped code does.
        """
        if not self._last_success_output:
            return None
        return str(self._run_output_folder(self._last_success_output))

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
        self.run_state_changed.emit(False)
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

        last_success_path = self._last_output_folder()

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
        has_tiff = fmt in ("ome-tiff", "both")
        # .ozx is only packaged from the sharded OME-Zarr writer.
        has_ozx = fmt in ("ome-zarr-sharded", "both")
        self._ozx_cb.setEnabled(has_ozx)
        if not has_ozx:
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
                "Corrects uneven brightness across each tile's field of view,\n"
                "estimated from the tiles themselves (no calibration images\n"
                "needed). Turn it on when the same bright or dark pattern\n"
                "repeats in every tile, or tiles meet at brightness seams."
            )
        else:
            self._flat_field_cb.setChecked(False)
            self._flat_field_cb.setEnabled(False)
            self._flat_field_cb.setToolTip(
                "Flat-field correction is not installed yet.\n"
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
                "Removes the stripes and shadows light-sheet images often show\n"
                "along the sheet direction. Use it when stripes are visible in\n"
                "the tiles. Slow: filters every plane at full resolution before\n"
                "downsampling (check 'Fast' for test runs). Set the direction\n"
                "with the 'Dir: Auto' list, tune with 'Settings…', and check the\n"
                "result with the 'Preview…' button beside it."
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
                "Destriping is unavailable: the destripe component failed to load.\n"
                f"    {type(pystripe_err).__name__}: {pystripe_err}\n"
                "Please report this message."
            )

        # --- Deconvolution (pycudadecon or RedLionfish) ---
        from flamingo_stitcher import deconvolution as _decon

        if _decon.is_available():
            self._deconv_cb.setEnabled(True)
            self._deconv_cb.setToolTip(
                "Sharpens each tile on the GPU before stitching (needs an NVIDIA\n"
                "or other OpenCL graphics card). Slow, and unless the stitching\n"
                "config names a measured PSF file it uses a calculated blur\n"
                "model. Try it on a small test run first and compare with the\n"
                "result without it."
            )
        else:
            self._deconv_cb.setChecked(False)
            self._deconv_cb.setEnabled(False)
            self._deconv_cb.setToolTip(
                "Deconvolution needs a GPU library that is not installed in\n"
                "this build. See 'For developers / Python users' in the README."
            )

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
                        "Combines the two light-sheet sides by image content.\n"
                        "Needs a GPU. Experimental."
                    )
                else:
                    item.setToolTip(
                        "Not available: Leonardo fusion needs its own GPU setup, which\n"
                        "'Set up flat-field…' does not install."
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
        s.setValue("blend_width_overlap", self._blend_width_combo.currentData())
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
        s.setValue(
            "intensity_adjustment", self._intensity_adjust_combo.currentData()
        )
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
        s.setValue("reg_z_spacing_um", self._registration_z_spacing())
        s.setValue("reg_upsample", self._reg_upsample_combo.currentData())
        s.setValue("solve_tolerance_px", self._solve_tol_spin.value())
        s.setValue("solve_transform", self._solve_model_combo.currentData())
        s.setValue("z_refine", self._z_refine_cb.isChecked())
        s.setValue("z_snap_to_plane", self._z_snap_cb.isChecked())
        s.setValue("stitching_approach", self._approach_combo.currentData())
        s.setValue("verbose_alignment", self._verbose_align_cb.isChecked())
        s.setValue(
            "fusion_region_gb", float(self._fusion_region_combo.currentData())
        )
        s.setValue("z_refine_range_um", self._z_refine_range_spin.value())
        s.setValue("registration_report", self._reg_report_cb.isChecked())
        s.setValue("proc_options_expanded", self._proc_toggle.isChecked())
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
        self._set_combo_by_data(
            self._blend_width_combo, s.value("blend_width_overlap", 1.0, type=float)
        )

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
        _ia_idx = self._intensity_adjust_combo.findData(
            s.value("intensity_adjustment", "off", type=str)
        )
        if _ia_idx >= 0:
            self._intensity_adjust_combo.setCurrentIndex(_ia_idx)

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
        self._set_registration_binning({"y": _rb_xy, "x": _rb_xy})
        self._restore_reg_z_spacing(s)
        self._set_combo_by_data(
            self._reg_upsample_combo, s.value("reg_upsample", 0, type=int)
        )
        self._solve_tol_spin.setValue(s.value("solve_tolerance_px", 3.5, type=float))
        self._set_combo_by_data(
            self._solve_model_combo, s.value("solve_transform", "translation", type=str)
        )
        self._z_refine_cb.setChecked(s.value("z_refine", False, type=bool))
        # Default TRUE: a run that has never seen this setting must not
        # silently keep sub-plane Z shifts.
        self._z_snap_cb.setChecked(s.value("z_snap_to_plane", True, type=bool))
        self._set_combo_by_data(
            self._approach_combo, s.value("stitching_approach", "default", type=str)
        )
        self._verbose_align_cb.setChecked(
            s.value("verbose_alignment", True, type=bool)
        )
        self._set_combo_by_data(
            self._fusion_region_combo, s.value("fusion_region_gb", 4.0, type=float)
        )
        self._z_refine_range_spin.setValue(
            s.value("z_refine_range_um", 40.0, type=float)
        )
        self._reg_report_cb.setChecked(
            s.value("registration_report", True, type=bool)
        )

        proc_expanded = s.value("proc_options_expanded", False, type=bool)
        self._proc_toggle.setChecked(proc_expanded)
        # Always closed for a new run: the settings get the window until
        # there is output worth reading. _focus_log_for_run opens it.
        self._log_toggle.setChecked(False)

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
        self.run_state_changed.emit(False)
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
        s.setValue("blend_width_overlap", self._blend_width_combo.currentData())
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
        s.setValue(
            "intensity_adjustment", self._intensity_adjust_combo.currentData()
        )
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
        s.setValue("reg_z_spacing_um", self._registration_z_spacing())
        s.setValue("reg_upsample", self._reg_upsample_combo.currentData())
        s.setValue("solve_tolerance_px", self._solve_tol_spin.value())
        s.setValue("solve_transform", self._solve_model_combo.currentData())
        s.setValue("z_refine", self._z_refine_cb.isChecked())
        s.setValue("z_snap_to_plane", self._z_snap_cb.isChecked())
        s.setValue("stitching_approach", self._approach_combo.currentData())
        s.setValue("verbose_alignment", self._verbose_align_cb.isChecked())
        s.setValue(
            "fusion_region_gb", float(self._fusion_region_combo.currentData())
        )
        s.setValue("z_refine_range_um", self._z_refine_range_spin.value())
        s.setValue("registration_report", self._reg_report_cb.isChecked())
        s.setValue("proc_options_expanded", self._proc_toggle.isChecked())
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
        self._set_combo_by_data(
            self._blend_width_combo, s.value("blend_width_overlap", 1.0, type=float)
        )

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
        _ia_idx = self._intensity_adjust_combo.findData(
            s.value("intensity_adjustment", "off", type=str)
        )
        if _ia_idx >= 0:
            self._intensity_adjust_combo.setCurrentIndex(_ia_idx)

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
        self._set_registration_binning({"y": _rb_xy, "x": _rb_xy})
        self._restore_reg_z_spacing(s)
        self._set_combo_by_data(
            self._reg_upsample_combo, s.value("reg_upsample", 0, type=int)
        )
        self._solve_tol_spin.setValue(s.value("solve_tolerance_px", 3.5, type=float))
        self._set_combo_by_data(
            self._solve_model_combo, s.value("solve_transform", "translation", type=str)
        )
        self._z_refine_cb.setChecked(s.value("z_refine", False, type=bool))
        # Default TRUE: a run that has never seen this setting must not
        # silently keep sub-plane Z shifts.
        self._z_snap_cb.setChecked(s.value("z_snap_to_plane", True, type=bool))
        self._set_combo_by_data(
            self._approach_combo, s.value("stitching_approach", "default", type=str)
        )
        self._verbose_align_cb.setChecked(
            s.value("verbose_alignment", True, type=bool)
        )
        self._set_combo_by_data(
            self._fusion_region_combo, s.value("fusion_region_gb", 4.0, type=float)
        )
        self._z_refine_range_spin.setValue(
            s.value("z_refine_range_um", 40.0, type=float)
        )
        self._reg_report_cb.setChecked(
            s.value("registration_report", True, type=bool)
        )

        proc_expanded = s.value("proc_options_expanded", False, type=bool)
        self._proc_toggle.setChecked(proc_expanded)
        # Always closed for a new run: the settings get the window until
        # there is output worth reading. _focus_log_for_run opens it.
        self._log_toggle.setChecked(False)

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

    # The one tab whose configuration is NOT portable: _build_config forces
    # multiview_fusion on and adds the rotation sign/centre, which describe a
    # multi-angle acquisition and would be wrong anywhere else.
    _workflow_kind = "multi-view"

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
            "angles into one volume. For normal single-angle acquisitions use "
            "the ‘Single Workflow’ or ‘Multi-Acquisition’ tab instead."
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
        self._mv_center_combo.setToolTip(
            "The point the views rotate about, in stage coordinates.\n"
            "'Auto (tile centroid)': the average X and middle Z of all tiles.\n"
            "Choose 'Manual (X, Z)' and type it in mm if the views don't line\n"
            "up with Auto."
        )
        self._mv_center_combo.currentIndexChanged.connect(
            self._mv_update_center_enabled
        )
        center_row.addWidget(self._mv_center_combo)
        center_row.addWidget(QLabel("X"))
        self._mv_cx_spin = QDoubleSpinBox()
        self._mv_cx_spin.setRange(-1000.0, 1000.0)
        self._mv_cx_spin.setDecimals(3)
        self._mv_cx_spin.setSuffix(" mm")
        self._mv_cx_spin.setToolTip(
            "X position of the rotation axis, in stage millimeters.\n"
            "Used only with 'Manual (X, Z)'."
        )
        center_row.addWidget(self._mv_cx_spin)
        center_row.addWidget(QLabel("Z"))
        self._mv_cz_spin = QDoubleSpinBox()
        self._mv_cz_spin.setRange(-1000.0, 1000.0)
        self._mv_cz_spin.setDecimals(3)
        self._mv_cz_spin.setSuffix(" mm")
        self._mv_cz_spin.setToolTip(
            "Z position of the rotation axis, in stage millimeters.\n"
            "Used only with 'Manual (X, Z)'."
        )
        center_row.addWidget(self._mv_cz_spin)
        center_row.addStretch()
        v.addLayout(center_row)

        sign_row = QHBoxLayout()
        sign_row.addWidget(QLabel("Rotation sign:"))
        self._mv_sign_combo = QComboBox()
        self._mv_sign_combo.addItem("+1", 1.0)
        self._mv_sign_combo.addItem("−1", -1.0)
        self._mv_sign_combo.setToolTip(
            "Which way a stage angle turns the views when they are fused.\n"
            "Leave at '+1'. If a two-angle test comes out mirrored or\n"
            "misaligned, try '−1'."
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
