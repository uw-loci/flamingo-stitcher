"""The 0.14.0 layout of the stitching tabs.

Basics at the top (what every run needs), then two tabs, 'Image quality' and
'Registration and overlap', so only one set of options shows at a time, then
the 'This run' panel. Spec: the mockups in claude-reports/design/
gui-rework-mockups (agreed with the user 2026-11-05).

``_setup_ui`` still builds every control, with its tooltip and wiring, in the
order it always has; :meth:`ReworkedLayoutMixin._arrange_reworked_layout` then
moves the controls into this layout and removes the old group boxes. Keeping
construction and arrangement apart means no control loses a connection in the
move.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from PyQt5.QtCore import QEvent, QObject, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

SLOW_LEGEND = "✱ = significantly increases processing time"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def take(widget: QWidget) -> QWidget:
    """Remove ``widget`` from whatever layout holds it, without a Qt warning."""
    parent = widget.parentWidget()
    if parent is not None and parent.layout() is not None:
        _remove_from(parent.layout(), widget)
    return widget


def _remove_from(layout: QLayout, widget: QWidget) -> bool:
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is widget:
            layout.takeAt(i)
            return True
        if item.layout() is not None and _remove_from(item.layout(), widget):
            return True
    return False


def fit_to_contents(widget: QWidget, chars: Optional[int] = None) -> QWidget:
    """Make an input as wide as what it holds, no wider.

    Lists are as wide as their longest choice; spin boxes and text boxes get
    room for about ``chars`` characters.
    """
    if isinstance(widget, QComboBox):
        widget.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    elif chars is not None:
        fm = widget.fontMetrics()
        extra = 36 if isinstance(widget, QAbstractSpinBox) else 14
        widget.setFixedWidth(fm.horizontalAdvance("0" * chars) + extra)
    return widget


def row(*items, stretch: bool = True, spacing: int = 8) -> QHBoxLayout:
    """A line of controls; a str becomes a label, an int a gap in pixels."""
    h = QHBoxLayout()
    h.setSpacing(spacing)
    h.setContentsMargins(0, 0, 0, 0)
    for it in items:
        if isinstance(it, str):
            h.addWidget(QLabel(it))
        elif isinstance(it, int):
            h.addSpacing(it)
        elif isinstance(it, QLayout):
            h.addLayout(it)
        elif it is not None:
            h.addWidget(it)
    if stretch:
        h.addStretch(1)
    return h


def heading(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet("font-weight: bold;")
    return lab


class SegmentedChoice(QWidget):
    """A row of large buttons standing in for a QComboBox.

    The combo stays the source of truth (every handler, saved setting and
    configuration file still reads it); clicking a button selects that entry,
    and the buttons follow the combo's selection and enabled state.
    """

    def __init__(self, combo: QComboBox, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._combo = combo
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons: List[QToolButton] = []
        for i in range(combo.count()):
            b = QToolButton()
            b.setText(combo.itemText(i))
            b.setCheckable(True)
            b.setToolTip(combo.toolTip())
            b.setProperty("segmented", True)
            b.setStyleSheet(
                "QToolButton { font-size: 12pt; min-width: 44px; min-height: 30px;"
                " border: 1px solid #9aa7b8; background: #fbfbfb; }"
                "QToolButton:checked { background: #2f6db5; color: white;"
                " border-color: #2f6db5; font-weight: bold; }"
                "QToolButton:disabled { color: #aaa; background: #f0f0f0;"
                " border-color: #ccc; }"
            )
            self._group.addButton(b, i)
            lay.addWidget(b)
            self.buttons.append(b)
        self._group.buttonClicked.connect(
            lambda b: combo.setCurrentIndex(self._group.id(b))
        )
        combo.currentIndexChanged.connect(self._sync)
        combo.installEventFilter(self)
        self.setToolTip(combo.toolTip())
        self._sync()
        self.setEnabled(combo.isEnabled())

    def _sync(self, *_):
        i = self._combo.currentIndex()
        if 0 <= i < len(self.buttons):
            self.buttons[i].setChecked(True)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self._combo and event.type() == QEvent.EnabledChange:
            self.setEnabled(self._combo.isEnabled())
        return False


# --------------------------------------------------------------------------- #
# The arrangement
# --------------------------------------------------------------------------- #


class ReworkedLayoutMixin:
    """Arranges the controls ``_setup_ui`` built into the 0.14.0 layout."""

    def _arrange_reworked_layout(self, layout, content_layout, extras_count: int,
                                 old_containers: Sequence[QWidget]) -> None:
        """``layout``: the tab's main layout; ``content_layout``: the scroll area's.

        ``extras_count`` items at the top of ``content_layout`` came from
        ``_add_dialog_extras`` (Multi-View) and stay first.
        """
        # ---- Output row: folder, then size and free space on the same line ----
        out_row = self._output_dir_edit.parentWidget().layout()
        take(self._output_info_label)
        for i in range(out_row.count()):
            item = out_row.itemAt(i)
            if item.layout() is not None and _contains(item.layout(), self._output_dir_edit):
                item.layout().addWidget(self._output_info_label)
                break

        # ---- Acquisitions ----
        queue_group = self._queue_table.parentWidget()
        if isinstance(queue_group, QGroupBox):
            queue_group.setTitle("Acquisitions")

        # A line under the queue says what a settings change will do (or why
        # the settings are grayed out); empty until there is something to say.
        self._selection_note = QLabel("")
        self._selection_note.setWordWrap(True)
        self._selection_note.setStyleSheet("color: #1d3f6e; font-size: 9pt;")
        self._selection_note.hide()

        basics = self._build_basics_group()
        tabs = self._build_settings_tabs()
        self._run_summary = self._build_run_summary_panel()

        # Scroll area: Multi-View extras, Acquisitions, note, Basics, tabs, panel.
        keep = [content_layout.takeAt(0) for _ in range(extras_count)]
        while content_layout.count():
            item = content_layout.takeAt(0)
            w = item.widget()
            if w is not None and w is not queue_group:
                w.hide()
        for item in keep:
            if item.widget() is not None:
                content_layout.addWidget(item.widget())
            elif item.layout() is not None:
                content_layout.addLayout(item.layout())
        content_layout.addWidget(queue_group)
        content_layout.addWidget(self._selection_note)
        content_layout.addWidget(basics)
        content_layout.addWidget(tabs)
        content_layout.addWidget(self._run_summary)
        content_layout.addStretch(1)

        # Every setting in Basics is locked as a unit during a run; the tab
        # pages are locked page by page so the tab bar still switches.
        self._config_container = basics
        self._proc_widget = tabs

        for w in old_containers:
            w.hide()
            w.deleteLater()

        # ---- Buttons: Run, Cancel, legend, Help ----
        take(self._setup_env_btn)
        self._setup_env_btn.hide()  # lives on the Options tab now
        btn_row = self._run_btn.parentWidget().layout()
        for i in range(btn_row.count()):
            item = btn_row.itemAt(i)
            if item.layout() is not None and _contains(item.layout(), self._run_btn):
                lay = item.layout()
                legend = QLabel(SLOW_LEGEND)
                legend.setStyleSheet("color: #c25e00; font-style: italic; font-size: 11px;")
                idx = lay.indexOf(self._help_btn)
                lay.insertWidget(idx, legend)
                break

        self._wire_run_summary()

    # ---- Basics -------------------------------------------------------------

    def _build_basics_group(self) -> QGroupBox:
        b = QGroupBox("Basics (every run needs these)")
        g = QVBoxLayout(b)
        g.setSpacing(6)

        big = QLabel("Downsample")
        f = big.font()
        f.setPointSize(f.pointSize() + 4)
        f.setBold(True)
        big.setFont(f)
        self._downsample_xy_buttons = SegmentedChoice(take(self._downsample_xy_combo))
        self._downsample_z_buttons = SegmentedChoice(take(self._downsample_z_combo))
        self._downsample_xy_combo.hide()
        self._downsample_z_combo.hide()
        holder = QWidget()  # the combos still need a parent widget
        hl = QHBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self._downsample_xy_combo)
        hl.addWidget(self._downsample_z_combo)
        holder.hide()
        g.addLayout(row(big, 12, "XY", self._downsample_xy_buttons, 18, "Z",
                        self._downsample_z_buttons, holder))

        # What the choice does to the data: voxel size, memory, size and time.
        impact = QFrame()
        impact.setObjectName("downsampleImpact")
        impact.setStyleSheet(
            "#downsampleImpact { background: #eef3fa; border: 1px solid #c9d8ec;"
            " border-radius: 4px; }"
        )
        il = QVBoxLayout(impact)
        il.setContentsMargins(10, 5, 10, 5)
        il.setSpacing(2)
        for w in (self._voxel_readout_label, self._memory_label, self._time_label):
            take(w)
        il.addLayout(row(self._voxel_readout_label, self._time_label))
        il.addWidget(self._memory_label)
        g.addWidget(impact)

        # About the image
        fit_to_contents(self._pixel_size_spin, 7)
        fit_to_contents(self._z_step_spin, 6)
        fit_to_contents(self._frame_size_combo)
        fit_to_contents(self._channels_edit, 14)
        self._image_hint.setWordWrap(True)
        hint_row = row(heading("About the image"), 12, stretch=False)
        hint_row.addWidget(take(self._image_hint), 1)
        g.addLayout(hint_row)
        g.addLayout(row(
            "Pixel size (µm):", take(self._pixel_size_spin), 18,
            "Z step (µm):", take(self._z_step_spin), 18,
            "Frame (AOI):", take(self._frame_size_combo), 18,
            "Channels:", take(self._channels_edit),
        ))

        # Saving
        fit_to_contents(self._format_combo)
        fit_to_contents(self._compression_combo)
        g.addWidget(heading("Saving"))
        g.addLayout(row(
            "Output format:", take(self._format_combo), take(self._format_help), 18,
            "Compression:", take(self._compression_combo), 18,
            take(self._tiff_pyramids_cb), take(self._ozx_cb),
        ))

        # Memory management: every run uses these.
        for c in (self._streaming_combo, self._chunk_size_combo, self._fusion_region_combo):
            fit_to_contents(c)
        self._fusion_region_label = QLabel("Fusion region size:")
        self._fusion_region_label.setToolTip(self._fusion_region_combo.toolTip())
        g.addWidget(heading("Memory management"))
        g.addLayout(row(
            "Memory mode:", take(self._streaming_combo), take(self._memory_indicator), 18,
            "Fusion chunk size:", take(self._chunk_size_combo), 18,
            self._fusion_region_label, take(self._fusion_region_combo),
        ))
        fit_to_contents(self._scratch_dir_edit, 40)
        g.addLayout(row("Scratch dir:", take(self._scratch_dir_edit),
                        take(self._scratch_browse_btn)))
        self._streaming_combo.currentIndexChanged.connect(self._update_fusion_region_enabled)
        self._update_fusion_region_enabled()
        return b

    def _update_fusion_region_enabled(self, *_):
        """'Fusion region size' is used only when the run streams."""
        in_memory = self._streaming_combo.currentData() is False
        for w in (self._fusion_region_combo, self._fusion_region_label):
            w.setEnabled(not in_memory)
        tip = (
            "Grayed out: used only when the run streams ('Auto' or 'Streaming "
            "(low memory)' in 'Memory mode').\n\n" if in_memory else ""
        )
        self._fusion_region_combo.setToolTip(tip + self._fusion_region_tip)
        self._fusion_region_label.setToolTip(tip + self._fusion_region_tip)

    # ---- The two tabs -------------------------------------------------------

    def _build_settings_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setObjectName("settingsTabs")
        f = tabs.tabBar().font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        tabs.tabBar().setFont(f)
        tabs.setStyleSheet(
            "QTabWidget#settingsTabs::pane { border: 1px solid #b9c3cf;"
            " border-top: 2px solid #2f6db5; top: -1px; }"
            "QTabWidget#settingsTabs > QTabBar::tab { padding: 6px 24px;"
            " background: #e9edf2; border: 1px solid #b9c3cf; border-bottom: none;"
            " margin-right: 3px; }"
            "QTabWidget#settingsTabs > QTabBar::tab:selected { background: white;"
            " color: #1d3f6e; }"
        )
        tabs.addTab(self._build_image_quality_page(), "Image quality")
        tabs.addTab(self._build_registration_page(), "Registration and overlap")
        self._settings_tabs = tabs
        self._reg_tab_index = 1

        # 'Hide settings' at the right end of the tab bar (the old Processing
        # Options toggle, re-used: checked = settings showing).
        t = self._proc_toggle
        take(t)
        t.setStyleSheet(
            "QPushButton { border: none; padding: 4px 8px; color: #333; }"
            "QPushButton:hover { color: #000; text-decoration: underline; }"
        )
        t.setToolTip("Fold both tabs down to their names, to give the log more room. "
                     "Click either tab to open it again.")
        tabs.setCornerWidget(t, Qt.TopRightCorner)
        tabs.tabBarClicked.connect(lambda _i: self._proc_toggle.setChecked(True))
        self._skip_reg_cb.toggled.connect(self._update_reg_tab_title)
        return tabs

    def _update_reg_tab_title(self, *_):
        title = "Registration and overlap"
        if self._skip_reg_cb.isChecked():
            title += "  (skipped)"
        self._settings_tabs.setTabText(self._reg_tab_index, title)

    def _set_settings_folded(self, folded: bool) -> None:
        """Fold the tabs to their names (folded) or show the open page."""
        tabs = self._settings_tabs
        if folded:
            tabs.setMaximumHeight(tabs.tabBar().sizeHint().height() + 2)
        else:
            tabs.setMaximumHeight(16777215)
        self._proc_toggle.setText("▼ Show settings" if folded else "▲ Hide settings")

    def _set_tab_pages_enabled(self, enabled: bool) -> None:
        for i in range(self._settings_tabs.count()):
            self._settings_tabs.widget(i).setEnabled(enabled)

    def _build_image_quality_page(self) -> QWidget:
        page = QWidget()
        g = QGridLayout(page)
        g.setContentsMargins(10, 10, 10, 10)
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(10)

        lp = QGroupBox("Light paths (left and right sheets)")
        lv = QVBoxLayout(lp)
        fit_to_contents(self._fusion_combo)
        lv.addLayout(row("Illum. fusion:", take(self._fusion_combo)))
        lv.addLayout(row(12, take(self._illum_align_cb)))
        g.addWidget(lp, 0, 0)

        br = QGroupBox("Brightness")
        bv = QVBoxLayout(br)
        bv.addLayout(row(take(self._flat_field_cb)))
        self._flat_field_install_note = _install_note()
        bv.addWidget(self._flat_field_install_note)
        fit_to_contents(self._intensity_adjust_combo)
        bv.addLayout(row("Intensity adjustment:", take(self._intensity_adjust_combo)))
        g.addWidget(br, 0, 1)

        sp = QGroupBox("Stripes")
        sv = QVBoxLayout(sp)
        fit_to_contents(self._destripe_dir_combo)
        sv.addLayout(row(take(self._destripe_cb), take(self._destripe_fast_cb),
                         take(self._destripe_dir_combo), take(self._destripe_settings_btn),
                         take(self._destripe_preview_btn)))
        g.addWidget(sp, 1, 0)

        sh = QGroupBox("Sharpness")
        shv = QVBoxLayout(sh)
        shv.addLayout(row(take(self._deconv_cb)))
        self._deconv_install_note = _install_note()
        shv.addWidget(self._deconv_install_note)
        g.addWidget(sh, 1, 1)

        bz = QGroupBox("Background (lossy)")
        bzv = QVBoxLayout(bz)
        bzv.setContentsMargins(6, 4, 6, 4)
        self._bg_zero_panel.set_always_open()
        bzv.addWidget(take(self._bg_zero_panel))
        g.addWidget(bz, 2, 0, 1, 2)

        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)
        g.setRowStretch(3, 1)
        return page

    def _build_registration_page(self) -> QWidget:
        page = QWidget()
        g = QGridLayout(page)
        g.setContentsMargins(10, 10, 10, 10)
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(10)

        rg = QGroupBox("Registration (where each tile goes)")
        r = QGridLayout(rg)
        r.setHorizontalSpacing(8)
        r.addWidget(take(self._skip_reg_cb), 0, 0, 1, 3)
        for c in (self._reg_binning_xy_combo, self._reg_z_spacing_combo,
                  self._reg_upsample_combo, self._approach_combo, self._solve_model_combo):
            fit_to_contents(c)
        fit_to_contents(self._solve_tol_spin, 11)
        fit_to_contents(self._z_refine_range_spin, 8)
        self._approach_label = QLabel("Approach:")
        lines = [
            (take(self._reg_binning_label), row(
                take(self._reg_binning_xy_label), take(self._reg_binning_xy_combo), 14,
                take(self._reg_z_spacing_label), take(self._reg_z_spacing_combo))),
            (take(self._reg_upsample_label), row(take(self._reg_upsample_combo))),
            (self._approach_label, row(take(self._approach_combo))),
            (take(self._solve_model_label), row(take(self._solve_model_combo))),
            (take(self._solve_tol_label), row(take(self._solve_tol_spin))),
        ]
        for i, (lab, lay) in enumerate(lines, start=1):
            r.addWidget(lab, i, 0)
            r.addLayout(lay, i, 1, 1, 2)
        n = len(lines) + 1
        r.addWidget(take(self._z_snap_cb), n, 0, 1, 3)
        r.addLayout(row(take(self._z_refine_cb), 14, take(self._z_refine_range_label),
                        take(self._z_refine_range_spin)), n + 1, 0, 1, 3)
        r.addWidget(take(self._reg_z_cost_label), n + 2, 0, 1, 3)
        note = take(self._shift_limits_note)
        note.setText("Shift limits and seam choice are per microscope: set them on the "
                     "Options tab ('Maximum lateral correction', 'Maximum axial correction').")
        note.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        r.addWidget(note, n + 3, 0, 1, 3)
        r.setColumnStretch(2, 1)
        g.addWidget(rg, 0, 0, 2, 1)

        ov = QGroupBox("Tile overlap (how neighboring tiles are joined)")
        ovl = QVBoxLayout(ov)
        fit_to_contents(self._tile_fusion_combo)
        fit_to_contents(self._blend_width_combo)
        ovl.addLayout(row("Tile overlap:", take(self._tile_fusion_combo),
                          take(self._blend_width_combo)))
        ovl.addLayout(row(take(self._content_fusion_cb)))
        ovl.addStretch(1)
        g.addWidget(ov, 0, 1)

        ck = QGroupBox("Checks and reports")
        ckl = QVBoxLayout(ck)
        fit_to_contents(self._border_qc_mode_combo)
        ckl.addLayout(row(take(self._verbose_align_cb), 18, take(self._reg_report_cb)))
        ckl.addLayout(row(take(self._border_qc_cb), 14, take(self._border_qc_label),
                          take(self._border_qc_mode_combo)))
        ckl.addStretch(1)
        g.addWidget(ck, 1, 1)

        g.setColumnStretch(0, 3)
        g.setColumnStretch(1, 2)
        g.setRowStretch(2, 1)
        return page

    # ---- 'This run' panel ----------------------------------------------------

    def _build_run_summary_panel(self) -> QFrame:
        from flamingo_stitcher.gui.run_summary import RunSummaryPanel

        return RunSummaryPanel()

    def _wire_run_summary(self) -> None:
        """Refresh the panel shortly after any setting changes."""
        self._run_summary_timer = QTimer(self)
        self._run_summary_timer.setSingleShot(True)
        self._run_summary_timer.setInterval(120)
        self._run_summary_timer.timeout.connect(self._refresh_run_summary)
        for w in self._summary_inputs():
            for sig in ("currentIndexChanged", "toggled", "valueChanged", "textChanged"):
                s = getattr(w, sig, None)
                if s is not None:
                    try:
                        s.connect(lambda *_: self._run_summary_timer.start())
                    except TypeError:
                        pass
                    break
        QTimer.singleShot(0, self._refresh_run_summary)

    def _summary_inputs(self) -> Iterable[QWidget]:
        return (
            self._downsample_xy_combo, self._downsample_z_combo, self._format_combo,
            self._compression_combo, self._streaming_combo, self._fusion_combo,
            self._illum_align_cb, self._flat_field_cb, self._destripe_cb,
            self._destripe_fast_cb, self._intensity_adjust_combo, self._deconv_cb,
            self._skip_reg_cb, self._solve_model_combo, self._approach_combo,
            self._solve_tol_spin, self._tile_fusion_combo, self._blend_width_combo,
            self._content_fusion_cb, self._z_refine_cb, self._bg_zero_panel._enable_cb,
            self._pixel_size_spin, self._z_step_spin,
        )

    def _refresh_run_summary(self) -> None:
        from flamingo_stitcher.gui.run_summary import chips_from_dialog

        panel = getattr(self, "_run_summary", None)
        if panel is None:
            return
        panel.show_groups(*chips_from_dialog(self))


def _contains(layout: QLayout, widget: QWidget) -> bool:
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is widget:
            return True
        if item.layout() is not None and _contains(item.layout(), widget):
            return True
    return False


def _install_note() -> QLabel:
    """A one-line pointer to where a missing part can be installed; hidden until needed."""
    lab = QLabel("")
    lab.setWordWrap(True)
    lab.setTextFormat(Qt.RichText)
    lab.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
    lab.hide()
    return lab
