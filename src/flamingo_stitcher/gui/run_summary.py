"""The 'This run' panel: what the selected acquisition's run will do, as tags.

Three columns named after the dialog's sections ('Basics', 'Image quality',
'Registration and overlap') so the tab that is not showing is never a surprise
before a long run. Orange tags (marked with a star) are the slow options, red
is lossy, yellow is skipped or not used, gray dashed is a main option left off.
The user asked for this in place of a one-line sentence that was "too hard to
read and will just be skipped" (2026-11-05).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

STAR = " ✱"

CHIP_STYLE = {
    "on": "background:#eef1f5;color:#222;border:1px solid #d3d9e2;",
    "slow": "background:#fff4e5;color:#8a4b00;border:1px solid #f0b46a;",
    "lossy": "background:#fdecec;color:#9b1c1c;border:1px solid #e9a3a3;",
    "skipped": "background:#fff8d6;color:#6b5600;border:1px solid #e6cf6a;font-style:italic;",
    "off": "background:transparent;color:#888;border:1px dashed #c8c8c8;",
}

Chip = Tuple[str, str]  # (text, kind)


class FlowLayout(QLayout):
    """Lays widgets left to right and wraps them (the standard Qt example)."""

    def __init__(self, parent=None, spacing: int = 5):
        super().__init__(parent)
        self._items = []
        self._sp = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._do(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._do(rect, False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        return size

    def _do(self, rect, test):
        x, y, line = rect.x(), rect.y(), 0
        for it in self._items:
            hint = it.sizeHint()
            nx = x + hint.width() + self._sp
            if nx - self._sp > rect.right() and line > 0:
                x, y = rect.x(), y + line + self._sp
                nx = x + hint.width() + self._sp
                line = 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x = nx
            line = max(line, hint.height())
        return y + line - rect.y()


def chip(text: str, kind: str = "on") -> QLabel:
    c = QLabel(text)
    c.setStyleSheet(CHIP_STYLE[kind] + "border-radius:10px;padding:2px 9px;")
    return c


class RunSummaryPanel(QFrame):
    """'This run' and three columns of tags."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("runSummary")
        self.setStyleSheet(
            "#runSummary { background:#fafbfc; border:1px solid #d6dbe1; border-radius:6px; }"
        )
        self._h = QHBoxLayout(self)
        self._h.setContentsMargins(10, 8, 10, 8)
        self._h.setSpacing(16)
        self._title = QLabel()
        self._title.setAlignment(Qt.AlignTop)
        self._title.setTextFormat(Qt.RichText)
        self._title.setMinimumWidth(80)
        self._h.addWidget(self._title)
        self._body = QWidget()
        self._body_layout = QHBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(16)
        self._h.addWidget(self._body, 1)
        self.groups: List[Tuple[str, List[Chip]]] = []
        self.message = ""
        self.show_groups("", [], "Nothing queued yet.")

    def show_groups(self, subtitle: str, groups: Sequence[Tuple[str, List[Chip]]],
                    message: str = "") -> None:
        """Show ``groups`` (name, tags), or ``message`` in their place."""
        self.groups = [(n, list(c)) for n, c in groups]
        self.message = message
        self._title.setText(
            "<span style='font-weight:bold;font-size:11pt;color:#1d3f6e'>This run</span>"
            + (f"<br><span style='color:#555'>{subtitle}</span>" if subtitle else "")
        )
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        if message or not groups:
            lab = QLabel(message)
            lab.setWordWrap(True)
            lab.setStyleSheet("color:#555;")
            self._body_layout.addWidget(lab, 1)
            return
        for name, chips in groups:
            col = QWidget()
            v = QVBoxLayout(col)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(4)
            head = QLabel(name)
            head.setStyleSheet("color:#666;font-size:8pt;font-weight:bold;")
            v.addWidget(head)
            holder = QWidget()
            holder.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            flow = FlowLayout(holder)
            for text, kind in chips:
                flow.addWidget(chip(text, kind))
            v.addWidget(holder)
            v.addStretch(1)
            self._body_layout.addWidget(col, 1)


# --------------------------------------------------------------------------- #
# Tags from the dialog's controls
# --------------------------------------------------------------------------- #


def _text(combo) -> str:
    return combo.currentText()


def _short(text: str) -> str:
    """'Content (weight by local detail)' -> 'Content'."""
    return text.split(" (")[0]


def chips_from_dialog(d) -> Tuple[str, List[Tuple[str, List[Chip]]], str]:
    """``(subtitle, groups, message)`` for the panel, from the dialog's controls."""
    if not getattr(d, "_queue", None):
        return "", [], ("Nothing queued yet. The settings the run will use appear here, "
                        "grouped like the sections above.")
    xy = d._downsample_xy_combo.currentData()
    z = d._downsample_z_combo.currentData()
    basics: List[Chip] = []
    if xy == -1:
        basics.append(("Downsample iso (near-cubic voxels)", "on"))
    else:
        basics.append((f"Downsample XY {xy}x, Z {z}x", "on"))
        px = d._pixel_size_spin.value()
        zs = d._z_step_spin.value()
        if px > 0:
            zt = f"{zs * z:.2f}" if zs > 0 else "auto"
            basics.append((f"{px * xy:.2f} × {px * xy:.2f} × {zt} µm", "on"))
    comp = _short(_text(d._compression_combo))
    basics.append((f"{_short(_text(d._format_combo))}, {comp}", "on"))
    basics.append((f"Memory: {_short(_text(d._streaming_combo))}", "on"))

    iq: List[Chip] = []
    illum = _short(_text(d._fusion_combo))
    aligned = d._illum_align_cb.isChecked() and d._illum_align_cb.isEnabled()
    iq.append((f"Light paths: {illum}" + (" + aligned" if aligned else ""), "on"))
    if d._flat_field_cb.isChecked() and d._flat_field_cb.isEnabled():
        iq.append(("Flat-field", "on"))
    else:
        iq.append(("Flat-field off", "off"))
    if d._destripe_cb.isChecked():
        fast = d._destripe_fast_cb.isChecked() and d._destripe_fast_cb.isEnabled()
        iq.append(("Destripe (fast)" + STAR if fast else "Destripe" + STAR, "slow"))
    else:
        iq.append(("Destripe off", "off"))
    ia = d._intensity_adjust_combo.currentData()
    iq.append({"off": ("Brightness matching off", "off"),
               "measure": ("Brightness matching: measure only", "on"),
               "apply": ("Brightness matching: Active", "on")}.get(
        ia, (f"Brightness matching: {ia}", "on")))
    if d._deconv_cb.isChecked() and d._deconv_cb.isEnabled():
        iq.append(("Deconvolution" + STAR, "slow"))
    bg = d._bg_zero_panel
    if bg.is_enabled():
        th = bg.thresholds()
        if th:
            iq.append((", ".join(f"Zero background ≤ {v} (ch {k})" for k, v in
                                 sorted(th.items())) + " (lossy)", "lossy"))
        else:
            iq.append(("Zero background: no threshold set", "skipped"))

    reg: List[Chip] = []
    if d._skip_reg_cb.isChecked():
        reg.append(("Registration skipped: stage positions", "skipped"))
    else:
        reg.append(("Rigid" if d._solve_model_combo.currentData() == "rigid" else "Shift only", "on"))
        reg.append((_text(d._approach_combo) if d._approach_combo.currentData() != "default"
                    else "Default approach", "on"))
        reg.append((f"Tolerance {d._solve_tol_spin.value():g} px", "on"))
        if d._z_refine_cb.isChecked():
            reg.append(("Refine Z" + STAR, "slow"))
    tf = d._tile_fusion_combo.currentData()
    if tf == "blend":
        reg.append((f"Blend, {_text(d._blend_width_combo).replace('Fade: ', '')}", "on"))
        if d._content_fusion_cb.isChecked():
            reg.append(("Content-based blending" + STAR, "slow"))
    else:
        reg.append((_text(d._tile_fusion_combo), "on"))

    subtitle = getattr(d, "_run_summary_subtitle", "")
    return subtitle, [("Basics", basics), ("Image quality", iq),
                      ("Registration and overlap", reg)], ""
