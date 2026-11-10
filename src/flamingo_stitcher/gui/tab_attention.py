"""A soft pulsing halo on tabs the user has not opened yet.

After 'Discover Tiles', the 'Image quality' and 'Registration and overlap' tabs
hold settings a new user may not know are there. A tab glows light blue until
the user clicks it (or switches to it from the keyboard), so the hint goes away
as soon as it has done its job. Discovering again re-arms both tabs.

The halo is painted by a transparent child of the tab bar that ignores the
mouse, so the tab bar, its style sheet and its clicks are unchanged. It pulses
slowly (one cycle every ``PERIOD_S``) and never reaches full strength, because
it is meant to say "there is something here", not "something is wrong".
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional, Set

from PyQt5.QtCore import QElapsedTimer, QEvent, QObject, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QTabWidget, QWidget

# Light blue, the same family as the 'Discover Tiles' accent (#1E88E5) but paler.
GLOW_RGB = (100, 181, 246)
# Strongest alpha at the top of the pulse (of 255). Kept well below opaque.
PEAK_ALPHA = 170
# Share of PEAK_ALPHA used for the wash inside the tab.
FILL_SHARE = 0.28
# One full fade-in, fade-out cycle, in seconds.
PERIOD_S = 2.4
# Repaint interval while a tab is glowing (about 20 frames per second).
FRAME_MS = 50

GLOW_TOOLTIP = (
    "You have not opened this tab since the last 'Discover Tiles'. "
    "Click it to check its settings before you click 'Run All'; "
    "the blue glow stops once you have."
)


class _GlowOverlay(QWidget):
    """Transparent layer over the tab bar that paints the halos."""

    def __init__(self, owner: "TabAttentionGlow", parent: QWidget):
        super().__init__(parent)
        self._owner = owner
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.NoFocus)

    def paintEvent(self, _event):  # noqa: N802 (Qt override)
        strength = self._owner.strength()
        if strength <= 0.0:
            return
        bar = self.parentWidget()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r, g, b = GLOW_RGB
        for index in sorted(self._owner.pending):
            if not (0 <= index < bar.count()):
                continue
            rect = QRectF(bar.tabRect(index)).adjusted(2.5, 2.5, -2.5, -1.5)
            # A faint wash inside the tab, then three strokes from wide and
            # faint to narrow and stronger: a halo rather than a hard outline.
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(r, g, b, int(PEAK_ALPHA * strength * FILL_SHARE)))
            p.drawRoundedRect(rect, 4.0, 4.0)
            for width, share in ((7.0, 0.25), (4.0, 0.5), (2.0, 1.0)):
                c = QColor(r, g, b, int(PEAK_ALPHA * strength * share))
                p.setPen(QPen(c, width))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(rect, 4.0, 4.0)
        p.end()


class TabAttentionGlow(QObject):
    """Pulse a light-blue halo on tabs of ``tabs`` until each is clicked.

    ``suppressed`` (optional) is asked on every frame; while it returns True
    nothing glows, but the tabs stay pending. The dialog uses it so the halo
    never competes with a flashing next-step button ('Add…' or 'Discover
    Tiles').
    """

    def __init__(self, tabs: QTabWidget, suppressed: Optional[Callable[[], bool]] = None):
        super().__init__(tabs)
        self._tabs = tabs
        self._bar = tabs.tabBar()
        self._suppressed = suppressed
        self.pending: Set[int] = set()
        self._plain_tips: Dict[int, str] = {}
        self._clock = QElapsedTimer()
        self._overlay = _GlowOverlay(self, self._bar)
        self._overlay.setGeometry(self._bar.rect())
        self._overlay.hide()
        self._bar.installEventFilter(self)
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)
        # A click on a tab, including the one already showing, or a keyboard
        # switch to another tab, counts as having looked.
        tabs.tabBarClicked.connect(self.acknowledge)
        tabs.currentChanged.connect(self._on_current_changed)

    # --- state ---------------------------------------------------------------

    def arm(self, indices: Optional[Iterable[int]] = None) -> None:
        """Make these tabs (default: all) glow until each is clicked."""
        if indices is None:
            indices = range(self._tabs.count())
        for i in indices:
            if 0 <= i < self._tabs.count() and i not in self.pending:
                self.pending.add(i)
                self._plain_tips[i] = self._bar.tabToolTip(i)
                self._bar.setTabToolTip(i, GLOW_TOOLTIP)
        self._update_running()

    def acknowledge(self, index: int) -> None:
        """The user opened tab ``index``: stop its glow."""
        if index in self.pending:
            self.pending.discard(index)
            self._bar.setTabToolTip(index, self._plain_tips.pop(index, ""))
            self._update_running()

    def clear(self) -> None:
        """Stop every glow (a run started, or the dialog is resetting)."""
        for i in list(self.pending):
            self.acknowledge(i)
        self._update_running()

    def is_glowing(self, index: int) -> bool:
        """True while tab ``index`` is pending and not suppressed."""
        return index in self.pending and not self._is_suppressed()

    def strength(self) -> float:
        """Current pulse level, 0 (off) to 1 (peak)."""
        if not self.pending or self._is_suppressed() or not self._clock.isValid():
            return 0.0
        t = self._clock.elapsed() / 1000.0
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * t / PERIOD_S)

    # --- internals -----------------------------------------------------------

    def _is_suppressed(self) -> bool:
        if self._suppressed is None:
            return False
        try:
            return bool(self._suppressed())
        except Exception:  # a hint must never break the dialog
            return False

    def _on_current_changed(self, index: int) -> None:
        # Only a switch the user can see counts; a hidden dialog changing its
        # own tab (none does today) is not a look.
        if self._bar.isVisible():
            self.acknowledge(index)

    def _update_running(self) -> None:
        if self.pending:
            if not self._clock.isValid():
                self._clock.start()
            self._overlay.setGeometry(self._bar.rect())
            self._overlay.show()
            self._overlay.raise_()
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
            self._clock.invalidate()
            self._overlay.hide()

    def _tick(self) -> None:
        if self._bar.isVisible():
            self._overlay.update()

    def eventFilter(self, obj, event):  # noqa: N802 (Qt override)
        if obj is self._bar and event.type() in (QEvent.Resize, QEvent.LayoutRequest):
            self._overlay.setGeometry(self._bar.rect())
        return False
