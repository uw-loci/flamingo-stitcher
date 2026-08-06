"""Stop the mouse wheel silently changing dropdowns and spinboxes.

Qt's default is that a combo box or spin box under the cursor eats wheel
events, so scrolling a tall settings dialog quietly rewrites whatever
happened to be under the pointer on the way past. In a stitching dialog that
means a changed output format, pixel size, or fusion mode discovered hours
later in the result — a class of misconfiguration the user cannot see and has
no reason to suspect.

The guard makes wheel events pass through to the scroll area unless the widget
is deliberately focused, which is the behaviour almost everyone expects. A
focused widget still scrolls normally, so keyboard-driven editing is unchanged.
"""

from __future__ import annotations

import logging

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QAbstractSpinBox, QComboBox, QSlider

logger = logging.getLogger(__name__)

# Widget types whose value a stray wheel event can change.
_GUARDED = (QComboBox, QAbstractSpinBox, QSlider)


class WheelGuard(QObject):
    """Event filter that drops wheel events on unfocused value widgets."""

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        if event.type() == QEvent.Wheel and not obj.hasFocus():
            # Ignore so the event keeps bubbling to the scroll area: the dialog
            # scrolls, the widget's value does not change.
            event.ignore()
            return True
        return False


def install_wheel_guard(root) -> int:
    """Protect every dropdown/spinbox/slider under ``root``. Returns the count.

    Call AFTER the UI is built. Also relaxes focus policy from WheelFocus to
    StrongFocus, so these widgets take focus by click or tab but never merely
    by being scrolled over — otherwise the first scroll focuses the widget and
    the second one changes it.
    """
    try:
        guard = getattr(root, "_wheel_guard", None)
        if guard is None:
            # Parented to root so it lives exactly as long as the dialog.
            guard = WheelGuard(root)
            root._wheel_guard = guard

        count = 0
        for widget in root.findChildren(_GUARDED):
            if widget.focusPolicy() == Qt.WheelFocus:
                widget.setFocusPolicy(Qt.StrongFocus)
            widget.installEventFilter(guard)
            count += 1
        return count
    except Exception as e:  # noqa: BLE001 - a UI nicety must never break a dialog
        logger.debug(f"Could not install wheel guard: {e!r}")
        return 0
