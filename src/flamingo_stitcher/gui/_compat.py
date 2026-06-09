"""Standalone compatibility shims for the stitching GUI.

When the stitching dialog runs **inside** the Py2Flamingo control app it relies
on a few host-app services (window-geometry persistence, the app icon, the
ntfy notification service). This module vendors minimal, dependency-free
equivalents so the dialog also runs as a **standalone** app.

Design goal: the dialog imports these names unconditionally. Behaviour:
- Standalone: a vendored ``WindowGeometryManager`` (set up by ``gui/app.py``)
  persists geometry; ``get_notification_service`` finds nothing and returns
  ``None`` (the dialog already handles ``None``).
- Embedded in Py2Flamingo: the host passes its own geometry manager to the
  dialog, and ``get_notification_service`` walks the widget parent chain and
  finds the app's real ``NotificationService`` — so notifications keep working
  with no special wiring.

Only PyQt5 is required here — no napari, no py2flamingo.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QMainWindow, QSplitter, QWidget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application icon
# ---------------------------------------------------------------------------

def get_app_icon() -> QIcon:
    """Return the Flamingo Stitcher window icon (or an empty icon)."""
    icon_path = Path(__file__).parent / "flamingo_icon.png"
    if icon_path.exists():
        return QIcon(str(icon_path))
    return QIcon()


# ---------------------------------------------------------------------------
# Window geometry persistence (vendored from py2flamingo)
# ---------------------------------------------------------------------------

_default_geometry_manager: Optional["WindowGeometryManager"] = None


def set_default_geometry_manager(manager: "WindowGeometryManager") -> None:
    """Set the app-wide default geometry manager.

    Call once at startup. All ``PersistentDialog``/``PersistentWidget``
    instances use it automatically unless an explicit manager is passed.
    """
    global _default_geometry_manager
    _default_geometry_manager = manager


class WindowGeometryManager:
    """Saves/restores window geometry + splitter sizes via JSON storage."""

    def __init__(self, config_file: str = "window_geometry.json"):
        self.config_file = Path(config_file)
        self._data: Dict[str, Any] = {"version": "1.0", "windows": {}}
        self._load_from_json()

    def _load_from_json(self) -> None:
        if not self.config_file.exists():
            return
        try:
            with open(self.config_file, "r") as f:
                self._data = json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not read geometry file, starting fresh: %s", e)
            self._data = {"version": "1.0", "windows": {}}

    def _save_to_json(self) -> None:
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.error("Error saving geometry file: %s", e)

    def _get_window_data(self, window_id: str) -> Dict[str, Any]:
        windows = self._data.setdefault("windows", {})
        if window_id not in windows:
            windows[window_id] = {
                "geometry": None,
                "state": None,
                "splitters": {},
                "dialog_state": {},
            }
        return windows[window_id]

    def save_geometry(self, window_id: str, widget: QWidget) -> None:
        try:
            window_data = self._get_window_data(window_id)
            window_data["geometry"] = base64.b64encode(
                widget.saveGeometry().data()
            ).decode("ascii")
            if isinstance(widget, QMainWindow):
                window_data["state"] = base64.b64encode(
                    widget.saveState().data()
                ).decode("ascii")
        except Exception as e:  # noqa: BLE001
            logger.error("Error saving geometry for '%s': %s", window_id, e)

    def restore_geometry(self, window_id: str, widget: QWidget) -> bool:
        try:
            windows = self._data.get("windows", {})
            if window_id not in windows:
                return False
            window_data = windows[window_id]
            geometry_b64 = window_data.get("geometry")
            if geometry_b64:
                widget.restoreGeometry(QByteArray(base64.b64decode(geometry_b64)))
            if isinstance(widget, QMainWindow):
                state_b64 = window_data.get("state")
                if state_b64:
                    widget.restoreState(QByteArray(base64.b64decode(state_b64)))
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Error restoring geometry for '%s': %s", window_id, e)
            return False

    def save_splitter_state(
        self, window_id: str, splitter_id: str, splitter: QSplitter
    ) -> None:
        try:
            window_data = self._get_window_data(window_id)
            window_data.setdefault("splitters", {})[splitter_id] = splitter.sizes()
        except Exception as e:  # noqa: BLE001
            logger.error("Error saving splitter state: %s", e)

    def restore_splitter_state(
        self, window_id: str, splitter_id: str, splitter: QSplitter
    ) -> bool:
        try:
            windows = self._data.get("windows", {})
            if window_id not in windows:
                return False
            splitters = windows[window_id].get("splitters", {})
            if splitter_id not in splitters:
                return False
            sizes = splitters[splitter_id]
            if sizes and len(sizes) == splitter.count():
                splitter.setSizes(sizes)
                return True
            return False
        except Exception as e:  # noqa: BLE001
            logger.error("Error restoring splitter state: %s", e)
            return False

    def save_all(self) -> None:
        self._save_to_json()


class PersistentDialog(QDialog):
    """QDialog with automatic geometry persistence.

    The geometry manager is resolved as: explicit ``geometry_manager`` kwarg,
    then the module-level default set via ``set_default_geometry_manager``.
    When neither exists, persistence is silently skipped (the dialog still
    works). The window ID defaults to the class name.
    """

    def __init__(
        self,
        *args,
        geometry_manager: Optional["WindowGeometryManager"] = None,
        window_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setWindowIcon(get_app_icon())
        self._geometry_manager = geometry_manager or _default_geometry_manager
        self._window_id = window_id or self.__class__.__name__
        self._geometry_restored = False

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._geometry_restored and self._geometry_manager:
            self._geometry_manager.restore_geometry(self._window_id, self)
            self._geometry_restored = True

    def hideEvent(self, event) -> None:
        if self._geometry_manager:
            self._geometry_manager.save_geometry(self._window_id, self)
            self._geometry_manager.save_all()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        if self._geometry_manager:
            self._geometry_manager.save_geometry(self._window_id, self)
            self._geometry_manager.save_all()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Notification service lookup (vendored — pure widget-tree walk)
# ---------------------------------------------------------------------------

def get_notification_service(widget) -> Optional[Any]:
    """Walk a widget's parent chain to find a host ``NotificationService``.

    Returns the first object found via a ``notification_service`` /
    ``_notification_service`` attribute (directly or via an ``app`` / ``_app``
    attribute), or ``None``. In a standalone launch nothing in the chain
    exposes one, so this returns ``None`` and the dialog skips notifications.
    """
    seen = set()
    current = widget
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("notification_service", "_notification_service"):
            svc = getattr(current, attr, None)
            if svc is not None:
                return svc
        for attr in ("app", "_app"):
            app = getattr(current, attr, None)
            if app is not None:
                svc = getattr(app, "notification_service", None)
                if svc is not None:
                    return svc
        try:
            current = current.parent()
        except Exception:  # noqa: BLE001
            return None
    return None
