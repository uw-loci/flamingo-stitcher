"""Standalone Flamingo Stitcher application.

Launches a single window with two tabs — "Multi-Acquisition" and
"Single Workflow" — each hosting the corresponding stitching dialog. This is
the entry point for both the ``flamingo-stitch-gui`` console script and the
frozen Windows executable.

The same dialog classes are used in-app (inside Py2Flamingo) and here; the only
difference is that here we set up a vendored window-geometry manager and a
QApplication so they can run on their own.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _user_config_dir() -> Path:
    """Per-user directory for geometry/settings JSON (created if needed)."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home())
        d = Path(base) / "FlamingoStitcher"
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / "FlamingoStitcher"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        d = Path(base) / "flamingo-stitcher"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    """Launch the standalone stitching GUI. Returns the Qt exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from PyQt5.QtCore import QSettings, QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QTabWidget,
    )

    from flamingo_stitcher.gui._compat import (
        WindowGeometryManager,
        get_app_icon,
        set_default_geometry_manager,
    )
    from flamingo_stitcher.gui.stitching_dialog import (
        NativeStitchingDialog,
        StitchingDialog,
    )
    from flamingo_stitcher.gui.updater import UpdatePanel

    # QSettings location (the dialogs persist their own settings via QSettings).
    QApplication.setOrganizationName("Flamingo")
    QApplication.setApplicationName("FlamingoStitcher")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setWindowIcon(get_app_icon())

    # Window-geometry persistence for the embedded PersistentDialogs.
    cfg_dir = _user_config_dir()
    set_default_geometry_manager(
        WindowGeometryManager(str(cfg_dir / "window_geometry.json"))
    )

    window = QMainWindow()
    window.setWindowTitle("Flamingo Stitcher")
    window.setWindowIcon(get_app_icon())

    tabs = QTabWidget()
    # The dialogs are QWidget subclasses, so they embed directly as tab pages.
    multi = StitchingDialog(parent=window)
    single = NativeStitchingDialog(parent=window)
    tabs.addTab(multi, "Multi-Acquisition")
    tabs.addTab(single, "Single Workflow")

    # Updates tab — checks GitHub Releases for newer installers (standalone
    # build only; the in-app Py2Flamingo dialogs ship their own updates).
    settings = QSettings()
    update_panel = UpdatePanel(window, tabs, settings)
    tabs.addTab(update_panel, "Updates")
    tabs.currentChanged.connect(
        lambda idx: update_panel.on_tab_changed(tabs.widget(idx))
    )

    window.setCentralWidget(tabs)

    window.resize(1100, 800)
    window.show()

    # Re-apply a pending-update badge from a prior session, then run the
    # throttled launch auto-check once the event loop is up.
    update_panel.restore_badge_from_cache()
    QTimer.singleShot(0, update_panel.maybe_auto_check)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
