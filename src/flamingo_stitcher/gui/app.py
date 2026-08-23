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


def _setup_logging() -> Path:
    """Configure console + persistent file logging. Returns the log file path.

    A per-launch timestamped file under ``<config dir>/logs/`` captures the
    full run log so it survives a hard crash / power loss (the GUI text panel
    is RAM-only). ``logging.FileHandler`` flushes after every record, so all
    but the last fraction of a line is on disk even on an abrupt shutdown.
    Old logs are pruned to the most recent 20.
    """
    from datetime import datetime

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    log_path = None
    try:
        log_dir = _user_config_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Prune to the 19 newest so this launch's file makes 20.
        existing = sorted(log_dir.glob("flamingo-stitcher_*.log"))
        for stale in existing[:-19]:
            try:
                stale.unlink()
            except OSError:
                pass
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"flamingo-stitcher_{stamp}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception as e:  # never let logging setup break startup
        logging.getLogger(__name__).warning(f"File logging unavailable: {e}")
    return log_path


def _splash_text(message: str) -> None:
    """Update the PyInstaller bootloader splash status line, if present.

    ``pyi_splash`` only exists inside the frozen build; in dev runs this is a
    no-op. Best-effort — never let splash chrome break startup.
    """
    try:
        import pyi_splash  # type: ignore

        pyi_splash.update_text(message)
    except Exception:
        pass


def _splash_close() -> None:
    """Close the bootloader splash once the main window is up (frozen only)."""
    try:
        import pyi_splash  # type: ignore

        pyi_splash.close()
    except Exception:
        pass


def _install_exception_guard() -> None:
    """Route unhandled exceptions to the log instead of crashing the process.

    PyQt5 aborts the whole application (qFatal → Windows exception 0xc0000409)
    when an unhandled Python exception escapes a slot invoked from C++ — e.g. a
    queued signal delivered from the worker thread — *and* sys.excepthook is
    still the default one. Installing a custom hook both makes such errors
    visible in the persistent log and prevents the hard crash: a slot failure
    degrades to a logged error rather than taking down a running stitch.
    (This is exactly how a missing import in a progress slot killed early
    builds with no traceback.)
    """
    import traceback

    _log = logging.getLogger("flamingo_stitcher.unhandled")

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        _log.error(
            "Unhandled exception (caught by guard; application kept alive):\n%s",
            "".join(traceback.format_exception(exc_type, exc, tb)),
        )

    sys.excepthook = _hook


def main() -> int:
    """Launch the standalone stitching GUI. Returns the Qt exit code."""
    log_path = _setup_logging()
    _log = logging.getLogger(__name__)
    # Log the version + environment on the FIRST line of every run — before any
    # dialog is built or a stitch starts — so a log is never version-ambiguous,
    # even if the app crashes at launch or the user never starts a run.
    try:
        import platform

        from flamingo_stitcher._version import __version__

        frozen = " (frozen)" if getattr(sys, "frozen", False) else ""
        _log.info(
            "Flamingo Stitcher %s%s starting — Python %s on %s",
            __version__,
            frozen,
            platform.python_version(),
            platform.platform(),
        )
    except Exception:
        pass
    if log_path:
        _log.info(f"Log file: {log_path}")
    _install_exception_guard()

    _splash_text("Loading the stitching engine...")

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
        MultiViewStitchingDialog,
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

    from flamingo_stitcher import __version__

    window = QMainWindow()
    window.setWindowTitle(f"Flamingo Stitcher {__version__}")
    window.setWindowIcon(get_app_icon())

    tabs = QTabWidget()
    # The dialogs are QWidget subclasses, so they embed directly as tab pages.
    multi = StitchingDialog(parent=window)
    single = NativeStitchingDialog(parent=window)
    multiview = MultiViewStitchingDialog(parent=window)
    # Standalone has no 3D Sample View, so hide the "Load … into Sample View"
    # completion button (the load_stitched_requested signal has no receiver here).
    multi.set_sample_view_available(False)
    single.set_sample_view_available(False)
    multiview.set_sample_view_available(False)
    tabs.addTab(multi, "Multi-Acquisition")
    tabs.addTab(single, "Single Workflow")
    tabs.addTab(multiview, "Multi-View")

    # Options — per-microscope, per-objective registration tuning. Its own tab
    # rather than a panel inside each dialog because the values belong to an
    # instrument, not to a run, and all three dialogs consume the same store.
    from flamingo_stitcher.gui.options_panel import OptionsPanel

    tabs.addTab(OptionsPanel(window), "Options")

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

    # Window is up — tear down the bootloader splash.
    _splash_close()

    # Re-apply a pending-update badge from a prior session, then run the
    # throttled launch auto-check once the event loop is up.
    update_panel.restore_badge_from_cache()
    QTimer.singleShot(0, update_panel.maybe_auto_check)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
