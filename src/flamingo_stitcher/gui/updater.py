"""In-app update checker for the standalone Flamingo Stitcher GUI.

Ported from the TRACE updater (github.com/alexmpdx/TRACE, by Alex M.) and
trimmed to fit this leaner app — no theme system, no startup-log dependency.

What it does:
  * Queries the GitHub ``releases/latest`` API on a background thread (never
    blocks the GUI) and compares the latest tag against the installed
    ``flamingo_stitcher.__version__``.
  * Surfaces an "Updates" tab with the installed version, a Check button, a
    one-click Install button (frozen Windows build only), and a "View all
    releases…" fallback.
  * Auto-checks on launch (throttled, opt-out) and, when a newer release
    exists, badges the Updates tab and pops a centered notification dialog.
  * One-click install downloads the version-stamped
    ``FlamingoStitcher-Setup-<ver>.exe`` asset, verifies its size, launches it
    detached, and quits so Inno Setup can upgrade in place (stable AppId).

Only wired into the standalone window (``gui/app.py``). The dialogs embedded in
the Py2Flamingo control app are unaffected — that app ships its own updates.
"""

from __future__ import annotations

import os
import ssl
import sys
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# --- Repository coordinates -------------------------------------------------
GITHUB_OWNER = "uw-loci"
GITHUB_REPO = "flamingo-stitcher"
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
LATEST_RELEASE_API = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)

# Release installer assets are named ``FlamingoStitcher-Setup-<ver>.exe``
# (Inno Setup ``OutputBaseFilename`` includes the version), so match by the
# stable prefix + ``.exe`` suffix rather than a fixed full name.
_ASSET_PREFIX = "FlamingoStitcher-Setup"

# Plain colors (this app has no theme manager). Chosen to read on both light
# and dark Qt palettes.
_C_SUCCESS = "#2e7d32"
_C_WARNING = "#e65100"
_C_ERROR = "#c62828"
_C_MUTED = "#888888"
_C_ACCENT = "#1976d2"

# QSettings keys.
_KEY_AUTO = "updates/auto_check_enabled"
_KEY_LAST = "updates/last_check_time"
_KEY_CACHED = "updates/cached_latest_version"

# Anti-spam window for the launch auto-check (seconds). GitHub's anonymous
# limit is 60 req/hr/IP; one request per launch under this window stays well
# inside it even on a tight relaunch loop.
_AUTO_CHECK_THROTTLE_S = 60


def make_ssl_context() -> ssl.SSLContext:
    """SSL context that works inside a PyInstaller bundle.

    Frozen Windows builds have no system CA store, so urllib otherwise fails
    with "CERTIFICATE_VERIFY_FAILED" the moment it talks to GitHub. Point at
    certifi's bundled cacert.pem when available; fall back to ssl defaults for
    dev runs where the system store is reachable.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _version_is_newer(candidate: str, installed: str) -> bool:
    """Return True iff ``candidate`` is strictly newer than ``installed``.

    Parses dotted version strings as tuples of ints and compares element-wise
    ("0.2.0" > "0.1.44"). On any parse failure, returns False — preferring to
    under-announce updates over offering the user a downgrade. This also makes
    the "running a local dev build ahead of the latest tag" case resolve to
    "up to date" rather than a spurious update prompt.
    """
    if not candidate or not installed:
        return False
    try:
        cand_parts = [int(x) for x in candidate.split(".")]
        inst_parts = [int(x) for x in installed.split(".")]
    except (ValueError, AttributeError):
        return False
    return cand_parts > inst_parts


def fetch_latest_release_info(api_url: str) -> dict:
    """Pure network + parse helper. No UI access. Raises on failure.

    Returns a dict with keys ``tag`` (raw GitHub tag, e.g. ``v0.1.4``),
    ``latest_version`` (the bare semver with a leading ``v`` stripped, for
    direct comparison with ``__version__``), ``html_url`` (the release page),
    ``asset_url`` (installer download URL, or None), and ``asset_size``
    (bytes, or None).
    """
    import json
    import urllib.request

    req = urllib.request.Request(api_url, headers={"User-Agent": "FlamingoStitcher-update-check"})
    with urllib.request.urlopen(req, timeout=10, context=make_ssl_context()) as resp:
        data = json.load(resp)

    latest_tag = str(data.get("tag_name") or "")
    latest_version = latest_tag[1:] if latest_tag.startswith("v") else latest_tag

    asset_url: Optional[str] = None
    asset_size: Optional[int] = None
    for asset in data.get("assets") or []:
        name = str(asset.get("name") or "")
        if name.startswith(_ASSET_PREFIX) and name.endswith(".exe"):
            asset_url = asset.get("browser_download_url") or asset.get("url")
            try:
                asset_size = int(asset.get("size") or 0) or None
            except Exception:
                asset_size = None
            break

    return {
        "tag": latest_tag,
        "latest_version": latest_version,
        "html_url": str(data.get("html_url") or RELEASES_PAGE_URL),
        "asset_url": asset_url,
        "asset_size": asset_size,
    }


class _UpdateCheckThread(QThread):
    """Runs the GitHub releases/latest query off the GUI thread.

    Emits a single ``result`` dict back to the GUI thread: ``{"ok": True, ...}``
    on success (merged with :func:`fetch_latest_release_info`'s payload) or
    ``{"ok": False, "error": str}`` on any failure. The caller decides whether
    to surface the error (manual click) or swallow it (silent auto-check).
    """

    result = pyqtSignal(dict)

    def __init__(self, api_url: str, parent=None):
        super().__init__(parent)
        self._api_url = api_url

    def run(self):
        try:
            payload = fetch_latest_release_info(self._api_url)
            self.result.emit({"ok": True, **payload})
        except Exception as exc:  # noqa: BLE001
            self.result.emit({"ok": False, "error": str(exc)})


def _installed_version() -> str:
    try:
        from flamingo_stitcher import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _can_install_in_place(asset_url: Optional[str]) -> bool:
    """True only when a one-click install is actually possible: a frozen
    Windows build with a downloadable installer asset. Dev runs and
    non-Windows fall back to the release-page link."""
    return bool(asset_url) and getattr(sys, "frozen", False) and sys.platform == "win32"


class UpdatePanel(QWidget):
    """The "Updates" tab: installed version, check/install controls, and the
    badge + notification machinery.

    Owns all update state. The host window passes its :class:`QTabWidget` so
    the panel can badge its own tab, and itself so notification dialogs can be
    centered over the main window.
    """

    def __init__(self, main_window, tab_widget, settings, parent=None):
        super().__init__(parent)
        self._window = main_window
        self._tabs = tab_widget
        self._settings = settings

        # Populated when a check finds a newer release.
        self._latest_url: Optional[str] = None
        self._latest_size: Optional[int] = None
        self._latest_version: Optional[str] = None
        # Set when the user clicks "Update now" in the launch notification
        # before the asset URL is known; the next result consumes it and fires
        # the install instead of just updating the tab.
        self._install_after_next_check = False
        # In-flight guard so a manual click colliding with the auto-check
        # doesn't fire two concurrent requests.
        self._thread: Optional[_UpdateCheckThread] = None
        # Session-only de-dup so the launch auto-check colliding with the
        # cached-restore dialog doesn't pop the same dialog twice.
        self._dialog_fired_for: Optional[str] = None
        self._pending_dialog: Optional[QDialog] = None
        self._badge_icon: Optional[QIcon] = None

        self._build_ui()

    # -- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Updates")
        tf = QFont(title.font())
        tf.setPointSize(tf.pointSize() + 3)
        tf.setBold(True)
        title.setFont(tf)
        layout.addWidget(title)

        self._version_label = QLabel(
            f"<span style='color:{_C_MUTED};'>Installed version:</span> "
            f"<b>{_installed_version()}</b>"
        )
        layout.addWidget(self._version_label)

        blurb = QLabel(
            "Flamingo Stitcher checks GitHub Releases for newer installers. "
            "Running a newer installer upgrades your existing copy in place — "
            "no need to uninstall first; your settings are preserved."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color:{_C_MUTED};")
        layout.addWidget(blurb)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setOpenExternalLinks(True)
        self._status_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        layout.addWidget(self._status_label)

        row = QHBoxLayout()
        self.btn_check = QPushButton("Check for updates")
        self.btn_check.setToolTip(
            "Query GitHub Releases for the latest Flamingo Stitcher installer "
            "and report whether you're up to date."
        )
        self.btn_check.clicked.connect(lambda: self.check_for_updates(silent=False))
        row.addWidget(self.btn_check)

        self.btn_install = QPushButton("Install update")
        self.btn_install.setToolTip(
            "Download the latest installer and launch it. It upgrades over the "
            "current install — your settings are preserved."
        )
        self.btn_install.setVisible(False)
        self.btn_install.clicked.connect(self.install_update)
        row.addWidget(self.btn_install)

        self.btn_releases = QPushButton("View all releases…")
        self.btn_releases.setToolTip(
            "Open the Releases page in your browser to download an installer manually."
        )
        self.btn_releases.clicked.connect(self._open_releases_page)
        row.addWidget(self.btn_releases)
        row.addStretch(1)
        layout.addLayout(row)

        self.chk_auto = QCheckBox("Auto-check for updates on launch")
        self.chk_auto.setToolTip(
            "When checked, Flamingo Stitcher silently queries Releases on launch "
            "(throttled) and flags this tab if an update is available."
        )
        self.chk_auto.setChecked(self._settings.value(_KEY_AUTO, True, type=bool))
        self.chk_auto.toggled.connect(
            lambda on: self._settings.setValue(_KEY_AUTO, bool(on))
        )
        layout.addWidget(self.chk_auto)

        layout.addSpacing(14)
        links_title = QLabel("Help &amp; links")
        lf = QFont(links_title.font())
        lf.setPointSize(lf.pointSize() + 2)
        lf.setBold(True)
        links_title.setFont(lf)
        layout.addWidget(links_title)

        repo = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}"
        links = QLabel(
            f"&bull; <a href='{repo}#readme' style='color:{_C_ACCENT};'>README &amp; usage guide</a><br>"
            f"&bull; <a href='{repo}/blob/main/src/flamingo_stitcher/docs/"
            f"stitching_hardware_troubleshooting.md' style='color:{_C_ACCENT};'>"
            f"Hardware &amp; troubleshooting guide</a><br>"
            f"&bull; <a href='{repo}/issues/new' style='color:{_C_ACCENT};'>Report an issue</a> "
            f"<span style='color:{_C_MUTED};'>(GitHub account required)</span><br>"
            f"&bull; <a href='{repo}' style='color:{_C_ACCENT};'>Project on GitHub</a>"
        )
        links.setOpenExternalLinks(True)
        links.setTextInteractionFlags(Qt.TextBrowserInteraction)
        links.setWordWrap(True)
        layout.addWidget(links)

        layout.addStretch(1)

    def _open_releases_page(self) -> None:
        try:
            QDesktopServices.openUrl(QUrl(RELEASES_PAGE_URL))
        except Exception:
            import webbrowser

            webbrowser.open(RELEASES_PAGE_URL)

    # -- Check -------------------------------------------------------------
    def check_for_updates(self, *, silent: bool = False) -> None:
        """Kick off the release query on a background thread.

        ``silent=False`` (manual button) shows a "Checking…" hint and surfaces
        errors. ``silent=True`` (launch auto-check) is invisible unless an
        update is found. At most one check runs at a time.
        """
        try:
            running = self._thread is not None and self._thread.isRunning()
        except RuntimeError:
            # Previous thread's C++ object was deleted by deleteLater; the
            # Python attribute is a stale sip wrapper. Treat as "no thread".
            running = False
            self._thread = None
        if running:
            return

        if not silent:
            self._status_label.setText(
                f"<span style='color:{_C_MUTED};'>Checking for updates…</span>"
            )
        self._thread = _UpdateCheckThread(LATEST_RELEASE_API, parent=self)
        self._thread.result.connect(
            lambda payload: self._apply_result(payload, silent=silent)
        )
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_thread_ref)
        self._thread.start()

    def _clear_thread_ref(self) -> None:
        self._thread = None

    def _apply_result(self, payload: dict, *, silent: bool) -> None:
        self._settings.setValue(_KEY_LAST, int(time.time()))

        if not payload.get("ok"):
            if not silent:
                err = payload.get("error", "")
                self._status_label.setText(
                    f"<span style='color:{_C_ERROR};'>Could not check for updates: {err}</span><br>"
                    f"<a href='{RELEASES_PAGE_URL}' style='color:{_C_ACCENT};'>"
                    f"Open the Releases page manually</a>"
                )
            return

        installed = _installed_version()
        latest = payload.get("latest_version") or ""
        if not latest:
            if not silent:
                self._status_label.setText(
                    f"<span style='color:{_C_MUTED};'>No releases found on GitHub yet. "
                    f"<a href='{RELEASES_PAGE_URL}' style='color:{_C_ACCENT};'>Releases page</a>.</span>"
                )
            return

        if not _version_is_newer(latest, installed):
            self._status_label.setText(
                f"<span style='color:{_C_SUCCESS};'>✓ You're up to date (installed: {installed}).</span>"
            )
            self.btn_install.setVisible(False)
            self._latest_url = self._latest_size = self._latest_version = None
            self.clear_badge(clear_cache=True)
            return

        # Newer release available.
        self._latest_url = payload.get("asset_url")
        self._latest_size = payload.get("asset_size")
        self._latest_version = latest

        if _can_install_in_place(self._latest_url):
            self.btn_install.setText(f"Install update {latest}")
            self.btn_install.setVisible(True)
            size_mb = (self._latest_size or 0) // (1024 * 1024)
            size_blurb = f" ({size_mb} MB)" if size_mb else ""
            self._status_label.setText(
                f"<span style='color:{_C_WARNING};'>Update available: <b>{latest}</b> "
                f"(you have {installed}).</span><br>"
                f"<span style='color:{_C_MUTED};'>Click <b>Install update {latest}</b> "
                f"to download{size_blurb} and launch the new installer.</span>"
            )
        else:
            self.btn_install.setVisible(False)
            self._status_label.setText(
                f"<span style='color:{_C_WARNING};'>A newer version is available: <b>{latest}</b> "
                f"(you have {installed}).</span><br>"
                f"<a href='{payload.get('html_url', RELEASES_PAGE_URL)}' style='color:{_C_ACCENT};'>"
                f"Open the release page and download the installer</a>"
            )

        self.show_badge(latest)
        self.show_update_dialog(latest)

        if self._install_after_next_check and _can_install_in_place(self._latest_url):
            self._install_after_next_check = False
            self.install_update()
        else:
            self._install_after_next_check = False

    # -- Badge -------------------------------------------------------------
    def _badge(self) -> QIcon:
        if self._badge_icon is None:
            size = 12
            pix = QPixmap(size, size)
            pix.fill(Qt.transparent)
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(QColor(_C_ACCENT))
            p.setPen(Qt.NoPen)
            p.drawEllipse(0, 0, size, size)
            p.end()
            self._badge_icon = QIcon(pix)
        return self._badge_icon

    def show_badge(self, latest_version: str) -> None:
        """Put a dot on the Updates tab and cache the latest version so the
        badge survives a relaunch until the user actually upgrades."""
        idx = self._tabs.indexOf(self)
        if idx >= 0:
            self._tabs.setTabIcon(idx, self._badge())
        self._settings.setValue(_KEY_CACHED, latest_version)

    def clear_badge(self, *, clear_cache: bool = False) -> None:
        idx = self._tabs.indexOf(self)
        if idx >= 0:
            self._tabs.setTabIcon(idx, QIcon())
        if clear_cache:
            self._settings.remove(_KEY_CACHED)

    # -- Launch hooks ------------------------------------------------------
    def maybe_auto_check(self) -> None:
        """Launch-time auto-check, throttled and opt-out."""
        if not self._settings.value(_KEY_AUTO, True, type=bool):
            return
        last = int(self._settings.value(_KEY_LAST, 0, type=int) or 0)
        if time.time() - last < _AUTO_CHECK_THROTTLE_S:
            return
        self.check_for_updates(silent=True)

    def restore_badge_from_cache(self) -> None:
        """Re-apply the badge + notification on launch if a prior session saw
        an update the user hasn't installed yet. A stale cache (installed has
        caught up) is dropped so it stops re-firing."""
        cached = str(self._settings.value(_KEY_CACHED, "") or "")
        if not cached:
            return
        if not _version_is_newer(cached, _installed_version()):
            self._settings.remove(_KEY_CACHED)
            return
        self.show_badge(cached)
        QTimer.singleShot(0, lambda: self.show_update_dialog(cached))

    def on_tab_changed(self, widget: QWidget) -> None:
        """Clear the badge when the user actually visits the Updates tab."""
        if widget is self:
            self.clear_badge()

    # -- Notification dialog ----------------------------------------------
    def show_update_dialog(self, latest_version: str) -> None:
        """Centered, non-modal "update available" notification with Dismiss
        and Update-now. De-duped per version within a session."""
        if self._dialog_fired_for == latest_version:
            return
        installed = _installed_version()

        dlg = QDialog(self._window)
        dlg.setWindowTitle("Update available")
        dlg.setWindowIcon(self._window.windowIcon())
        dlg.setMinimumWidth(380)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        head = QLabel("Update available")
        hf = QFont(head.font())
        hf.setBold(True)
        hf.setPointSize(hf.pointSize() + 1)
        head.setFont(hf)
        layout.addWidget(head)

        body = QLabel(
            f"Flamingo Stitcher <b>{latest_version}</b> is available — you're "
            f"running <b>{installed}</b>.<br><br>"
            "Click <b>Update now</b> to download and launch the new installer. "
            "Your settings are preserved."
        )
        body.setWordWrap(True)
        layout.addWidget(body)

        footer = QHBoxLayout()
        footer.addStretch(1)
        btn_dismiss = QPushButton("Dismiss")
        btn_dismiss.clicked.connect(dlg.reject)
        footer.addWidget(btn_dismiss)
        btn_now = QPushButton("Update now")
        btn_now.setDefault(True)

        def _on_now() -> None:
            dlg.accept()
            # Switch to the Updates tab so the user sees progress / fallbacks.
            idx = self._tabs.indexOf(self)
            if idx >= 0:
                self._tabs.setCurrentIndex(idx)
            if self._latest_url:
                self.install_update()
            else:
                # Asset URL not known yet (cached-restore path) — chain the
                # install onto the next check's result.
                self._install_after_next_check = True
                self.check_for_updates(silent=True)

        btn_now.clicked.connect(_on_now)
        footer.addWidget(btn_now)
        layout.addLayout(footer)

        self._dialog_fired_for = latest_version

        # Keep a reference so show() (non-blocking) doesn't let it be GC'd.
        self._pending_dialog = dlg
        dlg.finished.connect(lambda _r: setattr(self, "_pending_dialog", None))

        dlg.adjustSize()
        host = self._window.frameGeometry()
        dlg.move(
            host.center().x() - dlg.width() // 2,
            host.center().y() - dlg.height() // 2,
        )
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    # -- Install -----------------------------------------------------------
    def install_update(self) -> None:
        """Download the latest installer and launch it, then quit so it can
        overwrite the running .exe."""
        url = self._latest_url
        version = self._latest_version or "?"
        if not url:
            self._status_label.setText(
                f"<span style='color:{_C_ERROR};'>No installer URL — run Check for updates first.</span>"
            )
            return

        import tempfile

        # Unique temp name per attempt — a stale handle on a fixed name (AV
        # mid-scan, a prior aborted download) would lock the path and the next
        # open() fails with Permission denied.
        fd, dst_str = tempfile.mkstemp(suffix="-FlamingoStitcher-Setup.exe")
        os.close(fd)
        dst = Path(dst_str)

        dlg = QProgressDialog(f"Downloading Flamingo Stitcher {version}…", "Cancel", 0, 100, self)
        dlg.setWindowTitle("Flamingo Stitcher update")
        dlg.setWindowIcon(self._window.windowIcon())
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()

        cancelled = False
        try:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "FlamingoStitcher-update"})
            with urllib.request.urlopen(req, timeout=30, context=make_ssl_context()) as resp:
                total = int(resp.headers.get("Content-Length") or 0) or (self._latest_size or 0)
                downloaded = 0
                last_pct = -1
                with open(dst, "wb") as out:
                    while True:
                        if dlg.wasCanceled():
                            cancelled = True
                            break
                        buf = resp.read(1 << 20)  # 1 MB
                        if not buf:
                            break
                        out.write(buf)
                        downloaded += len(buf)
                        if total:
                            pct = int(downloaded * 100 / total)
                            if pct != last_pct:
                                last_pct = pct
                                mb_done = downloaded // (1024 * 1024)
                                mb_total = total // (1024 * 1024)
                                dlg.setLabelText(
                                    f"Downloading Flamingo Stitcher {version}…\n"
                                    f"{mb_done} / {mb_total} MB ({pct}%)"
                                )
                                dlg.setValue(pct)
                        QApplication.processEvents()
        except Exception as e:  # noqa: BLE001
            dlg.close()
            self._safe_unlink(dst)
            QMessageBox.critical(
                self,
                "Update download failed",
                f"Could not download Flamingo Stitcher {version}:\n\n{e}\n\n"
                f"Check your connection and try again, or use 'View all releases…' "
                f"to download manually.",
            )
            return
        dlg.close()

        if cancelled:
            self._safe_unlink(dst)
            self._status_label.setText(
                f"<span style='color:{_C_MUTED};'>Update download cancelled.</span>"
            )
            return

        # Guard against a truncated download that didn't raise.
        if self._latest_size and dst.stat().st_size != self._latest_size:
            self._safe_unlink(dst)
            QMessageBox.critical(
                self,
                "Update download incomplete",
                f"Downloaded size ({dst.stat().st_size} bytes) does not match the "
                f"expected size ({self._latest_size} bytes). Try again, or download "
                f"manually from the Releases page.",
            )
            return

        # Launch the installer detached and quit so it can replace files.
        try:
            import subprocess

            if sys.platform == "win32":
                DETACHED_PROCESS = 0x00000008
                subprocess.Popen([str(dst)], creationflags=DETACHED_PROCESS, close_fds=True)
            else:
                os.startfile(str(dst))  # type: ignore[attr-defined]  # noqa: SIM117 - dev convenience only
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Could not launch installer",
                f"Downloaded the installer to:\n\n{dst}\n\nbut couldn't launch it:\n{e}\n\n"
                f"Run it manually from the location above.",
            )
            return

        QApplication.quit()

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
