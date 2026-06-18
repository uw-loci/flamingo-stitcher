"""One-click builder for the isolated flat-field preprocessing environment.

basicpy (flat-field correction) hard-pins scipy<1.13 while the stitcher needs
scipy>=1.14, so it runs in a separate Python environment. This module builds
that environment with **pixi** — a single self-contained binary that downloads
its own Python and resolves the packages — so the **user needs nothing
pre-installed** (no system Python, no env vars, no terminal).

Flow (all driven from the "Set up flat-field…" button):
    1. ensure_pixi()  → download the pinned pixi binary if absent
    2. build_env()    → stage the manifest + worker script, run `pixi install`
    3. env_python()   → the resulting interpreter, used by isolated_service

This mirrors the Appose/pixi pattern used by the QPSC QuPath extensions
(self-bootstrapping env, pinned tool version). CPU-only: flat-field needs no GPU.
"""

from __future__ import annotations

import logging
import os
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Pinned pixi version — keep reproducible rather than tracking "latest".
PIXI_VERSION = "0.70.2"

# Per-platform release asset (raw single-file binaries from prefix-dev/pixi).
_PIXI_ASSETS = {
    "win32": "pixi-x86_64-pc-windows-msvc.exe",
    "linux": "pixi-x86_64-unknown-linux-musl",
    "darwin-arm64": "pixi-aarch64-apple-darwin",
    "darwin-x86_64": "pixi-x86_64-apple-darwin",
}

ProgressFn = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


def _no_window_kwargs() -> dict:
    """subprocess kwargs that prevent a console window flashing on Windows."""
    if sys.platform != "win32":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
def _default_base() -> Path:
    """Default root for Flamingo preprocessing assets (under the user profile)."""
    if sys.platform == "win32":
        root = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(root) / "Flamingo"
    return Path.home() / ".flamingo"


def _pointer_file() -> Path:
    """Tiny file (always in the default base) recording the chosen install root.

    Kept at a fixed location so both the GUI builder and the runtime service
    resolve the same env even when the user installed it on another drive.
    """
    return _default_base() / "install_location.txt"


def install_root() -> Path:
    """Root under which the pixi binary + flat-field env live.

    The user can relocate this (e.g. to a drive with more space); the choice is
    persisted via :func:`set_install_root`. Falls back to the default base.
    """
    try:
        ptr = _pointer_file()
        if ptr.is_file():
            chosen = Path(ptr.read_text(encoding="utf-8").strip())
            if str(chosen):
                return chosen
    except Exception:
        pass
    return _default_base()


def set_install_root(path: Path) -> None:
    """Persist the chosen install root (created lazily). Pass a directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ptr = _pointer_file()
    ptr.parent.mkdir(parents=True, exist_ok=True)
    ptr.write_text(str(path), encoding="utf-8")


# Back-compat alias.
def base_dir() -> Path:
    return install_root()


def pixi_exe() -> Path:
    """Path to the (bundled-or-downloaded) pixi binary."""
    name = "pixi.exe" if sys.platform == "win32" else "pixi"
    return install_root() / "pixi" / name


def env_dir() -> Path:
    """Pixi project directory for the flat-field environment."""
    return install_root() / "flatfield_env"


def pixi_cache_dir() -> Path:
    """Package cache dir — kept under install_root so the chosen drive holds
    everything (the conda/PyPI cache is multi-GB and would otherwise land on C:)."""
    return install_root() / "pixi_cache"


def env_python() -> Path:
    """Path to the Python interpreter inside the built pixi environment."""
    envroot = env_dir() / ".pixi" / "envs" / "default"
    if sys.platform == "win32":
        return envroot / "python.exe"
    return envroot / "bin" / "python"


def staged_worker() -> Path:
    """Path to the standalone worker script staged into the env directory."""
    return env_dir() / "flamingo_isolated_worker.py"


def ensure_worker_staged() -> Optional[Path]:
    """Refresh the staged worker script from the (bundled) source if stale.

    The worker is copied into the env dir at *build* time, so an env built by
    an older app version keeps that older worker on disk — an app update alone
    would not pick up worker fixes without a manual "Reinstall flat-field…".
    Re-staging here (a tiny file copy) guarantees the running app always uses
    its own bundled worker. No-op if the env dir does not exist yet.
    """
    try:
        dst = staged_worker()
        if not dst.parent.is_dir():
            return None
        src = worker_source()
        new = src.read_bytes()
        if not dst.is_file() or dst.read_bytes() != new:
            dst.write_bytes(new)
            logger.info(f"Re-staged isolated worker -> {dst}")
        return dst
    except Exception as e:  # never let staging break a run
        logger.warning(f"Could not re-stage isolated worker: {e}")
        return None


def is_built() -> bool:
    """True once the environment's Python interpreter exists."""
    return env_python().is_file()


# ---------------------------------------------------------------------------
# Bundled-resource lookup (works in source tree and PyInstaller frozen build)
# ---------------------------------------------------------------------------
def _resource_path(relative: str) -> Path:
    """Locate a packaged resource by its path relative to the package root."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        p = Path(base) / "flamingo_stitcher" / relative
        if p.is_file():
            return p
    try:
        from importlib.resources import files

        p = Path(str(files("flamingo_stitcher").joinpath(relative)))
        if p.is_file():
            return p
    except Exception:
        pass
    p = Path(__file__).resolve().parent / relative
    if p.is_file():
        return p
    raise FileNotFoundError(f"Packaged resource not found: {relative}")


def manifest_source() -> Path:
    return _resource_path("preprocessing/pixi.toml")


def worker_source() -> Path:
    return _resource_path("isolated_worker.py")


# ---------------------------------------------------------------------------
# pixi bootstrap
# ---------------------------------------------------------------------------
def _pixi_asset_name() -> str:
    if sys.platform == "win32":
        return _PIXI_ASSETS["win32"]
    if sys.platform == "darwin":
        import platform

        arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
        return _PIXI_ASSETS[f"darwin-{arch}"]
    return _PIXI_ASSETS["linux"]


def pixi_download_url() -> str:
    return (
        f"https://github.com/prefix-dev/pixi/releases/download/"
        f"v{PIXI_VERSION}/{_pixi_asset_name()}"
    )


def ensure_pixi(progress: Optional[ProgressFn] = None) -> Path:
    """Download the pinned pixi binary if it isn't already present.

    Returns the path to the pixi executable.
    """
    progress = progress or _noop
    exe = pixi_exe()
    if exe.is_file():
        progress(f"pixi already present ({exe})")
        return exe

    url = pixi_download_url()
    exe.parent.mkdir(parents=True, exist_ok=True)
    tmp = exe.with_suffix(exe.suffix + ".part")
    progress(f"Downloading pixi {PIXI_VERSION} …")
    logger.info(f"Downloading pixi from {url}")
    try:
        with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted host)
            total = int(resp.headers.get("Content-Length", 0))
            if total:
                progress(f"Downloading pixi ({total // 1_000_000} MB)…")
            got = 0
            last_quarter = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    # Report only at 25% milestones (≤3 lines), not every MB.
                    if total:
                        quarter = int(got * 4 / total)
                        if quarter != last_quarter and quarter < 4:
                            last_quarter = quarter
                            progress(f"  pixi download {quarter * 25}%…")
        tmp.replace(exe)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    if sys.platform != "win32":
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    progress("pixi ready.")
    return exe


# ---------------------------------------------------------------------------
# Environment build
# ---------------------------------------------------------------------------
def _looks_like_windows_file_lock(text: str) -> bool:
    """Detect the conda/pixi Windows file-lock failure (os error 32)."""
    t = (text or "").lower()
    return "os error 32" in t or "being used by another process" in t


def build_env(progress: Optional[ProgressFn] = None) -> Path:
    """Build the flat-field environment with pixi. Returns the env's Python path.

    Stages the bundled manifest + standalone worker into the project dir, then
    runs ``pixi install`` (which downloads Python + basicpy). Idempotent: a
    second call re-syncs against the manifest.
    """
    progress = progress or _noop
    exe = ensure_pixi(progress)

    proj = env_dir()
    proj.mkdir(parents=True, exist_ok=True)

    # Stage manifest + the self-contained worker script.
    import shutil

    shutil.copyfile(manifest_source(), proj / "pixi.toml")
    shutil.copyfile(worker_source(), staged_worker())

    progress("Resolving and installing packages (this can take several minutes)…")
    cmd = [str(exe), "install", "--manifest-path", str(proj / "pixi.toml")]
    logger.info(f"Running: {' '.join(cmd)}")

    # Keep pixi's (multi-GB) package cache on the chosen drive too.
    sub_env = dict(os.environ)
    sub_env["PIXI_CACHE_DIR"] = str(pixi_cache_dir())

    # Stream pixi output line-by-line into the progress log.
    captured: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(proj),
            env=sub_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **_no_window_kwargs(),
        )
    except OSError as e:
        raise RuntimeError(f"Could not launch pixi: {e}") from e

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            captured.append(line)
            progress(line)
    rc = proc.wait()

    if rc != 0:
        blob = "\n".join(captured[-40:])
        if _looks_like_windows_file_lock(blob):
            raise RuntimeError(
                "pixi install failed due to a Windows file lock (os error 32). "
                "Another process is using a file in the environment. Close other "
                "Flamingo/Python windows and any antivirus real-time scan of "
                f"{proj}, then try again.\n\n{blob}"
            )
        raise RuntimeError(f"pixi install failed (exit {rc}):\n{blob}")

    py = env_python()
    if not py.is_file():
        raise RuntimeError(
            f"pixi install finished but the environment Python is missing at {py}."
        )
    progress("Flat-field environment ready.")
    return py
