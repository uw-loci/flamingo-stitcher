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


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
def base_dir() -> Path:
    """Root for the shared Flamingo preprocessing assets (pixi + env)."""
    if sys.platform == "win32":
        root = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(root) / "Flamingo"
    return Path.home() / ".flamingo"


def pixi_exe() -> Path:
    """Path to the (bundled-or-downloaded) pixi binary."""
    name = "pixi.exe" if sys.platform == "win32" else "pixi"
    return base_dir() / "pixi" / name


def env_dir() -> Path:
    """Pixi project directory for the flat-field environment."""
    return base_dir() / "flatfield_env"


def env_python() -> Path:
    """Path to the Python interpreter inside the built pixi environment."""
    envroot = env_dir() / ".pixi" / "envs" / "default"
    if sys.platform == "win32":
        return envroot / "python.exe"
    return envroot / "bin" / "python"


def staged_worker() -> Path:
    """Path to the standalone worker script staged into the env directory."""
    return env_dir() / "flamingo_isolated_worker.py"


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
            got = 0
            last_mb = -1
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    mb = got // 1_000_000
                    if total and mb != last_mb:  # report at most once per MB
                        last_mb = mb
                        progress(f"Downloading pixi … {mb}/{total // 1_000_000} MB")
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

    # Stream pixi output line-by-line into the progress log.
    captured: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(proj),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
