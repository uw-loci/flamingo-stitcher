"""Unit tests for the pixi flat-field environment builder (pure logic only).

The full build (download pixi → install basicpy) is validated manually on the
target; here we lock down the path/URL/resource-resolution logic so a refactor
can't silently break the bootstrap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flamingo_stitcher import preprocessing_env as pe  # noqa: E402


def test_pixi_url_is_pinned_and_per_platform():
    url = pe.pixi_download_url()
    assert f"/v{pe.PIXI_VERSION}/" in url
    assert url.startswith("https://github.com/prefix-dev/pixi/releases/download/")
    assert pe._pixi_asset_name() in url


def test_asset_name_known_targets():
    assert pe._PIXI_ASSETS["win32"] == "pixi-x86_64-pc-windows-msvc.exe"
    assert pe._PIXI_ASSETS["linux"] == "pixi-x86_64-unknown-linux-musl"


def test_env_python_under_env_dir():
    py = pe.env_python()
    assert pe.env_dir() in py.parents
    assert ".pixi" in py.parts and "envs" in py.parts


def test_pixi_exe_name_matches_platform():
    name = pe.pixi_exe().name
    assert name == ("pixi.exe" if sys.platform == "win32" else "pixi")


def test_bundled_resources_resolve():
    # Manifest + standalone worker must be locatable in the source tree.
    assert pe.manifest_source().is_file()
    assert pe.worker_source().is_file()
    text = pe.manifest_source().read_text()
    assert "basicpy" in text and "scipy" in text


def test_windows_file_lock_detector():
    assert pe._looks_like_windows_file_lock("error: os error 32 blah")
    assert pe._looks_like_windows_file_lock("file being used by another process")
    assert not pe._looks_like_windows_file_lock("some unrelated error")
