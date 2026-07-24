"""Discovery must survive a corrupt/unreadable per-tile metadata file.

An acquisition on an external (USB-C) drive can have a single corrupt
``*_Settings.txt`` companion — the OS raises ``OSError`` (on Windows,
``[WinError 1392] The file or directory is corrupted and unreadable``) when it
is read, even though every other file on the drive reads fine. Before the fix
that one read propagated out of ``discover_flat_tiles`` and the GUI reported
"Discovered 0 tiles across 0/1 directories" — i.e. one bad metadata file threw
away the entire acquisition.

Two guarantees are locked in here:
  1. ``_read_text_resilient`` retries a transient read error and recovers.
  2. ``discover_flat_tiles`` degrades a tile whose ``_Settings.txt`` stays
     unreadable to the grid-computed position (from the root Workflow.txt) and
     still returns *every* tile, so the mosaic can stitch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from flamingo_stitcher import pipeline  # noqa: E402
from flamingo_stitcher.pipeline import (  # noqa: E402
    _read_text_resilient,
    discover_flat_tiles,
)

_AOI = 4  # tiny 4x4 frame keeps the synthetic .raw files trivially small
_N_PLANES = 2

_ROOT_WORKFLOW = """<Workflow Settings>
  <Experiment Settings>
    Plane spacing (um) = 10
  </Experiment Settings>
  <Camera Settings>
    AOI width = 4
    AOI height = 4
  </Camera Settings>
  <Start Position>
    X (mm) = 1.000
    Y (mm) = 2.000
    Z (mm) = 10.000
  </Start Position>
  <End Position>
    X (mm) = 3.000
    Y (mm) = 2.000
    Z (mm) = 12.000
  </End Position>
</Workflow Settings>
"""


def _settings_text(x_mm: float, y_mm: float) -> str:
    return (
        "<Workflow Settings>\n"
        "  <Start Position>\n"
        f"    X (mm) = {x_mm}\n"
        f"    Y (mm) = {y_mm}\n"
        "    Z (mm) = 10.000\n"
        "  </Start Position>\n"
        "  <End Position>\n"
        "    Z (mm) = 12.000\n"
        "  </End Position>\n"
        "</Workflow Settings>\n"
    )


def _raw_name(x_idx: int, y_idx: int) -> str:
    return (
        f"S000_t000000_V000_R0000_X{x_idx:03d}_Y{y_idx:03d}"
        f"_C02_I0_D1_P{_N_PLANES:05d}.raw"
    )


def _write_flat_acq(tmp_path: Path) -> Path:
    """Two flat tiles (X000_Y000, X001_Y000) with valid raw + settings files."""
    acq = tmp_path / "20260723_beadstiling"
    acq.mkdir()
    (acq / "Workflow.txt").write_text(_ROOT_WORKFLOW)

    raw_bytes = b"\x00" * (_N_PLANES * _AOI * _AOI * 2)  # uint16 payload
    for x_idx in (0, 1):
        raw = acq / _raw_name(x_idx, 0)
        raw.write_bytes(raw_bytes)
        settings = raw.with_name(raw.stem + "_Settings.txt")
        # Real settings positions are deliberately far from the grid values so
        # the test can tell which source a tile's position came from.
        settings.write_text(_settings_text(x_mm=50.0 + x_idx, y_mm=60.0))
    return acq


def test_read_text_resilient_recovers_transient(tmp_path, monkeypatch):
    """A transient OSError on read is retried and the read ultimately succeeds."""
    p = tmp_path / "meta.txt"
    p.write_text("payload")

    real_read = Path.read_text
    state = {"failures": 2}

    def flaky(self, *args, **kwargs):
        if self == p and state["failures"] > 0:
            state["failures"] -= 1
            raise OSError(1392, "The file or directory is corrupted and unreadable")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _s: None)  # no real delay

    assert _read_text_resilient(p) == "payload"
    assert state["failures"] == 0  # both transient failures were consumed


def test_read_text_resilient_reraises_persistent(tmp_path, monkeypatch):
    """A persistently corrupt file re-raises OSError after exhausting retries."""
    p = tmp_path / "meta.txt"
    p.write_text("payload")

    def always_corrupt(self, *args, **kwargs):
        raise OSError(1392, "The file or directory is corrupted and unreadable")

    monkeypatch.setattr(Path, "read_text", always_corrupt)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _s: None)

    with pytest.raises(OSError):
        _read_text_resilient(p, retries=3)


def test_discovery_survives_one_corrupt_settings_file(tmp_path, monkeypatch):
    """One corrupt _Settings.txt must not drop the whole acquisition.

    The corrupt tile falls back to the grid-computed position; every tile is
    still discovered so the mosaic can stitch.
    """
    acq = _write_flat_acq(tmp_path)
    corrupt_name = _raw_name(0, 0).replace(".raw", "_Settings.txt")

    real_read = Path.read_text

    def corrupt_one(self, *args, **kwargs):
        if self.name == corrupt_name:
            raise OSError(1392, "The file or directory is corrupted and unreadable")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", corrupt_one)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _s: None)

    tiles = discover_flat_tiles(acq)

    # Both tiles present — discovery was NOT aborted by the corrupt file.
    assert len(tiles) == 2

    # The healthy tile (X001) kept its real settings position (~51.0).
    healthy = next(t for t in tiles if t.x_mm > 40.0)
    assert healthy.x_mm == pytest.approx(51.0)

    # The corrupt tile fell back to the grid position from Workflow.txt
    # (Start X = 1.0, so tile index 0 -> 1.0), NOT its (unreadable) 50.0.
    fell_back = next(t for t in tiles if t.x_mm < 40.0)
    assert fell_back.x_mm == pytest.approx(1.0)

    # The degraded tile is FLAGGED so the GUI/CLI can warn the user visibly,
    # not just log it. The healthy tile carries no warning.
    assert fell_back.metadata_warning is not None
    assert "metadata" in fell_back.metadata_warning.lower()
    assert healthy.metadata_warning is None


def test_raw_size_warning_flags_truncated_and_ok(tmp_path):
    """A short .raw is flagged as truncated; a correctly-sized one is not."""
    from flamingo_stitcher.pipeline import _raw_size_warning

    fw = fh = 8
    n = 2
    expected = n * fw * fh * 2  # 256 bytes

    ok = tmp_path / "ok.raw"
    ok.write_bytes(b"\x00" * expected)
    assert _raw_size_warning(ok, n, fw, fh) is None

    short = tmp_path / "short.raw"
    short.write_bytes(b"\x00" * (expected // 2))
    w = _raw_size_warning(short, n, fw, fh)
    assert w is not None and "truncated" in w.lower()

    # Non-raw (e.g. compressed TIFF) is not size-checked.
    tif = tmp_path / "x.tif"
    tif.write_bytes(b"\x00" * 4)
    assert _raw_size_warning(tif, n, fw, fh) is None
