"""Phase-1 destriping hardening (ASLM: destriping is now essential).

Locks in three fixes:
  * a missing pystripe FAILS LOUD (no silent un-destriped output),
  * non-fast destripe runs PER ILLUMINATION SIDE, BEFORE fusion,
  * the nested per-plane destripe pool is sized against the concurrent
    preprocess-worker count (no oversubscription).
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from flamingo_stitcher import pipeline as pl  # noqa: E402
from flamingo_stitcher.pipeline import (  # noqa: E402
    RawTileInfo,
    StitchingConfig,
    StitchingPipeline,
    destripe_volume,
)


# --------------------------------------------------------------------------- #
# 1. Loud failure when the destripe backend can't load (never a silent no-op)
# --------------------------------------------------------------------------- #
def test_backend_failure_raises_not_silent(monkeypatch):
    # Simulate the vendored backend failing to import (e.g. pywt missing in a
    # broken frozen build) by shadowing it with a module lacking filter_streaks.
    broken = types.ModuleType("flamingo_stitcher._pystripe_core")
    monkeypatch.setitem(sys.modules, "flamingo_stitcher._pystripe_core", broken)
    with pytest.raises(RuntimeError, match="backend"):
        destripe_volume(np.zeros((2, 4, 4), dtype=np.uint16))


# --------------------------------------------------------------------------- #
# 2. Nested-pool budget divides by concurrent preprocess workers
# --------------------------------------------------------------------------- #
def test_destripe_worker_budget_accounts_for_nesting():
    cores = os.cpu_count() or 1

    p = StitchingPipeline(StitchingConfig(destripe_workers=None))
    p._active_preprocess_workers = 1
    assert p._destripe_worker_budget() is None  # sequential → auto-size

    p._active_preprocess_workers = 4
    assert p._destripe_worker_budget() == max(1, cores // 4)

    # explicit cap is still honored, and still bounded by the nesting budget
    p2 = StitchingPipeline(StitchingConfig(destripe_workers=2))
    p2._active_preprocess_workers = 8
    assert p2._destripe_worker_budget() == max(1, min(2, cores // 8))


# --------------------------------------------------------------------------- #
# 3. Non-fast destripe runs per side, before fusion
# --------------------------------------------------------------------------- #
def _install_spies(monkeypatch):
    """Replace load/destripe/fuse with order-recording spies."""
    calls = []

    def fake_load(path, n_planes, w, h):
        return np.zeros((3, 8, 8), dtype=np.uint16)

    def fake_destripe(volume, max_workers=None):
        # record that destripe saw a SINGLE-side-shaped volume
        calls.append(("destripe", tuple(volume.shape)))
        return volume

    def fake_fuse(volumes, method="max"):
        calls.append(("fuse", len(volumes)))
        return np.asarray(list(volumes.values())[0])

    monkeypatch.setattr(pl, "load_tile_volume", fake_load)
    monkeypatch.setattr(pl, "destripe_volume", fake_destripe)
    monkeypatch.setattr(pl, "fuse_illumination_sides", fake_fuse)
    return calls


def _tile(sides):
    return RawTileInfo(
        folder=Path("."), x_mm=0.0, y_mm=0.0, z_min_mm=0.0, z_max_mm=0.0,
        n_planes=3, illumination_sides=list(sides),
        raw_files={0: {s: Path(f"I{s}.raw") for s in sides}},
        frame_width=8, frame_height=8,
    )


def test_destripe_per_side_before_fusion_dual(monkeypatch):
    calls = _install_spies(monkeypatch)
    cfg = StitchingConfig(destripe=True, destripe_fast=False,
                          downsample_xy=1, downsample_z=1)
    pipe = StitchingPipeline(cfg)
    pipe._preprocess_single_tile(_tile([0, 1]), ch_id=0)

    # Two destripes (one per side), THEN one fuse — destripe precedes fuse.
    kinds = [c[0] for c in calls]
    assert kinds == ["destripe", "destripe", "fuse"], calls
    # each destripe saw a single-side volume (3,8,8), not a fused/stacked one
    assert calls[0][1] == (3, 8, 8) and calls[1][1] == (3, 8, 8)


def test_destripe_single_side_no_fusion(monkeypatch):
    calls = _install_spies(monkeypatch)
    cfg = StitchingConfig(destripe=True, destripe_fast=False,
                          downsample_xy=1, downsample_z=1)
    pipe = StitchingPipeline(cfg)
    pipe._preprocess_single_tile(_tile([0]), ch_id=0)

    # One side (ASLM): destripe once, no fusion at all.
    assert [c[0] for c in calls] == ["destripe"], calls
