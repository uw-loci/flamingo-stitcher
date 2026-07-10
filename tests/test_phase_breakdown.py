"""Tests for the end-of-run per-phase time breakdown."""

from __future__ import annotations

import time

from flamingo_stitcher.multi_phase_estimator import MultiPhaseEstimator
from flamingo_stitcher.timing_cache import StitchingTimingCache, StitchingTimingKey


def _make_estimator(tmp_path):
    cache = StitchingTimingCache(path=tmp_path / "timing.json")
    key = StitchingTimingKey(
        n_tiles=6,
        n_channels=1,
        n_pyramid_levels=1,
        n_timepoints=1,
        output_format="ome-zarr-sharded",
        fusion_method="content_based",
        skip_registration=False,
        planes_per_tile=200,
    )
    return MultiPhaseEstimator(cache, key)


def _seed_durations(est, durations):
    """Inject known phase durations without waiting real wall time."""
    est.start()
    est._phase_durations = dict(durations)
    # Pin elapsed to the sum so shares are deterministic.
    est._start_t = time.monotonic() - sum(durations.values())


def test_breakdown_lists_phases_in_execution_order(tmp_path):
    est = _make_estimator(tmp_path)
    _seed_durations(
        est,
        {"fuse": 60.0, "discover": 2.0, "register": 10.0, "write": 28.0},
    )
    lines = est.format_breakdown()
    body = "\n".join(lines)

    assert lines[0].startswith("=== Time breakdown ===")
    # Execution order (discover before register before fuse before write),
    # not insertion order.
    idx = {
        name: body.index(name)
        for name in ("Discover tiles", "Register tiles", "Fuse", "Write output")
    }
    assert idx["Discover tiles"] < idx["Register tiles"] < idx["Fuse"] < idx["Write output"]
    assert lines[-1].strip().startswith("TOTAL")


def test_breakdown_shares_sum_and_total(tmp_path):
    est = _make_estimator(tmp_path)
    _seed_durations(est, {"fuse": 75.0, "register": 25.0})
    lines = est.format_breakdown()

    # No untracked gap here (elapsed == sum of phases), so shares are 75/25.
    fuse_line = next(l for l in lines if "Fuse" in l)
    reg_line = next(l for l in lines if "Register tiles" in l)
    assert "75.0%" in fuse_line
    assert "25.0%" in reg_line
    # TOTAL renders the full wall time (1:40).
    assert "1:40" in lines[-1]


def test_breakdown_reports_untracked_gap(tmp_path):
    est = _make_estimator(tmp_path)
    est.start()
    est._phase_durations = {"fuse": 40.0}
    # 60s wall, only 40s attributed -> 20s Other/setup.
    est._start_t = time.monotonic() - 60.0
    lines = est.format_breakdown()
    assert any("Other/setup" in l for l in lines)


def test_breakdown_empty_for_trivial_run(tmp_path):
    est = _make_estimator(tmp_path)
    est.start()  # no phases, ~0s elapsed
    assert est.format_breakdown() == []


def test_phase_breakdown_folds_in_running_phase(tmp_path):
    est = _make_estimator(tmp_path)
    est.start_phase("fuse")
    est._current_phase_start = time.monotonic() - 5.0
    bd = est.phase_breakdown()
    assert bd["fuse"] >= 5.0
