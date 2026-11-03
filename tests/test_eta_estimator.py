"""Tests for the progress-anchored whole-run ETA.

The ETA extrapolates linearly from the pipeline's monotone global progress
fraction (remaining = elapsed*(1-f)/f), with a cold-start prior. These tests
pin the two properties that matter to the user: it tracks a steady pace
accurately, and it does NOT swing wildly (the old share-division bug).
"""

from __future__ import annotations

import time

import pytest

from flamingo_stitcher.multi_phase_estimator import MultiPhaseEstimator
from flamingo_stitcher.timing_cache import StitchingTimingCache, StitchingTimingKey


def _make(tmp_path, prior=None):
    cache = StitchingTimingCache(path=tmp_path / "t.json")
    key = StitchingTimingKey(
        n_tiles=176,
        n_channels=1,
        n_pyramid_levels=1,
        n_timepoints=1,
        output_format="imaris",
        fusion_method="cosine",
        skip_registration=True,
        planes_per_tile=1662,
    )
    return MultiPhaseEstimator(cache, key, prior_total_s=prior)


def _at(est, elapsed_s, frac):
    """Force a given elapsed time + progress fraction, return remaining."""
    est.start()
    est._start_t = time.monotonic() - elapsed_s
    est.update_fraction(frac)
    return est.remaining_seconds()


def test_cold_start_uses_prior(tmp_path):
    est = _make(tmp_path, prior=600.0)
    # No progress yet, 10 s in -> prior minus elapsed.
    rem = _at(est, 10.0, 0.0)
    assert 580 <= rem <= 600


def test_linear_extrapolation_when_progressing(tmp_path):
    # At f=0.5 after 100 s, the remaining half should take ~another 100 s.
    est = _make(tmp_path, prior=600.0)
    rem = _at(est, 100.0, 0.5)
    assert abs(rem - 100.0) < 1.0


def test_fraction_is_monotone(tmp_path):
    est = _make(tmp_path)
    est.update_fraction(0.4)
    est.update_fraction(0.2)  # a writer's local percent must not drag it back
    assert est._frac == 0.4


def test_tracks_steady_pace_without_swinging(tmp_path):
    # A run whose true total is ~1000 s, sampled at a CONSTANT pace. Each
    # estimate (fresh estimator, so no EMA carryover) should land near the true
    # remaining -- never the 3 min -> 30 min -> 5 h explosion of the old scheme.
    true_total = 1000.0
    for elapsed, f in [(250, 0.25), (500, 0.50), (750, 0.75), (900, 0.90)]:
        est = _make(tmp_path, prior=600.0)  # deliberately WRONG prior
        rem = _at(est, elapsed, f)
        expected = true_total - elapsed
        # Within 15% of truth despite the bad prior -- because past _F_HI the
        # live extrapolation dominates.
        assert abs(rem - expected) <= 0.15 * true_total, (elapsed, f, rem)


def test_estimating_when_no_signal(tmp_path):
    est = _make(tmp_path, prior=None)
    # 2 s in, no progress, no prior -> nothing to say.
    est.start()
    est._start_t = time.monotonic() - 2.0
    assert est.remaining_seconds() is None
    assert est.format_label() == "estimating..."


def test_smoothing_damps_a_phase_boundary_step(tmp_path):
    # A downward step in fraction-implied remaining is smoothed, not instant.
    est = _make(tmp_path, prior=1000.0)
    est.start()
    est._start_t = time.monotonic() - 400.0
    est.update_fraction(0.40)
    first = est.remaining_seconds()  # ~600 (live)
    est.update_fraction(0.55)  # crossed into next phase; live drops
    second = est.remaining_seconds()
    # EMA: the second reading moved toward the new value but not all the way.
    live_new = 400.0 * (1 - 0.55) / 0.55
    assert live_new < second < first


def test_each_phase_start_logs_its_prediction_and_the_end_checks_it(tmp_path, caplog):
    """The live ETA was never logged, so its accuracy could not be checked.

    2026-10-27: only the opening prior reached the log. Each phase start now
    logs the predicted total, and the end of the run compares every one of
    them with what the run actually took.
    """
    import logging

    est = _make(tmp_path, prior=1000.0)
    caplog.set_level(logging.INFO, logger="flamingo_stitcher.multi_phase_estimator")
    _at(est, 300.0, 0.5)  # on pace for 600 s
    est.start_phase("fuse")
    assert any("at the start of fuse" in r.message for r in caplog.records)
    labels = [label for label, _ in est._predictions]
    assert labels == ["start", "fuse"]
    assert est._predictions[1][1] < est._predictions[0][1]

    line = est.format_prediction_check(600.0)
    assert "start 16:40 (actual/predicted 0.60)" in line
    assert "fuse" in line


def test_a_phase_started_too_early_to_extrapolate_is_not_logged(tmp_path):
    est = _make(tmp_path, prior=1000.0)
    est.start_phase("discover")  # elapsed ~0
    assert [label for label, _ in est._predictions] == ["start"]


# --------------------------------------------------------------------------
# A phase that reports its own progress is extrapolated on its own clock
# --------------------------------------------------------------------------


def _make_phased(tmp_path, prior_phase_s):
    cache = StitchingTimingCache(path=tmp_path / "t.json")
    key = StitchingTimingKey(
        n_tiles=42, n_channels=1, n_pyramid_levels=1, n_timepoints=1,
        output_format="ome-tiff", fusion_method="content_based",
        skip_registration=False, planes_per_tile=2999,
    )
    return MultiPhaseEstimator(
        cache, key, prior_total_s=sum(prior_phase_s.values()),
        prior_phase_s=prior_phase_s,
    )


RUN35_PRIOR = {"preprocess": 11000.0, "register": 11000.0, "fuse": 44000.0, "write": 15000.0}


def test_fusion_is_extrapolated_from_its_regions_not_the_whole_run(tmp_path):
    """Run 35: 72 of 90 regions after 70 h of fusion, 6 h of earlier phases.

    The whole-run extrapolation, with bands from a prior that put fusion at
    a quarter of its real length, kept projecting the earlier phases' pace.
    Fusion's own pace says 17.5 h of fusion left, then the write."""
    est = _make_phased(tmp_path, RUN35_PRIOR)
    est.start()
    est._start_t = time.monotonic() - (6 + 70) * 3600.0
    est.start_phase("fuse")
    est.set_phase_progress("fuse", 72 / 90, 70 * 3600.0)
    est.update_fraction(0.80)
    rem = est.remaining_seconds()
    assert rem == pytest.approx(17.5 * 3600 + 15000.0, rel=0.01)


def test_the_phase_estimate_counts_down_between_reports(tmp_path):
    est = _make_phased(tmp_path, RUN35_PRIOR)
    est.start()
    est.start_phase("fuse")
    est.set_phase_progress("fuse", 0.5, 1000.0)
    est._phase_progress = (*est._phase_progress[:3], time.monotonic() - 300.0)
    assert est.phase_remaining_seconds("fuse") == pytest.approx(700.0, abs=1.0)


def test_the_first_phase_report_replaces_the_smoothed_estimate(tmp_path):
    """EMA smoothing must not drag the old, wrong number along for hours."""
    est = _make_phased(tmp_path, RUN35_PRIOR)
    est.start()
    est._start_t = time.monotonic() - 40000.0
    est.start_phase("fuse")
    est.update_fraction(0.5)                      # whole-run: ~40000 s left
    est.set_phase_progress("fuse", 0.1, 3600.0)  # fusion's pace: 32400 s + write
    est.update_fraction(0.51)
    assert est.remaining_seconds() == pytest.approx(32400.0 + 15000.0, rel=0.01)


def test_another_phase_falls_back_to_the_whole_run_fraction(tmp_path):
    est = _make_phased(tmp_path, RUN35_PRIOR)
    est.start()
    est._start_t = time.monotonic() - 100.0
    est.start_phase("fuse")
    est.set_phase_progress("fuse", 0.5, 50.0)
    est.start_phase("write")
    est.update_fraction(0.5)
    assert est.remaining_seconds() == pytest.approx(100.0, rel=0.05)


def test_later_phases_come_from_the_prior(tmp_path):
    est = _make_phased(tmp_path, RUN35_PRIOR)
    assert est._later_phases_s("fuse") == pytest.approx(15000.0)
    assert est._later_phases_s("write") == 0.0
