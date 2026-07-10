"""Tests for the progress-anchored whole-run ETA.

The ETA extrapolates linearly from the pipeline's monotone global progress
fraction (remaining = elapsed*(1-f)/f), with a cold-start prior. These tests
pin the two properties that matter to the user: it tracks a steady pace
accurately, and it does NOT swing wildly (the old share-division bug).
"""

from __future__ import annotations

import time

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
