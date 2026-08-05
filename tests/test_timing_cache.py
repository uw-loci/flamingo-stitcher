"""Tests for StitchingTimingCache, its key bucketing, and the estimator's
cache-recording lifecycle.

The whole-run ETA *math* (progress-fraction extrapolation, cold-start prior,
smoothing) lives in test_eta_estimator.py. This file covers the pieces that
feed it: how runs are keyed and bucketed, how the persistent timing cache
records/blends/reloads runs, and how MultiPhaseEstimator's phase tracking
writes a finished run back into the cache.

The estimator reads ``time.monotonic``; a FakeClock is monkeypatched in so
multi-phase timings are deterministic without real waits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flamingo_stitcher.multi_phase_estimator import (
    MultiPhaseEstimator,
    _format_duration,
)
from flamingo_stitcher.timing_cache import (
    PHASE_ORDER,
    StitchingTimingCache,
    StitchingTimingKey,
    _bucket_planes,
    _bucket_tiles,
)


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fc = FakeClock()
    monkeypatch.setattr(
        "flamingo_stitcher.multi_phase_estimator.time.monotonic", fc
    )
    return fc


def make_key(**overrides):
    base = dict(
        n_tiles=20,
        n_channels=2,
        n_pyramid_levels=4,
        n_timepoints=1,
        output_format="ome-zarr-sharded",
        fusion_method="cosine",
        skip_registration=False,
        planes_per_tile=200,
    )
    base.update(overrides)
    return StitchingTimingKey(**base)


# ---------------------------------------------------------------------------
# Key bucketing + serialization
# ---------------------------------------------------------------------------


class TestKeyBucketing:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, "1-4"),
            (4, "1-4"),
            (5, "5-9"),
            (9, "5-9"),
            (10, "10-24"),
            (24, "10-24"),
            (25, "25-49"),
            (49, "25-49"),
            (50, "50-99"),
            (99, "50-99"),
            (100, "100-249"),
            (249, "100-249"),
            (250, "250+"),
            (1000, "250+"),
        ],
    )
    def test_bucket_tiles(self, n, expected):
        assert _bucket_tiles(n) == expected

    def test_bucket_planes_bounds(self):
        assert _bucket_planes(1) == "1-50"
        assert _bucket_planes(50) == "1-50"
        assert _bucket_planes(51) == "51-150"
        assert _bucket_planes(2000) == "1000+"

    def test_key_serialization_stable(self):
        assert make_key().serialize() == make_key().serialize()

    def test_different_keys_serialize_differently(self):
        assert make_key(n_tiles=20).serialize() != make_key(n_tiles=200).serialize()

    def test_neighboring_counts_share_bucket(self):
        # 15 and 20 both fall in the 10-24 bucket -> same key
        assert make_key(n_tiles=15).serialize() == make_key(n_tiles=20).serialize()


# ---------------------------------------------------------------------------
# Persistent timing cache
# ---------------------------------------------------------------------------


class TestTimingCache:
    def test_empty_cache_returns_none(self, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        assert cache.get_total_s(make_key()) is None
        assert cache.get_phase_shares(make_key()) == {}

    def test_record_run_stores_total_and_shares(self, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        cache.record_run(
            make_key(),
            total_s=1000.0,
            phase_durations_s={
                "discover": 5.0,
                "register": 200.0,
                "fuse": 600.0,
                "write": 195.0,
            },
        )
        assert cache.get_total_s(make_key()) == pytest.approx(1000.0)
        shares = cache.get_phase_shares(make_key())
        assert shares["discover"] == pytest.approx(0.005)
        assert shares["fuse"] == pytest.approx(0.6)
        assert shares["write"] == pytest.approx(0.195)

    def test_ema_blends_runs(self, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        cache.record_run(make_key(), 1000.0, {"fuse": 600.0})
        cache.record_run(make_key(), 2000.0, {"fuse": 1000.0})
        # alpha=0.3: total = 0.3*2000 + 0.7*1000 = 1300
        assert cache.get_total_s(make_key()) == pytest.approx(1300.0)
        # share: first 0.6, then 0.5; EMA = 0.3*0.5 + 0.7*0.6 = 0.57
        assert cache.get_phase_shares(make_key())["fuse"] == pytest.approx(0.57)

    def test_persistence_round_trip(self, tmp_path: Path):
        path = tmp_path / "c.json"
        c1 = StitchingTimingCache(path=path)
        c1.record_run(make_key(), 500.0, {"fuse": 250.0})
        c2 = StitchingTimingCache(path=path)
        assert c2.get_total_s(make_key()) == pytest.approx(500.0)

    def test_different_keys_isolated(self, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        cache.record_run(make_key(n_tiles=20), 1000.0, {"fuse": 600.0})
        cache.record_run(make_key(n_tiles=200), 5000.0, {"fuse": 3000.0})
        assert cache.get_total_s(make_key(n_tiles=20)) == pytest.approx(1000.0)
        assert cache.get_total_s(make_key(n_tiles=200)) == pytest.approx(5000.0)

    def test_zero_total_ignored(self, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        cache.record_run(make_key(), 0.0, {"fuse": 100.0})
        assert cache.get_total_s(make_key()) is None

    def test_corrupt_file_recovers(self, tmp_path: Path):
        path = tmp_path / "c.json"
        path.write_text("not json")
        cache = StitchingTimingCache(path=path)
        assert cache.get_total_s(make_key()) is None
        cache.record_run(make_key(), 100.0, {"fuse": 50.0})
        assert cache.get_total_s(make_key()) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Estimator -> cache recording lifecycle (phase tracking writes runs back)
# ---------------------------------------------------------------------------


class TestEstimatorCacheRecording:
    def test_no_estimate_without_data(self, clock, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        est = MultiPhaseEstimator(cache, make_key())
        est.start()
        clock.advance(10.0)
        # No phase completed, no cache: nothing to estimate from.
        assert est.remaining_seconds() is None
        assert est.format_label() == "estimating..."

    def test_cold_start_from_cache(self, clock, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        cache.record_run(make_key(), 600.0, {"fuse": 300.0})
        est = MultiPhaseEstimator(cache, make_key())
        est.start()
        clock.advance(60.0)
        # Cached total 600s seeds the prior; 60s in -> ~540s remaining.
        assert est.remaining_seconds() == pytest.approx(540.0)

    def test_format_label_with_cached_total(self, clock, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        cache.record_run(make_key(), 120.0, {"fuse": 60.0})
        est = MultiPhaseEstimator(cache, make_key())
        est.start()
        label = est.format_label()
        assert "remaining" in label
        assert "Done at ~" in label

    def test_finalize_records_to_cache(self, clock, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        est = MultiPhaseEstimator(cache, make_key())
        est.start_phase("discover")
        clock.advance(10.0)
        est.start_phase("fuse")
        clock.advance(100.0)
        est.start_phase("write")
        clock.advance(90.0)
        est.finalize(success=True)
        assert cache.get_total_s(make_key()) == pytest.approx(200.0)
        shares = cache.get_phase_shares(make_key())
        assert shares["discover"] == pytest.approx(0.05)
        assert shares["fuse"] == pytest.approx(0.5)
        assert shares["write"] == pytest.approx(0.45)

    def test_finalize_failure_does_not_pollute_cache(self, clock, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        est = MultiPhaseEstimator(cache, make_key())
        est.start_phase("fuse")
        clock.advance(50.0)
        est.finalize(success=False)
        assert cache.get_total_s(make_key()) is None

    def test_phase_re_entry_accumulates(self, clock, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        est = MultiPhaseEstimator(cache, make_key())
        est.start_phase("fuse")
        clock.advance(30.0)
        est.start_phase("write")
        clock.advance(10.0)
        est.start_phase("fuse")  # re-entry (e.g. multi-channel)
        clock.advance(20.0)
        est.finalize(success=True)
        shares = cache.get_phase_shares(make_key())
        total = 30 + 10 + 20
        assert shares["fuse"] == pytest.approx(50.0 / total)  # 30 + 20
        assert shares["write"] == pytest.approx(10.0 / total)

    def test_short_run_not_recorded(self, clock, tmp_path: Path):
        cache = StitchingTimingCache(path=tmp_path / "c.json")
        est = MultiPhaseEstimator(cache, make_key())
        est.start_phase("discover")
        clock.advance(0.5)  # < 1s threshold
        est.finalize(success=True)
        assert cache.get_total_s(make_key()) is None


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


class TestFormatDuration:
    @pytest.mark.parametrize(
        "secs,expected",
        [
            (0, "0s"),
            (59, "59s"),
            (60, "1:00"),
            (3599, "59:59"),
            (3600, "1:00:00"),
            (3725, "1:02:05"),
        ],
    )
    def test_basics(self, secs, expected):
        assert _format_duration(secs) == expected


class TestPhaseOrderInvariants:
    def test_phase_order_has_expected_entries(self):
        for p in ("discover", "register", "preprocess", "fuse", "write"):
            assert p in PHASE_ORDER


# --- Tile-count scaling and preprocessing key axes -------------------------


def _key(n_tiles, **kw):
    """A realistic key; overrides let a test vary one axis at a time."""
    from flamingo_stitcher.timing_cache import StitchingTimingKey

    base = dict(
        n_tiles=n_tiles,
        n_channels=1,
        n_pyramid_levels=-1,
        n_timepoints=1,
        output_format="ome-tiff",
        fusion_method="cosine",
        skip_registration=True,
        planes_per_tile=34,
        downsample_xy=2,
        downsample_z=1,
        source_drive="D:",
        dest_drive="G:",
    )
    base.update(kw)
    return StitchingTimingKey(**base)


class TestTileCountScaling:
    """Tile counts are bucketed, and the widest bucket spans 2.5x.

    Returning a bucket's mean unscaled let a 249-tile run's timing seed a
    135-tile run: 204 s predicted for a run that took 115 s, almost exactly the
    249/135 ratio. Wall time is near-linear in tile count for fixed geometry.
    """

    def test_cached_total_scales_to_the_actual_tile_count(self, tmp_path):
        from flamingo_stitcher.timing_cache import StitchingTimingCache

        cache = StitchingTimingCache(path=tmp_path / "timing.json")
        # Both counts live in the same "100-249" bucket.
        cache.record_run(_key(249), 204.0, {"preprocess": 126.0}, n_tiles=249)

        scaled = cache.get_total_s(_key(135), n_tiles=135)

        assert scaled is not None
        # 204 * 135/249 = 110.6, against a real measured 115 s.
        assert 100.0 < scaled < 120.0, scaled

    def test_unscaled_when_the_entry_predates_tile_tracking(self, tmp_path):
        from flamingo_stitcher.timing_cache import StitchingTimingCache

        cache = StitchingTimingCache(path=tmp_path / "timing.json")
        cache.record_run(_key(249), 204.0, {"preprocess": 126.0})  # no n_tiles

        assert cache.get_total_s(_key(135), n_tiles=135) == 204.0

    def test_unscaled_when_the_caller_gives_no_tile_count(self, tmp_path):
        from flamingo_stitcher.timing_cache import StitchingTimingCache

        cache = StitchingTimingCache(path=tmp_path / "timing.json")
        cache.record_run(_key(249), 204.0, {"preprocess": 126.0}, n_tiles=249)

        assert cache.get_total_s(_key(135)) == 204.0

    def test_implausible_ratio_falls_back_instead_of_extrapolating(self, tmp_path):
        """The 250+ bucket is unbounded, so guard the linearity assumption."""
        from flamingo_stitcher.timing_cache import StitchingTimingCache

        cache = StitchingTimingCache(path=tmp_path / "timing.json")
        cache.record_run(_key(250), 200.0, {"preprocess": 120.0}, n_tiles=250)

        # 2000/250 = 8x, past the 4x trust limit → raw mean, not 1600 s.
        assert cache.get_total_s(_key(2000), n_tiles=2000) == 200.0


class TestPreprocessingKeyAxes:
    """Destripe/deconv/flat-field dominate preprocess, so they must key apart.

    Otherwise a destripe-on and a destripe-off run average into one number that
    fits neither — the same reasoning that already puts downsample in the key.
    """

    def test_destripe_changes_the_key(self):
        assert _key(135).serialize() != _key(135, destripe=True).serialize()

    def test_fast_destripe_is_its_own_bucket(self):
        """It filters the DOWNSAMPLED tile — a different cost class."""
        slow = _key(135, destripe=True).serialize()
        fast = _key(135, destripe=True, destripe_fast=True).serialize()
        assert slow != fast

    def test_deconvolution_and_flat_field_change_the_key(self):
        base = _key(135).serialize()
        assert _key(135, deconvolution=True).serialize() != base
        assert _key(135, flat_field=True).serialize() != base

    def test_destripe_runs_do_not_share_a_cache_entry(self, tmp_path):
        from flamingo_stitcher.timing_cache import StitchingTimingCache

        cache = StitchingTimingCache(path=tmp_path / "timing.json")
        cache.record_run(_key(135), 70.0, {"preprocess": 30.0}, n_tiles=135)

        on = _key(135, destripe=True, destripe_fast=True)
        assert cache.get_total_s(on, n_tiles=135) is None  # its own cold start

    def test_defaults_keep_the_key_stable_for_a_plain_run(self):
        """A run with no preprocessing must serialize the off-state, not vary."""
        s = _key(135).serialize()
        assert "ds=00" in s and "dec=0" in s and "ff=0" in s
