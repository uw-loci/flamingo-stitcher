"""Multi-phase stitching ETA estimator.

The whole-run "remaining time" is anchored to the pipeline's own **global
progress fraction** (the monotone 0–100% the pipeline already emits, which
advances per-tile through preprocessing and per-region through fusion), via
honest linear extrapolation from elapsed wall time::

    remaining = elapsed * (1 - f) / f

This is self-calibrating (it uses how long the run has *actually* taken to
reach fraction ``f``, so a slow disk or a slow machine is absorbed
automatically) and monotone in ``f`` — it cannot do the wild 3 min → 5 h
swings the old share-division scheme produced. That scheme inferred the total
by dividing a phase's *observed time* by its cached *share of total*, which,
with no within-phase progress signal, made the estimate balloon the longer any
one phase ran.

Two refinements keep the early estimate sane:

* **Cold-start prior.** Before there's meaningful progress (``f`` tiny /
  little elapsed), fall back to a prior total — the cached mean wall time for
  this config if we have run it before, else a rough model passed in by the
  caller. The live extrapolation is blended in as ``f`` grows past a small
  threshold, so there's no discontinuity when it takes over.
* **Smoothing.** The reported value is EMA-smoothed so per-emit jitter and the
  small step at a phase boundary don't jump the clock around.

Per-phase wall clocks are still tracked (``start_phase`` / ``end_phase``) — the
memory watchdog and the end-of-run time breakdown depend on them — they just no
longer drive the ETA.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

from flamingo_stitcher.timing_cache import (
    PHASE_ORDER,
    StitchingTimingCache,
    StitchingTimingKey,
)

logger = logging.getLogger(__name__)

# Progress-fraction band over which the live extrapolation takes over from the
# cold-start prior. Below _F_LO the run is too early to trust elapsed/f; above
# _F_HI the extrapolation is used alone; between, the two are blended.
_F_LO = 0.05
_F_HI = 0.25

# EMA smoothing for the reported remaining time (per query). Higher = snappier.
_SMOOTH_ALPHA = 0.35

# Don't report a live ETA until this much wall time has passed — a couple of
# seconds in, elapsed is too small to extrapolate from.
_MIN_ELAPSED_S = 5.0


class MultiPhaseEstimator:
    """Live ETA across the discrete phases of a stitching run.

    Lifecycle::

        est = MultiPhaseEstimator(cache, key)
        est.start()                  # at run start
        est.start_phase("discover")  # entering a phase
        est.end_phase("discover")    # leaving a phase (or start_phase
                                     # of the next phase auto-ends it)
        ...
        eta = est.format_label()     # any time after start()
        est.finalize()               # at run end (writes to cache)
    """

    def __init__(
        self,
        cache: StitchingTimingCache,
        key: StitchingTimingKey,
        prior_total_s: Optional[float] = None,
    ):
        self._cache = cache
        self._key = key
        self._start_t: Optional[float] = None
        self._current_phase: Optional[str] = None
        self._current_phase_start: Optional[float] = None
        self._phase_durations: Dict[str, float] = {}

        # Live global progress fraction (0..1), fed by the pipeline's monotone
        # percent. Kept monotone non-decreasing so a writer that reports its own
        # local 0..100 can't drag the whole-run fraction backwards.
        self._frac: float = 0.0
        self._smoothed_remaining: Optional[float] = None

        # Pass the ACTUAL tile count so a bucketed cache entry is scaled to
        # this run's size instead of inherited whole (see get_total_s).
        self._cached_total_s = cache.get_total_s(key, n_tiles=key.n_tiles)
        self._cached_shares = cache.get_phase_shares(key)
        # Cold-start prior: prefer the measured mean for this config; else the
        # caller's rough model. Used until live progress can carry the estimate.
        self._prior_total_s = self._cached_total_s or prior_total_s
        if self._cached_total_s:
            logger.info(
                f"Stitching ETA: seeded from cache (total ~{self._cached_total_s:.0f}s)"
            )
        elif prior_total_s:
            logger.info(
                f"Stitching ETA: cold start, rough prior ~{prior_total_s:.0f}s "
                f"(refined live from progress)"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._start_t is None:
            self._start_t = time.monotonic()

    def start_phase(self, phase: str) -> None:
        """Mark the start of a phase. If another phase was running it's
        ended first."""
        if phase not in PHASE_ORDER:
            logger.debug(f"Unknown phase '{phase}', tracking anyway")
        if self._current_phase is not None:
            self.end_phase(self._current_phase)
        self.start()
        self._current_phase = phase
        self._current_phase_start = time.monotonic()

    def end_phase(self, phase: str) -> None:
        """End the named phase. No-op if it isn't the current phase."""
        if self._current_phase != phase or self._current_phase_start is None:
            return
        dur = time.monotonic() - self._current_phase_start
        # If we re-enter the same phase later (e.g. multi-channel fuse
        # reports the same status repeatedly), accumulate rather than
        # overwrite.
        self._phase_durations[phase] = self._phase_durations.get(phase, 0.0) + max(
            0.0, dur
        )
        self._current_phase = None
        self._current_phase_start = None

    def finalize(self, success: bool = True) -> None:
        """End the current phase and, on success, push timings to cache."""
        if self._current_phase is not None:
            self.end_phase(self._current_phase)
        if not success:
            return  # don't poison cache with failed/cancelled runs
        elapsed = self.elapsed_seconds()
        if elapsed is None or elapsed < 1.0:
            return  # not worth recording (presumably aborted)
        self._cache.record_run(
            self._key,
            elapsed,
            dict(self._phase_durations),
            n_tiles=self._key.n_tiles,
        )
        logger.info(
            f"Stitching ETA: recorded run "
            f"(total={elapsed:.0f}s, phases={self._phase_durations})"
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def elapsed_seconds(self) -> Optional[float]:
        if self._start_t is None:
            return None
        return time.monotonic() - self._start_t

    def update_fraction(self, frac: float) -> None:
        """Feed the pipeline's global progress fraction (0..1) and re-estimate.

        The fraction is kept monotone non-decreasing (a writer reporting its own
        local percent must not drag the whole-run fraction backwards). The EMA
        commit happens here — once per progress emit — so ``remaining_seconds``
        can be a side-effect-free read that callers may hit multiple times.
        """
        try:
            f = float(frac)
        except (TypeError, ValueError):
            return
        if f != f:  # NaN
            return
        f = min(1.0, max(0.0, f))
        if f > self._frac:
            self._frac = f
        raw = self._raw_remaining()
        if raw is None:
            return
        if self._smoothed_remaining is None:
            self._smoothed_remaining = raw
        else:
            self._smoothed_remaining = (
                _SMOOTH_ALPHA * raw + (1.0 - _SMOOTH_ALPHA) * self._smoothed_remaining
            )

    def _raw_remaining(self) -> Optional[float]:
        """Unsmoothed remaining-seconds estimate from current elapsed + fraction.
        Pure (no state mutation)."""
        elapsed = self.elapsed_seconds()
        if elapsed is None:
            return None

        f = self._frac
        # Live linear extrapolation from actual progress: reaching fraction f in
        # `elapsed` seconds implies elapsed*(1-f)/f left. Self-calibrating,
        # monotone in f.
        live = None
        if f > 1e-6 and elapsed >= _MIN_ELAPSED_S:
            live = elapsed * (1.0 - f) / f

        # Cold-start prior (cached mean or rough model), decremented by elapsed.
        prior = None
        if self._prior_total_s:
            prior = max(0.0, self._prior_total_s - elapsed)

        # Blend: prior early, live once there's enough progress to trust it.
        if live is not None and prior is not None:
            if f >= _F_HI:
                raw = live
            elif f <= _F_LO:
                raw = prior
            else:
                w = (f - _F_LO) / (_F_HI - _F_LO)
                raw = (1.0 - w) * prior + w * live
        elif live is not None:
            raw = live
        elif prior is not None:
            raw = prior
        else:
            return None
        return max(0.0, raw)

    def remaining_seconds(self) -> Optional[float]:
        """Smoothed remaining-seconds estimate (side-effect-free read).

        Returns the EMA committed by :meth:`update_fraction`. Before any
        progress emit it falls back to the raw prior estimate so an early
        caller still gets a number rather than ``None`` once enough wall time
        has passed.
        """
        if self._smoothed_remaining is not None:
            return max(0.0, self._smoothed_remaining)
        return self._raw_remaining()

    def eta_clock(self) -> Optional[datetime]:
        rem = self.remaining_seconds()
        if rem is None:
            return None
        return datetime.now() + timedelta(seconds=rem)

    def format_remaining(self) -> str:
        rem = self.remaining_seconds()
        if rem is None:
            return "estimating..."
        return _format_duration(rem)

    def format_eta(self) -> str:
        clock = self.eta_clock()
        if clock is None:
            return "--:--"
        if clock.date() == datetime.now().date():
            return clock.strftime("%H:%M")
        return clock.strftime("%a %H:%M")

    def format_label(self) -> str:
        """``"M:SS remaining (Done at ~HH:MM)"`` or ``"estimating..."``."""
        rem = self.remaining_seconds()
        if rem is None:
            return "estimating..."
        return f"{_format_duration(rem)} remaining (Done at ~{self.format_eta()})"

    def phase_breakdown(self) -> Dict[str, float]:
        """Accumulated wall seconds per phase, with the in-progress phase's
        partial time folded in so a breakdown read mid-phase is complete."""
        out = dict(self._phase_durations)
        if self._current_phase is not None and self._current_phase_start is not None:
            out[self._current_phase] = out.get(self._current_phase, 0.0) + max(
                0.0, time.monotonic() - self._current_phase_start
            )
        return out

    def format_breakdown(self, title: str = "Time breakdown") -> list:
        """Human-readable per-phase wall-time breakdown for the run log.

        One line per observed phase (in execution order) with its wall time
        and share of the tracked total, a trailing ``Other/setup`` line for
        wall time not attributed to any phase (setup, gaps between status
        messages, teardown), and a ``TOTAL`` line. Returns ``[]`` if the run
        was too short to be worth reporting.
        """
        total = self.elapsed_seconds()
        if total is None or total < 1.0:
            return []
        durations = self.phase_breakdown()
        if not durations:
            return []

        labels = {
            "discover": "Discover tiles",
            "register": "Register tiles",
            "preprocess": "Load + preprocess",
            "fuse": "Fuse",
            "write": "Write output",
            "metadata": "Write metadata",
        }
        # Known phases first (execution order), then any unexpected ones.
        ordered = [p for p in PHASE_ORDER if p in durations]
        ordered += [p for p in durations if p not in PHASE_ORDER]

        tracked = sum(durations.values())
        untracked = max(0.0, total - tracked)

        rows = [(labels.get(p, p), durations[p]) for p in ordered]
        if untracked >= 1.0:
            rows.append(("Other/setup", untracked))
        name_w = max(len(name) for name, _ in rows)

        lines = [f"=== {title} ==="]
        for name, dur in rows:
            pct = (100.0 * dur / total) if total > 0 else 0.0
            lines.append(
                f"  {name:<{name_w}}  {_format_duration(dur):>9}  ({pct:5.1f}%)"
            )
        lines.append(f"  {'TOTAL':<{name_w}}  {_format_duration(total):>9}")
        return lines


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}:{s:02d}"
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}"
