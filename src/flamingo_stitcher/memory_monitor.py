"""Background RSS sampler + peak tracker.

Bounded peak RAM is a guarantee the stitching pipeline must keep as new stages
are added (see docs / the memory-boundedness plan). This module is the shared
measurement primitive behind both halves of "independent checking":

  * the offline **scaling test** (``tests/test_memory_scaling.py``) samples peak
    RSS at several dataset sizes and asserts it stays flat — any new O(dataset)
    allocation makes the slope go positive and fails CI; and
  * the in-run **watchdog** warns (visually, via the GUI) when live RSS exceeds
    the projected bound, so an unmodeled allocation surfaces loud + attributable
    instead of an opaque OS OOM.

The monitor is deliberately dependency-light (psutil only, already required) and
degrades to a no-op when psutil is unavailable, so importing it can never break
a headless or frozen run.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Optional

try:
    import psutil

    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover - psutil is a hard dep, but never crash
    _HAVE_PSUTIL = False


class MemoryMonitor:
    """Sample this process's RSS on a background thread and track the peak.

    Usage (context manager)::

        with MemoryMonitor() as mon:
            mon.set_phase("fuse")
            ... work ...
        print(mon.peak_delta_bytes, mon.phase_peaks_delta())

    Optionally pass ``threshold_bytes`` + ``on_exceed`` to get a one-shot
    callback the first time RSS crosses the threshold — this is how the runtime
    watchdog raises its (non-blocking, warn-only) popup without aborting the run.
    """

    def __init__(
        self,
        interval_s: float = 0.05,
        threshold_bytes: Optional[int] = None,
        on_exceed: Optional[Callable[[int, Optional[str]], None]] = None,
        metric: str = "uss",
    ) -> None:
        """
        metric: "uss" (unique set size — private/committed memory, the right
            proxy for "will I OOM"; excludes shared + file-backed memmap page
            cache) or "rss" (resident, includes memmap page cache — overcounts
            spilled tiles). USS needs /proc/smaps so it's sampled a bit slower;
            falls back to RSS automatically if unavailable on the platform.
        """
        self.interval_s = interval_s
        self.threshold_bytes = threshold_bytes
        self.on_exceed = on_exceed
        self.metric = metric
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._phase: Optional[str] = None
        self._phase_peak: Dict[str, int] = {}
        self._exceeded = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._proc = psutil.Process() if _HAVE_PSUTIL else None
        self._use_uss = metric == "uss"

    # -- sampling ---------------------------------------------------------
    def _rss(self) -> int:
        """Sample the chosen memory metric (USS preferred, RSS fallback)."""
        if self._proc is None:
            return 0
        if self._use_uss:
            try:
                return int(self._proc.memory_full_info().uss)
            except Exception:
                # USS unsupported here (e.g. permissions / platform) — degrade
                # to RSS for the rest of this monitor's life.
                self._use_uss = False
        try:
            return int(self._proc.memory_info().rss)
        except Exception:
            return 0

    def start(self) -> "MemoryMonitor":
        if not _HAVE_PSUTIL:
            return self
        self.baseline_bytes = self._rss()
        self.peak_bytes = self.baseline_bytes
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="MemoryMonitor", daemon=True
        )
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss = self._rss()
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            phase = self._phase
            if phase is not None and rss > self._phase_peak.get(phase, 0):
                self._phase_peak[phase] = rss
            # Compare the WORKING-SET DELTA (rss - baseline) against the
            # threshold, not absolute USS: the projected peak the caller passes
            # is a working-set allocation figure and excludes the fixed
            # interpreter/library/CUDA baseline. Comparing absolute USS would
            # cry wolf on small jobs (baseline already near threshold) and, for
            # near-limit jobs, sit above physical RAM (never fires).
            if (
                self.threshold_bytes
                and not self._exceeded
                and (rss - self.baseline_bytes) > self.threshold_bytes
            ):
                self._exceeded = True
                if self.on_exceed is not None:
                    try:
                        self.on_exceed(rss, phase)
                    except Exception:
                        pass  # a broken callback must never crash a run
            self._stop.wait(self.interval_s)

    def stop(self) -> "MemoryMonitor":
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None
        return self

    # -- phase attribution ------------------------------------------------
    def set_phase(self, phase: Optional[str]) -> None:
        """Tag subsequent samples with ``phase`` so peaks are attributable.

        Records a floor for the phase immediately so a phase that allocates and
        frees between samples still shows a non-zero peak.
        """
        self._phase = phase
        if phase is not None:
            self._phase_peak.setdefault(phase, self._rss())

    # -- results ----------------------------------------------------------
    @property
    def available(self) -> bool:
        return _HAVE_PSUTIL

    @property
    def peak_delta_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)

    def phase_peaks_delta(self) -> Dict[str, int]:
        return {
            p: max(0, v - self.baseline_bytes) for p, v in self._phase_peak.items()
        }

    # -- context manager --------------------------------------------------
    def __enter__(self) -> "MemoryMonitor":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False
