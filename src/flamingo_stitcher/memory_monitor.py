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

import os
import threading
from typing import Callable, Dict, List, Optional, Tuple

try:
    import psutil

    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover - psutil is a hard dep, but never crash
    _HAVE_PSUTIL = False


def _linux_smaps_rollup() -> Dict[str, int]:
    """`/proc/self/smaps_rollup` as {key: bytes}. Empty on any failure."""
    out: Dict[str, int] = {}
    try:
        with open("/proc/self/smaps_rollup", "r") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if len(parts) == 2 and parts[1] == "kB":
                    out[key.strip()] = int(parts[0]) * 1024
    except Exception:
        return {}
    return out


def memory_split(proc=None) -> Tuple[int, int]:
    """``(private_commit_bytes, mapped_resident_bytes)`` for this process.

    These are two genuinely different things and the pipeline's memory model
    only predicts the first:

    * **private commit** — anonymous memory this process allocated. If it
      exceeds RAM the process dies. This is what `estimate_memory_usage`
      projects, and the only figure worth comparing against it.
    * **mapped resident** — pages of memory-mapped FILES currently in the
      working set: the tile spill and the multi-hundred-GB `fused.dat`. These
      are backed by disk, the OS reclaims them under pressure, and they do not
      by themselves cause an OOM.

    Why this exists: the watchdog used USS, and USS counts both. On Windows
    `QueryWorkingSetEx` marks a page shared only when another process also maps
    it, so a 400 GB fused memmap owned by one process reads as entirely
    private — and the watchdog reported a working set 13x the projection on a
    run that completed comfortably. Flushing the memmap per region does not
    help: `flush()` writes pages back and makes them CLEAN, it does not evict
    them, so they stay resident and stay counted.

    Mapped-resident is not noise, though, and is reported rather than
    discarded: when the scratch disk is too slow to absorb write-back, that
    backlog is what freezes the machine.
    """
    if not _HAVE_PSUTIL:
        return (0, 0)
    proc = proc or psutil.Process()
    try:
        info = proc.memory_info()
    except Exception:
        return (0, 0)

    # Windows: PrivateUsage — private commit, excludes mapped files by
    # construction. rss here is WorkingSetSize, which includes them.
    private = getattr(info, "private", None)
    if private is not None:
        rss = int(getattr(info, "rss", 0) or 0)
        return (int(private), max(0, rss - int(private)))

    # Linux: smaps_rollup separates anonymous from file-backed directly.
    rollup = _linux_smaps_rollup()
    if rollup:
        anon = rollup.get("Anonymous", 0)
        rss = rollup.get("Rss", 0)
        return (int(anon), max(0, int(rss) - int(anon)))

    # Nothing better available (macOS, restricted platforms): fall back to USS,
    # which over-counts by the mapped set rather than under-counting.
    try:
        return (int(proc.memory_full_info().uss), 0)
    except Exception:
        return (int(getattr(info, "rss", 0) or 0), 0)


def top_memory_consumers(
    limit: int = 5, min_gb: float = 1.0, exclude_self: bool = True
) -> List[Tuple[str, float, int]]:
    """Biggest RAM users on this machine as ``[(name, gb, n_processes), ...]``.

    Aggregated **by process name** and sorted descending, because a browser or
    Fiji typically spans dozens of PIDs and a per-PID list buries the real
    culprit. Only entries at or above ``min_gb`` are returned.

    Used to make "waiting for memory" actionable: the user can see that e.g.
    Fiji is holding 21 GB and close it, instead of watching a stalled run.
    Best-effort — returns ``[]`` if psutil is unavailable, and silently skips
    processes that vanish or deny access mid-scan.
    """
    if not _HAVE_PSUTIL:
        return []
    skip = set()
    if exclude_self:
        # Don't tell the user to close the stitcher (or its own helper workers).
        try:
            me = psutil.Process()
            skip = {me.pid} | {c.pid for c in me.children(recursive=True)}
        except Exception:
            skip = {os.getpid()}

    totals: Dict[str, List[float]] = {}
    try:
        procs = psutil.process_iter(["name", "memory_info"])
    except Exception:
        return []
    for proc in procs:
        try:
            if proc.pid in skip:
                continue
            info = proc.info
            mem = info.get("memory_info")
            if mem is None:
                continue
            name = info.get("name") or f"pid {proc.pid}"
            entry = totals.setdefault(name, [0.0, 0])
            entry[0] += float(mem.rss)
            entry[1] += 1
        except Exception:
            continue  # process died / access denied — skip it

    out = [
        (name, rss / (1024**3), int(count))
        for name, (rss, count) in totals.items()
        if rss / (1024**3) >= min_gb
    ]
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:limit]


def format_memory_consumers(rows: List[Tuple[str, float, int]]) -> str:
    """One-per-line ``"  name — 21.3 GB (28 processes)"`` rendering."""
    lines = []
    for name, gb, count in rows:
        suffix = f" ({count} processes)" if count > 1 else ""
        lines.append(f"  {name} — {gb:.1f} GB{suffix}")
    return "\n".join(lines)


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
        metric:
            "private" (**default, and the one to use against a projection**) —
                private commit only: anonymous memory this process allocated.
                Excludes memory-mapped files, so the tile spill and `fused.dat`
                do not inflate it. This is the quantity
                `estimate_memory_usage` predicts.
            "uss" — unique set size. On Windows this *includes* the resident
                pages of a memmap owned by one process, so on a streaming run
                it measures mostly on-disk scratch and is not comparable to any
                projection. Kept for callers that genuinely want total private
                footprint.
            "rss" — resident set, includes all mapped file pages.

        `mapped_resident_bytes` is tracked alongside whichever metric is chosen,
        so a caller can report "and N GB of that is reclaimable file cache"
        without a second sampling pass.
        """
        self.interval_s = interval_s
        self.threshold_bytes = threshold_bytes
        self.on_exceed = on_exceed
        self.metric = metric
        self.baseline_bytes = 0
        self.peak_bytes = 0
        # Peak resident memory-mapped file pages seen (tile spill + fused.dat).
        # Not part of the threshold — it is disk-backed and reclaimable — but
        # reported, because a scratch disk too slow to absorb write-back is the
        # documented way this pipeline freezes a machine.
        self.peak_mapped_bytes = 0
        self._phase: Optional[str] = None
        self._phase_peak: Dict[str, int] = {}
        self._exceeded = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._proc = psutil.Process() if _HAVE_PSUTIL else None
        self._use_uss = metric == "uss"

    # -- sampling ---------------------------------------------------------
    def _rss(self) -> int:
        """Sample the chosen memory metric, tracking mapped-file residency too."""
        if self._proc is None:
            return 0
        if self.metric == "private":
            private, mapped = memory_split(self._proc)
            if mapped > self.peak_mapped_bytes:
                self.peak_mapped_bytes = mapped
            return private
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
