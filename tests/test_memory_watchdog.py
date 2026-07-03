"""Unit tests for the memory watchdog mechanism (MemoryMonitor).

The pipeline-side wiring (StitchingPipeline._start_memory_watchdog ->
worker.memory_warning signal -> dialog popup) is exercised for import/no-crash
by test_memory_scaling.py, which runs the real pipeline with the watchdog code
present. Here we test the trip mechanism directly: threshold crossing fires the
callback, with the correct phase attribution, exactly once.
"""

from __future__ import annotations

import time
import unittest

try:
    import psutil  # noqa: F401

    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False

from flamingo_stitcher.memory_monitor import MemoryMonitor


@unittest.skipUnless(_HAVE_PSUTIL, "needs psutil")
class TestMemoryMonitor(unittest.TestCase):
    def test_threshold_fires_once_with_phase(self):
        fired = []

        def on_exceed(used_bytes, phase):
            fired.append((used_bytes, phase))

        # Threshold of 1 byte -> current RSS/USS always exceeds it immediately.
        mon = MemoryMonitor(
            interval_s=0.01, threshold_bytes=1, on_exceed=on_exceed
        )
        mon.start()
        mon.set_phase("fuse")
        time.sleep(0.1)
        mon.stop()

        self.assertEqual(len(fired), 1, "callback must fire exactly once")
        used, phase = fired[0]
        self.assertGreater(used, 1)
        self.assertEqual(phase, "fuse")

    def test_no_fire_below_threshold(self):
        fired = []
        mon = MemoryMonitor(
            interval_s=0.01,
            threshold_bytes=10 * 1024**4,  # 10 TB — never crossed
            on_exceed=lambda *a: fired.append(a),
        )
        with mon:
            time.sleep(0.05)
        self.assertEqual(fired, [])

    def test_peak_and_phase_tracking(self):
        mon = MemoryMonitor(interval_s=0.01)
        mon.start()
        mon.set_phase("preprocess")
        blob = bytearray(20 * 1024 * 1024)  # 20 MB, keep it resident
        mon.set_phase("fuse")
        time.sleep(0.05)
        mon.stop()
        self.assertGreaterEqual(mon.peak_delta_bytes, 0)
        peaks = mon.phase_peaks_delta()
        self.assertIn("fuse", peaks)
        del blob

    def test_uss_falls_back_to_rss(self):
        # metric="uss" must not raise even if USS is unavailable; it silently
        # degrades to RSS. We just assert a sample succeeds and is positive.
        mon = MemoryMonitor(interval_s=0.01, metric="uss")
        mon.start()
        time.sleep(0.03)
        mon.stop()
        self.assertGreater(mon.peak_bytes, 0)


if __name__ == "__main__":
    unittest.main()
