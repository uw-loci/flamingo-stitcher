"""The window freeze watchdog (2026-10-27: a finished stitch left the window
"Not Responding" and nothing in the log said where it was stuck)."""

import logging
import threading

from flamingo_stitcher.freeze_watchdog import FreezeWatchdog


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_no_dump_while_the_window_beats(caplog):
    clock = _Clock()
    wd = FreezeWatchdog(threshold_s=60, clock=clock)
    for _ in range(10):
        clock.t += 30
        wd.beat()
        assert wd.check() is False


def test_a_stall_dumps_every_stack_once(caplog):
    clock = _Clock()
    wd = FreezeWatchdog(threshold_s=60, clock=clock)
    wd.beat()
    caplog.set_level(logging.ERROR, logger="flamingo_stitcher.freeze_watchdog")
    clock.t = 61
    # check() runs on the watchdog's own thread in the app; the dump leaves
    # out the calling thread, so call it from another one here too.
    results = []
    for _ in range(2):
        t = threading.Thread(target=lambda: results.append(wd.check()))
        t.start()
        t.join()
    assert results == [True, False]  # once per freeze
    text = caplog.records[-1].getMessage()
    assert "not responding for 61 s" in text
    assert "window thread" in text


def test_recovery_rearms_the_dump(caplog):
    clock = _Clock()
    wd = FreezeWatchdog(threshold_s=60, clock=clock)
    clock.t = 100
    assert wd.check() is True
    wd.beat()  # window recovered
    clock.t = 200
    assert wd.check() is True


def test_the_dump_names_a_busy_threads_frame():
    started, release = threading.Event(), threading.Event()

    def _stuck_in_here():
        started.set()
        release.wait(5)

    t = threading.Thread(target=_stuck_in_here, name="busy")
    t.start()
    started.wait(5)
    try:
        assert "_stuck_in_here" in FreezeWatchdog().format_stacks()
    finally:
        release.set()
        t.join()
