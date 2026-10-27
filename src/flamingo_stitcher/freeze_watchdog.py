"""Write every thread's stack to the log when the window stops responding.

2026-10-27: a 4.5 h stitch finished cleanly, then the window went "Not
Responding" with one core busy and no disk activity. The log file ended at the
pipeline's last line and nothing said where the window's thread was, so the
cause could not be found without a debugger on the rig.

The window's thread calls :meth:`FreezeWatchdog.beat` from a timer. A daemon
thread checks the age of the last beat; once it passes ``threshold_s`` it logs
the stack of every Python thread, once per freeze, then again only after the
window has recovered and frozen anew. Qt-free so the rule is testable without
a display.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# How long the window's thread may go without a beat before its stack is
# dumped. Long enough that no ordinary handler trips it (the slowest seen,
# loading a stitched preview, takes a few seconds), short enough that a user
# who waits a minute before killing the app still leaves the dump behind.
FREEZE_THRESHOLD_S = 60.0
# How often the watchdog thread looks.
CHECK_INTERVAL_S = 5.0


class FreezeWatchdog:
    def __init__(
        self,
        threshold_s: float = FREEZE_THRESHOLD_S,
        clock: Callable[[], float] = time.monotonic,
        log: Optional[logging.Logger] = None,
    ):
        self._threshold_s = float(threshold_s)
        self._clock = clock
        self._log = log or logger
        self._last_beat = clock()
        self._watched_ident = threading.get_ident()
        self._dumped = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def beat(self) -> None:
        """Called from the watched (window) thread while it is responsive."""
        if self._dumped:
            self._log.warning(
                f"Window responsive again after "
                f"{self._clock() - self._last_beat:.0f} s."
            )
        self._last_beat = self._clock()
        self._watched_ident = threading.get_ident()
        self._dumped = False

    def check(self) -> bool:
        """Dump the stacks if the watched thread is overdue. True if dumped."""
        stalled = self._clock() - self._last_beat
        if stalled < self._threshold_s or self._dumped:
            return False
        self._dumped = True
        self._log.error(
            f"Window not responding for {stalled:.0f} s. Stack of every thread "
            f"(the window's thread is marked):\n" + self.format_stacks()
        )
        return True

    def format_stacks(self) -> str:
        names = {t.ident: t.name for t in threading.enumerate()}
        parts = []
        for ident, frame in sys._current_frames().items():
            if ident == threading.get_ident():
                continue  # the watchdog itself
            mark = "  <-- window thread" if ident == self._watched_ident else ""
            parts.append(
                f"--- {names.get(ident, 'thread')} ({ident}){mark}\n"
                + "".join(traceback.format_stack(frame))
            )
        return "\n".join(parts)

    def start(self, interval_s: float = CHECK_INTERVAL_S) -> None:
        if self._thread is not None:
            return

        def _loop():
            while not self._stop.wait(interval_s):
                try:
                    self.check()
                except Exception:  # a diagnostic must never take the app down
                    pass

        self._thread = threading.Thread(
            target=_loop, name="freeze-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
