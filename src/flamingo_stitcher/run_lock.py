"""One stitching run at a time, across every stitching tab of the app.

'Single Workflow', 'Multi-Acquisition' and 'Multi-View' are separate tabs with
separate queues, and until 0.14.0 nothing stopped a run in each at once. Two
runs share the computer's memory (one large run peaked near 79 GB of 191) and
its cores (destriping uses all of them), and before 0.14.0 also the same
scratch files. So a tab takes this lock when its run starts, and the others
gray out 'Run All' until it is released.

Qt-free so it can be tested on its own. Holders are any hashable object (the
dialog); listeners are called with no arguments whenever the holder changes.
"""

from __future__ import annotations

import weakref
from typing import Callable, List, Optional, Tuple

_holder: Optional[Tuple[int, str]] = None  # (id of the holder, its tab label)
_listeners: List[weakref.ref] = []


def holder_label(exclude: object = None) -> Optional[str]:
    """The tab name of the run in progress, or None. ``exclude``: ignore this holder."""
    if _holder is None or (exclude is not None and _holder[0] == id(exclude)):
        return None
    return _holder[1]


def acquire(owner: object, label: str) -> bool:
    """Take the lock for ``owner``. False if another holder has it."""
    global _holder
    if _holder is not None and _holder[0] != id(owner):
        return False
    _holder = (id(owner), label)
    _notify()
    return True


def release(owner: object) -> None:
    """Give the lock back, if ``owner`` holds it."""
    global _holder
    if _holder is not None and _holder[0] == id(owner):
        _holder = None
        _notify()


def subscribe(callback: Callable[[], None]) -> None:
    """Call ``callback()`` whenever a run starts or ends in any tab.

    A bound method is held weakly, so a closed dialog drops out by itself.
    """
    try:
        ref = weakref.WeakMethod(callback)
    except TypeError:
        ref = weakref.ref(callback)
    _listeners.append(ref)


def _notify() -> None:
    alive = []
    for ref in _listeners:
        cb = ref()
        if cb is None:
            continue
        try:
            cb()
        except RuntimeError:
            # The Qt object behind a bound method was deleted.
            continue
        alive.append(ref)
    _listeners[:] = alive
