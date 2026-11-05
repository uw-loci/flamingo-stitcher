"""Flush part of a memory-mapped array to disk without holding the GIL.

``numpy.memmap.flush()`` calls Python's ``mmap.flush()``, which keeps the GIL
for the whole flush, so no other Python thread runs until it returns. On the
rig that froze the window for 84-95 s after every fused region of a large run
(run 35, 2026-10-30): the flush wrote a region's tens of GB while the GUI, the
log and the freeze watchdog all waited. Measured off-rig (Linux, Python 3.12.3,
2 GB of dirty pages): a 1 ms ticker thread ran once during ``mmap.flush`` and
418 times out of ~463 during the same flush made here.

:func:`flush` calls the same operating-system function Python uses
(``FlushViewOfFile`` on Windows, ``msync(MS_SYNC)`` on Linux and macOS) through
``ctypes``, which releases the GIL for the length of the call, and only over the
bytes the given view spans. The flush still takes as long; it just no longer
stops the rest of the program while it runs. On any other platform, or if the
native call fails, it falls back to the memmap's own ``flush()``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import mmap
import sys
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

PAGE_SIZE = mmap.PAGESIZE

# msync's MS_SYNC flag differs by platform and Python's mmap module does not
# export it. Values from <sys/mman.h>.
_MS_SYNC = {"linux": 4, "darwin": 16}


def _load_native() -> Optional[Callable[[int, int], None]]:
    """The OS flush function as ``call(address, length)``, or None if unknown."""
    try:
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            func = kernel32.FlushViewOfFile
            func.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            func.restype = ctypes.c_int  # BOOL

            def call(address: int, length: int) -> None:
                if not func(address, length):
                    err = ctypes.get_last_error()
                    raise OSError(err, f"FlushViewOfFile failed (error {err})")

            call.native = func  # type: ignore[attr-defined]
            return call

        flag = next((v for k, v in _MS_SYNC.items() if sys.platform.startswith(k)), None)
        name = ctypes.util.find_library("c")
        if flag is None or name is None:
            return None
        libc = ctypes.CDLL(name, use_errno=True)
        func = libc.msync
        func.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        func.restype = ctypes.c_int

        def call(address: int, length: int) -> None:
            if func(address, length, flag) != 0:
                err = ctypes.get_errno()
                raise OSError(err, f"msync failed (errno {err})")

        call.native = func  # type: ignore[attr-defined]
        return call
    except (OSError, AttributeError):
        return None


_native = _load_native()
_warned = False


def _byte_bounds(arr: np.ndarray):
    try:
        from numpy.lib.array_utils import byte_bounds  # numpy >= 2.0
    except ImportError:  # pragma: no cover - numpy 1.x
        byte_bounds = np.byte_bounds
    return byte_bounds(arr)


def flush_range(arr: np.ndarray):
    """The (page-aligned address, length) :func:`flush` would pass to the OS."""
    low, high = _byte_bounds(arr)
    start = low - (low % PAGE_SIZE)
    return start, high - start


def flush(arr: np.ndarray) -> None:
    """Write ``arr``'s dirty pages to its file without blocking other threads.

    ``arr`` is a ``numpy.memmap`` or any view of one (a region, a slab, one
    channel). Only the bytes the view spans are flushed. An array that is not
    backed by a memory map is left alone.
    """
    global _warned
    owner = arr if getattr(arr, "_mmap", None) is not None else None
    if owner is None or arr.size == 0:
        return
    if _native is not None:
        try:
            _native(*flush_range(arr))
            return
        except OSError as exc:
            if not _warned:
                logger.warning(
                    "Native memmap flush failed (%s); using numpy's flush, which "
                    "pauses the window until it finishes.", exc
                )
                _warned = True
    owner.flush()
