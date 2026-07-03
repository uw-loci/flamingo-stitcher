"""Cancellation aborts a running dask compute promptly (not only between stages).

The pipeline polls the cancel flag between stages, but a single big fuse/write
compute can run for a long time with no stage boundary. _CancelCallback raises
inside the dask scheduler so the compute tears down at the next task boundary.
"""

import dask.array as da
import pytest

from flamingo_stitcher.pipeline import PipelineCancelled, _CancelCallback


def test_cancel_callback_aborts_running_compute():
    """A compute in progress is torn down once the cancel flag flips."""
    ticks = {"n": 0}

    def cancelled():
        ticks["n"] += 1
        return ticks["n"] > 3  # cancel after a few tasks have been scheduled

    # 100 chunks -> many tasks, so cancellation happens well before completion.
    arr = da.ones((1000, 1000), chunks=(100, 100))
    graph = (arr + 1).sum()

    with pytest.raises(PipelineCancelled):
        with _CancelCallback(cancelled):
            graph.compute(scheduler="synchronous")

    # It stopped early, not after visiting every task.
    assert ticks["n"] <= 6


def test_no_cancel_completes_normally():
    """With the flag never set, the callback is a no-op and the compute runs."""
    arr = da.ones((100, 100), chunks=(50, 50))
    with _CancelCallback(lambda: False):
        result = int((arr + 1).sum().compute(scheduler="synchronous"))
    assert result == 100 * 100 * 2


def test_cancel_callback_threaded_scheduler():
    """Also aborts under the threaded scheduler (what the fuse/store use)."""
    state = {"go": False}

    def cancelled():
        return state["go"]

    arr = da.ones((400, 400), chunks=(50, 50))
    graph = (arr + 1).sum()
    state["go"] = True  # cancel from the very first task
    with pytest.raises(PipelineCancelled):
        with _CancelCallback(cancelled):
            graph.compute(scheduler="threads", num_workers=2)
