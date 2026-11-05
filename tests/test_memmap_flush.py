"""memmap_flush: the region flush that no longer freezes the window.

numpy's memmap.flush() holds the GIL for the whole flush (84-95 s window
freezes per fused region on run 35). memmap_flush.flush calls the same OS
function through ctypes, which releases the GIL. These tests check that the
call really goes through a GIL-releasing ctypes function, that it covers exactly
the view's bytes, that the data reaches the file, and the fallbacks.

Runs on Linux in the main CI job and on Windows in the windows-flush job, so
both FlushViewOfFile and msync are exercised for real.
"""

import ctypes
import sys

import numpy as np
import pytest

from flamingo_stitcher import memmap_flush


@pytest.fixture
def mm(tmp_path):
    arr = np.memmap(tmp_path / "fused.dat", dtype=np.uint16, mode="w+", shape=(2, 40, 300, 500))
    yield arr
    del arr


def test_native_flush_is_available_on_this_platform():
    if sys.platform == "win32" or sys.platform.startswith(("linux", "darwin")):
        assert memmap_flush._native is not None


def test_native_call_releases_the_gil():
    # ctypes releases the GIL for CDLL/WinDLL functions and keeps it only for
    # PyDLL ones, which carry the PYTHONAPI flag.
    func = memmap_flush._native.native
    assert not (func._flags_ & ctypes._FUNCFLAG_PYTHONAPI)


def test_region_data_reaches_the_file(mm, tmp_path):
    region = mm[1][5:17, 40:200, 100:450]
    region[:] = np.arange(region.size, dtype=np.uint16).reshape(region.shape)
    memmap_flush.flush(region)
    on_disk = np.fromfile(tmp_path / "fused.dat", dtype=np.uint16).reshape(mm.shape)
    np.testing.assert_array_equal(on_disk[1, 5:17, 40:200, 100:450], region)


def test_flushes_only_the_views_bytes(mm, monkeypatch):
    calls = []
    monkeypatch.setattr(memmap_flush, "_native", lambda a, n: calls.append((a, n)))
    region = mm[1][5:17, 40:200, 100:450]
    memmap_flush.flush(region)
    (address, length), = calls
    base = mm.ctypes.data
    first = region.ctypes.data
    last = first + region.strides[0] * (region.shape[0] - 1) \
        + region.strides[1] * (region.shape[1] - 1) + region.strides[2] * region.shape[2]
    assert address % memmap_flush.PAGE_SIZE == 0
    assert base <= address <= first < first + 1 <= address + length
    assert address + length == last
    assert address > base  # not the whole map: channel 0 is left out
    assert length < mm.nbytes // 2


def test_plain_arrays_are_left_alone(monkeypatch):
    calls = []
    monkeypatch.setattr(memmap_flush, "_native", lambda a, n: calls.append((a, n)))
    memmap_flush.flush(np.zeros((10, 10), dtype=np.uint16))
    assert calls == []


def test_falls_back_to_numpy_flush_when_the_native_call_fails(mm, monkeypatch):
    def broken(address, length):
        raise OSError(22, "msync failed")

    flushed = []
    monkeypatch.setattr(memmap_flush, "_native", broken)
    monkeypatch.setattr(np.memmap, "flush", lambda self: flushed.append(True))
    memmap_flush.flush(mm[0][:2])
    assert flushed == [True]


def test_falls_back_when_no_native_function(mm, monkeypatch):
    flushed = []
    monkeypatch.setattr(memmap_flush, "_native", None)
    monkeypatch.setattr(np.memmap, "flush", lambda self: flushed.append(True))
    memmap_flush.flush(mm[0])
    assert flushed == [True]
