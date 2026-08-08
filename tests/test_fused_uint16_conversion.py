"""The fused-volume uint16 conversion must happen IN the dask graph.

A 104-tile brain (26 GB fused output) tripped the memory watchdog on every run:
projected 82.7 GB, live 128.1 GB. The in-memory path converted the fused volume
*after* materialising it::

    vol = np.asarray(fused_sim.data.compute())
    vol = np.clip(vol, 0, 65535).astype(np.uint16)

which keeps three full-size copies alive at once — the computed array, the clip
temporary, and the astype result — for a conversion that is a no-op anyway
(``multiview_stitcher.fusion`` already returns ``sims[0].dtype``, i.e. uint16).
The streaming and preview paths had always done it lazily; only the in-memory
path had drifted.

These tests pin the saving by measuring it, using ``tracemalloc`` — which tracks
numpy's data allocator (NEP 49) exactly, so the numbers are deterministic rather
than RSS noise. The peak test compares the shipped helper against the eager
formulation on the *same* input, so reverting the helper makes the two identical
and the test fails.
"""

from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

da = pytest.importorskip("dask.array")

from flamingo_stitcher.pipeline import lazy_uint16  # noqa: E402


def _fused_like(shape=(1, 1, 48, 512, 512), chunks=(1, 1, 16, 256, 256), dtype=np.uint16):
    """A stand-in for ``fused_sim.data``: leading singleton dims, chunked."""
    return da.zeros(shape, chunks=chunks, dtype=dtype)


class TestShapeAndDtype:
    def test_squeezes_leading_singleton_dims(self):
        out = lazy_uint16(_fused_like())
        assert out.ndim == 3
        assert out.shape == (48, 512, 512)

    def test_result_is_uint16_and_still_lazy(self):
        out = lazy_uint16(_fused_like(dtype=np.float64))
        assert out.dtype == np.uint16
        # Still a dask array — the whole point is that nothing was materialised.
        assert isinstance(out, da.Array)

    def test_clamps_out_of_range_float_values(self):
        # 4-D so the squeeze is exercised too: (1,1,1,4) -> (1,1,4).
        src = np.array([[[[-5.0, 0.0, 1.5, 70000.0]]]], dtype=np.float64)
        out = lazy_uint16(da.from_array(src, chunks=(1, 1, 1, 2))).compute()
        assert out.dtype == np.uint16
        np.testing.assert_array_equal(
            out, np.array([[[0, 0, 1, 65535]]], dtype=np.uint16)
        )

    def test_uint16_input_passes_through_unchanged(self):
        """The common case: mvs already returns uint16, so this is a no-op...

        ...in VALUE. It must not be a no-op in *cost*, which is what the peak
        test below covers.
        """
        src = (np.arange(24, dtype=np.uint16) * 1000).reshape(1, 4, 6)
        out = lazy_uint16(da.from_array(src, chunks=(1, 2, 3))).compute()
        np.testing.assert_array_equal(out, src)


def _peak_bytes(fn) -> int:
    """Peak traced allocation of ``fn()``, isolated from the caller's."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = fn()
        _cur, peak = tracemalloc.get_traced_memory()
        del result
        return peak
    finally:
        tracemalloc.stop()


class TestItActuallySavesACopy:
    """The regression: converting eagerly costs one extra full-size volume."""

    # 48 x 512 x 512 uint16 = 25.2 MB fused — small enough to be a fast unit
    # test, large enough that one extra copy dwarfs interpreter noise.
    SHAPE = (1, 1, 48, 512, 512)
    CHUNKS = (1, 1, 16, 256, 256)

    @staticmethod
    def _nbytes():
        return 48 * 512 * 512 * 2

    def _lazy(self):
        return np.asarray(lazy_uint16(_fused_like(self.SHAPE, self.CHUNKS)).compute())

    def _eager(self):
        """The formulation this fix removed, kept here as the control."""
        vol = np.asarray(_fused_like(self.SHAPE, self.CHUNKS).compute())
        while vol.ndim > 3:
            vol = vol[0]
        return np.clip(vol, 0, 65535).astype(np.uint16)

    def test_both_produce_the_same_volume(self):
        np.testing.assert_array_equal(self._lazy(), self._eager())

    def test_lazy_peak_is_at_least_one_volume_lower(self):
        nbytes = self._nbytes()
        lazy_peak = _peak_bytes(self._lazy)
        eager_peak = _peak_bytes(self._eager)
        saved = eager_peak - lazy_peak
        # The eager form holds computed + clip + astype simultaneously, so it
        # must exceed the lazy form by ~one full volume. Demand 80% of one
        # volume so the assertion doesn't hinge on allocator rounding.
        assert saved > 0.8 * nbytes, (
            f"converting in the graph saved only {saved / 1e6:.1f} MB; expected "
            f">{0.8 * nbytes / 1e6:.1f} MB (one {nbytes / 1e6:.1f} MB volume). "
            f"lazy peak={lazy_peak / 1e6:.1f} MB eager peak={eager_peak / 1e6:.1f} MB. "
            "Has the uint16 conversion moved back out of the dask graph?"
        )

    def test_lazy_peak_stays_near_the_materialization_floor(self):
        """Absolute bound, so this fails even if someone 'fixes' the control.

        ``.compute()`` inherently holds the per-chunk results and the
        concatenated output at once, so ~2x the volume is the floor. Anything
        near 3x means a whole extra copy is back.
        """
        nbytes = self._nbytes()
        lazy_peak = _peak_bytes(self._lazy)
        assert lazy_peak < 2.6 * nbytes, (
            f"in-graph conversion peaked at {lazy_peak / nbytes:.2f}x the fused "
            f"volume ({lazy_peak / 1e6:.1f} MB); the compute floor is ~2x."
        )
