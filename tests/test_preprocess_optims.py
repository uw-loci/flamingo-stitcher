"""Bit-identity regression tests for the memory-efficiency optimisations.

These fixes reduce per-tile RAM (per-plane float32 instead of whole-volume
upcasts) and must not change output. Locks in:
  * downsample_volume  — per-plane XY zoom == whole-volume zoom (factor_z==1)
  * fuse_illumination_sides "mean" — per-plane == whole-volume
  * _apply_flatfield_volume — per-plane == whole-volume flat-field
  * ome_zarr_writer._downsample_2x_to_memmap — streamed == in-RAM _downsample_2x
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from flamingo_stitcher.pipeline import (
    StitchingConfig,
    StitchingPipeline,
    downsample_volume,
    fuse_illumination_sides,
)


class TestDownsampleIdentical(unittest.TestCase):
    def test_xy_only_matches_whole_volume(self):
        from scipy.ndimage import zoom

        rng = np.random.default_rng(0)
        for fxy in (2, 4, 8):
            vol = rng.integers(0, 65535, size=(7, 130, 97), dtype=np.uint16)
            got = downsample_volume(vol, factor_xy=fxy, factor_z=1)
            ref = np.clip(
                zoom(vol.astype(np.float32), (1.0, 1.0 / fxy, 1.0 / fxy), order=1),
                0,
                65535,
            ).astype(np.uint16)
            np.testing.assert_array_equal(got, ref, err_msg=f"xy={fxy}")

    def test_z_downsample_path_unchanged(self):
        from scipy.ndimage import zoom

        rng = np.random.default_rng(1)
        vol = rng.integers(0, 65535, size=(8, 64, 64), dtype=np.uint16)
        got = downsample_volume(vol, factor_xy=2, factor_z=2)
        ref = np.clip(
            zoom(vol.astype(np.float32), (0.5, 0.5, 0.5), order=1), 0, 65535
        ).astype(np.uint16)
        np.testing.assert_array_equal(got, ref)


class TestIllumMeanIdentical(unittest.TestCase):
    def test_mean_matches_whole_volume(self):
        rng = np.random.default_rng(2)
        left = rng.integers(0, 65535, size=(5, 40, 40), dtype=np.uint16)
        right = rng.integers(0, 65535, size=(5, 40, 40), dtype=np.uint16)
        got = fuse_illumination_sides({0: left, 1: right}, method="mean")
        ref = ((left.astype(np.float32) + right.astype(np.float32)) / 2).astype(
            np.uint16
        )
        np.testing.assert_array_equal(got, ref)
        # max path unchanged
        np.testing.assert_array_equal(
            fuse_illumination_sides({0: left, 1: right}, method="max"),
            np.maximum(left, right),
        )


class TestFlatfieldApplyIdentical(unittest.TestCase):
    def test_apply_matches_whole_volume(self):
        rng = np.random.default_rng(3)

        class _M:
            flatfield = rng.uniform(0.5, 1.5, size=(40, 40)).astype(np.float32)
            darkfield = rng.uniform(0, 50, size=(40, 40)).astype(np.float32)

        vol = rng.integers(0, 4000, size=(5, 40, 40), dtype=np.uint16)
        p = StitchingPipeline(StitchingConfig())
        got = p._apply_flatfield_volume(vol, _M())
        ff = np.where(_M.flatfield > 0.001, _M.flatfield, 1.0)
        ref = np.empty_like(vol)
        for z in range(vol.shape[0]):
            ref[z] = np.clip(
                (vol[z].astype(np.float32) - _M.darkfield) / ff, 0, 65535
            ).astype(np.uint16)
        np.testing.assert_array_equal(got, ref)


class TestZarrV2DownsampleIdentical(unittest.TestCase):
    def test_streamed_matches_in_ram(self):
        from flamingo_stitcher.writers import ome_zarr_writer as w

        rng = np.random.default_rng(4)
        for shp in [(9, 66, 50), (2, 9, 66, 50)]:
            a = rng.integers(0, 65535, size=shp, dtype=np.uint16)
            ref = w._downsample_2x(a)
            with tempfile.TemporaryDirectory() as d:
                mm = w._downsample_2x_to_memmap(a, Path(d) / "l.dat")
                np.testing.assert_array_equal(np.asarray(mm), ref, err_msg=str(shp))
                del mm


if __name__ == "__main__":
    unittest.main()
