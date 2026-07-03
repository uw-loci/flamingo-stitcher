"""Correctness tests for the streaming OME-TIFF pyramid (item A).

The pyramid is now built into on-disk memmaps one Z-plane at a time instead of
holding every downsampled level in RAM (~0.33x the output). This must be
BIT-IDENTICAL to the old whole-volume downsample, and the written file must read
back with the right full-res + pyramid levels.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from flamingo_stitcher.writers import ome_tiff_writer as w


class TestStreamingDownsampleIdentical(unittest.TestCase):
    def test_3d_matches_whole_volume(self):
        rng = np.random.default_rng(0)
        vol = rng.integers(0, 4000, size=(5, 64, 48), dtype=np.uint16)
        for factor in (2, 4):
            whole = w._downsample_yx(vol, factor)
            with tempfile.TemporaryDirectory() as d:
                mm = w._downsample_yx_to_memmap(vol, factor, Path(d) / "l.dat")
                np.testing.assert_array_equal(np.asarray(mm), whole)
                del mm

    def test_4d_matches_whole_volume(self):
        rng = np.random.default_rng(1)
        vol = rng.integers(0, 4000, size=(2, 4, 64, 64), dtype=np.uint16)
        for factor in (2, 4):
            whole = w._downsample_yx(vol, factor)
            with tempfile.TemporaryDirectory() as d:
                mm = w._downsample_yx_to_memmap(vol, factor, Path(d) / "l.dat")
                np.testing.assert_array_equal(np.asarray(mm), whole)
                del mm


class TestOmeTiffRoundTrip(unittest.TestCase):
    def test_write_and_read_pyramid(self):
        try:
            import tifffile
        except Exception:
            self.skipTest("tifffile not installed")

        rng = np.random.default_rng(2)
        vol = rng.integers(0, 4000, size=(3, 512, 512), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "img.ome.tif"
            w.write_pyramidal_ome_tiff(
                vol,
                out,
                voxel_size_um={"z": 5.0, "y": 1.0, "x": 1.0},
                pyramid_levels=2,
            )
            self.assertTrue(out.exists())
            # Temp pyramid dir must be cleaned up.
            self.assertFalse(
                (Path(d) / f".stitch_pyr_tmp_{out.stem}").exists(),
                "streaming pyramid temp dir was not cleaned up",
            )
            # Full-res reads back identical.
            with tifffile.TiffFile(str(out)) as tf:
                base = tf.series[0].levels[0].asarray()
                np.testing.assert_array_equal(base, vol)
                # Two pyramid levels present at half/quarter YX.
                levels = tf.series[0].levels
                self.assertEqual(len(levels), 3)  # full + 2
                self.assertEqual(levels[1].asarray().shape, (3, 256, 256))
                self.assertEqual(levels[2].asarray().shape, (3, 128, 128))


if __name__ == "__main__":
    unittest.main()
