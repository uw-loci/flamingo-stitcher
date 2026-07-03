"""End-to-end: super-block batched fusion must produce identical output to
whole-output fusion when run through the real StitchingPipeline.

This exercises the pipeline integration (region iteration, chunk-aligned slice
math, memmap allocation from full stack properties) — not just the MVS-level
equivalence in test_superblock_fusion.py.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import multiview_stitcher  # noqa: F401
    import tifffile

    from _synth_acq import write_synth_acquisition
    from flamingo_stitcher.pipeline import (
        StitchingConfig,
        StitchingPipeline,
        discover_tiles,
    )

    _HAVE = True
except Exception:
    _HAVE = False


def _stitch(acq, out_dir, superblock, overlap_mode):
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.skip_registration = True  # stage-position fusion → deterministic
    cfg.streaming_mode = True
    cfg.output_format = "ome-tiff"
    cfg.resource_guard_enabled = False
    cfg.tile_overlap_fusion = overlap_mode
    cfg.content_based_fusion = overlap_mode == "blend"
    # Small chunks so even a tiny mosaic spans several super-block regions.
    cfg.output_chunksize = {"z": 4, "y": 16, "x": 16}
    cfg.fusion_superblock_chunks = superblock
    StitchingPipeline(cfg).run(acq, out_dir)
    tif = next(Path(out_dir).glob("*.ome.tif"))
    with tifffile.TiffFile(str(tif)) as tf:
        return tf.series[0].levels[0].asarray()


@unittest.skipUnless(_HAVE, "needs multiview_stitcher + tifffile")
class TestSuperblockPipelineIdentical(unittest.TestCase):
    def _run_mode(self, overlap_mode):
        with tempfile.TemporaryDirectory() as d:
            acq = write_synth_acquisition(
                Path(d) / "acq",
                grid=(3, 3),
                n_planes=8,
                channels=(1,),
                frame_size=(48, 48),
                overlap=0.2,
            )
            self.assertEqual(len(discover_tiles(acq)), 9)
            whole = _stitch(acq, Path(d) / "whole", superblock=0, overlap_mode=overlap_mode)
            reg = _stitch(acq, Path(d) / "sb", superblock=1, overlap_mode=overlap_mode)
            self.assertEqual(whole.shape, reg.shape, overlap_mode)
            np.testing.assert_array_equal(
                whole,
                reg,
                err_msg=f"{overlap_mode}: super-block output differs from whole-output",
            )

    def test_max(self):
        self._run_mode("max")

    def test_blend(self):
        self._run_mode("blend")


if __name__ == "__main__":
    unittest.main()
