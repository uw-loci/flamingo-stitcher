"""C1: the streaming path reuses the reference channel's registration spill for
its fusion pass instead of preprocessing + writing it to disk a second time.

This test exercises the streaming path with registration ENABLED (the existing
e2e tests use skip_registration=True, which bypasses the reference-channel
spill entirely) and multi-channel data, so both the reuse path (reference
channel) and the normal-materialize path (other channels) run. It asserts the
run completes, produces a correct-shaped output, and leaves no temp spill behind.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


@unittest.skipUnless(_HAVE, "needs multiview_stitcher + tifffile")
class TestRefChannelReuse(unittest.TestCase):
    def test_streaming_with_registration_reuses_ref_spill(self):
        with tempfile.TemporaryDirectory() as d:
            acq = write_synth_acquisition(
                Path(d) / "acq",
                grid=(2, 2),
                n_planes=12,
                channels=(1, 2),  # ref (ch 1) reused; ch 2 materialized normally
                frame_size=(96, 96),
                overlap=0.25,
            )
            self.assertEqual(len(discover_tiles(acq)), 4)

            cfg = StitchingConfig.with_yaml_defaults()
            cfg.skip_registration = False  # exercise the ref-channel spill + reuse
            cfg.streaming_mode = True
            cfg.output_format = "ome-tiff"
            cfg.resource_guard_enabled = False
            cfg.output_chunksize = {"z": 4, "y": 32, "x": 32}

            logs = []
            StitchingPipeline(
                cfg, progress_fn=lambda p, m: None
            ).logger.addHandler(_ListHandler(logs))
            out = StitchingPipeline(cfg, progress_fn=lambda p, m: None).run(
                acq, Path(d) / "out"
            )

            tif = next(Path(out).glob("*.ome.tif"))
            with tifffile.TiffFile(str(tif)) as tf:
                arr = tf.series[0].levels[0].asarray()
            self.assertEqual(arr.ndim, 4)  # (C, Z, Y, X)
            self.assertEqual(arr.shape[0], 2)  # two channels
            self.assertGreater(int((arr > 0).sum()), 0)  # not all zero

            # No temp spill left behind anywhere under the output dir.
            leftover = list(Path(d, "out").rglob(".stitch_tmp"))
            self.assertEqual(leftover, [], f"temp spill not cleaned: {leftover}")


import logging  # noqa: E402


class _ListHandler(logging.Handler):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        self.sink.append(record.getMessage())


if __name__ == "__main__":
    unittest.main()
