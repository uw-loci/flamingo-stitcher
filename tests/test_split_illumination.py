"""split_illumination: output each light path as its own channel.

Two layers:
  * Pure unit test of ``_output_channel_units`` (channel expansion) — no I/O.
  * End-to-end through the real streaming pipeline: a two-illumination-side
    acquisition, stitched with split on, must yield two output channels
    (Channel_<ch>_I0 / _I1) that differ; with split off it yields one fused
    channel — byte-identical channel_ids to the historical behaviour.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


def _fake_tile(raw_files):
    return SimpleNamespace(raw_files=raw_files)


@unittest.skipUnless(_HAVE, "needs flamingo_stitcher import")
class TestOutputChannelUnits(unittest.TestCase):
    """Pure expansion logic, no files touched."""

    def _pipeline(self, split):
        cfg = StitchingConfig.with_yaml_defaults()
        cfg.split_illumination = split
        return StitchingPipeline(cfg)

    def test_unsplit_is_identity_ints(self):
        # One channel, two sides — but split off → single fused unit, int label.
        tiles = [_fake_tile({1: {0: "a", 1: "b"}})]
        units = self._pipeline(False)._output_channel_units(tiles, [1])
        self.assertEqual(units, [(1, 1, None)])

    def test_split_expands_two_sided_channel(self):
        tiles = [_fake_tile({3: {0: "a", 1: "b"}})]
        units = self._pipeline(True)._output_channel_units(tiles, [3])
        self.assertEqual(units, [("3_I0", 3, 0), ("3_I1", 3, 1)])

    def test_split_leaves_single_sided_channel_alone(self):
        # Only one illumination side present → nothing to split, stays fused.
        tiles = [_fake_tile({2: {0: "a"}})]
        units = self._pipeline(True)._output_channel_units(tiles, [2])
        self.assertEqual(units, [(2, 2, None)])

    def test_split_multichannel_order_and_sides(self):
        tiles = [_fake_tile({1: {0: "a", 1: "b"}, 2: {0: "c"}})]
        units = self._pipeline(True)._output_channel_units(tiles, [1, 2])
        # ch1 splits into two sides; ch2 (single side) stays fused. Order kept.
        self.assertEqual(
            units, [("1_I0", 1, 0), ("1_I1", 1, 1), (2, 2, None)]
        )


def _stitch(acq, out_dir, split):
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.skip_registration = True
    cfg.streaming_mode = True
    cfg.output_format = "ome-tiff"
    cfg.resource_guard_enabled = False
    cfg.split_illumination = split
    cfg.output_chunksize = {"z": 4, "y": 16, "x": 16}
    StitchingPipeline(cfg).run(acq, out_dir)
    meta = json.loads((Path(out_dir) / "stitch_metadata.json").read_text())
    tif = next(Path(out_dir).glob("*.ome.tif"))
    with tifffile.TiffFile(str(tif)) as tf:
        arr = tf.series[0].levels[0].asarray()
    return meta, arr


@unittest.skipUnless(_HAVE, "needs multiview_stitcher + tifffile")
class TestSplitIlluminationPipeline(unittest.TestCase):
    def test_split_yields_two_channels_that_differ(self):
        with tempfile.TemporaryDirectory() as d:
            acq = write_synth_acquisition(
                Path(d) / "acq",
                grid=(2, 2),
                n_planes=8,
                channels=(1,),
                illum_sides=(0, 1),  # two light paths (side 1 is 1.1x side 0)
                frame_size=(32, 32),
                overlap=0.2,
            )
            self.assertEqual(len(discover_tiles(acq)), 4)

            # Split OFF: one fused channel, historical int channel id.
            meta_f, arr_f = _stitch(acq, Path(d) / "fused", split=False)
            self.assertEqual(meta_f["channel_ids"], [1])

            # Split ON: two channels, one per light path.
            meta_s, arr_s = _stitch(acq, Path(d) / "split", split=True)
            self.assertEqual(meta_s["channel_ids"], ["1_I0", "1_I1"])
            self.assertEqual(
                list(meta_s["channels"].keys()), ["1_I0", "1_I1"]
            )

            # The split output must carry 2 channels vs the fused 1, and the two
            # light-path channels must actually differ (side bias 1.0 vs 1.1),
            # otherwise the split collapsed back to one image.
            c_axis_s = arr_s.shape[0] if arr_s.ndim == 4 else 1
            self.assertEqual(c_axis_s, 2)
            i0 = arr_s[0].astype(np.int32)
            i1 = arr_s[1].astype(np.int32)
            self.assertTrue(
                np.any(i0 != i1), "the two light-path channels are identical"
            )
