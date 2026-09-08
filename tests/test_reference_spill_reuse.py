"""The registration spill is kept only when something can actually reuse it.

After registering, the reference channel's preprocessed tiles are held so the
fusion loop does not materialise them a second time. But that spill holds
ILLUMINATION-FUSED tiles (side = None), and a unit only reuses a spill whose
side matches. When the run splits the light paths, the fusion loop asks for
side 0 and side 1 — so the spill stands in for neither and sits untouched until
the end of the run holding a full copy of every tile.

On the 2026-09-06 7x7 that is 246 GB of dead scratch, and the estimate does not
count it: the run needed ~880 GB against a stated ~620 GB. At the 4x-in-Z sizes
it is 984 GB, and 3520 GB against 3230 GB free is a run that dies on disk hours
in, after the registration has already been paid for.

Run: python -m pytest tests/test_reference_spill_reuse.py -q
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import pytest

try:
    from _synth_acq import write_synth_acquisition
    from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

    import multiview_stitcher  # noqa: F401

    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False

pytestmark = pytest.mark.skipif(not _HAVE, reason="optional deps missing")


def _config(**overrides):
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.skip_registration = False
    cfg.streaming_mode = True
    cfg.output_format = "ome-tiff"
    cfg.resource_guard_enabled = False
    cfg.flat_field_correction = False
    cfg.reg_channel = 1
    cfg.pixel_size_um = 2.0
    cfg.auto_pixel_size = False
    cfg.verbose_alignment_log = False
    cfg.output_chunksize = {"z": 4, "y": 32, "x": 32}
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class TestTheSpillIsDroppedWhenUnusable(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _acq(self, sides):
        return write_synth_acquisition(
            self.tmp / f"acq{len(sides)}",
            grid=(2, 2), overlap=0.2, n_planes=8,
            frame_size=(64, 64), pixel_size_um=2.0,
            illum_sides=sides, z_texture=True,
        )

    def _run(self, acq, out_name, **overrides):
        pipe = StitchingPipeline(_config(**overrides))
        with self.assertLogs(pipe.logger, level=logging.INFO) as captured:
            pipe.run(acq, self.tmp / out_name)
        return "\n".join(captured.output)

    def test_splitting_the_sides_drops_it(self):
        text = self._run(
            self._acq((0, 1)), "out_split", split_illumination=True
        )
        assert "dropped the reference spill" in text, text[-2000:]
        assert "reusing ref-channel spill" not in text

    def test_fusing_the_sides_keeps_it(self):
        """The saving is real when it can be spent: fusing the sides means the
        spill matches the one unit, and re-materialising would be a wasted pass
        over every tile."""
        text = self._run(
            self._acq((0, 1)), "out_fused", split_illumination=False
        )
        assert "reusing ref-channel spill" in text, text[-2000:]
        assert "dropped the reference spill" not in text

    def test_a_single_sided_acquisition_keeps_it_even_when_splitting(self):
        """Nothing to split, so the spill still matches the only unit."""
        text = self._run(
            self._acq((0,)), "out_single", split_illumination=True
        )
        assert "reusing ref-channel spill" in text, text[-2000:]
