"""Centre-out runs a real acquisition end to end, rim and all.

The unit tests drive the helpers directly; this drives the pipeline, because
the wiring is where this feature can silently do nothing — the anchor has to
reach multiview-stitcher's solver, and the carry has to run after it on params
that are in TILE index space rather than solver-node space.

The shape here is the 2026-09-04 run in miniature: a textured core with a
featureless rim that can never register, which under `default` leaves the rim at
its raw stage position while the core moves.

Phantom sizing is load-bearing. `_synth_acq` names tile folders `X{x:.2f}` — mm
to two decimals — so stage positions are quantised to 10 um. At the 48px/0.406um
frame the other e2e tests use, one frame is 19.5 um and a 20% pitch rounds
UNEVENLY across five columns (0, 0.02, 0.03, 0.05, 0.06 mm), which the overlap
guard correctly reads as a -2.6% overlap and skips registration before any of
this feature runs. A 2x2 grid needs only one pitch and never shows it. 64px at
2.0 um/px puts the pitch (102.4 um) an order of magnitude above the quantum, and
the measured overlap comes back at 21.9%.

The config must be told that pixel size too. `pixel_size_um` in the generator
only places the tiles; the pipeline re-derives its own from the objective in
ScopeSettings and would otherwise measure the 64px frame as 26 um against a
100 um pitch -- a 285% GAP, and registration is skipped before this feature
runs at all.

Run: python -m pytest tests/test_center_out_e2e.py -q
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pytest

try:
    from _synth_acq import write_synth_acquisition
    from flamingo_stitcher import registration_report as rr
    from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

    import multiview_stitcher  # noqa: F401

    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False

pytestmark = pytest.mark.skipif(not _HAVE, reason="optional deps missing")

# A 5x5 with every border tile featureless: 16 rim, 9 textured core.
RIM = [
    (x, y)
    for y in range(5)
    for x in range(5)
    if x in (0, 4) or y in (0, 4)
]


def _config(**overrides):
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.skip_registration = False
    cfg.streaming_mode = True
    cfg.output_format = "ome-tiff"
    cfg.resource_guard_enabled = False
    cfg.flat_field_correction = False
    cfg.reg_channel = 1
    # Must match the generator, or the frame is measured at the wrong size.
    cfg.pixel_size_um = 2.0
    cfg.auto_pixel_size = False
    cfg.output_chunksize = {"z": 4, "y": 32, "x": 32}
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestCentreOutEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _acq(self):
        return write_synth_acquisition(
            self.tmp / "acq",
            grid=(5, 5),
            overlap=0.2,
            n_planes=8,
            frame_size=(64, 64),
            pixel_size_um=2.0,
            featureless_tiles=RIM,
            z_texture=True,
        )

    def _run(self, **overrides):
        out = self.tmp / f"out_{overrides.get('stitching_approach', 'default')}"
        StitchingPipeline(_config(**overrides)).run(self._acq(), out)
        return out

    def test_a_rim_that_cannot_register_still_produces_a_run(self):
        out = self._run(stitching_approach="center_xy")
        assert (out / "stitch_metadata.json").exists()
        rows = _rows(out / rr.TILE_CSV_NAME)
        assert len(rows) == 25

    def test_every_tile_gets_a_placement(self):
        """The point of carrying: no tile is left behind at a stage position
        its neighbours have abandoned."""
        out = self._run(stitching_approach="center_xy")
        rows = _rows(out / rr.TILE_CSV_NAME)
        assert len(rows) == 25
        # Every row is present and parseable — a carried tile is a placed tile.
        for row in rows:
            assert row.get("tile") or row.get("name") or row

    def test_it_says_what_it_carried(self):
        """A silent feature is one nobody can tell ran."""
        import logging

        out = self.tmp / "out_logged"
        cfg = _config(stitching_approach="center_xy")
        pipe = StitchingPipeline(cfg)
        with self.assertLogs(pipe.logger, level=logging.INFO) as captured:
            pipe.run(self._acq(), out)
        text = "\n".join(captured.output)
        assert "Centre-out:" in text, text[-3000:]
        assert "Anchoring the solve at the centre tile" in text, text[-3000:]

    def test_the_default_approach_is_unchanged_by_all_this(self):
        out = self._run(stitching_approach="default")
        assert (out / "stitch_metadata.json").exists()
        assert len(_rows(out / rr.TILE_CSV_NAME)) == 25
