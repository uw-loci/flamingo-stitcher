"""Tiles may differ in DEPTH; they may not differ in FRAME SIZE.

An acquisition with per-tile Z ranges — what Collect Tiles produces, so the
scope images only the Z span where the sample actually is — gives every tile
its own plane count. The 97-tile run on 2026-08-08 died 43 seconds in with::

    RuntimeError: Tile 7 shape (1502, 1024, 1024) != expected (1287, 1024, 1024)

``_materialize_tiles_to_disk`` had probed ``tiles[0]`` and imposed its shape on
all 97. Nothing downstream needed that: ``_build_fusion_inputs`` places each
tile at its own ``z_min_mm`` and multiview-stitcher accepts views of differing
shape. The uniform-shape check was the only obstacle, and the header line
("Planes per tile: 1287") reported tile 0 as if it spoke for the set, so the
log gave no hint the tiles were ever different sizes.

Frame size is the real invariant: Y/X disagreeing means the AOI or binning
changed mid-acquisition, and no placement can reconcile that.

Run: python -m pytest tests/test_per_tile_z_depth.py -q
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import multiview_stitcher  # noqa: F401

    from _synth_acq import write_synth_acquisition
    from flamingo_stitcher.pipeline import (
        StitchingConfig,
        StitchingPipeline,
        discover_tiles,
    )

    _HAVE = True
except Exception:
    _HAVE = False


def _pipeline(config=None):
    """A pipeline instance we can call individual methods on."""
    return StitchingPipeline(config or StitchingConfig())


@unittest.skipUnless(_HAVE, "multiview-stitcher not installed")
class TestDepthMayVaryPerTile(unittest.TestCase):
    def test_tiles_of_differing_depth_each_keep_their_own(self):
        """The exact shape of the failure: one tile deeper than the probe."""
        pipe = _pipeline()

        class _T:
            def __init__(self, n):
                self.n_planes = n
                self.frame_width = 8
                self.frame_height = 8
                self.x_mm = 0.0
                self.y_mm = 0.0
                self.folder = Path(f"X0.00_Y0.00_p{n}")
                self.raw_files = {1: {0: Path("unused")}}

        tiles = [_T(6), _T(9), _T(4)]
        made = {}

        def fake_preprocess(tile, ch_id, illum_side=None):
            return np.full((tile.n_planes, 8, 8), tile.n_planes, dtype=np.uint16)

        pipe._preprocess_single_tile = fake_preprocess

        with tempfile.TemporaryDirectory() as td:
            out = pipe._materialize_tiles_to_disk(
                tiles, 1, (6, 8, 8), Path(td) / "spill"
            )
            self.assertEqual(len(out), 3)
            for lazy, tile in out:
                made[tile.n_planes] = lazy.shape
                # The spilled bytes must round-trip, not just the shape.
                self.assertTrue(
                    np.all(np.asarray(lazy) == tile.n_planes),
                    f"tile of {tile.n_planes} planes did not round-trip",
                )

        self.assertEqual(made, {6: (6, 8, 8), 9: (9, 8, 8), 4: (4, 8, 8)})

    def test_a_changed_frame_size_still_raises(self):
        """Depth is data; frame size is a mid-run hardware change."""
        pipe = _pipeline()

        class _T:
            def __init__(self, n, w):
                self.n_planes = n
                self.frame_width = w
                self.frame_height = 8
                self.x_mm = 0.0
                self.y_mm = 0.0
                self.folder = Path(f"X0.00_Y0.00_w{w}")
                self.raw_files = {1: {0: Path("unused")}}

        tiles = [_T(6, 8), _T(6, 16)]

        def fake_preprocess(tile, ch_id, illum_side=None):
            return np.zeros((tile.n_planes, 8, tile.frame_width), dtype=np.uint16)

        pipe._preprocess_single_tile = fake_preprocess

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError) as ctx:
                pipe._materialize_tiles_to_disk(
                    tiles, 1, (6, 8, 8), Path(td) / "spill"
                )
        msg = str(ctx.exception)
        self.assertIn("frame", msg.lower())
        # The message has to tell the user which axis is allowed to vary,
        # or they will "fix" the wrong thing.
        self.assertIn("Depth may vary", msg)


@unittest.skipUnless(_HAVE, "multiview-stitcher not installed")
class TestTheLogDescribesTheSpread(unittest.TestCase):
    """Reporting tile 0's depth as "planes per tile" is what hid this."""

    def _summarise(self, tiles):
        pipe = _pipeline()
        records = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        pipe.logger = logging.getLogger(f"test.summary.{id(tiles)}")
        pipe.logger.handlers = [_Cap()]
        pipe.logger.setLevel(logging.INFO)
        pipe.logger.propagate = False
        pipe._log_tile_summary(tiles)
        return records

    def _tile(self, n, z0=10.0, step_um=5.0):
        class _T:
            pass

        t = _T()
        t.n_planes = n
        t.x_mm = 0.0
        t.y_mm = 0.0
        t.z_min_mm = z0
        t.z_max_mm = z0 + (n - 1) * step_um / 1000.0
        t.channels = [1]
        t.illumination_sides = [0]
        t.raw_files = {}
        t.z_step_mm = (t.z_max_mm - t.z_min_mm) / (n - 1) if n > 1 else 0.0
        return t

    def test_uniform_depth_reads_as_a_single_number(self):
        lines = self._summarise([self._tile(16), self._tile(16)])
        planes = [ln for ln in lines if "Planes per tile" in ln]
        self.assertTrue(planes)
        self.assertIn("16", planes[0])
        self.assertNotIn("varies", planes[0])

    def test_varying_depth_reads_as_a_range(self):
        lines = self._summarise(
            [self._tile(16), self._tile(24), self._tile(20)]
        )
        planes = [ln for ln in lines if "Planes per tile" in ln]
        self.assertTrue(planes)
        self.assertIn("16", planes[0])
        self.assertIn("24", planes[0])
        self.assertIn("varies", planes[0])

    def test_a_differing_z_step_is_called_out(self):
        """Depth may vary. The STEP may not — one voxel size covers all tiles."""
        tiles = [self._tile(16, step_um=5.0), self._tile(16, step_um=8.0)]
        lines = self._summarise(tiles)
        warned = [ln for ln in lines if "Z step differs" in ln]
        self.assertTrue(warned, f"no Z-step warning in: {lines}")
        self.assertIn("stretched", warned[0])
        # The consequence has to be quantified, or "differs" is unactionable.
        self.assertIn("planes of drift", warned[0])

    def test_a_shared_z_step_stays_quiet(self):
        tiles = [self._tile(16), self._tile(24)]
        lines = self._summarise(tiles)
        self.assertFalse([ln for ln in lines if "Z step differs" in ln])

    def test_rounding_noise_in_the_z_bounds_does_not_cry_wolf(self):
        """The real numbers from the 2026-08-08 97-tile run.

        Each tile's step is derived as (z_max - z_min) / (n_planes - 1) from
        bounds the acquisition rounds, so tiles land a few nanometres apart.
        5.0022–5.0047 µm is 0.05% — under one plane of drift across even the
        deepest tile. An absolute threshold (the first cut used 1e-6 mm) fires
        on every ragged acquisition and trains the user to ignore the warning.
        """
        tiles = [
            self._tile(1287, step_um=5.0022),
            self._tile(1931, step_um=5.0047),
            self._tile(1500, step_um=5.0035),
        ]
        lines = self._summarise(tiles)
        self.assertFalse(
            [ln for ln in lines if "Z step differs" in ln],
            "0.05% step spread is arithmetic noise, not a different Z step",
        )

    def test_the_threshold_sits_between_noise_and_a_real_change(self):
        """Guard the constant itself: noise is ~0.05%, a real change ~60%."""
        from flamingo_stitcher.pipeline import Z_STEP_SPREAD_WARN_FRAC

        self.assertGreater(Z_STEP_SPREAD_WARN_FRAC, 0.001)
        self.assertLess(Z_STEP_SPREAD_WARN_FRAC, 0.10)


@unittest.skipUnless(_HAVE, "multiview-stitcher not installed")
class TestEndToEndWithPerTileZRanges(unittest.TestCase):
    """The whole point: such an acquisition must stitch, not abort."""

    def test_a_ragged_acquisition_fuses_to_the_deepest_extent(self):
        with tempfile.TemporaryDirectory() as td:
            acq = write_synth_acquisition(
                Path(td) / "acq",
                grid=(2, 2),
                n_planes=16,
                frame_size=(32, 32),
                # Row-major (ix, iy): three depths across four tiles, which is
                # what per-tile Z ranges look like on disk.
                tile_planes={(0, 0): 16, (1, 0): 24, (0, 1): 20, (1, 1): 24},
            )
            tiles = discover_tiles(acq)
            self.assertEqual(len(tiles), 4)
            depths = sorted(t.n_planes for t in tiles)
            self.assertEqual(depths, [16, 20, 24, 24], "synth data isn't ragged")

            cfg = StitchingConfig.with_yaml_defaults()
            cfg.skip_registration = True
            cfg.streaming_mode = True
            cfg.output_format = "ome-tiff"
            cfg.resource_guard_enabled = False
            cfg.output_chunksize = {"z": 8, "y": 32, "x": 32}

            out = StitchingPipeline(cfg, progress_fn=lambda p, m: None).run(
                acq, Path(td) / "out"
            )

            import tifffile

            tif = next(Path(out).glob("*.ome.tif"))
            with tifffile.TiffFile(str(tif)) as tf:
                fused = tf.series[0].levels[0].asarray()
            fused = np.squeeze(fused)
            self.assertEqual(fused.ndim, 3)
            # Z must span the DEEPEST tile — a fuse that silently truncated to
            # the probe tile's 16 planes would lose a third of the volume.
            self.assertEqual(
                fused.shape[0],
                24,
                f"fused Z {fused.shape[0]} should cover the deepest tile (24)",
            )
            self.assertTrue(fused.max() > 0, "fused volume is empty")
            # The deep planes exist only in the deeper tiles, so they must
            # carry signal rather than being an all-zero pad.
            self.assertTrue(
                fused[20:].max() > 0,
                "planes beyond the shallowest tile came out empty",
            )


if __name__ == "__main__":
    unittest.main()
