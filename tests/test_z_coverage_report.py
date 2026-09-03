"""The fused stack is the UNION of the registered tiles, so its edge is ragged.

A 3x3 run on 2026-09-03 came back with big black rectangles in plane 162 of 164
and read as a broken stitch. It was not: registration had shifted the tiles in Z
by -26.0 .. +12.3 um (a 10 um Z step), so the union bounding box is ~4 planes
taller than any single tile, and in those extra planes only the tiles shifted
that way have data. multiview-stitcher transforms with ``cval=np.nan`` and ends
``fuse_field`` with ``np.nan_to_num``, so an uncovered voxel is a hard 0.

Tile-overlap fusion is not the cause and cannot be: ``max_fusion`` is a plain
``np.nanmax`` and ``fuse_field`` only builds blending weights when the fusion
func declares a ``blending_weights`` kwarg, which ``max_fusion`` does not. A
max-fused voxel is never dimmer than the dimmest tile covering it.

So the run has to SAY which planes are fully covered — otherwise the edge of the
volume is indistinguishable from missing data.

Run: python -m pytest tests/test_z_coverage_report.py -q
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("multiview_stitcher")
from multiview_stitcher import param_utils  # noqa: E402

from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline  # noqa: E402

_VOXEL = {"z": 10.0, "y": 2.095, "x": 2.095}  # the 2026-09-03 run's voxel


def _params(dzs):
    return [
        param_utils.affine_from_translation(np.asarray((dz, 0.0, 0.0), float))
        for dz in dzs
    ]


def _tile_data(n, n_planes=160, z_min_mm=16.931):
    vol = np.zeros((n_planes, 8, 8), np.uint16)  # only .shape is read
    return [(vol, SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=z_min_mm))] * n


def _log(pipe, params, td, voxel=_VOXEL, caplog=None):
    with caplog.at_level(logging.INFO, logger=pipe.logger.name):
        pipe._log_z_coverage(params, td, voxel)
    return "\n".join(r.getMessage() for r in caplog.records)


class TestRaggedEdgeIsReported:
    def test_reproduces_the_2026_09_03_geometry(self, caplog):
        """dz spanning -26.0..+12.3 um turns 160 tile planes into 164 fused."""
        pipe = StitchingPipeline(StitchingConfig())
        dzs = [0.0, -4.5, -14.5, -26.0, -0.6, +12.3, -8.0, +3.0, -20.0]
        out = _log(pipe, _params(dzs), _tile_data(9), caplog=caplog)

        assert "164 planes" in out, out
        # The tiles are 160 planes each; a 38.3 um spread (3.83 Z steps) both
        # adds 4 planes to the union and takes 4 off the fully covered band.
        assert "(156)" in out and "every tile" in out, out
        assert "not a fusion artifact" in out, out

    def test_plane_162_of_164_is_named_as_outside_the_covered_range(self, caplog):
        """The plane the user opened must fall outside the range we report."""
        pipe = StitchingPipeline(StitchingConfig())
        dzs = [0.0, -4.5, -14.5, -26.0, -0.6, +12.3, -8.0, +3.0, -20.0]
        out = _log(pipe, _params(dzs), _tile_data(9), caplog=caplog)

        # The covered band is planes 4-159, so 162 sits in the ragged top —
        # exactly where the black rectangles were.
        assert "planes 4-159" in out, out
        covered = range(4, 160)
        assert 162 not in covered


class TestNoFalseAlarm:
    def test_unshifted_tiles_report_full_coverage(self, caplog):
        pipe = StitchingPipeline(StitchingConfig())
        out = _log(pipe, _params([0.0] * 4), _tile_data(4), caplog=caplog)
        assert "all 160 fused planes are covered by every tile" in out, out

    def test_subplane_shifts_do_not_invent_a_ragged_edge(self, caplog):
        """Shifts well under one Z step round to zero extra planes."""
        pipe = StitchingPipeline(StitchingConfig())
        out = _log(pipe, _params([0.0, 0.9, -1.2, 2.0]), _tile_data(4), caplog=caplog)
        assert "every tile" in out, out


class TestDoesNotKillTheRun:
    def test_no_tiles_is_silent(self, caplog):
        pipe = StitchingPipeline(StitchingConfig())
        assert _log(pipe, [], [], caplog=caplog) == ""

    def test_unusable_tile_info_is_swallowed(self, caplog):
        pipe = StitchingPipeline(StitchingConfig())
        td = [(np.zeros((4, 8, 8)), SimpleNamespace())]  # no z_min_mm
        assert _log(pipe, _params([0.0]), td, caplog=caplog) == ""

    def test_zero_z_voxel_is_silent(self, caplog):
        pipe = StitchingPipeline(StitchingConfig())
        out = _log(
            pipe, _params([0.0]), _tile_data(1), voxel={"z": 0.0}, caplog=caplog
        )
        assert out == ""


class TestPerTileZDepths:
    def test_shorter_tile_shrinks_the_covered_band(self, caplog):
        """Collect Tiles gives every tile its own depth; the shallowest wins."""
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data(2) + [
            (np.zeros((150, 8, 8), np.uint16), SimpleNamespace(z_min_mm=16.931))
        ]
        out = _log(pipe, _params([0.0, 0.0, 0.0]), td, caplog=caplog)
        assert "160 planes" in out, out
        assert "(150)" in out, out
