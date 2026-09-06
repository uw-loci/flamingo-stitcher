"""Centre-anchored placement, and carrying the tiles that cannot register.

A default single-workflow collection images a rectangle, so a round sample
leaves a rim of empty tiles. multiview-stitcher solves each connected component
of the seam graph independently and hands an edgeless tile the identity, so the
core moves and the rim stays at its stage position: the seam between them tears,
and the registered-seam share drops under the guard that decides whether to
trust the run at all. The 7x7 on 2026-09-04 registered 41 of 84 seams, came out
as 16 tile groups with 15 tiles connected to nothing, and was discarded after
9h33m of measuring it.

Phase 1 stays a simultaneous solve, only anchored at the centre. It is NOT a
greedy outward walk: placing tiles one at a time commits the answer in visit
order, so a tile whose left and bottom neighbours disagree can never satisfy
both. BigStitcher's answer is to optimise all links at once and drop the worst
link (keeping connectivity) rather than to sequence the placement, and
multiview-stitcher implements the same thing.

Run: python -m pytest tests/test_center_out_approach.py -q
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from flamingo_stitcher import center_out  # noqa: E402
from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline  # noqa: E402

PITCH = 1.8225


def _param(translation):
    m = np.eye(4)
    m[:3, 3] = translation
    return m


def _t(param):
    return np.asarray(param)[:3, 3]


def _grid(cols=7, rows=7, indexed=True):
    """A 7x7 like the run: X index increases with stage X, Y with stage Y."""
    tiles = []
    for row in range(rows):
        for col in range(cols):
            tiles.append(
                SimpleNamespace(
                    x_mm=1.33 + col * PITCH,
                    y_mm=14.04 + row * PITCH,
                    z_min_mm=15.13,
                    tile_index=(col, row) if indexed else None,
                )
            )
    return tiles


def _neighbours(cols=7, rows=7):
    pairs = []
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            if col + 1 < cols:
                pairs.append((i, i + 1, "x"))
            if row + 1 < rows:
                pairs.append((i, i + cols, "y"))
    return pairs


def _core_indices(cols=7, rows=7):
    """The interior 5x5 — what actually contains sample."""
    return [
        row * cols + col
        for row in range(1, rows - 1)
        for col in range(1, cols - 1)
    ]


class TestTheAnchor:
    def test_the_centre_tile_is_the_one_nearest_the_centroid(self):
        tiles = _grid()
        assert center_out.centre_tile_index(tiles) == 24  # col 3, row 3 of 7x7

    def test_a_partly_covered_grid_anchors_on_where_the_tiles_are(self):
        """The arithmetic middle of a grid can be background; the centroid is not."""
        tiles = _grid()[:10]
        index = center_out.centre_tile_index(tiles)
        assert index is not None and 0 <= index < 10

    def test_no_tiles_has_no_anchor(self):
        assert center_out.centre_tile_index([]) is None


class TestCarryingTheRim:
    def test_every_tile_outside_the_core_is_placed(self):
        tiles = _grid()
        core = _core_indices()
        params = [_param((0.0, 0.0, 0.0)) for _ in tiles]
        for i in core:
            params[i] = _param((0.0, 7.0, -3.0))
        out = center_out.carry_deferred_tiles(params, 49, core, _neighbours())
        assert len(out.carried) + len(out.orphans) == 49 - len(core)
        assert not out.orphans, "a 7x7 rim always touches the 5x5 core"

    def test_a_carried_tile_takes_its_own_neighbours_correction(self):
        """Not the mosaic median — the neighbours it actually touches.

        This is the whole difference from the `default` binding: on a mosaic
        whose corrections vary across the grid, a median leaves a rim tile out
        of register with the specific tiles beside it.
        """
        tiles = _grid()
        core = _core_indices()
        params = [_param((0.0, 0.0, 0.0)) for _ in tiles]
        # A correction that varies across the grid: +2 um per column.
        for i in core:
            col = i % 7
            params[i] = _param((0.0, 0.0, 2.0 * col))
        out = center_out.carry_deferred_tiles(params, 49, core, _neighbours())
        # Rim tile at col 0, row 3 touches exactly one core tile: col 1, row 3.
        carried = _t(out.params[3 * 7 + 0])
        assert carried[2] == pytest.approx(2.0 * 1)
        # And the median of the core would have been col 3's value, 6.0.
        assert carried[2] != pytest.approx(6.0)

    def test_it_spreads_outward_one_ring_per_pass(self):
        """A two-deep rim must not strand its outer ring."""
        tiles = _grid(9, 9)
        core = [r * 9 + c for r in range(3, 6) for c in range(3, 6)]
        params = [_param((0.0, 0.0, 0.0)) for _ in tiles]
        for i in core:
            params[i] = _param((0.0, 5.0, 0.0))
        out = center_out.carry_deferred_tiles(params, 81, core, _neighbours(9, 9))
        assert out.rounds >= 3
        assert not out.orphans
        assert _t(out.params[0])[1] == pytest.approx(5.0)

    def test_a_tile_touching_nothing_moves_with_the_mosaic(self):
        """Better than a stage position its neighbours have abandoned."""
        tiles = _grid(3, 3)
        core = [4]
        params = [_param((0.0, 0.0, 0.0)) for _ in tiles]
        params[4] = _param((0.0, 9.0, 0.0))
        # Only the centre's 4-neighbours are adjacent; corners touch nothing.
        pairs = [(1, 4, "y"), (3, 4, "x"), (4, 5, "x"), (4, 7, "y")]
        out = center_out.carry_deferred_tiles(params, 9, core, pairs)
        assert sorted(out.orphans) == [0, 2, 6, 8]
        assert _t(out.params[0])[1] == pytest.approx(9.0)

    def test_the_core_is_never_moved(self):
        tiles = _grid()
        core = _core_indices()
        params = [_param((0.0, float(i), 0.0)) for i in range(49)]
        out = center_out.carry_deferred_tiles(params, 49, core, _neighbours())
        for i in core:
            assert _t(out.params[i])[1] == pytest.approx(float(i))

    def test_a_rotated_param_is_left_alone(self):
        """Averaging translations is only valid for pure translations."""
        tiles = _grid(3, 3)
        params = [_param((0.0, 0.0, 0.0)) for _ in tiles]
        spun = np.eye(4)
        spun[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        params[4] = spun
        out = center_out.carry_deferred_tiles(params, 9, [4], _neighbours(3, 3))
        assert out.carried == [] and out.rounds == 0


class TestTensionAlerts:
    """The L-shape: left and bottom neighbours disagree about one tile."""

    @staticmethod
    def _seam(a, b, residual, status="registered"):
        return SimpleNamespace(
            index_a=a, index_b=b, residual_px=residual, status=status
        )

    def test_a_tile_whose_seams_disagree_is_named(self):
        tiles = _grid()
        seams = [self._seam(10, 11, 0.4), self._seam(11, 18, 9.2)]
        messages = center_out.tension_alerts(seams, tiles, tolerance_um=3.5)
        assert len(messages) == 1
        assert "not able to resolve all overlaps for tile" in messages[0]
        assert "X004 Y001" in messages[0]  # tile 11 = col 4, row 1
        assert "9.2" in messages[0]

    def test_one_seam_is_never_a_disagreement(self):
        """With a single neighbour there is nothing to be inconsistent with."""
        tiles = _grid()
        seams = [self._seam(10, 11, 99.0)]
        assert center_out.tension_alerts(seams, tiles, tolerance_um=3.5) == []

    def test_seams_within_tolerance_are_silent(self):
        tiles = _grid()
        seams = [self._seam(10, 11, 0.4), self._seam(11, 18, 1.1)]
        assert center_out.tension_alerts(seams, tiles, tolerance_um=3.5) == []

    def test_unregistered_seams_do_not_count(self):
        tiles = _grid()
        seams = [
            self._seam(10, 11, 40.0, status="below_quality"),
            self._seam(11, 18, 40.0, status="pruned"),
        ]
        assert center_out.tension_alerts(seams, tiles, tolerance_um=3.5) == []

    def test_multi_acquisition_tiles_are_named_by_position(self):
        """Folder-layout tiles have no grid index; position is their identity."""
        tiles = _grid(indexed=False)
        seams = [self._seam(10, 11, 0.4), self._seam(11, 18, 9.2)]
        messages = center_out.tension_alerts(seams, tiles, tolerance_um=3.5)
        assert len(messages) == 1
        assert "X=" in messages[0] and "Y=" in messages[0]


class TestTheApproachSwitch:
    def test_default_is_the_default(self):
        assert StitchingConfig().stitching_approach == "default"
        assert StitchingPipeline(StitchingConfig())._center_out_approach() is False

    def test_center_xy_is_recognised(self):
        pipe = StitchingPipeline(StitchingConfig(stitching_approach="center_xy"))
        assert pipe._center_out_approach() is True

    def test_the_anchor_is_only_passed_for_center_xy(self, caplog):
        tiles = _grid()
        plain = StitchingPipeline(StitchingConfig())
        assert "reference_view" not in plain._groupwise_kwargs(tiles, None)

        pipe = StitchingPipeline(StitchingConfig(stitching_approach="center_xy"))
        with caplog.at_level(logging.INFO, logger=pipe.logger.name):
            kwargs = pipe._groupwise_kwargs(tiles, None)
        assert kwargs["reference_view"] == 24
        assert "Anchoring the solve at the centre tile" in caplog.text

    def test_the_anchor_is_translated_into_solver_node_space(self):
        """Solver nodes are positions in the registered subset, not tile ids."""
        tiles = _grid()
        pipe = StitchingPipeline(StitchingConfig(stitching_approach="center_xy"))
        index_map = [24, 10, 11]  # tile 24 sits at position 0
        assert pipe._groupwise_kwargs(tiles, index_map)["reference_view"] == 0

    def test_a_held_out_centre_falls_back_to_the_solver_s_own_reference(self):
        tiles = _grid()
        pipe = StitchingPipeline(StitchingConfig(stitching_approach="center_xy"))
        assert "reference_view" not in pipe._groupwise_kwargs(tiles, [10, 11])

    def test_the_tolerances_survive_either_way(self):
        for approach in ("default", "center_xy"):
            kwargs = StitchingPipeline(
                StitchingConfig(stitching_approach=approach)
            )._groupwise_kwargs(_grid(), None)
            assert "abs_tol" in kwargs and "rel_tol" in kwargs
