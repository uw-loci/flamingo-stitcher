"""Dropping a physically impossible seam BEFORE the global solve.

The quality threshold cannot catch this. Quality is a Spearman rank correlation
over the overlap (multiview_stitcher/registration.py:110), which stays high when
an elongated structure slides along its own axis — so a confident wrong peak
scores *well*. The run that motivated this accepted a pairwise dx of -345.7 µm
across a 156.7 µm overlap at quality 0.477: at that shift the two tiles do not
overlap at all, so it cannot be a measurement of their relative position.

Why before the solve rather than after: the post-solve clamp can only revert
whole TILES, and a solved component is only meaningful whole. On that run the
clamp reverted 4 tiles out of an 8-tile component and tore three seams that had
genuinely been measured.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("multiview_stitcher")
pytest.importorskip("networkx")
import networkx as nx  # noqa: E402
from multiview_stitcher import param_utils  # noqa: E402

from flamingo_stitcher import registration_report as rr  # noqa: E402
from flamingo_stitcher.pipeline import (  # noqa: E402
    StitchingConfig,
    StitchingPipeline,
)

# 3 tiles in a row, frame 100 µm (voxel 1 µm), pitch 80 µm -> 20 µm overlap.
_VOXEL = {"z": 1.0, "y": 1.0, "x": 1.0}


def _tile_data(xs, shape=(400, 100, 100)):
    vol = np.zeros(shape, np.uint16)  # only .shape is read
    return [(vol, SimpleNamespace(x_mm=x, y_mm=0.0, z_min_mm=0.0)) for x in xs]


def _edge(translation, quality=0.9):
    return {
        "transform": param_utils.affine_from_translation(
            np.asarray(translation, float)
        ),
        "quality": quality,
    }


class TestTheBound:
    def _reject(self, **config):
        pipe = StitchingPipeline(StitchingConfig(**config))
        return pipe._pairwise_shift_reject(_tile_data([0.0, 0.08, 0.16]), _VOXEL)

    def test_a_shift_larger_than_the_overlap_is_rejected(self):
        reject = self._reject()
        assert reject(0, 1, _edge((0, 0, 60))) is not None

    def test_a_plausible_shift_survives(self):
        reject = self._reject()
        assert reject(0, 1, _edge((0, 0, 4))) is None

    def test_the_reason_says_the_tiles_would_not_overlap(self):
        reason = self._reject()(0, 1, _edge((0, 0, 60)))
        assert "no longer overlap" in reason

    def test_a_high_quality_score_does_not_save_it(self):
        # The whole point: the peak is confident AND wrong.
        assert self._reject()(0, 1, _edge((0, 0, 60), quality=0.99)) is not None

    def test_z_is_bounded_separately_from_lateral(self):
        reject = self._reject()
        # Well inside the lateral bound, far outside the axial one.
        assert reject(0, 1, _edge((300, 0, 1))) is not None

    def test_an_explicit_bound_overrides_the_measured_overlap(self):
        # The Options tab exists so a rig can state what its stage can do.
        strict = self._reject(max_registration_shift_um=5.0)
        assert strict(0, 1, _edge((0, 0, 10))) is not None
        assert self._reject()(0, 1, _edge((0, 0, 10))) is None

    def test_an_edge_with_no_transform_is_left_alone(self):
        assert self._reject()(0, 1, {"quality": 0.9}) is None

    def test_no_bound_can_be_sized_means_no_predicate(self):
        pipe = StitchingPipeline(StitchingConfig())
        assert pipe._pairwise_shift_reject([], _VOXEL) is None


class TestTheShim:
    """The rejection has to happen inside multiview-stitcher's own call, at the
    last point before the global solve."""

    def _graph(self):
        g = nx.Graph()
        g.add_nodes_from([0, 1, 2])
        g.add_edge(0, 1, **_edge((0, 0, 3)))
        g.add_edge(1, 2, **_edge((0, 0, 345)))  # the impossible one
        return g

    def test_it_removes_the_impossible_edge_and_keeps_the_good_one(self):
        pipe = StitchingPipeline(StitchingConfig())
        reject = pipe._pairwise_shift_reject(_tile_data([0.0, 0.08, 0.16]), _VOXEL)
        sink = {}
        from multiview_stitcher import mv_graph

        with rr.capture_prefilter_graph(sink, reject):
            out = mv_graph.filter_edges(self._graph(), threshold=0.0, weight_key="quality")
        assert (0, 1) in out.edges
        assert (1, 2) not in out.edges
        assert list(sink["rejected"]) == [(1, 2)]
        assert "no longer overlap" in sink["rejected"][(1, 2)]

    def test_the_prefilter_graph_still_holds_every_edge(self):
        # The report needs the rejected edge's own quality score.
        pipe = StitchingPipeline(StitchingConfig())
        reject = pipe._pairwise_shift_reject(_tile_data([0.0, 0.08, 0.16]), _VOXEL)
        sink = {}
        from multiview_stitcher import mv_graph

        with rr.capture_prefilter_graph(sink, reject):
            mv_graph.filter_edges(self._graph(), threshold=0.0, weight_key="quality")
        assert set(sink["prefilter"].edges) == {(0, 1), (1, 2)}

    def test_without_a_predicate_nothing_is_removed(self):
        sink = {}
        from multiview_stitcher import mv_graph

        with rr.capture_prefilter_graph(sink):
            out = mv_graph.filter_edges(self._graph(), threshold=0.0, weight_key="quality")
        assert set(out.edges) == {(0, 1), (1, 2)}
        assert "rejected" not in sink

    def test_the_attribute_is_restored_afterwards(self):
        from multiview_stitcher import mv_graph

        before = mv_graph.filter_edges
        with rr.capture_prefilter_graph({}, lambda *a: None):
            pass
        assert mv_graph.filter_edges is before


class TestSeamStatus:
    def test_a_rejected_seam_is_not_reported_as_below_quality(self):
        # Distinct status, because the fix is opposite: lowering the quality
        # threshold makes an implausible-shift seam MORE likely, not less.
        tiles = [
            SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=0.0, folder=None, tile_index=0),
            SimpleNamespace(x_mm=0.08, y_mm=0.0, z_min_mm=0.0, folder=None, tile_index=1),
        ]
        prefilter = nx.Graph()
        prefilter.add_edge(0, 1, quality=0.88)
        seams = rr.extract_seams(
            tiles=tiles,
            voxel_size_um=_VOXEL,
            reg_dict={"pairwise_registration": {"graph": nx.Graph()}},
            prefilter_graph=prefilter,
            rejected_edges={(0, 1): "proposed a lateral shift of 345.0 µm"},
            frame_extent_um={"x": 100.0, "y": 100.0, "z": 400.0},
        )
        assert len(seams) == 1
        assert seams[0].status == rr.STATUS_IMPLAUSIBLE_SHIFT
        assert seams[0].quality == pytest.approx(0.88)
        assert "345.0" in seams[0].note

    def test_a_rejected_seam_does_not_connect_the_mosaic(self):
        seam = rr.SeamResult(
            tile_a="a", tile_b="b", index_a=0, index_b=1, axis="x",
            status=rr.STATUS_IMPLAUSIBLE_SHIFT,
        )
        cov = rr.mosaic_coverage(2, [seam])
        assert cov.n_registered_seams == 0
        assert not cov.is_safe_to_apply
