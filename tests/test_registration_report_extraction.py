"""Reading multiview-stitcher's registration output without trusting its shape.

The seam statuses come from three different places in MVS's return value, and
one of the facts we most want — what a rejected seam actually scored — is not in
it at all, because `register()` filters those edges out before returning. So
this file pins both the happy path and every degradation: a missing graph, a
missing metrics block, a graph that is not networkx, and the difference between
"below the threshold" (we know the number) and "no edge" (we do not, and must
not invent one).

The graph is read structurally through `.edges` / `.get_edge_data`, so these
tests hand it a SimpleNamespace and never import multiview-stitcher.

Run: python3 -m pytest tests/test_registration_report_extraction.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flamingo_stitcher import registration_report as rr

_VOXEL = {"z": 10.0, "y": 0.8, "x": 0.8}


def _tile(name, x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(
        folder=SimpleNamespace(name=name), x_mm=x, y_mm=y, z_min_mm=z
    )


def _row(n=2, pitch=0.9):
    return [_tile(f"X{i * pitch:.2f}_Y0.00", x=i * pitch) for i in range(n)]


def _affine(dz=0.0, dy=0.0, dx=0.0):
    mat = np.eye(4)
    mat[0, 3], mat[1, 3], mat[2, 3] = dz, dy, dx
    return mat


def _graph(edges: dict):
    """A stand-in for the networkx graph MVS returns."""
    return SimpleNamespace(
        edges=list(edges),
        get_edge_data=lambda a, b: edges.get((a, b)) or edges.get((b, a)),
    )


def _reg_dict(graph=None, used_edges=None, residuals=None):
    return {
        "pairwise_registration": {"graph": graph},
        "groupwise_resolution": {
            "metrics": {
                "used_edges": {0: used_edges} if used_edges is not None else None,
                "edge_residuals": {0: residuals} if residuals is not None else None,
            }
        },
    }


def _seams(tiles, **kwargs):
    return rr.extract_seams(tiles=tiles, voxel_size_um=_VOXEL, **kwargs)


class TestSurvivingEdges:
    def test_an_edge_that_fed_the_global_solve_reads_registered(self):
        graph = _graph({(0, 1): {"quality": 0.83, "transform": _affine(dz=12.0)}})
        seams = _seams(_row(2), reg_dict=_reg_dict(graph, used_edges=[(0, 1)]))
        assert seams[0].status == rr.STATUS_REGISTERED
        assert seams[0].quality == pytest.approx(0.83)

    def test_the_pairwise_shift_is_carried_through_in_both_units(self):
        graph = _graph({(0, 1): {"quality": 0.9, "transform": _affine(dz=25.0, dx=4.0)}})
        seams = _seams(_row(2), reg_dict=_reg_dict(graph, used_edges=[(0, 1)]))
        assert seams[0].dz_um == pytest.approx(25.0)
        assert seams[0].dz_frames == pytest.approx(2.5)
        assert seams[0].dx_px == pytest.approx(5.0)

    def test_an_edge_present_but_unused_reads_pruned(self):
        # It passed the quality filter and was then dropped by the global
        # optimization's edge pruning — a different diagnosis from a bad score.
        graph = _graph({(0, 1): {"quality": 0.9, "transform": _affine()}})
        seams = _seams(_row(2), reg_dict=_reg_dict(graph, used_edges=[]))
        assert seams[0].status == rr.STATUS_PRUNED
        assert "pruning" in seams[0].note

    def test_edge_orientation_does_not_matter(self):
        graph = _graph({(1, 0): {"quality": 0.7, "transform": _affine()}})
        seams = _seams(_row(2), reg_dict=_reg_dict(graph, used_edges=[(1, 0)]))
        assert seams[0].status == rr.STATUS_REGISTERED

    def test_the_residual_is_picked_up_when_present(self):
        graph = _graph({(0, 1): {"quality": 0.9, "transform": _affine()}})
        seams = _seams(
            _row(2),
            reg_dict=_reg_dict(graph, used_edges=[(0, 1)], residuals={(0, 1): 2.5}),
        )
        assert seams[0].residual_px == pytest.approx(2.5)

    def test_absent_used_edges_does_not_demote_everything_to_pruned(self):
        # An older MVS, or a resolver that reports no used_edges, must not make
        # a whole successful run read as "nothing was used".
        graph = _graph({(0, 1): {"quality": 0.9, "transform": _affine()}})
        seams = _seams(_row(2), reg_dict=_reg_dict(graph))
        assert seams[0].status == rr.STATUS_REGISTERED


class TestRejectedEdges:
    def test_a_captured_prefilter_graph_supplies_the_rejected_score(self):
        # The whole reason capture_prefilter_graph exists: this row is the most
        # actionable one in the report and MVS deletes it before returning.
        before = _graph({(0, 1): {"quality": 0.31, "transform": _affine()}})
        seams = _seams(
            _row(2),
            reg_dict=_reg_dict(_graph({})),
            prefilter_graph=before,
            quality_threshold=0.4,
        )
        assert seams[0].status == rr.STATUS_BELOW_QUALITY
        assert seams[0].quality == pytest.approx(0.31)
        assert "0.4" in seams[0].note

    def test_without_the_prefilter_graph_the_score_stays_unknown(self):
        # Never invent it. "dropped, quality unrecorded" is a true statement;
        # "quality 0.0" would send someone hunting a dead overlap.
        seams = _seams(_row(2), reg_dict=_reg_dict(_graph({})))
        assert seams[0].status == rr.STATUS_DROPPED
        assert seams[0].quality is None

    def test_the_dropped_note_explains_why_the_reason_is_missing(self):
        seams = _seams(_row(2), reg_dict=_reg_dict(_graph({})))
        assert "does not report which" in seams[0].note

    def test_the_report_warns_that_dropped_statuses_are_inferred(self):
        report = rr.build_report(
            tiles=_row(2),
            params=[_affine(), _affine()],
            voxel_size_um=_VOXEL,
            reg_dict=_reg_dict(_graph({})),
        )
        assert any("inferred rather than read" in w for w in report.warnings)


class TestDegradation:
    @pytest.mark.parametrize(
        "reg_dict",
        [
            None,
            {},
            {"pairwise_registration": None},
            {"pairwise_registration": {"graph": None}},
            {"pairwise_registration": {}, "groupwise_resolution": 7},
            {"groupwise_resolution": {"metrics": {"used_edges": "nonsense"}}},
        ],
    )
    def test_a_malformed_return_value_yields_rows_not_an_exception(self, reg_dict):
        seams = _seams(_row(3), reg_dict=reg_dict)
        assert len(seams) == 2
        assert all(s.status == rr.STATUS_DROPPED for s in seams)

    def test_a_graph_that_raises_on_iteration_is_survivable(self):
        class Hostile:
            @property
            def edges(self):
                raise RuntimeError("no")

        seams = _seams(_row(2), reg_dict=_reg_dict(Hostile()))
        assert seams[0].status == rr.STATUS_DROPPED

    def test_an_xarray_like_quality_is_reduced_to_a_scalar(self):
        # MVS stores quality as a DataArray over t; a bare float must not be
        # assumed. np.asarray covers both.
        graph = _graph(
            {(0, 1): {"quality": np.array([0.62]), "transform": _affine()}}
        )
        seams = _seams(_row(2), reg_dict=_reg_dict(graph, used_edges=[(0, 1)]))
        assert seams[0].quality == pytest.approx(0.62)

    def test_a_nan_quality_reads_as_unknown_not_as_zero(self):
        graph = _graph({(0, 1): {"quality": float("nan"), "transform": _affine()}})
        seams = _seams(_row(2), reg_dict=_reg_dict(graph, used_edges=[(0, 1)]))
        assert seams[0].quality is None


class TestPrefilterCapture:
    def test_it_is_a_no_op_when_multiview_stitcher_is_absent(self, monkeypatch):
        # Import failure must degrade to "no scores", never to a broken run.
        monkeypatch.setitem(__import__("sys").modules, "multiview_stitcher", None)
        sink = {}
        with rr.capture_prefilter_graph(sink):
            pass
        assert sink == {}

    def test_it_restores_the_original_function(self):
        mv_graph = pytest.importorskip("multiview_stitcher.mv_graph")
        original = mv_graph.filter_edges
        with rr.capture_prefilter_graph({}):
            assert mv_graph.filter_edges is not original
        assert mv_graph.filter_edges is original

    def test_it_restores_even_when_the_body_raises(self):
        mv_graph = pytest.importorskip("multiview_stitcher.mv_graph")
        original = mv_graph.filter_edges
        with pytest.raises(ValueError):
            with rr.capture_prefilter_graph({}):
                raise ValueError("boom")
        assert mv_graph.filter_edges is original

    def test_it_captures_the_graph_the_filter_was_handed(self):
        mv_graph = pytest.importorskip("multiview_stitcher.mv_graph")
        nx = pytest.importorskip("networkx")
        original = mv_graph.filter_edges
        graph = nx.Graph()
        graph.add_edge(0, 1, quality=0.31)
        sink = {}
        try:
            with rr.capture_prefilter_graph(sink):
                mv_graph.filter_edges(graph, threshold=0.4, weight_key="quality")
        finally:
            mv_graph.filter_edges = original
        assert list(sink["prefilter"].edges) == [(0, 1)]
