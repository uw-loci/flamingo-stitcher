"""Seams whose SHARED STRIP is empty, and the sampling bug that hid them.

Two tiles can both be full of sample while the strip they actually share is
empty medium -- the sample simply ends partway across the mosaic. The whole-tile
content gate passes both, so the pair is scheduled for a full pairwise
registration over nothing. Phase correlation there does not fail loudly: it
returns a confident peak drawn from noise, which then has to be caught
downstream by the quality threshold or the shift bound, and is counted as a
FAILED seam either way. Not asking is cheaper and more honest.

The measure this rides on was also broken. `_subsample` derived its Z stride
from the voxel budget, and on a real 643 x 2048 x 2048 tile one plane already
fills that budget -- so every tile was scored on PLANE 0, which in a light-sheet
stack is empty medium.

Run: python -m pytest tests/test_seam_content_gate.py -q
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from flamingo_stitcher import registration_report as rr
from flamingo_stitcher import tile_content as tc
from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

RNG = np.random.default_rng(11)

_NZ, _NY, _NX = 8, 64, 64
_VOXEL = {"z": 10.0, "y": 1.0, "x": 1.0}


def _blobs(shape):
    seed = RNG.random(shape).astype(np.float32)
    out = ndimage.gaussian_filter(seed, sigma=2.0)
    return (out - out.min()) / max(1e-9, float(np.ptp(out)))


def _tile(content_x=None):
    """A tile that is empty except for a structured band in X.

    `content_x` is a (lo, hi) slice of columns that carries sample; None means
    the whole tile is empty medium.
    """
    vol = RNG.normal(3000, 40, (_NZ, _NY, _NX)).astype(np.float32)
    if content_x is not None:
        lo, hi = content_x
        vol[:, :, lo:hi] += 2500.0 * _blobs((_NZ, _NY, hi - lo))
    return vol.astype(np.uint16)


# --------------------------------------------------------------------------- #
# The sampling bug the gate is built on
# --------------------------------------------------------------------------- #


class TestTheSampleSpansTheStack:
    def test_a_real_tile_is_not_collapsed_to_a_single_plane(self):
        # 643 x 2048 x 2048 is the acquisition that exposed this.
        out = tc._subsample(np.zeros((643, 2048, 2048), np.uint16))
        assert out.shape[0] > 1, "scored one plane of a 643-plane stack"
        assert out.shape[0] >= min(643, tc._MIN_SAMPLE_PLANES)

    def test_the_sample_reaches_the_far_end_of_the_stack(self):
        vol = np.zeros((643, 512, 512), np.uint16)
        vol[600:] = 1  # content only near the bottom
        assert tc._subsample(vol).max() == 1, "never looked past the top of Z"

    def test_it_still_respects_the_voxel_budget(self):
        assert tc._subsample(np.zeros((4000, 512, 512), np.uint16)).size <= (
            tc._MAX_SAMPLE_VOXELS
        )

    def test_a_sample_slab_in_the_middle_is_seen(self):
        # The realistic light-sheet case: sample in a slab, medium either side.
        vol = np.zeros((400, 256, 256), np.float32)
        vol[180:220] = 1000.0 * _blobs((40, 256, 256))
        vol += RNG.normal(0, 5, vol.shape).astype(np.float32)
        assert tc.structure_score(vol) is not None
        assert tc.structure_score(vol) > tc.DEFAULT_MIN_STRUCTURE


# --------------------------------------------------------------------------- #
# Edge bands
# --------------------------------------------------------------------------- #


class TestEdgeScores:
    def test_content_on_the_high_side_shows_there_and_not_on_the_low_side(self):
        edges = tc.edge_structure(_tile(content_x=(48, 64)), fraction=0.2)
        assert edges.x_high > tc.DEFAULT_MIN_STRUCTURE
        assert edges.x_low < tc.DEFAULT_MIN_STRUCTURE

    def test_content_on_the_low_side_shows_there(self):
        edges = tc.edge_structure(_tile(content_x=(0, 16)), fraction=0.2)
        assert edges.x_low > tc.DEFAULT_MIN_STRUCTURE
        assert edges.x_high < tc.DEFAULT_MIN_STRUCTURE

    def test_an_unmeasurable_band_counts_as_content(self):
        # Same convention as the tile gate: only exclude what we are sure about.
        assert tc.EdgeContent().has_content("x", True) is True

    def test_score_tiles_carries_the_edges_when_asked(self):
        results = tc.score_tiles([_tile((48, 64))], edge_fraction=0.2)
        assert results[0].edges is not None
        assert results[0].edges.x_high > results[0].edges.x_low

    def test_score_tiles_leaves_edges_alone_by_default(self):
        assert tc.score_tiles([_tile((48, 64))])[0].edges is None


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def _msims(n, pitch_um):
    from multiview_stitcher import msi_utils
    from multiview_stitcher import spatial_image_utils as si_utils

    out = []
    for i in range(n):
        sim = si_utils.get_sim_from_array(
            np.zeros((_NZ, _NY, _NX), np.uint16),
            dims=["z", "y", "x"],
            scale=_VOXEL,
            translation={"z": 0.0, "y": 0.0, "x": i * pitch_um},
            transform_key="metadata",
        )
        out.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
    return out


def _content(volumes, fraction=0.2):
    return tc.score_tiles(volumes, edge_fraction=fraction)


def _pipe(**kwargs):
    return StitchingPipeline(StitchingConfig(**kwargs))


# 85% pitch -> a 15% overlap, as the rig collects.
PITCH_UM = _NX * 0.85 * _VOXEL["x"]


class TestItDropsOnlyEmptySeams:
    def test_a_seam_with_sample_on_both_sides_survives(self):
        pipe = _pipe()
        # Tile 0 has content at its high edge, tile 1 at its low edge: they
        # meet in the middle, which is exactly what the seam shares.
        vols = [_tile((48, 64)), _tile((0, 16))]
        pairs, dropped = pipe._empty_overlap_pairs(
            _msims(2, PITCH_UM), _content(vols), None,
            [(0.0, 0.0), (PITCH_UM, 0.0)], "metadata",
        )
        assert dropped == []
        assert pairs is None, "nothing dropped should leave MVS's graph alone"

    def test_a_seam_whose_shared_strip_is_empty_on_one_side_is_dropped(self):
        # The real shape of this: the sample ends partway across the mosaic.
        # Tile 0 carries sample only at its LOW edge, away from tile 1, so the
        # 0-1 strip is empty -- while 1-2 is a perfectly good seam.
        pipe = _pipe()
        vols = [_tile((0, 16)), _tile((0, 16)) + _tile((48, 64)), _tile((0, 16))]
        pairs, dropped = pipe._empty_overlap_pairs(
            _msims(3, PITCH_UM), _content(vols), None,
            [(0.0, 0.0), (PITCH_UM, 0.0), (2 * PITCH_UM, 0.0)], "metadata",
        )
        assert dropped == [(0, 1)]
        assert pairs == [(1, 2)], "dropped a seam that had sample on both sides"

    def test_both_tiles_pass_the_whole_tile_gate_first(self):
        # Otherwise this is just the existing content gate wearing a hat.
        vols = [_tile((0, 16)), _tile((0, 16))]
        assert all(c.has_content for c in _content(vols))

    def test_the_gate_off_drops_nothing(self):
        pipe = _pipe(registration_seam_content_gate=False)
        assert pipe.config.registration_seam_content_gate is False


class TestItIsReportedAsNeverAttempted:
    def _seams(self, gated):
        tiles = [
            SimpleNamespace(x_mm=0.0, y_mm=0.0, folder=None),
            SimpleNamespace(x_mm=PITCH_UM / 1000.0, y_mm=0.0, folder=None),
        ]
        return rr.extract_seams(
            tiles=tiles,
            voxel_size_um=_VOXEL,
            reg_dict=None,
            empty_overlap_pairs=gated,
            frame_extent_um={"x": _NX * _VOXEL["x"], "y": _NY * _VOXEL["y"]},
        )

    def test_a_gated_seam_gets_its_own_status(self):
        seams = self._seams([(0, 1)])
        assert seams and seams[0].status == rr.STATUS_EMPTY_OVERLAP

    def test_it_says_the_pair_was_not_attempted(self):
        assert "not attempted" in self._seams([(0, 1)])[0].note

    def test_it_does_not_count_against_mosaic_coverage(self):
        # The whole point: a seam nobody could measure is not a failed seam.
        gated = rr.mosaic_coverage(2, self._seams([(0, 1)]))
        plain = rr.mosaic_coverage(2, self._seams([]))
        assert gated.n_expected_seams == 0
        assert plain.n_expected_seams == 1


class TestItNeverFailsTheRun:
    def test_a_broken_graph_leaves_registration_untouched(self, caplog):
        pipe = _pipe()
        with caplog.at_level(logging.WARNING):
            pairs, dropped = pipe._empty_overlap_pairs(
                ["not an msim"], _content([_tile((0, 16))]), None,
                [(0.0, 0.0)], "metadata",
            )
        assert (pairs, dropped) == (None, [])

    def test_it_refuses_to_leave_no_seams_at_all(self, caplog):
        pipe = _pipe()
        # Three tiles in a row, all empty at every edge that matters.
        vols = [_tile(None), _tile(None), _tile(None)]
        content = _content(vols)
        for c in content:  # force them past the whole-tile gate
            c.has_content = True
        with caplog.at_level(logging.WARNING):
            pairs, dropped = pipe._empty_overlap_pairs(
                _msims(3, PITCH_UM), content, None,
                [(0.0, 0.0), (PITCH_UM, 0.0), (2 * PITCH_UM, 0.0)], "metadata",
            )
        assert pairs is None, "gated a mosaic down to nothing"
        assert "NO seams to register" in caplog.text


class TestTheBandWidth:
    def test_it_is_derived_from_the_measured_overlap(self):
        pipe = _pipe()
        tiles = [
            SimpleNamespace(x_mm=0.0, y_mm=0.0),
            SimpleNamespace(x_mm=0.085, y_mm=0.0),
        ]
        frac = pipe._seam_band_fraction(tiles, {"x": 100.0, "y": 100.0})
        assert 0.1 <= frac <= 0.4

    def test_it_falls_back_rather_than_raising(self):
        assert 0.1 <= _pipe()._seam_band_fraction([], {}) <= 0.4
