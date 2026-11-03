"""Split/blend hand over ONCE across the acquisition, not once per tile.

The 2026-09-18 7x7 split every tile at its own frame midline, so the output
carried a seam down every tile column. Worse than cosmetic: a sheet entering
from the left has to cross all the tissue to the left of the current tile
before it arrives, so on the left edge of a 12 mm sample the left sheet is good
across the WHOLE frame and the right sheet has just crossed the entire sample.
Splitting that tile at its midline throws away the good half and substitutes
the one that travelled furthest.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from flamingo_stitcher.pipeline import (
    StitchingPipeline,
    fuse_illumination_sides,
    side_weight_profile,
)

FRAME, PITCH = 256, 200


def _span(ncol, frame=FRAME, pitch=PITCH):
    return pitch * (ncol - 1) + frame


# --------------------------------------------------------------------------
# The requirement: one handover, and it is a tile seam
# --------------------------------------------------------------------------


def _owners(ncol, frame=FRAME, pitch=PITCH):
    """Per column: 1.0 (low-end sheet), 0.0 (high-end sheet) or None (cut)."""
    span = _span(ncol, frame, pitch)
    out = []
    for c in range(ncol):
        w = side_weight_profile(frame, 0.5, span=span, offset=c * pitch)
        out.append(float(w[0]) if np.all(w == w[0]) else None)
    return out


@pytest.mark.parametrize("ncol", [2, 3, 4, 5, 6, 7, 12])
def test_a_hard_split_cuts_no_tile(ncol):
    """Each tile is one sheet whole.

    A cut inside a frame sits at the same stage-relative place in both tiles
    of the seam it falls in. Ket_Trial1 run 35 (2026-11-03): phase correlation
    locked onto it and measured exactly 0 µm on the dim handover seams, where
    Max had measured -11 to -14 µm on the same data.
    """
    assert None not in _owners(ncol)


@pytest.mark.parametrize("ncol", [2, 3, 4, 5, 6, 7, 12])
def test_there_is_exactly_one_handover_between_two_columns(ncol):
    owners = _owners(ncol)
    changes = [c for c in range(ncol - 1) if owners[c] != owners[c + 1]]
    assert len(changes) == 1
    assert owners[0] == 1.0 and owners[-1] == 0.0


def test_the_handover_follows_the_tile_centres():
    """The tile whose centre is past the middle of the acquisition takes the
    high-end sheet; a centre exactly on the middle (odd column count) stays
    with the low end."""
    for ncol in (2, 3, 4, 5, 6, 7):
        span = _span(ncol)
        owners = _owners(ncol)
        for c, owner in enumerate(owners):
            centre2 = 2 * c * PITCH + FRAME
            assert owner == (1.0 if centre2 <= span else 0.0), (ncol, c)


def test_the_middle_column_of_an_odd_mosaic_goes_whole_to_the_low_end():
    assert _owners(5) == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_tiles_away_from_the_centre_are_pure_and_carry_no_seam():
    span = _span(7)
    for c, expect in ((0, 1.0), (1, 1.0), (5, 0.0), (6, 0.0)):
        w = side_weight_profile(FRAME, 0.5, span=span, offset=c * PITCH)
        assert np.all(w == expect), f"column {c} should be one sheet outright"


def test_fusing_a_mosaic_tile_under_split_copies_one_side():
    lo = np.full((2, 4, FRAME), 100, np.uint16)
    hi = np.full((2, 4, FRAME), 900, np.uint16)
    span = _span(4)
    first = fuse_illumination_sides({0: hi, 1: lo}, "split", axis=-1, low_side=1,
                                    span=span, offset=0)
    last = fuse_illumination_sides({0: hi, 1: lo}, "split", axis=-1, low_side=1,
                                   span=span, offset=3 * PITCH)
    assert np.all(first == 100) and np.all(last == 900)


# --------------------------------------------------------------------------
# The single-FOV case must not move
# --------------------------------------------------------------------------


def test_a_single_field_of_view_is_byte_identical_to_the_old_behaviour():
    """The case already tested on the rig. A one-tile mosaic is the special
    case of the general profile, not a separate code path."""
    for frac in (0.0, 0.2, 0.35, 0.5):
        old_style = side_weight_profile(FRAME, frac)
        as_mosaic = side_weight_profile(FRAME, frac, span=FRAME, offset=0)
        np.testing.assert_array_equal(old_style, as_mosaic)


def test_fuse_without_a_span_still_splits_the_frame():
    lo = np.full((2, 4, FRAME), 100, np.uint16)
    hi = np.full((2, 4, FRAME), 900, np.uint16)
    out = fuse_illumination_sides({0: hi, 1: lo}, "split", axis=-1, low_side=1)
    assert out[..., 0] .max() == 100 and out[..., -1].min() == 900


# --------------------------------------------------------------------------
# Blend spreads its handover over the acquisition too
# --------------------------------------------------------------------------


def test_blend_ramps_across_the_mosaic_not_inside_each_tile():
    span = _span(7)
    ramping = []
    for c in range(7):
        w = side_weight_profile(FRAME, 0.35, span=span, offset=c * PITCH)
        if 0.0 < w.min() or w.max() < 1.0:
            if not (np.all(w == w[0])):
                ramping.append(c)
    # A 35% pure share each side leaves a band in the middle of the ACQUISITION,
    # so only the central columns ramp; the outer ones stay pure.
    assert ramping, "something must ramp"
    assert 0 not in ramping and 6 not in ramping


def test_each_tiles_window_is_monotone():
    span = _span(7)
    for c in range(7):
        w = side_weight_profile(FRAME, 0.35, span=span, offset=c * PITCH)
        assert np.all(np.diff(w) <= 1e-6), f"column {c} is not monotone"


def test_adjacent_tiles_agree_on_the_weight_where_they_overlap():
    """This is what removes the seam.

    Tiles overlap by FRAME-PITCH px. Splitting per tile gave each of them its
    own midline, so two tiles disagreed about which sheet owns the shared
    strip; one profile over the acquisition makes them agree exactly.
    """
    span = _span(7)
    ovl = FRAME - PITCH
    for c in range(6):
        a = side_weight_profile(FRAME, 0.35, span=span, offset=c * PITCH)
        b = side_weight_profile(FRAME, 0.35, span=span, offset=(c + 1) * PITCH)
        np.testing.assert_allclose(a[-ovl:], b[:ovl], atol=1e-6,
                                   err_msg=f"columns {c}/{c+1} disagree in their overlap")


def test_per_tile_splitting_is_what_used_to_disagree():
    """Guards the claim above: without the span the same two tiles conflict."""
    ovl = FRAME - PITCH
    a = side_weight_profile(FRAME, 0.35)
    b = side_weight_profile(FRAME, 0.35)
    assert not np.allclose(a[-ovl:], b[:ovl], atol=1e-6)


# --------------------------------------------------------------------------
# Placing a tile inside the acquisition
# --------------------------------------------------------------------------


def _pipeline(tiles, *, reverse_x=False, orientation="identity"):
    p = StitchingPipeline.__new__(StitchingPipeline)
    p.config = SimpleNamespace(
        illumination_fusion="split",
        illumination_low_side=1,
        illumination_pure_frac=0.35,
        pixel_size_um=1.0,
        tile_orientation=orientation,
        camera_x_inverted=False,
        reverse_x_tiles=reverse_x,
        reverse_y_tiles=False,
    )
    p.logger = mock.MagicMock()
    p._illum_tiles = tiles
    p._illum_span_logged = False
    return p


def _tiles(n, pitch_mm=0.2):
    return [SimpleNamespace(x_mm=1.0 + i * pitch_mm, y_mm=5.0) for i in range(n)]


def test_offsets_walk_the_mosaic_in_stage_order():
    tiles = _tiles(5)
    p = _pipeline(tiles)
    vol = np.zeros((1, 4, FRAME), np.uint16)
    offs = [p._illumination_mosaic_window(t, vol, -1)[1] for t in tiles]
    assert offs == sorted(offs) and offs[0] == 0
    assert offs == [0, 200, 400, 600, 800]


def test_reversed_tile_order_reverses_the_offsets():
    """`reverse_x_tiles` is how this package handles X handedness on n7 (the
    tile ORDER flips, the pixels do not). The handover has to follow it."""
    tiles = _tiles(5)
    fwd = [_pipeline(tiles)._illumination_mosaic_window(t, np.zeros((1, 4, FRAME)), -1)[1]
           for t in tiles]
    rev = [_pipeline(tiles, reverse_x=True)._illumination_mosaic_window(
        t, np.zeros((1, 4, FRAME)), -1)[1] for t in tiles]
    assert rev == fwd[::-1]


def test_a_single_tile_acquisition_asks_for_no_window():
    p = _pipeline(_tiles(1))
    assert p._illumination_mosaic_window(p._illum_tiles[0], np.zeros((1, 4, FRAME)), -1) == (None, 0)


def test_the_span_covers_every_tile():
    tiles = _tiles(6)
    p = _pipeline(tiles)
    vol = np.zeros((1, 4, FRAME), np.uint16)
    for t in tiles:
        span, off = p._illumination_mosaic_window(t, vol, -1)
        assert 0 <= off and off + FRAME <= span


def test_the_orientation_probe_finds_the_axis_and_its_direction():
    """Probed by orienting a ramp rather than read from a name table: a
    transposing orientation moves the illumination axis to the other output
    axis and a flipping one reverses it."""
    assert _pipeline([], orientation="identity")._orientation_maps_frame_axis(-1) == (-1, False)
    assert _pipeline([], orientation="flip_h")._orientation_maps_frame_axis(-1) == (-1, True)
