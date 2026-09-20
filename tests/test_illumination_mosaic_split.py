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


def _lines(span, frame, pitch, ncol, pure_frac=0.5):
    """X positions in the mosaic where the winning sheet changes."""
    out = []
    for c in range(ncol):
        off = c * pitch
        w = side_weight_profile(frame, pure_frac, span=span, offset=off)
        for i in np.flatnonzero(np.diff(w) != 0):
            out.append(off + int(i) + 1)
    return sorted(set(out))


def _span(ncol, frame=FRAME, pitch=PITCH):
    return pitch * (ncol - 1) + frame


# --------------------------------------------------------------------------
# The requirement: one line
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ncol", [2, 3, 4, 5, 6, 7, 12])
def test_a_hard_split_draws_exactly_one_line_in_the_whole_mosaic(ncol):
    span = _span(ncol)
    assert len(_lines(span, FRAME, PITCH, ncol)) == 1


def test_the_line_sits_at_the_middle_of_the_acquisition():
    for ncol in (3, 4, 7):
        span = _span(ncol)
        (line,) = _lines(span, FRAME, PITCH, ncol)
        assert abs(line - span // 2) <= 1


def test_an_odd_column_count_splits_a_tile_and_an_even_one_lands_on_a_seam():
    """Exactly what a single handover means geometrically."""
    # Odd: the centre column carries the handover inside its own frame.
    span = _span(5)
    (line,) = _lines(span, FRAME, PITCH, 5)
    centre_start, centre_end = 2 * PITCH, 2 * PITCH + FRAME
    assert centre_start < line < centre_end

    # Even: the handover falls where two columns meet, not mid-frame.
    span = _span(4)
    (line,) = _lines(span, FRAME, PITCH, 4)
    assert abs(line - span // 2) <= 1


def test_tiles_away_from_the_centre_are_pure_and_carry_no_seam():
    span = _span(7)
    for c, expect in ((0, 1.0), (1, 1.0), (5, 0.0), (6, 0.0)):
        w = side_weight_profile(FRAME, 0.5, span=span, offset=c * PITCH)
        assert np.all(w == expect), f"column {c} should be one sheet outright"


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
