"""Tile-spacing gap detection (_detect_tile_spacing_gaps).

Tiles are drawn at their acquired stage positions, sized ``frame_dim ×
pixel_size``. If the acquisition stepped an axis farther apart than one frame
covers, the mosaic has real blank gaps (missing data) — most often a non-square
AOI whose tile step was computed from a square, width-only field of view (the
"full-width tiles with black bands between rows" failure). The detector must
flag the gapped axis, name the square-FOV cause, and stay silent when tiles
overlap normally.
"""

from types import SimpleNamespace

from flamingo_stitcher.pipeline import _detect_tile_spacing_gaps


def _tiles(coords):
    """Duck-typed tiles carrying only the x_mm/y_mm the detector reads."""
    return [SimpleNamespace(x_mm=x, y_mm=y) for x, y in coords]


# A 2048×1024 (non-square) AOI at ~1.0475 µm/px:
#   X coverage = 2048 * 1.0475 / 1000 = 2.145 mm
#   Y coverage = 1024 * 1.0475 / 1000 = 1.073 mm
_PX = 1.0475
_FW, _FH = 2048, 1024


def test_no_warning_when_tiles_overlap_on_both_axes():
    # 20% overlap: step = 0.8 * coverage on each axis -> tiles touch/overlap.
    xs = [0.0, 0.8 * 2.145, 1.6 * 2.145]
    ys = [0.0, 0.8 * 1.073, 1.6 * 1.073]
    coords = [(x, y) for y in ys for x in xs]
    assert _detect_tile_spacing_gaps(_tiles(coords), _FW, _FH, _PX) == []


def test_flags_y_gap_when_short_axis_stepped_as_square():
    # The reported failure: Y stepped for a 2048-tall frame (0.8 * X coverage)
    # while the real AOI is 1024 tall -> gaps between rows. X stepped correctly.
    x_step = 0.8 * 2.145  # correct X step (overlap)
    y_step = 0.8 * 2.145  # WRONG: Y stepped as if the frame were square (2048)
    xs = [0.0, x_step, 2 * x_step]
    ys = [0.0, y_step, 2 * y_step, 3 * y_step]
    coords = [(x, y) for y in ys for x in xs]

    msgs = _detect_tile_spacing_gaps(_tiles(coords), _FW, _FH, _PX)

    assert len(msgs) == 1, msgs
    m = msgs[0]
    assert m.startswith("Y tiles do not overlap")
    assert "blank gaps" in m
    assert "missing acquired data" in m
    # Names the square-FOV cause and the non-square AOI dimensions.
    assert "as if the frame were square" in m
    assert "2048×1024" in m


def test_square_aoi_gap_reported_without_square_hint():
    # A genuine gap on a SQUARE frame (e.g. under-tiled region) is still flagged,
    # but must NOT claim a square-FOV cause (there is no short/long axis).
    cov = 2048 * _PX / 1000.0  # 2.145 mm
    step = cov * 1.5  # clear gap
    ys = [0.0, step, 2 * step]
    coords = [(0.0, y) for y in ys]

    msgs = _detect_tile_spacing_gaps(_tiles(coords), 2048, 2048, _PX)

    assert len(msgs) == 1
    assert "Y tiles do not overlap" in msgs[0]
    assert "as if the frame were square" not in msgs[0]


def test_single_tile_has_no_step_to_compare():
    # One tile: no consecutive step on either axis -> nothing to warn about.
    assert _detect_tile_spacing_gaps(_tiles([(0.0, 0.0)]), _FW, _FH, _PX) == []


def test_axis_with_one_row_is_ignored_but_the_other_still_checked():
    # A single Y row (Y constant) but X columns spaced with a real gap: the
    # constant Y axis yields no step, while the gapped X axis is still flagged.
    x_step = 2048 * _PX / 1000.0 * 1.5  # clear X gap
    coords = [(0.0, 0.0), (x_step, 0.0), (2 * x_step, 0.0)]
    msgs = _detect_tile_spacing_gaps(_tiles(coords), _FW, _FH, _PX)
    assert len(msgs) == 1
    assert msgs[0].startswith("X tiles do not overlap")


def test_guards_bad_inputs():
    assert _detect_tile_spacing_gaps([], _FW, _FH, _PX) == []
    assert _detect_tile_spacing_gaps(_tiles([(0.0, 0.0)]), _FW, _FH, 0.0) == []
