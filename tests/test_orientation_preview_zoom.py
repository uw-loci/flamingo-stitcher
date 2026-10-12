"""The orientation preview must be readable.

Picking the right orientation means reading the tile labels burned into each
panel to see which arrangement has neighbours CONNECTED. The grid renders eight
panels at _CELL_PX (260), while the arrays behind them are built at
`target_long_px` (1000) — so the thumbnail throws away ~4x the detail the
decision needs, and there was no way to get it back. The MIP overview has this
for free because it lives in napari.

PyQt5 is installed neither here nor in the release workflow (which only builds
the installer), so the dialog's behaviour cannot be exercised in automation.
The wiring is checked statically — the pattern the other GUI tests here use —
and the resolution claim the whole fix rests on is measured for real.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src/flamingo_stitcher"
DIALOG = SRC / "gui/orientation_preview_dialog.py"


def _tree():
    return ast.parse(DIALOG.read_text(encoding="utf-8"))


def _node(name):
    for n in ast.walk(_tree()):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            return n
    return None


# --------------------------------------------------------------------------
# The premise: there is real detail being discarded
# --------------------------------------------------------------------------


def test_the_thumbnail_discards_several_times_the_detail():
    """If the arrays were already thumbnail-sized, zooming would show nothing
    and this whole dialog would be theatre."""
    import re

    src = DIALOG.read_text(encoding="utf-8")
    cell = int(re.search(r"^_CELL_PX = (\d+)", src, re.M).group(1))

    ori = (SRC / "orientation.py").read_text(encoding="utf-8")
    target = int(re.search(r"target_long_px: int = (\d+)", ori).group(1))

    assert target / cell > 2.0, (
        f"panels render at {target}px and display at {cell}px — only "
        f"{target / cell:.1f}x, so a zoom view would add little"
    )


def test_the_labels_are_burned_in_by_default():
    """The text being zoomed into has to exist. If `label_tiles` ever defaults
    off, the zoom shows a bigger unlabelled picture."""
    ori = (SRC / "orientation.py").read_text(encoding="utf-8")
    assert "label_tiles: bool = True" in ori


# --------------------------------------------------------------------------
# The wiring
# --------------------------------------------------------------------------


def test_the_zoom_dialog_exists():
    assert _node("OrientationZoomDialog") is not None


def test_panels_are_clickable_and_open_the_zoom():
    body = ast.dump(_node("OrientationPreviewDialog") or _tree())
    assert "_ClickableLabel" in body, "the preview cells must be clickable"
    assert "_open_zoom" in body


def test_clicking_a_cell_passes_that_cells_own_array():
    """A late-binding lambda in a loop is the classic way every cell ends up
    opening the LAST panel."""
    src = DIALOG.read_text(encoding="utf-8")
    assert "lambda k=key, a=arr:" in src, (
        "bind key/arr as defaults, or every cell opens the same orientation"
    )


def test_choosing_in_the_zoom_selects_it_in_the_grid():
    """Otherwise you zoom in, decide, close, and have to find it again."""
    body = ast.dump(_node("_open_zoom"))
    assert "setChecked" in body
    assert "_selected_name" in body
    assert "_apply_btn" in body


def test_closing_the_zoom_without_choosing_changes_nothing():
    body = ast.dump(_node("_open_zoom"))
    assert "chosen" in body, "must distinguish 'Use this' from 'Close'"


def test_an_empty_panel_does_not_open_a_blank_window():
    body = ast.dump(_node("_open_zoom"))
    assert "size" in body


def test_zooming_past_one_to_one_does_not_smooth_the_text():
    """Smoothing an enlarged burned-in label blurs the very text the dialog
    exists to make readable."""
    body = ast.dump(_node("_render"))
    assert "FastTransformation" in body and "SmoothTransformation" in body


def test_ctrl_wheel_zooms():
    fn = _node("wheelEvent")
    assert fn is not None
    assert "ControlModifier" in ast.dump(fn)


def test_the_zoom_range_spans_both_sides_of_one_to_one():
    import re

    src = DIALOG.read_text(encoding="utf-8")
    steps = eval(re.search(r"_STEPS = (\([^)]*\))", src).group(1))
    assert min(steps) < 1.0 < max(steps)
    assert 1.0 in steps, "Reset needs a 1:1 step to return to"
    assert max(steps) >= 3.0


# --------------------------------------------------------------------------
# Labels: outside the image, and present at all
# --------------------------------------------------------------------------

import numpy as np  # noqa: E402

from flamingo_stitcher.orientation import (  # noqa: E402
    _add_axis_gutters,
    _derive_tile_indices,
    _tile_label,
)


class _Tile:
    def __init__(self, x, y, index=None):
        self.x_mm, self.y_mm, self.tile_index = x, y, index


def test_a_folder_layout_acquisition_gets_labels_at_all():
    """`tile_index` exists only for the flat FILENAME layout. The microscope
    writes the FOLDER layout, so `_tile_label` returned "" and the rig's own
    previews carried no labels — the text the operator was trying to read was
    never drawn."""
    t = _Tile(1.33, 14.04)
    assert _tile_label(t) == "", "unchanged when nothing is derived"
    assert _tile_label(t, (0, 0)) == "X0Y0"


def test_indices_are_derived_from_the_stage_positions():
    tiles = [_Tile(1.0 + c * 1.9, 5.0 + r * 1.9) for r in range(3) for c in range(4)]
    idx = _derive_tile_indices(tiles)
    assert {idx[id(t)] for t in tiles} == {(c, r) for r in range(3) for c in range(4)}


def test_positions_are_clustered_not_rounded():
    """A coordinate on a rounding boundary would split one row in two and
    misnumber every row below it."""
    tiles = [_Tile(1.0, 5.0), _Tile(1.0 + 2e-6, 5.0), _Tile(2.9, 5.0)]
    idx = _derive_tile_indices(tiles)
    assert idx[id(tiles[0])] == idx[id(tiles[1])] == (0, 0)
    assert idx[id(tiles[2])] == (1, 0)


def test_an_explicit_tile_index_still_wins():
    assert _tile_label(_Tile(1.0, 5.0, index=(3, 4)), (9, 9)) == "X3Y4"


def test_the_gutter_does_not_cover_the_image():
    """The constraint: the preview is what is being judged, so labels big
    enough to read must not sit on top of it."""
    canvas = np.full((900, 900), 0.8, dtype=np.float32)
    spots = [
        (f"X{c}Y{r}", 50 + r * 300, 50 + c * 300) for r in range(3) for c in range(3)
    ]
    out = _add_axis_gutters(canvas, spots)
    assert out.shape[0] > canvas.shape[0] and out.shape[1] > canvas.shape[1]
    margin = out.shape[0] - canvas.shape[0]
    # Every original pixel survives untouched.
    np.testing.assert_array_equal(out[margin:, margin:], canvas)


def test_the_gutter_actually_carries_the_labels():
    canvas = np.full((900, 900), 0.8, dtype=np.float32)
    spots = [
        (f"X{c}Y{r}", 50 + r * 300, 50 + c * 300) for r in range(3) for c in range(3)
    ]
    out = _add_axis_gutters(canvas, spots)
    margin = out.shape[0] - canvas.shape[0]
    assert (out[:margin, margin:] > 0.9).any(), "no glyphs in the top gutter"
    assert (out[margin:, :margin] > 0.9).any(), "no glyphs in the left gutter"


def test_the_gutter_labels_survive_the_thumbnail_downscale():
    """They exist to be read at _CELL_PX, not at full size."""
    import re

    src = DIALOG.read_text(encoding="utf-8")
    cell = int(re.search(r"^_CELL_PX = (\d+)", src, re.M).group(1))
    canvas = np.full((1000, 1000), 0.8, dtype=np.float32)
    spots = [(f"X{c}Y{r}", 70 + r * 300, 70 + c * 300) for r in range(3) for c in range(3)]
    out = _add_axis_gutters(canvas, spots)
    margin = out.shape[0] - canvas.shape[0]
    glyph_px = 5 * max(2, margin // 7)          # bitmap font is 5 rows tall
    shown = glyph_px * cell / out.shape[0]
    assert shown >= 8, f"labels would be {shown:.1f}px tall in the preview grid"


def test_an_unlabelled_mosaic_is_left_alone():
    canvas = np.full((100, 100), 0.5, dtype=np.float32)
    np.testing.assert_array_equal(_add_axis_gutters(canvas, [("", 10, 10)]), canvas)


def test_a_two_digit_index_still_fits_the_gutter():
    """A ten-row mosaic draws "Y10" — one character longer than the scale was
    sized for. Fixed fractions of the margin break here; solving for the text
    does not."""
    from flamingo_stitcher.orientation import _gutter_scale, _label_box_px

    for margin in (28, 60, 111, 250):
        for text in ("X0", "Y9", "X10", "Y123"):
            assert _label_box_px(text, _gutter_scale(text, margin)) <= margin or (
                _gutter_scale(text, margin) == 2
            ), f"{text!r} overflows a {margin}px gutter"


def test_a_ten_row_mosaic_leaves_the_image_untouched():
    canvas = np.full((1000, 1000), 0.8, dtype=np.float32)
    spots = [
        (f"X{c}Y{r}", 50 + r * 95, 50 + c * 95) for r in range(10) for c in range(10)
    ]
    out = _add_axis_gutters(canvas, spots)
    margin = out.shape[0] - canvas.shape[0]
    np.testing.assert_array_equal(out[margin:, margin:], canvas)
