"""Border QC checks the seams where the tiles were PUT, not where the stage said.

2026-10-16, the 6x7 n7 run stitched twice on 0.13.9, registered and with
'Skip registration': the two QC blocks were identical, the same 34 seams with
the same shifts, 24 of them "AT SEARCH LIMIT". QC paired and ordered the tiles
by raw stage X, so on a reverse-X scope every X seam compared the two tiles'
far edges, and it never applied the registered shifts, so a registered run
reported the stage error that registration had already removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from flamingo_stitcher import border_qc

scipy_ndimage = pytest.importorskip("scipy.ndimage")

PX_UM = 1.0
Z, NY, TX = 6, 200, 40
PITCH = 30  # px; overlap = TX - PITCH = 10 px


class _Tile:
    def __init__(self, x_mm, y_mm, idx):
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.z_min_mm = 0.0
        self.z_max_mm = (Z - 1) * 0.005
        self.tile_index = idx


def _base(seed=3):
    """Blobby, high-contrast structure: wrong strips cannot look alike."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 1, size=(Z, NY + 20, TX + PITCH + 20))
    sm = scipy_ndimage.gaussian_filter(raw, sigma=(0, 5.0, 5.0))
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    return (1000.0 + 6000.0 * sm).astype(np.float32)


def _run(entries, **kw):
    tiles = [t for _v, t in entries]
    return border_qc.run_border_qc(
        {1: entries},
        tiles,
        pixel_size_um=PX_UM,
        ds_xy=1,
        ds_z=1,
        z_step_um=5.0,
        reg_channel=1,
        params=border_qc.BorderQCParams(mode="mip"),
        **kw,
    )


def _reverse_x_pair(base):
    """Image-left tile has the HIGHER stage X, as on n7 (world x = -stage x)."""
    left = base[:, :NY, 0:TX].copy()
    right = base[:, :NY, PITCH:PITCH + TX].copy()
    pitch_mm = PITCH * PX_UM / 1000.0
    return [
        (left, _Tile(5.0 + pitch_mm, 1.0, (1, 0))),
        (right, _Tile(5.0, 1.0, (0, 0))),
    ]


def test_a_reverse_x_mosaic_is_compared_on_the_edges_that_overlap():
    entries = _reverse_x_pair(_base())
    assert _run(entries, reverse_x=True).n_pairs_flagged == 0


def test_stage_order_on_a_reverse_x_mosaic_compares_the_wrong_edges():
    """The 0.13.9 behaviour, kept as the control: without the flip the same
    clean pair is flagged, which is what the rig's 34 seams were."""
    entries = _reverse_x_pair(_base())
    assert _run(entries, reverse_x=False).n_pairs_flagged >= 1


# QC re-aligns each seam itself by up to +/-8 px before looking for a step, so
# only an error past that window survives as a flagged seam.
ERR = 14


def _stage_error_pair(base, err_px):
    """Stage says the tiles line up in Y; B's content sits err_px further on."""
    a = base[:, 0:NY, 0:TX].copy()
    b = base[:, err_px:err_px + NY, PITCH:PITCH + TX].copy()
    pitch_mm = PITCH * PX_UM / 1000.0
    ta, tb = _Tile(5.0, 1.0, (0, 0)), _Tile(5.0 + pitch_mm, 1.0, (1, 0))
    return [(a, ta), (b, tb)], ta, tb


def test_the_placed_geometry_is_the_registered_one():
    """Across-seam overlap and along-seam offset come from stage + registration."""
    _entries, ta, tb = _stage_error_pair(_base(), ERR)
    vol = np.zeros((Z, NY, TX))
    pa = border_qc._Placed(ta, 1.0, 1.0, None)
    stage = border_qc._Placed(tb, 1.0, 1.0, None)
    moved = border_qc._Placed(tb, 1.0, 1.0, (0.0, ERR * PX_UM, 3 * PX_UM))
    assert border_qc._placed_overlap_px(pa, stage, vol, "x", PX_UM) == (TX - PITCH, 0)
    # B sits ERR further on in Y, so A's row ERR is B's row 0; 3 px further
    # right in X takes 3 px off the overlap.
    assert border_qc._placed_overlap_px(pa, moved, vol, "x", PX_UM) == (
        TX - PITCH - 3, -ERR,
    )


def test_the_placement_offset_is_applied_before_the_seam_is_compared():
    entries, _ta, _tb = _stage_error_pair(_base(), ERR)
    (a, _), (b, _) = entries
    params = border_qc.BorderQCParams(mode="mip", refine_shift=False)
    lined_up = border_qc.detect_border_steps(
        a, b, "x", TX - PITCH, params=params, geom_shift=(0, -ERR)
    )
    assert not lined_up.flagged
    assert lined_up.median_step_counts == 0.0


def test_registered_placement_is_used_when_offsets_are_given():
    entries, ta, tb = _stage_error_pair(_base(), ERR)
    offsets = {
        border_qc.position_key(ta): (0.0, 0.0, 0.0),
        border_qc.position_key(tb): (0.0, ERR * PX_UM, 0.0),
    }
    report = _run(entries, offsets_um=offsets)
    assert report.settings["placement"] == "registered"
    assert report.n_pairs_flagged == 0


def test_seams_are_named_by_grid_cell():
    _entries, ta, tb = _stage_error_pair(_base(), ERR)
    assert (border_qc._tile_name(ta), border_qc._tile_name(tb)) == (
        "X000 Y000", "X001 Y000",
    )


def test_the_report_says_which_placement_it_checked():
    entries, ta, tb = _stage_error_pair(_base(), ERR)
    text = border_qc.format_report_text(
        _run(entries, offsets_um={border_qc.position_key(tb): (0.0, float(ERR), 0.0)})
    )
    assert "Placement checked: registered" in text
