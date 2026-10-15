"""A drifting mosaic must register as one piece, not be cut up by its own guards.

The 2026-10-14 stitch of a 6x7 acquisition (2026-10-06, n7, 8x downsample) came
out with horizontal steps between rows and whole columns displaced by 200+ µm,
on BOTH illumination sides. The seams themselves were mostly fine. Two guards
turned measurements into damage:

  * multiview-stitcher's `abs_tol` is a PHYSICAL residual and we passed 3.5
    meaning pixels. At 8.38 µm/px that is 0.42 px, so the solver pruned every
    loop of seams and kept a spanning tree: 19 measured seams discarded, rows
    hung off one or two row-to-row seams each.
  * the post-solve clamp judged each tile against the MOSAIC median. The
    corrections drift 30-180 µm across the grid, so the far corner read as an
    outlier while every adjacent pair agreed within a seam, and three tiles were
    sent back from beside neighbours they had registered to at quality 0.95.

These tests pin both, plus the Z step that silently overrode the data.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("multiview_stitcher")
from multiview_stitcher import param_utils  # noqa: E402

from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline  # noqa: E402


def _params(translations):
    return [
        param_utils.affine_from_translation(np.asarray(t, float)) for t in translations
    ]


def _trans(param):
    arr = np.asarray(param)
    return (arr[0] if arr.ndim == 3 else arr)[:3, 3]


# --------------------------------------------------------------------------
# The 2026-10-14 run, as logged
# --------------------------------------------------------------------------

# Registration voxel and spill shape from the log: 8x XY, Z labelled 4 µm.
_VOX_1014 = {"z": 4.0, "y": 8.38, "x": 8.38}
_SHAPE_1014 = (750, 256, 256)

# (label, stage X mm, stage Y mm, dz, dy, dx) -- the PRE-clamp placement.
# Rows are taken from the log's TILE PLACEMENT table; the three tiles the clamp
# reverted are rebuilt from the registered seams they sat on, and the rebuild
# reproduces the log's own "max shift seen z=64.0 ... x=180.2 µm".
_PLACED_1014 = [
    ("X004 Y006", 3.665, 13.415, 12.0, -125.70, 104.75),
    ("X003 Y006", 5.596, 13.415, 16.0, -92.18, 100.56),
    ("X002 Y006", 7.527, 13.415, 12.0, -92.18, 79.61),
    ("X004 Y005", 3.665, 15.346, 16.0, -92.18, 100.56),
    ("X003 Y005", 5.596, 15.346, 12.0, -92.18, 96.37),
    ("X002 Y005", 7.527, 15.346, 8.0, -92.18, 96.37),
    ("X001 Y005", 9.458, 15.346, 0.0, -96.37, 100.56),
    ("X005 Y004", 1.735, 17.276, 20.0, -37.71, 75.42),
    ("X004 Y004", 3.665, 17.276, 16.0, -37.71, 75.42),
    ("X003 Y004", 5.596, 17.276, 12.0, -37.71, 75.42),
    ("X002 Y004", 7.527, 17.276, 8.0, -37.71, 75.42),
    ("X001 Y004", 9.458, 17.276, 4.0, -29.33, 71.23),
    ("X000 Y004", 11.389, 17.276, -8.0, -16.76, 62.85),
    ("X005 Y003", 1.735, 19.207, 24.0, -54.47, 58.66),
    ("X004 Y003", 3.665, 19.207, 16.0, -25.14, 58.66),
    ("X003 Y003", 5.596, 19.207, 12.0, -25.14, 54.47),
    ("X002 Y003", 7.527, 19.207, 4.0, -25.14, 50.28),
    ("X001 Y003", 9.458, 19.207, -4.0, -25.14, 41.90),
    ("X000 Y003", 11.389, 19.207, -16.0, -12.57, 33.52),
    ("X005 Y002", 1.735, 21.138, 16.0, -12.57, 20.95),
    ("X004 Y002", 3.665, 21.138, 8.0, -4.19, 8.38),
    ("X003 Y002", 5.596, 21.138, 0.0, 0.0, 0.0),
    ("X002 Y002", 7.527, 21.138, -12.0, 4.19, -16.76),
    ("X001 Y002", 9.458, 21.138, -24.0, 4.19, -37.71),
    ("X000 Y002", 11.389, 21.138, -40.0, 8.38, -62.85),     # clamped Z in the run
    ("X005 Y001", 1.735, 23.069, -8.0, 41.90, -83.80),
    ("X004 Y001", 3.665, 23.069, -16.0, 50.28, -104.75),
    ("X003 Y001", 5.596, 23.069, -24.0, 54.47, -121.51),
    ("X002 Y001", 7.527, 23.069, -36.0, 58.66, -142.46),
    ("X001 Y001", 9.458, 23.069, -48.0, 58.66, -163.41),    # clamped ZYX in the run
    ("X000 Y001", 11.389, 23.069, -64.0, 67.04, -180.17),   # clamped ZYX in the run
]


def _tile_data(rows, shape):
    vol = np.zeros(shape, np.uint16)  # only .shape is read
    return [(vol, SimpleNamespace(x_mm=x, y_mm=y, z_min_mm=0.0)) for _l, x, y, *_ in rows]


def test_the_2026_10_14_drift_is_not_clamped():
    """The run's three clamps, reproduced and removed.

    Under the mosaic-median rule X001 Y001 and X000 Y001 sat 218 and 235 µm from
    the median laterally (bound 214 µm) and 48/64 µm axially, and were reverted
    from beside a neighbour each was registered to at quality 0.95 -- leaving
    233 and 239 µm seams in the output. Against their neighbours they are off by
    one seam's worth: 21 µm.
    """
    pipe = StitchingPipeline(StitchingConfig())
    td = _tile_data(_PLACED_1014, _SHAPE_1014)
    params = _params([(dz, dy, dx) for _l, _x, _y, dz, dy, dx in _PLACED_1014])
    result = pipe._clamp_registration_shifts(params, td, _VOX_1014)

    clamped = [
        _PLACED_1014[r.index][0] for r in result.records if r.clamped_xy or r.clamped_z
    ]
    assert clamped == [], f"a drifting grid was cut up: {clamped}"
    for (label, *_rest, dz, dy, dx), out in zip(_PLACED_1014, result.params):
        assert _trans(out) == pytest.approx([dz, dy, dx], abs=1e-6), label


def _grid(nx, ny, pitch_mm=1.931):
    return [(f"X{i} Y{j}", i * pitch_mm, j * pitch_mm) for j in range(ny) for i in range(nx)]


def test_a_steady_drift_far_beyond_one_overlap_is_left_alone():
    """Each step drifts 40 µm laterally and 20 µm in Z: 200 µm and 100 µm
    across a 6x7 grid, well past the 214 µm overlap and the Z bound from the
    far corner to the median. No seam opens, so nothing may be clamped."""
    cells = _grid(6, 7)
    rows = [
        (lab, x, y, 20.0 * (i % 6), 40.0 * (i // 6), 40.0 * (i % 6))
        for i, (lab, x, y) in enumerate(cells)
    ]
    pipe = StitchingPipeline(StitchingConfig())
    result = pipe._clamp_registration_shifts(
        _params([r[3:] for r in rows]), _tile_data(rows, _SHAPE_1014), _VOX_1014
    )
    assert not any(r.clamped_xy or r.clamped_z for r in result.records)


def test_a_flung_tile_inside_a_drift_is_still_caught_and_rejoins_its_neighbours():
    """The clamp's real job survives: one tile 600 µm out of line with every
    neighbour, in a grid that is itself drifting. It must revert to where its
    NEIGHBOURS are, not to the mosaic median -- in a drift those differ."""
    cells = _grid(6, 7)
    shifts = [[0.0, 40.0 * (i // 6), 40.0 * (i % 6)] for i in range(len(cells))]
    victim = 2 * 6 + 3                      # an interior tile
    expected = list(shifts[victim])
    shifts[victim][2] += 600.0
    rows = [(lab, x, y, *s) for (lab, x, y), s in zip(cells, shifts)]

    pipe = StitchingPipeline(StitchingConfig())
    result = pipe._clamp_registration_shifts(
        _params(shifts), _tile_data(rows, _SHAPE_1014), _VOX_1014
    )
    flagged = [r.index for r in result.records if r.clamped_xy]
    assert flagged == [victim]
    placed = _trans(result.params[victim])
    # Neighbours' median: the tiles left/right/above/below on a linear ramp.
    assert placed[1] == pytest.approx(expected[1], abs=1e-6)
    assert placed[2] == pytest.approx(expected[2], abs=40.0 + 1e-6)


# --------------------------------------------------------------------------
# The solver tolerance
# --------------------------------------------------------------------------


def _pipe_with_tol(px=3.5):
    cfg = StitchingConfig()
    cfg.global_opt_abs_tol = px
    return StitchingPipeline(cfg)


def test_the_tolerance_is_converted_from_pixels_to_micrometres():
    tol = _pipe_with_tol()._global_opt_abs_tol_um(_VOX_1014, {"z": 2, "y": 1, "x": 1})
    assert tol == pytest.approx(3.5 * 8.38)


def test_registration_binning_scales_the_pixel():
    tol = _pipe_with_tol()._global_opt_abs_tol_um(
        {"z": 2.5, "y": 2.095, "x": 2.095}, {"z": 4, "y": 2, "x": 2}
    )
    # lateral px = 4.19 µm -> 14.67 µm; voxel diagonal sqrt(10^2+4.19^2*2) = 11.6
    assert tol == pytest.approx(3.5 * 4.19)


def test_it_never_falls_below_one_voxel_diagonal():
    """With Z far coarser than XY, a one-plane Z difference alone would exceed
    3.5 lateral pixels and prune a good seam. The solver's own default floor."""
    tol = _pipe_with_tol()._global_opt_abs_tol_um(
        {"z": 10.0, "y": 0.5, "x": 0.5}, {"z": 1, "y": 1, "x": 1}
    )
    assert tol == pytest.approx((10.0**2 + 0.5**2 * 2) ** 0.5)


def test_without_a_voxel_size_it_keeps_the_old_number():
    assert _pipe_with_tol()._global_opt_abs_tol_um({}, {}) == pytest.approx(3.5)


def _drifting_seam_graph(nx_, ny_, *, noise_px, outliers, seed=0):
    """A grid of tiles with known positions and noisy pairwise measurements,
    in the form multiview-stitcher's global optimisation consumes."""
    import networkx as nx
    import xarray as xr

    rng = np.random.default_rng(seed)
    px = 8.38
    truth = {}
    for j in range(ny_):
        for i in range(nx_):
            truth[(i, j)] = np.array([6.0 * i - 4.0 * j, 25.0 * j, 18.0 * i + 9.0 * j])
    index = {k: n for n, k in enumerate(sorted(truth))}
    g = nx.Graph()
    g.add_nodes_from(index.values())
    edges = []
    for (i, j) in truth:
        for (di, dj) in ((1, 0), (0, 1)):
            b = (i + di, j + dj)
            if b in truth:
                edges.append(((i, j), b))
    bad = set(range(0, len(edges), max(1, len(edges) // max(1, outliers))))
    bad = set(list(bad)[:outliers])
    for k, (a, b) in enumerate(edges):
        shift = truth[a] - truth[b] + rng.normal(0.0, noise_px * px, 3)
        if k in bad:
            shift[1] += 10 * px  # a wrong correlation peak, 10 px out
        affine = param_utils.affine_from_translation(shift)
        lower = np.zeros(3)
        upper = np.array([100.0, 214.0, 2145.0])
        g.add_edge(
            index[a], index[b],
            transform=affine,
            quality=0.9,
            overlap=1.0,
            bbox=xr.DataArray(np.stack([lower, upper])),
        )
    return g, truth, index, len(edges)


def _seam_error_vs_truth(params, g, truth, index):
    """Median and max placement error over grid seams, gauge-free."""
    inv = {v: k for k, v in index.items()}

    def t(n):
        arr = np.asarray(params[n])
        return (arr.reshape(-1, 4, 4)[0])[:3, 3]

    def errs(sign):
        return [
            np.linalg.norm(sign * (t(a) - t(b)) - (truth[inv[a]] - truth[inv[b]]))
            for a, b in g.edges
        ]

    e = min(errs(+1), errs(-1), key=np.median)
    return float(np.median(e)), float(np.max(e))


@pytest.mark.parametrize("seed", [0, 1])
def test_multiview_stitcher_reads_abs_tol_in_micrometres(seed):
    """The units claim, checked against the solver itself rather than our
    reading of it. A 6x7 grid (42 tiles, 71 seams), one pixel of honest noise,
    three planted wrong peaks. Measured on multiview-stitcher 0.1.59:

        abs_tol 3.5 (old pass-through)   keeps 45-48  median error 15-17 µm
        abs_tol 29.3 (3.5 px converted)  keeps 69     median error  9-12 µm

    The old value prunes to within a few seams of a spanning tree (41) -- the
    2026-10-14 run kept one more than a tree -- and the placement is worse for
    it. The converted value does NOT prune every planted outlier: one is partly
    absorbed (max error ~60 µm). That is the limit of prune-the-worst-edge, not
    of the tolerance, and is pinned here so a future change can see it.
    """
    from multiview_stitcher.param_resolution.global_optimization import (
        groupwise_resolution_global_optimization as solve,
    )

    g, truth, index, n_edges = _drifting_seam_graph(
        6, 7, noise_px=1.0, outliers=3, seed=seed
    )
    n_tiles = len(index)

    p_old, info_old = solve(g.copy(), transform="translation", abs_tol=3.5)
    tol = _pipe_with_tol()._global_opt_abs_tol_um(_VOX_1014, {"z": 2, "y": 1, "x": 1})
    p_new, info_new = solve(g.copy(), transform="translation", abs_tol=tol)

    kept_old, kept_new = len(info_old["used_edges"]), len(info_new["used_edges"])
    assert kept_old <= (n_tiles - 1) + 8, f"3.5 µm kept {kept_old}: not near-tree"
    assert kept_new >= n_edges - 6, f"{tol:.1f} µm kept only {kept_new} of {n_edges}"

    med_old, _ = _seam_error_vs_truth(p_old, g, truth, index)
    med_new, _ = _seam_error_vs_truth(p_new, g, truth, index)
    assert med_new < med_old, (
        f"converting the tolerance must improve placement: median seam error "
        f"{med_new:.1f} µm vs {med_old:.1f} µm before"
    )


# --------------------------------------------------------------------------
# The Z step
# --------------------------------------------------------------------------


def _z_check(configured, data_um, caplog):
    pipe = StitchingPipeline(StitchingConfig())
    tile = SimpleNamespace(z_step_mm=data_um / 1000.0)
    with caplog.at_level(logging.INFO):
        pipe._check_configured_z_step(configured, [tile])
    return caplog.records


def test_a_configured_z_step_that_disagrees_with_the_data_is_shouted(caplog):
    """2026-10-14: 1.000 µm configured, data 2.500 µm, and the log said nothing
    -- 'Z step from data' only prints when nothing is configured."""
    records = _z_check(1.0, 2.5, caplog)
    warnings = [r for r in records if r.levelno >= logging.WARNING]
    assert warnings, "a 2.5x Z scale error must not be silent"
    text = warnings[0].getMessage()
    assert "1.000" in text and "2.500" in text and "squashed" in text


def test_an_agreeing_configured_z_step_is_noted_quietly(caplog):
    records = _z_check(2.5, 2.5, caplog)
    assert not [r for r in records if r.levelno >= logging.WARNING]
    assert any("agrees with the data" in r.getMessage() for r in records)
