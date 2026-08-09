"""Tile-border artifact QC — detect sharp intensity steps along tile seams.

A stitching seam artifact shows up as a sharp ~1-pixel step in signal right
where two neighboring tiles meet (a flat-field/exposure mismatch, or a fusion
blend that never quite cancels). This module measures that, per neighbor pair,
and writes a plain-text report.

Design notes:
  * The detector (:func:`detect_border_steps`) is pure — numpy in, dataclass
    out — so it is unit-testable in isolation with tiny synthetic strips. It
    only slices the thin near-border slab of each tile (works for numpy or a
    dask/memmap-backed array: it slices first, then ``np.asarray`` the slab).
  * It distinguishes a genuine seam step from three confounders by comparing
    the cross-seam jump to the intra-tile gradient at the same place:
      - a smooth intensity gradient continuous across the seam → the two tiles
        agree in their overlap → no step;
      - a real biological edge crossing the seam → large intra-tile gradient →
        the ratio stays low → not flagged;
      - noise → killed by an absolute noise floor + median smoothing.
  * Modes: ``mip`` (default; Z max-project, report border *length*), ``full``
    (per-Z, report *area* + Z-range), ``pairs`` (offending pairs only).

Only numpy and scipy.ndimage are used (scipy is optional — the detector still
runs without it, just without connected-component speckle filtering / phase
correlation refinement).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

try:  # scipy is a hard dep of the package, but keep the detector importable.
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

if TYPE_CHECKING:  # structural: any object with these attrs works (see tests).
    from flamingo_stitcher.pipeline import RawTileInfo


# Array-dimension index of each seam-normal axis in a (Z, Y, X) volume.
_AXIS_TO_DIM = {"z": 0, "y": 1, "x": 2}
_VALID_MODES = ("mip", "full", "pairs")


@dataclass
class BorderQCParams:
    """Tunable knobs for the border-step detector."""

    mode: str = "mip"  # "mip" | "full" | "pairs"
    alpha: float = 4.0  # step must exceed alpha * local intra-tile gradient
    beta: float = 3.0  # ...and beta * per-pair noise sigma
    k_frac: float = 0.125  # fraction of overlap width used as the seam window
    grad_window: int = 4  # interior px used to estimate the local gradient
    min_component_px: int = 10  # drop flagged blobs smaller than this
    bg_percentile: float = 5.0  # background threshold percentile of the overlap
    bg_abs_floor: float = 0.0  # absolute background floor (counts)
    z_stride: int = 1  # subsample Z in full mode when huge
    refine_shift: bool = True  # integer phase-correlation alignment
    max_refine_shift_px: int = 8
    include_z_seams: bool = False  # also check Z-tiled mosaic seams


@dataclass
class BorderStepResult:
    """Outcome of comparing one neighbor pair across their shared seam."""

    flagged: bool
    axis: str  # "x" | "y" | "z"
    overlap_px: int
    area_px: int = 0
    border_length_px: Optional[int] = None  # set in mip mode
    z_index_range: Optional[Tuple[int, int]] = None  # array indices, full mode
    median_step_counts: float = 0.0
    max_step_counts: float = 0.0
    n_samples_evaluated: int = 0
    largest_component_px: int = 0
    used_shift: Tuple[int, int] = (0, 0)
    # True when used_shift hit the search limit: the real misalignment is at
    # least this large and was NOT measured.
    shift_clamped: bool = False
    note: str = ""


@dataclass
class PairReport:
    """A single flagged/checked neighbor pair for the human-readable report."""

    tile_a_name: str
    tile_b_name: str
    axis: str
    result: BorderStepResult
    area_um2: Optional[float] = None
    z_mm_range: Optional[Tuple[float, float]] = None
    z_plane_range: Optional[Tuple[int, int]] = None
    border_length_um: Optional[float] = None
    channel_id: Optional[int] = None


@dataclass
class BorderQCReport:
    params: BorderQCParams
    channel_id: Optional[int]
    pairs: List[PairReport] = field(default_factory=list)
    n_pairs_checked: int = 0
    n_pairs_flagged: int = 0
    elapsed_s: float = 0.0
    settings: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure detector
# ---------------------------------------------------------------------------


def _moveaxis_last(vol, dim):
    """Return a view with ``dim`` moved to the last axis."""
    return np.moveaxis(vol, dim, -1)


def _phase_corr_int_shift(a2d, b2d, max_shift):
    """Integer (d0, d1, clamped) shift aligning ``b2d`` onto ``a2d``.

    ``clamped`` is True when the peak landed at (or beyond) ``±max_shift`` on
    either axis, i.e. the returned number is the SEARCH LIMIT and the true
    misalignment is at least that large — not a measurement. Reporting a
    clamped value as if it were measured is how a mosaic misaligned by an
    unknown amount produced a tidy-looking "aligned shift: ds=8" for 57% of
    its flagged seams.
    """
    if a2d.shape != b2d.shape or a2d.size == 0:
        return (0, 0, False)
    a = np.asarray(a2d, dtype=np.float64)
    b = np.asarray(b2d, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    if not np.any(a) or not np.any(b):
        return (0, 0, False)
    axes = tuple(range(a.ndim))
    fa = np.fft.rfftn(a, axes=axes)
    fb = np.fft.rfftn(b, axes=axes)
    r = fa * np.conj(fb)
    mag = np.abs(r)
    mag[mag == 0] = 1e-12
    corr = np.fft.irfftn(r / mag, s=a.shape, axes=axes)
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    shifts = []
    clamped = False
    for p, n in zip(peak, a.shape):
        s = p if p <= n // 2 else p - n
        if abs(s) >= max_shift:
            clamped = True
        s = int(np.clip(s, -max_shift, max_shift))
        shifts.append(s)
    return (shifts[0], shifts[1], clamped)


def _refine_overlap_width(a_border, b_border, w0, search):
    """Pick the across-seam overlap width that best aligns tile structure.

    Stage positions are quantized/imperfect, so the geometry-derived overlap
    ``w0`` can be several px off. Search widths near ``w0`` and pick the one
    minimizing the along-seam-normal *gradient* mismatch — gradient is
    DC-invariant, so a genuine flat-field step (a constant offset) does NOT bias
    the alignment; only lateral structure does. Returns the best width.
    """
    L = a_border.shape[-1]
    w0 = int(min(max(w0, 1), L))
    if search <= 0:
        return w0
    lo = max(2, w0 - search)
    hi = min(L, w0 + search)
    costs = {}
    for wt in range(lo, hi + 1):
        a_ov = a_border[..., L - wt:]
        b_ov = b_border[..., :wt]
        ga = np.diff(a_ov, axis=-1)
        gb = np.diff(b_ov, axis=-1)
        costs[wt] = float(np.mean((ga - gb) ** 2))
    if not costs:
        return w0
    # Only override the geometry width when the structural evidence is clear:
    # among widths within 10% of the best cost (a plateau when there's no
    # lateral structure to lock onto), keep the one closest to w0.
    best_cost = min(costs.values())
    thresh = best_cost * 1.10 + 1e-9
    near = [wt for wt, c in costs.items() if c <= thresh]
    return min(near, key=lambda wt: (abs(wt - w0), wt))


def _align_z(va, vb, ta, tb):
    """Crop two tile volumes to the Z range they physically share.

    The seam detector compares slabs plane index by plane index, which only
    means anything if plane *k* of each tile is the same depth in the sample.
    That holds for a uniform acquisition and fails for one with per-tile Z
    ranges, where every tile has its own z_min and depth: the comparison would
    either measure two different depths against each other or, once the plane
    counts differ, raise on the shape mismatch and take the whole QC pass down
    with it.

    Returns ``(va, vb, a0)`` where ``a0`` is how many planes were trimmed off
    the front of ``va`` — callers need it to report depths in mm, which are
    measured from ``ta.z_min_mm``. Returns ``(None, None, 0)`` when the tiles
    overlap in X/Y but not in Z: nothing to compare, and saying so beats
    reporting a seam that was never measured.
    """
    za, zb = int(va.shape[0]), int(vb.shape[0])
    # Derive each tile's plane pitch from the tile itself, so this is correct
    # whatever Z downsample the volumes have already been through.
    span_a = float(ta.z_max_mm) - float(ta.z_min_mm)
    step_mm = (span_a / (za - 1)) if za > 1 and span_a > 0 else 0.0
    if step_mm <= 0:
        offset = 0
    else:
        offset = int(round((float(tb.z_min_mm) - float(ta.z_min_mm)) / step_mm))
    a0 = max(0, offset)
    b0 = max(0, -offset)
    common = min(za - a0, zb - b0)
    if common <= 0:
        return None, None, 0
    if a0 == 0 and b0 == 0 and common == za == zb:
        return va, vb, 0  # identical geometry — avoid a needless re-slice
    return va[a0 : a0 + common], vb[b0 : b0 + common], a0


def _robust_sigma(x):
    """1.4826 * MAD — robust std estimate. ``x`` already finite."""
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)) + 1e-9)


def detect_border_steps(
    vol_a,
    vol_b,
    axis: str,
    overlap_px: int,
    *,
    params: Optional[BorderQCParams] = None,
    geom_shift: Tuple[int, int] = (0, 0),
) -> BorderStepResult:
    """Detect sharp intensity steps along the shared seam of two tiles.

    ``vol_a`` / ``vol_b`` are (Z, Y, X) arrays (numpy or dask/memmap). ``vol_a``
    is the lower-coordinate tile along ``axis`` ("x": B is to the right; "y": B
    is below; "z": B is above). Only the near-seam slab of each is materialized.
    ``overlap_px <= 0`` selects the abutting comparison (last vs first face).
    """
    p = params or BorderQCParams()
    n = _AXIS_TO_DIM[axis]
    W = int(overlap_px)
    search = int(p.max_refine_shift_px) if p.refine_shift else 0
    # A geometry width <=0 can just be stage-position quantization eating a real
    # (small) overlap; if we can refine, search upward for it instead of
    # treating clearly-overlapping tiles as abutting.
    abutting = W < 1 and search <= 0
    margin = max(2, int(p.grad_window) + 2)
    # Materialize a generous near-seam border from each tile: the overlap plus
    # room to (a) refine the overlap width and (b) estimate the interior
    # gradient. Slicing first keeps dask/memmap reads to just this slab.
    D = max(W, 1) + margin + search + 1

    a_last = _moveaxis_last(vol_a, n)  # (P0, P1, N_a)
    b_last = _moveaxis_last(vol_b, n)  # (P0, P1, N_b)
    D = min(D, a_last.shape[-1], b_last.shape[-1])
    a_slab = np.asarray(a_last[..., -D:], dtype=np.float32)  # seam at the end
    b_slab = np.asarray(b_last[..., :D], dtype=np.float32)  # seam at the start

    # Collapse Z (the first plane axis) in mip mode for x/y seams.
    collapse = p.mode == "mip" and axis in ("x", "y") and a_slab.shape[0] > 1
    if collapse:
        a_slab = a_slab.max(axis=0, keepdims=True)
        b_slab = b_slab.max(axis=0, keepdims=True)
    elif p.mode == "full" and p.z_stride > 1 and axis in ("x", "y"):
        a_slab = a_slab[:: p.z_stride]
        b_slab = b_slab[:: p.z_stride]

    L = a_slab.shape[-1]
    dz = ds = 0
    shift_clamped = False
    if abutting:
        a_face = a_slab[..., -1]  # (P0, P1)
        b_face = b_slab[..., 0]
        step = np.abs(b_face - a_face)
        # Local along-seam-normal gradient from the two innermost columns.
        g_a = np.abs(a_slab[..., -1] - a_slab[..., -2]) if L >= 2 else np.zeros_like(step)
        g_b = np.abs(b_slab[..., 1] - b_slab[..., 0]) if L >= 2 else np.zeros_like(step)
        local_scale = np.maximum(g_a, g_b)
        a_ov = a_slab[..., -1:]
        b_ov = b_slab[..., :1]
        note = "abutting"
    else:
        # Refine the across-seam overlap width (stage positions are imperfect),
        # then align laterally (P0, P1) by integer phase correlation.
        W = _refine_overlap_width(a_slab, b_slab, max(W, 1), search)
        a_ov = a_slab[..., L - W:]  # (P0, P1, W)
        b_ov = b_slab[..., :W]
        if p.refine_shift and _ndi is not None:
            a_prof = a_ov.mean(axis=-1)  # (P0, P1)
            b_prof = b_ov.mean(axis=-1)
            dz, ds, shift_clamped = _phase_corr_int_shift(a_prof, b_prof, search)
            if dz or ds:
                b_ov = np.roll(b_ov, shift=(dz, ds), axis=(0, 1))
        kk = max(1, int(round(W * p.k_frac)))
        c0 = max(0, W // 2 - kk // 2)
        c1 = min(W, c0 + kk)
        step = np.abs(
            np.median(b_ov[..., c0:c1], axis=-1) - np.median(a_ov[..., c0:c1], axis=-1)
        )
        g_a = np.median(np.abs(np.diff(a_ov, axis=-1)), axis=-1)
        g_b = np.median(np.abs(np.diff(b_ov, axis=-1)), axis=-1)
        local_scale = np.maximum(g_a, g_b)
        note = ""

    # Per-pair noise sigma from interior first-differences (both tiles).
    diffs = np.concatenate(
        [np.diff(a_slab, axis=-1).ravel(), np.diff(b_slab, axis=-1).ravel()]
    )
    diffs = diffs[np.isfinite(diffs)]
    noise_sigma = _robust_sigma(diffs)

    # Background / NaN mask: require signal in both tiles at the seam.
    a_seam = np.median(a_ov, axis=-1)
    b_seam = np.median(b_ov, axis=-1)
    overlap_vals = np.concatenate([a_ov.ravel(), b_ov.ravel()])
    overlap_vals = overlap_vals[np.isfinite(overlap_vals)]
    if overlap_vals.size:
        bg = max(float(p.bg_abs_floor), float(np.percentile(overlap_vals, p.bg_percentile)))
    else:
        bg = float(p.bg_abs_floor)
    valid = np.isfinite(step) & np.isfinite(local_scale)
    valid &= (a_seam > bg) & (b_seam > bg)

    local_scale = np.where(valid, local_scale, 0.0)
    step = np.where(valid, step, 0.0)

    flag = (
        valid
        & (step > p.alpha * np.maximum(local_scale, noise_sigma))
        & (step > p.beta * noise_sigma)
    )

    # Connected-component speckle filter (needs scipy).
    largest = int(flag.sum())
    if _ndi is not None and flag.ndim >= 1 and flag.any():
        lbl, n_lbl = _ndi.label(flag)
        if n_lbl:
            sizes = np.bincount(lbl.ravel())
            sizes[0] = 0
            largest = int(sizes.max())
            keep = sizes >= p.min_component_px
            flag = keep[lbl]
    elif largest < p.min_component_px:
        flag = np.zeros_like(flag)
        largest = 0

    n_valid = int(valid.sum())
    flagged_steps = step[flag]
    med_step = float(np.median(flagged_steps)) if flagged_steps.size else 0.0
    max_step = float(flagged_steps.max()) if flagged_steps.size else 0.0

    res = BorderStepResult(
        flagged=bool(flag.any()),
        axis=axis,
        overlap_px=W,
        median_step_counts=med_step,
        max_step_counts=max_step,
        n_samples_evaluated=n_valid,
        largest_component_px=largest,
        used_shift=(int(dz), int(ds)),
        shift_clamped=bool(shift_clamped),
        note=note,
    )

    if collapse:  # mip: flag is (1, S)
        line = flag.reshape(-1)
        res.border_length_px = int(line.sum())
        res.area_px = int(flag.sum())
    else:
        res.area_px = int(flag.sum())
        if axis in ("x", "y") and flag.ndim == 2 and flag.any():
            zi = np.where(flag.any(axis=1))[0]
            stride = p.z_stride if p.mode == "full" else 1
            res.z_index_range = (int(zi.min()) * stride, int(zi.max()) * stride)
    return res


# ---------------------------------------------------------------------------
# Neighbor enumeration
# ---------------------------------------------------------------------------


def _group_by(vals, tol):
    """Cluster near-equal coordinate values; return {rounded_key: [indices]}."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    groups: List[List[int]] = []
    for i in order:
        if groups and abs(vals[i] - vals[groups[-1][-1]]) <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _min_pitch(vals):
    uniq = sorted({round(v, 4) for v in vals})
    gaps = [b - a for a, b in zip(uniq, uniq[1:]) if b - a > 1e-6]
    return min(gaps) if gaps else 0.0


def find_neighbor_pairs(tiles, *, include_z: bool = False):
    """Return (i, j, axis) for adjacent tiles, i the lower coordinate.

    X-seam: same row (y), adjacent in x. Y-seam: same column (x), adjacent in y.
    Z-seam (optional): same x,y, adjacent in z_min. A pitch tolerance skips
    gaps, so partial coverage simply yields no pair for the missing edge.
    """
    xs = [t.x_mm for t in tiles]
    ys = [t.y_mm for t in tiles]
    px, py = _min_pitch(xs), _min_pitch(ys)
    tol_x = 0.25 * px if px else 1e-6
    tol_y = 0.25 * py if py else 1e-6
    pairs: List[Tuple[int, int, str]] = []

    # X-seams: within each row (grouped by y), consecutive in x.
    for row in _group_by(ys, tol_y):
        row_sorted = sorted(row, key=lambda i: xs[i])
        for a, b in zip(row_sorted, row_sorted[1:]):
            gap = xs[b] - xs[a]
            if px and gap <= px * 1.5 and gap > tol_x:
                pairs.append((a, b, "x"))
    # Y-seams: within each column (grouped by x), consecutive in y.
    for col in _group_by(xs, tol_x):
        col_sorted = sorted(col, key=lambda i: ys[i])
        for a, b in zip(col_sorted, col_sorted[1:]):
            gap = ys[b] - ys[a]
            if py and gap <= py * 1.5 and gap > tol_y:
                pairs.append((a, b, "y"))
    if include_z:
        # Z-seams: same (x, y), adjacent z_min.
        cells: Dict[Tuple[float, float], List[int]] = {}
        for i, t in enumerate(tiles):
            cells.setdefault((round(t.x_mm, 3), round(t.y_mm, 3)), []).append(i)
        for idxs in cells.values():
            zs = sorted(idxs, key=lambda i: tiles[i].z_min_mm)
            for a, b in zip(zs, zs[1:]):
                if tiles[b].z_min_mm - tiles[a].z_min_mm > 1e-6:
                    pairs.append((a, b, "z"))
    return pairs


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _overlap_px(tile_a, tile_b, vol_a, axis, eff_pixel_um, eff_z_um):
    """Overlap in downsampled px along the seam-normal axis."""
    if axis == "x":
        extent = vol_a.shape[2]
        pitch_um = abs(tile_b.x_mm - tile_a.x_mm) * 1000.0
        return int(round(extent - pitch_um / eff_pixel_um))
    if axis == "y":
        extent = vol_a.shape[1]
        pitch_um = abs(tile_b.y_mm - tile_a.y_mm) * 1000.0
        return int(round(extent - pitch_um / eff_pixel_um))
    # z
    extent = vol_a.shape[0]
    pitch_um = abs(tile_b.z_min_mm - tile_a.z_min_mm) * 1000.0
    return int(round(extent - pitch_um / max(eff_z_um, 1e-9)))


def _tile_name(tile) -> str:
    try:
        return tile.folder.name
    except Exception:
        return str(getattr(tile, "folder", "?"))


def run_border_qc(
    channel_tile_data,
    tiles,
    *,
    pixel_size_um: float,
    ds_xy: int,
    ds_z: int,
    z_step_um: float,
    reg_channel: int,
    params: BorderQCParams,
    logger=None,
    cancelled_fn=None,
) -> BorderQCReport:
    """Run border QC over neighbor pairs of the reference channel.

    ``channel_tile_data`` is ``{ch_id: [(volume, tile_info), ...]}`` (volumes are
    the preprocessed, downsampled (Z,Y,X) arrays). ``pixel_size_um`` /
    ``z_step_um`` are the EFFECTIVE (post-downsample) sizes.
    """
    t0 = time.monotonic()
    ch = reg_channel if reg_channel in channel_tile_data else (
        sorted(channel_tile_data)[0] if channel_tile_data else None
    )
    report = BorderQCReport(
        params=params,
        channel_id=ch,
        settings={
            "pixel_size_um_effective": pixel_size_um,
            "z_step_um_effective": z_step_um,
            "downsample_xy": ds_xy,
            "downsample_z": ds_z,
            "mode": params.mode,
            "alpha": params.alpha,
            "beta": params.beta,
            "min_component_px": params.min_component_px,
        },
    )
    if ch is None:
        return report

    # Map tile identity -> (volume, tile) for this channel.
    entries = channel_tile_data[ch]
    vol_by_id = {id(t): v for (v, t) in entries}
    tiles_present = [t for (v, t) in entries]
    pairs = find_neighbor_pairs(tiles_present, include_z=params.include_z_seams)

    eff_um2 = pixel_size_um * pixel_size_um
    # Stage positions come from folder names quantized to ~0.01 mm, so the
    # geometry-derived overlap can be several px off — widen the alignment
    # search to cover that quantization (bigger at fine effective pixel sizes).
    quant_px = int(math.ceil(6.0 / max(pixel_size_um, 1e-6)))
    # An explicit request (> the 8 px default) is honoured as-is; the min(24,…)
    # cap only bounds the AUTO widening for fine pixel sizes.
    if int(params.max_refine_shift_px) > 8:
        search_px = int(params.max_refine_shift_px)
    else:
        search_px = int(min(24, max(params.max_refine_shift_px, quant_px, 8)))
    pair_params = replace(params, max_refine_shift_px=search_px)
    n_no_shared_z = 0
    for (ia, ib, axis) in pairs:
        if cancelled_fn is not None and cancelled_fn():
            break
        ta, tb = tiles_present[ia], tiles_present[ib]
        va, vb = vol_by_id.get(id(ta)), vol_by_id.get(id(tb))
        if va is None or vb is None:
            continue
        # X/Y neighbours need not span the same Z (per-tile Z ranges), and the
        # detector compares plane-for-plane. Trim to what they share.
        z_trim_a = 0
        if axis in ("x", "y"):
            va, vb, z_trim_a = _align_z(va, vb, ta, tb)
            if va is None:
                n_no_shared_z += 1
                continue
        report.n_pairs_checked += 1
        W = _overlap_px(ta, tb, va, axis, pixel_size_um, z_step_um)
        try:
            res = detect_border_steps(va, vb, axis, W, params=pair_params)
        except Exception as e:  # never let one pair kill the pass
            if logger is not None:
                logger.debug(f"Border QC pair {_tile_name(ta)}<->{_tile_name(tb)} failed: {e}")
            continue
        if not res.flagged:
            continue
        report.n_pairs_flagged += 1

        pr = PairReport(
            tile_a_name=_tile_name(ta),
            tile_b_name=_tile_name(tb),
            axis=axis,
            result=res,
            channel_id=ch,
        )
        if res.border_length_px is not None:
            pr.border_length_um = res.border_length_px * pixel_size_um
        if res.area_px:
            pr.area_um2 = res.area_px * eff_um2
        if res.z_index_range is not None and axis in ("x", "y"):
            # Indices are relative to the Z-ALIGNED slab, so add back whatever
            # _align_z trimmed off the front of tile A before naming a depth.
            z0, z1 = (i + z_trim_a for i in res.z_index_range)
            pr.z_plane_range = (z0 * ds_z, z1 * ds_z)
            # z_index_range is in the (possibly z-strided) downsampled frame;
            # convert using the native z step per downsampled plane.
            per_ds_plane_mm = (z_step_um / 1000.0)
            pr.z_mm_range = (
                ta.z_min_mm + z0 * per_ds_plane_mm,
                ta.z_min_mm + z1 * per_ds_plane_mm,
            )
        report.pairs.append(pr)

    # Sort worst-first per mode.
    def _severity(pr: PairReport):
        if params.mode == "mip":
            return (pr.result.border_length_px or 0, pr.result.median_step_counts)
        if params.mode == "full":
            return (pr.result.area_px, pr.result.median_step_counts)
        return (pr.result.median_step_counts,)

    report.pairs.sort(key=_severity, reverse=True)
    report.elapsed_s = time.monotonic() - t0
    if n_no_shared_z and logger is not None:
        # Not an error — with per-tile Z ranges some neighbours genuinely image
        # disjoint depths. Say it out loud so "0 seams flagged" is never read as
        # "every seam checked".
        logger.info(
            f"  Border QC skipped {n_no_shared_z} neighbour pair"
            f"{'s' if n_no_shared_z != 1 else ''}: adjacent in X/Y but sharing "
            f"no Z range (per-tile Z ranges), so there is no common volume to "
            f"compare."
        )
    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report_text(report: BorderQCReport, *, acquisition: str = "") -> str:
    s = report.settings
    lines = []
    lines.append("=" * 71)
    lines.append(" TILE-BORDER ARTIFACT QC" + (f" — {acquisition}" if acquisition else ""))
    ch = report.channel_id
    reg = "on" if True else "off"
    mode_desc = {
        "mip": "mip (Z max-projection)",
        "full": "full (per-Z)",
        "pairs": "pairs (list only)",
    }.get(report.params.mode, report.params.mode)
    lines.append(
        f" Mode: {mode_desc}   Channel: {ch} (reference)"
    )
    lines.append(
        f" Effective px: {s.get('pixel_size_um_effective', 0):.3f} um XY "
        f"(downsample_xy={s.get('downsample_xy')})   "
        f"Pairs checked: {report.n_pairs_checked}   Flagged: {report.n_pairs_flagged}"
    )
    lines.append(
        f" Thresholds: alpha={report.params.alpha:g}  beta={report.params.beta:g}  "
        f"min_component={report.params.min_component_px} px   "
        f"Elapsed: {report.elapsed_s:.1f} s"
    )
    lines.append(" (Intra-tile illumination seams are not analyzed — tile-to-tile borders only.)")
    # Headline the clamp count. A run where most shifts hit the search limit is
    # not "slightly misaligned by 8 px" — it is misaligned by an unknown amount
    # the QC could not measure, and every number below understates it.
    _flagged = [pr.result for pr in report.pairs if pr.result.flagged]
    _with_shift = [r for r in _flagged if r.used_shift != (0, 0)]
    _clamped = [r for r in _with_shift if r.shift_clamped]
    if _clamped:
        _px = s.get("pixel_size_um_effective", 0) or 0
        lines.append(
            f" ** {len(_clamped)} of {len(_with_shift)} measured shifts hit the "
            f"+/-{report.params.max_refine_shift_px} px search limit "
            f"(+/-{report.params.max_refine_shift_px * _px:.1f} um). Those seams are "
            "misaligned by AT LEAST that much — the true offset was not measured."
        )
        lines.append(
            "    Raise border_qc_max_shift_px (or --border-qc-max-shift) to measure it. "
            "A large clamped fraction points at a systematic placement error "
            "(pixel size, overlap, or tile order), not random jitter."
        )
    if s.get("downsample_xy", 1) and s.get("downsample_xy", 1) > 2:
        lines.append(
            f" NOTE: downsample_xy={s.get('downsample_xy')} softens single-pixel steps; "
            "rerun at downsample_xy<=2 for the most sensitive detection."
        )
    lines.append("-" * 71)
    if not report.pairs:
        lines.append(" No tile pairs exceeded threshold.")
        lines.append("=" * 71)
        return "\n".join(lines)

    lines.append(" FLAGGED SEAMS (worst first)")
    for k, pr in enumerate(report.pairs, 1):
        r = pr.result
        seam = {"x": "X-seam", "y": "Y-seam", "z": "Z-seam"}[pr.axis]
        ov = "abutting" if r.note == "abutting" else f"{r.overlap_px} px"
        lines.append(f" {k}. {pr.tile_a_name}  <->  {pr.tile_b_name}   [{seam}]")
        if report.params.mode == "mip":
            length = f"{r.border_length_px} px"
            if pr.border_length_um:
                length += f" ({pr.border_length_um:.0f} um)"
            lines.append(
                f"       border length: {length}   "
                f"median step: {r.median_step_counts:.0f} counts   overlap: {ov}"
            )
        elif report.params.mode == "full":
            area = f"{r.area_px} px^2"
            if pr.area_um2:
                area += f" ({pr.area_um2:.0f} um^2)"
            zr = ""
            if pr.z_mm_range and pr.z_plane_range:
                zr = (
                    f" | Z: {pr.z_mm_range[0]:.2f}-{pr.z_mm_range[1]:.2f} mm "
                    f"(planes {pr.z_plane_range[0]}-{pr.z_plane_range[1]})"
                )
            lines.append(
                f"       area: {area}{zr}"
            )
            lines.append(
                f"       median step: {r.median_step_counts:.0f} counts "
                f"(max {r.max_step_counts:.0f})   overlap: {ov}"
            )
        else:  # pairs
            lines.append(
                f"       severity: {r.median_step_counts:.0f} counts   overlap: {ov}"
            )
        if r.used_shift != (0, 0):
            _sh = f"       aligned shift: (dz={r.used_shift[0]}, ds={r.used_shift[1]})"
            if r.shift_clamped:
                _sh += "   ** AT SEARCH LIMIT — true offset is LARGER, unmeasured **"
            lines.append(_sh)
    lines.append("=" * 71)
    return "\n".join(lines)


def report_to_json(report: BorderQCReport, *, acquisition: str = "") -> Dict[str, Any]:
    return {
        "version": 1,
        "acquisition": acquisition,
        "channel_id": report.channel_id,
        "mode": report.params.mode,
        "n_pairs_checked": report.n_pairs_checked,
        "n_pairs_flagged": report.n_pairs_flagged,
        "elapsed_s": round(report.elapsed_s, 3),
        "settings": report.settings,
        "pairs": [
            {
                "tile_a": pr.tile_a_name,
                "tile_b": pr.tile_b_name,
                "axis": pr.axis,
                "channel_id": pr.channel_id,
                "overlap_px": pr.result.overlap_px,
                "flagged": pr.result.flagged,
                "area_px": pr.result.area_px,
                "area_um2": pr.area_um2,
                "border_length_px": pr.result.border_length_px,
                "border_length_um": pr.border_length_um,
                "z_index_range": pr.result.z_index_range,
                "z_plane_range": pr.z_plane_range,
                "z_mm_range": pr.z_mm_range,
                "median_step_counts": pr.result.median_step_counts,
                "max_step_counts": pr.result.max_step_counts,
                "n_samples_evaluated": pr.result.n_samples_evaluated,
                "used_shift": list(pr.result.used_shift),
                "shift_clamped": bool(pr.result.shift_clamped),
                "note": pr.result.note,
            }
            for pr in report.pairs
        ],
    }
