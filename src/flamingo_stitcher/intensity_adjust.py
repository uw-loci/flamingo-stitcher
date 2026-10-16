"""Per-tile intensity adjustment, ported from BigStitcher.

Tiles that are correctly placed can still meet at a visible brightness step:
laser drift over a long acquisition, bleaching of each overlap by the first
tile that imaged it, depth. The flat-field fixes the shape of the field
*within* a frame, the same for every tile; nothing else in the stitcher
matched tile-to-tile levels, and the registration's own normalisations act on
throw-away copies used to measure shifts.

This is a port of BigStitcher's "Intensity Adjustment" (Fiji plugin
``Intensity_Adjustment`` -> ``IntensityAdjustmentTools.computeIntensityAdjustment``
in multiview-reconstruction 9.1.1, identical on master at the time of
porting), with the same parameters and defaults:

1. **Grid.** The maximal bounding box of all views in world pixels
   (``BoundingBoxMaximal``: floor/ceil of every view's transformed corners
   ``0 .. dim-1``), scaled by ``1/downsampling`` with ``Math.round``
   (``TransformVirtual.scaleBoundingBox``). Grid point ``g`` is world
   ``g * downsampling``.
2. **Sampling.** Each view is sampled at every grid point by LINEAR
   interpolation; a point counts as inside only STRICTLY within
   ``0 < t < dim-1`` on every axis, otherwise it is -1
   (``TransformView.transformView(.., -1, 1)``, ``intersectsLinearInterpolation``).
3. **Matches.** Wherever two or more views are inside, every pair (i < j)
   gets the match ``(value_i, value_j)``, in grid order (x fastest).
   Background is NOT excluded: BigStitcher does not.
4. **Thinning.** Pairs are visited in Java ``HashMap`` iteration order of
   their ``ValuePair(i, j)`` keys and each list is thinned with ONE shared
   ``java.util.Random(344)`` by ``remove(nextInt(size))`` while
   ``size >= max_matches`` -- so ``max_matches - 1`` survive. Reproduced
   exactly (same RNG stream, same order, same survivors); the removal uses a
   Fenwick tree instead of ``ArrayList.remove``, O(n log n) instead of O(n^2).
5. **Model.** Per view, ``a * v + b``. mpicbg
   ``InterpolatedAffineModel1D(InterpolatedAffineModel1D(Affine, Translation,
   offset_regularization), Identity, identity_regularization)``; with
   ``affine=False`` ``InterpolatedAffineModel1D(Translation, Identity, ..)``.
6. **Solve.** BigStitcher runs mpicbg's ``TileConfiguration``: shuffled,
   parallel Gauss-Seidel sweeps of "fit each tile to its neighbours' current
   values", 201-300 of them, no tile fixed. Every sweep is linear in the
   other tiles' parameters, so the configuration it converges to solves a
   linear system; this module solves that system directly. It is
   deterministic where BigStitcher's shuffled order is not.
7. **Offset shift.** All offsets are lowered by the smallest one
   (``runGlobal``: "corrected to avoid negative intensities"). Views with no
   overlap keep identity and take part in that minimum.
8. **Apply.** ``v * a + b`` per view before fusion (``IntensityAdjuster``).

Known, deliberate differences: BigStitcher reads the pyramid level
``DownsampleTools.openDownsampled`` picks; we sample the stitcher's own
working copy (the downsampled tile spill). Sampling arithmetic is float64
where imglib2's ``FloatType`` interpolation is float32, and a grid point lying
exactly on a view's edge can fall the other way under a last-bit difference
in the inverse transform.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# BigStitcher's own defaults (Intensity_Adjustment, multiview-reconstruction 9.1.1).
BIGSTITCHER_DEFAULT_DOWNSAMPLING = 10.0
BIGSTITCHER_DEFAULT_MAX_MATCHES = 10000
BIGSTITCHER_DEFAULT_AFFINE = True
BIGSTITCHER_DEFAULT_OFFSET_REGULARIZATION = 0.1
BIGSTITCHER_DEFAULT_IDENTITY_REGULARIZATION = 0.1
BIGSTITCHER_RANDOM_SEED = 344

# Grid points sampled at once per tile pair. A memory bound only: the slabs
# are consecutive in grid order, so the result does not depend on it.
_POINTS_PER_SLAB = 1 << 21


@dataclass
class IntensityAdjustParams:
    downsampling: float = BIGSTITCHER_DEFAULT_DOWNSAMPLING
    max_matches: int = BIGSTITCHER_DEFAULT_MAX_MATCHES
    affine: bool = BIGSTITCHER_DEFAULT_AFFINE
    offset_regularization: float = BIGSTITCHER_DEFAULT_OFFSET_REGULARIZATION
    identity_regularization: float = BIGSTITCHER_DEFAULT_IDENTITY_REGULARIZATION


@dataclass
class View:
    """One tile as the adjustment sees it.

    ``data`` is (Z, Y, X). ``affine`` is 3x4, mapping a voxel index
    (x, y, z) -- imglib2 axis order -- to world PIXELS (x, y, z).
    """

    data: np.ndarray
    affine: np.ndarray


@dataclass
class IntensityAdjustResult:
    scale: np.ndarray            # a per view
    offset: np.ndarray           # b per view (after the min-offset shift)
    min_offset: float            # what was subtracted from every offset
    pair_counts: Dict[Tuple[int, int], Tuple[int, int]]  # (before, after thinning)
    connected: np.ndarray        # bool per view: had any overlap
    bbox: Tuple[np.ndarray, np.ndarray]          # world px, inclusive
    grid: Tuple[np.ndarray, np.ndarray]          # scaled, inclusive


# ---------------------------------------------------------------------------
# java.util.Random and java.util.HashMap, exactly
# ---------------------------------------------------------------------------

_MULT = 0x5DEECE66D
_MASK = (1 << 48) - 1


class JavaRandom:
    """``java.util.Random``: same seed scramble, LCG and ``nextInt(bound)``."""

    def __init__(self, seed: int):
        self.seed = (seed ^ _MULT) & _MASK

    def _next(self, bits: int) -> int:
        self.seed = (self.seed * _MULT + 0xB) & _MASK
        r = self.seed >> (48 - bits)
        if r >= 1 << 31:  # Java int
            r -= 1 << 32
        return r

    def next_int(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        r = self._next(31)
        m = bound - 1
        if (bound & m) == 0:
            return (bound * r) >> 31
        u = r
        while True:
            r = u % bound
            if u - r + m < (1 << 31):  # no int overflow
                return r
            u = self._next(31)


def java_hashmap_order(m: int) -> List[Tuple[int, int]]:
    """Iteration order of a ``HashMap<ValuePair<Integer,Integer>, ..>`` filled
    with every (i, j), i < j < m, in nested-loop order -- BigStitcher's
    ``intensityMatches``.

    ``ValuePair.hashCode = 31 * (31 + i) + j``; ``HashMap`` spreads it with
    ``h ^ (h >>> 16)``, sizes the table to the next power of two >= 16 holding
    ``n / 0.75``, iterates buckets in index order and, within a bucket, keeps
    insertion order (resizes split lists preserving order). Bins only become
    trees at 8+ entries; that never happens for the view counts a stitch has,
    and is refused rather than guessed.
    """
    keys = [(i, j) for i in range(m - 1) for j in range(i + 1, m)]
    n = len(keys)
    cap = 16
    while n > cap * 0.75:
        cap *= 2
    buckets: Dict[int, List[Tuple[int, int]]] = {}
    for k in keys:
        h = (31 * (31 + k[0]) + k[1]) & 0xFFFFFFFF
        h ^= h >> 16
        buckets.setdefault(h & (cap - 1), []).append(k)
    if any(len(v) >= 8 for v in buckets.values()):
        raise NotImplementedError(
            "HashMap bin would be treeified; iteration order not reproduced"
        )
    return [k for b in sorted(buckets) for k in buckets[b]]


def thin_like_arraylist(n: int, max_matches: int, rnd: JavaRandom) -> np.ndarray:
    """Indices that survive ``while size >= max: list.remove(rnd.nextInt(size))``.

    Same random stream as the ArrayList loop, in O(n log n): a Fenwick tree of
    alive flags finds the k-th survivor.
    """
    if n < max_matches:
        return np.arange(n)
    tree = [0] * (n + 1)
    for i in range(1, n + 1):  # all alive: tree[i] = lowbit(i)
        tree[i] = i & (-i)
    alive = bytearray(b"\x01") * n
    top = 1 << (n.bit_length() - 1)
    size = n
    while size >= max_matches:
        k = rnd.next_int(size) + 1  # 1-based rank among survivors
        pos = 0
        step = top
        while step:
            nxt = pos + step
            if nxt <= n and tree[nxt] < k:
                pos = nxt
                k -= tree[nxt]
            step >>= 1
        idx = pos  # 0-based index of the k-th alive
        alive[idx] = 0
        i = idx + 1
        while i <= n:
            tree[i] -= 1
            i += i & (-i)
        size -= 1
    return np.flatnonzero(np.frombuffer(bytes(alive), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Grid + sampling
# ---------------------------------------------------------------------------


def _java_round(x: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)


def _corners(dims_xyz) -> np.ndarray:
    mx = [d - 1 for d in dims_xyz]
    return np.array(
        [[x, y, z] for x in (0, mx[0]) for y in (0, mx[1]) for z in (0, mx[2])],
        dtype=np.float64,
    )


def _apply(aff: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ aff[:, :3].T + aff[:, 3]


def _dims_xyz(v: View) -> Tuple[int, int, int]:
    z, y, x = v.data.shape
    return (x, y, z)


def view_bounds_world(v: View) -> Tuple[np.ndarray, np.ndarray]:
    c = _apply(np.asarray(v.affine, dtype=np.float64), _corners(_dims_xyz(v)))
    return c.min(axis=0), c.max(axis=0)


def bounding_box(views: Sequence[View]) -> Tuple[np.ndarray, np.ndarray]:
    """``BoundingBoxMaximal``: floor(min), ceil(max) in world pixels."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for v in views:
        a, b = view_bounds_world(v)
        lo = np.minimum(lo, a)
        hi = np.maximum(hi, b)
    return np.floor(lo).astype(np.int64), np.ceil(hi).astype(np.int64)


def sample_view(v: View, grid_pts_xyz: np.ndarray, downsampling: float) -> np.ndarray:
    """Linear sample of ``v`` at scaled grid points (N, 3); -1 outside.

    The grid point is world ``g * downsampling``; inside means strictly
    ``0 < t < dim - 1`` on every axis (``intersectsLinearInterpolation``).
    """
    aff = np.asarray(v.affine, dtype=np.float64)
    inv = np.linalg.inv(np.vstack([aff, [0, 0, 0, 1]]))[:3]
    world = grid_pts_xyz.astype(np.float64) * downsampling
    t = _apply(inv, world)
    dims = np.array(_dims_xyz(v), dtype=np.float64)
    inside = np.all((t > 0) & (t < dims - 1), axis=1)
    out = np.full(len(t), -1.0, dtype=np.float64)
    if not inside.any():
        return out
    ti = t[inside]
    f = np.floor(ti).astype(np.int64)
    w = ti - f
    # Read only the block these points need (a dask/memmap tile is never
    # pulled whole), then index inside it.
    lo = f.min(axis=0)
    hi = f.max(axis=0) + 2
    block = np.asarray(
        v.data[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]], dtype=np.float64
    )
    f = f - lo
    acc = np.zeros(len(ti), dtype=np.float64)
    for dz in (0, 1):
        wz = w[:, 2] if dz else 1.0 - w[:, 2]
        for dy in (0, 1):
            wy = w[:, 1] if dy else 1.0 - w[:, 1]
            for dx in (0, 1):
                wx = w[:, 0] if dx else 1.0 - w[:, 0]
                acc += wx * wy * wz * block[f[:, 2] + dz, f[:, 1] + dy, f[:, 0] + dx]
    out[inside] = acc
    return out


def _grid_box_for(v: View, downsampling: float, g_lo, g_hi):
    """Scaled-grid points that could be inside ``v`` (conservative)."""
    a, b = view_bounds_world(v)
    lo = np.maximum(np.floor(a / downsampling).astype(np.int64), g_lo)
    hi = np.minimum(np.ceil(b / downsampling).astype(np.int64), g_hi)
    return lo, hi


def collect_matches(
    views: Sequence[View], params: IntensityAdjustParams
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Tuple, Tuple]:
    """Every (value_i, value_j) pair, per view pair, in BigStitcher's order.

    BigStitcher walks the whole scaled box once and appends to every pair that
    covers a point. Restricting the walk to one pair's common box visits the
    same points in the same relative order, which is all a pair's list
    depends on.
    """
    ds = float(params.downsampling)
    use_ds = not (ds == 1.0 or math.isnan(ds))
    bb_lo, bb_hi = bounding_box(views)
    if use_ds:
        g_lo, g_hi = _java_round(bb_lo / ds), _java_round(bb_hi / ds)
        step = ds
    else:
        g_lo, g_hi = bb_lo.copy(), bb_hi.copy()
        step = 1.0
    boxes = [_grid_box_for(v, step, g_lo, g_hi) for v in views]
    m = len(views)
    matches: Dict[Tuple[int, int], np.ndarray] = {}
    for i in range(m - 1):
        for j in range(i + 1, m):
            lo = np.maximum(boxes[i][0], boxes[j][0])
            hi = np.minimum(boxes[i][1], boxes[j][1])
            if np.any(hi < lo):
                matches[(i, j)] = np.empty((0, 2))
                continue
            # x fastest, then y, then z: imglib2 cursor order. Walked in Z
            # slabs so memory stays bounded; slabs are consecutive in that
            # order, so the list comes out the same.
            ys = np.arange(lo[1], hi[1] + 1)
            xs = np.arange(lo[0], hi[0] + 1)
            per_plane = len(ys) * len(xs)
            slab = max(1, _POINTS_PER_SLAB // per_plane)
            parts = []
            for z0 in range(lo[2], hi[2] + 1, slab):
                zs = np.arange(z0, min(z0 + slab, hi[2] + 1))
                Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
                pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
                vi = sample_view(views[i], pts, step)
                vj = sample_view(views[j], pts, step)
                keep = (vi >= 0) & (vj >= 0)
                parts.append(np.stack([vi[keep], vj[keep]], axis=1))
            matches[(i, j)] = np.concatenate(parts) if parts else np.empty((0, 2))
    return matches, (bb_lo, bb_hi), (g_lo, g_hi)


def thin_matches(
    matches: Dict[Tuple[int, int], np.ndarray], m: int, max_matches: int
) -> Dict[Tuple[int, int], np.ndarray]:
    rnd = JavaRandom(BIGSTITCHER_RANDOM_SEED)
    out = {}
    for key in java_hashmap_order(m):
        arr = matches.get(key, np.empty((0, 2)))
        keep = thin_like_arraylist(len(arr), int(max_matches), rnd)
        out[key] = arr[keep]
    return out


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def solve(
    matches: Dict[Tuple[int, int], np.ndarray], m: int, params: IntensityAdjustParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fixed point of BigStitcher's per-tile fits, then the min-offset shift.

    Per tile i, from all of its matches ``x = own value``, ``q = a_j y + b_j``
    (the other view's current mapping of its value; unit weights):

        affine fit:       m = cov(x, q) / var(x),  c = mean(q) - m mean(x)
        translation fit:  t = mean(q) - mean(x)
        inner (l1):       s = (1-l1) m + l1,  o = (1-l1) c + l1 t
        outer (l2):       a_i = (1-l2) s + l2,  b_i = (1-l2) o

    ``affine=False`` is the same with l1 = 1. Every term is linear in the
    neighbours' (a_j, b_j), so the converged configuration solves one linear
    system.
    """
    l1 = float(params.offset_regularization) if params.affine else 1.0
    l2 = float(params.identity_regularization)
    per: List[List[Tuple[int, np.ndarray, np.ndarray]]] = [[] for _ in range(m)]
    for (i, j), arr in matches.items():
        if len(arr) == 0:
            continue
        per[i].append((j, arr[:, 0], arr[:, 1]))
        per[j].append((i, arr[:, 1], arr[:, 0]))
    connected = np.array([bool(p) for p in per])
    A = np.zeros((2 * m, 2 * m))
    d = np.zeros(2 * m)
    for i in range(m):
        ai, bi = 2 * i, 2 * i + 1
        if not per[i]:
            A[ai, ai] = A[bi, bi] = 1.0  # identity, untouched
            d[ai] = 1.0
            continue
        N = sum(len(x) for _, x, _ in per[i])
        sx = sum(float(x.sum()) for _, x, _ in per[i])
        sxx = sum(float((x * x).sum()) for _, x, _ in per[i])
        xbar = sx / N
        var = sxx - N * xbar * xbar
        if params.affine and var <= 0:
            raise ValueError(
                f"view {i}: all of its overlap intensities are equal; the "
                "affine fit is ill-defined (mpicbg IllDefinedDataPointsException)"
            )
        # coefficient rows (over the 2m unknowns) for mean(q) and m_aff
        qbar = np.zeros(2 * m)
        maff = np.zeros(2 * m)
        for j, x, y in per[i]:
            n_ij = len(x)
            sy = float(y.sum())
            qbar[2 * j] += sy / N
            qbar[2 * j + 1] += n_ij / N
            if params.affine:
                maff[2 * j] += (float((x * y).sum()) - xbar * sy) / var
                maff[2 * j + 1] += (float(x.sum()) - xbar * n_ij) / var
        # a_i = (1-l2)(1-l1) m_aff + (1-l2) l1 + l2
        A[ai] -= (1 - l2) * (1 - l1) * maff
        A[ai, ai] += 1.0
        d[ai] = (1 - l2) * l1 + l2
        # b_i = (1-l2) [ q̄ - (1-l1) x̄ m_aff - l1 x̄ ]
        A[bi] -= (1 - l2) * (qbar - (1 - l1) * xbar * maff)
        A[bi, bi] += 1.0
        d[bi] = -(1 - l2) * l1 * xbar
    theta = np.linalg.solve(A, d)
    a = theta[0::2].copy()
    b = theta[1::2].copy()
    min_off = float(b.min())
    b -= min_off
    return a, b, connected, min_off


def compute(views: Sequence[View], params: Optional[IntensityAdjustParams] = None) -> IntensityAdjustResult:
    params = params or IntensityAdjustParams()
    m = len(views)
    raw, bbox, grid = collect_matches(views, params)
    thinned = thin_matches(raw, m, params.max_matches)
    a, b, connected, min_off = solve(thinned, m, params)
    counts = {k: (len(raw[k]), len(thinned[k])) for k in raw}
    return IntensityAdjustResult(a, b, min_off, counts, connected, bbox, grid)


def apply_uint16(volume, scale: float, offset: float):
    """``v * a + b`` (``IntensityAdjuster``), rounded and clipped to uint16.

    Works on numpy or dask arrays. BigStitcher fuses in float; the stitcher's
    output is uint16, so values are rounded and clipped to 0..65535.
    """
    import dask.array as da

    if isinstance(volume, da.Array):
        return volume.map_blocks(
            _apply_block, scale, offset, dtype=np.uint16
        )
    return _apply_block(np.asarray(volume), scale, offset)


def _apply_block(block, scale, offset):
    out = np.asarray(block, dtype=np.float32) * np.float32(scale) + np.float32(offset)
    np.rint(out, out=out)
    np.clip(out, 0, 65535, out=out)
    return out.astype(np.uint16)
