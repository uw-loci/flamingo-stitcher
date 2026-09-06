"""Centre-anchored placement with a second pass for tiles that cannot register.

A default single-workflow collection images a rectangle, so a round sample
leaves a rim of empty tiles. Those tiles have nothing to correlate against —
they are not a registration failure, they are background — but multiview-
stitcher resolves each connected component of the seam graph INDEPENDENTLY and
gives an edgeless tile the identity transform. So the registered core moves and
the rim stays at its raw stage position, which both tears the seam between them
and drags the registered-seam fraction under the guard that decides whether to
trust the run at all. On a 7x7 with a brain in the middle: 41 of 84 seams,
16 tile groups, 15 tiles connected to nothing, registration discarded after
9h33m of measuring it.

This module adds the missing half. Phase 1 is unchanged in kind — the same
simultaneous least-squares solve — only ANCHORED at the centre-most tile, so the
gauge is the middle of the sample and accumulated error grows outward rather
than from whichever corner the solver happened to pick.

Phase 1 is deliberately NOT a greedy outward walk. Placing each tile from its
already-placed neighbours commits the answer in visit order: for a tile whose
left and bottom neighbours disagree, whichever was visited first wins and the
other seam can never be satisfied. BigStitcher solves this by never placing
tiles one at a time — it optimises all links at once and, when the worst edge
residual exceeds tolerance, DROPS that link (only if removing it does not
disconnect the graph) and re-solves. multiview-stitcher implements the same
thing as ``global_optimization``. Greedy growth would reintroduce exactly the
tension that design exists to avoid, so "build outward" is expressed as the
anchor, not as the visit order.

Phase 2 is the new part: every tile outside the core is CARRIED — placed at the
mean correction of its grid-adjacent already-placed neighbours, spreading
outward until nothing is left. A carried tile keeps exactly the overlap the
stage gave it with those neighbours, so no black gap opens between it and the
mosaic, and no tile is left behind at a raw stage position while its neighbours
have moved.

Tiles are carried individually rather than as solved components. A component
that registered internally but not to the core is, on this data, a pair of rim
tiles that correlated on noise; trusting its internal solve is how a block
slides 90-194 px past the tiles around it. Neighbour agreement is the safer
gauge, and the count is logged either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Per-edge residual (MICROMETRES — multiview-stitcher's edge_residuals are in
# physical units, despite the report column being named residual_px) above which
# a tile's seams are reported as irreconcilable. Matches the spirit of
# global_opt_abs_tol, which is the tolerance BigStitcher prunes links at.
DEFAULT_TENSION_UM = 3.5


def tile_label(tile, index: int) -> str:
    """``X006 Y004`` when the flat filename carries a grid index.

    Folder-layout (multi-acquisition) tiles have no index — their identity is
    where they sit — so those fall back to the stage position, which is what
    the operator sees in the queue.
    """
    idx = getattr(tile, "tile_index", None)
    if idx is not None:
        try:
            return f"X{int(idx[0]):03d} Y{int(idx[1]):03d}"
        except Exception:
            pass
    try:
        return f"X={float(tile.x_mm):.2f} Y={float(tile.y_mm):.2f}"
    except Exception:
        return f"tile {index}"


def centre_tile_index(tiles: Sequence) -> Optional[int]:
    """The tile nearest the mosaic's XY centroid — the anchor for the solve.

    The centroid rather than the grid's arithmetic middle: on a partly covered
    grid the middle cell can itself be empty, and anchoring on background is
    the one place the gauge should never sit.
    """
    if not tiles:
        return None
    try:
        xs = np.array([float(t.x_mm) for t in tiles], dtype=float)
        ys = np.array([float(t.y_mm) for t in tiles], dtype=float)
    except Exception:
        return None
    if not len(xs):
        return None
    d2 = (xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2
    return int(np.argmin(d2))


def _translation(param) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(param, dtype=float)
        mat = arr[0] if arr.ndim == 3 else arr
        if not np.allclose(mat[:3, :3], np.eye(3), atol=1e-9):
            return None  # not a pure translation; averaging would be meaningless
        return mat[:3, 3].copy()
    except Exception:
        return None


def _with_translation(param, translation: np.ndarray):
    """`param` rebuilt around `translation`, keeping its type/dims/coords."""
    updated = param.copy()
    buf = updated.values if hasattr(updated, "values") else updated
    mat = buf[0] if buf.ndim == 3 else buf
    mat[:3, 3] = translation
    return updated


@dataclass
class CarryResult:
    """What phase 2 did, in numbers the log can state plainly."""

    params: List = field(default_factory=list)
    core: Set[int] = field(default_factory=set)
    carried: List[int] = field(default_factory=list)
    orphans: List[int] = field(default_factory=list)
    rounds: int = 0
    max_step_um: float = 0.0

    def describe(self, n_tiles: int) -> str:
        parts = [
            f"{len(self.core)} of {n_tiles} tiles placed by registration",
            f"{len(self.carried)} carried by their neighbours "
            f"({self.rounds} pass(es), largest move {self.max_step_um:.1f} µm)",
        ]
        if self.orphans:
            parts.append(
                f"{len(self.orphans)} with no placed neighbour at all, moved "
                f"with the mosaic"
            )
        return "; ".join(parts)


def carry_deferred_tiles(
    params: Sequence,
    n_tiles: int,
    core: Sequence[int],
    neighbour_pairs: Sequence[Tuple[int, int]],
) -> CarryResult:
    """Place every tile outside `core` at its placed neighbours' mean correction.

    Spreads outward one ring per pass, so a tile two rings out is carried by
    tiles that were themselves carried — which is what keeps a wide empty rim
    continuous with the mosaic instead of stranding its outer edge.

    `neighbour_pairs` is GRID adjacency (whether or not a seam registered), so
    a tile with nothing to correlate against still has somewhere to take its
    placement from. Accepts ``(i, j)`` or ``(i, j, axis)``.
    """
    result = CarryResult(params=list(params), core=set(int(i) for i in core))
    if not params or n_tiles <= 0:
        return result

    adjacency: Dict[int, Set[int]] = {}
    for pair in neighbour_pairs or []:
        # border_qc.find_neighbor_pairs yields (i, j, axis); a plain (i, j) is
        # accepted too so tests and callers need not carry the axis.
        try:
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < n_tiles and 0 <= b < n_tiles:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    translations: Dict[int, np.ndarray] = {}
    for index in result.core:
        if index < len(result.params):
            value = _translation(result.params[index])
            if value is not None:
                translations[index] = value
    if not translations:
        return result  # nothing trustworthy to carry FROM

    placed = set(translations)
    pending = [i for i in range(n_tiles) if i not in placed]

    while pending:
        # Snapshot the frontier so the result does not depend on visit order:
        # every tile in this ring is carried by the PREVIOUS ring only.
        ring = [
            i for i in pending if adjacency.get(i, set()) & placed
        ]
        if not ring:
            break
        result.rounds += 1
        updates: Dict[int, np.ndarray] = {}
        for index in ring:
            sources = [translations[j] for j in adjacency[index] & placed]
            updates[index] = np.mean(np.stack(sources, axis=0), axis=0)
        for index, value in updates.items():
            if index < len(result.params):
                before = _translation(result.params[index])
                if before is not None:
                    result.max_step_um = max(
                        result.max_step_um, float(np.max(np.abs(value - before)))
                    )
                result.params[index] = _with_translation(
                    result.params[index], value
                )
            translations[index] = value
            placed.add(index)
            result.carried.append(index)
        pending = [i for i in pending if i not in placed]

    # Anything still unplaced touches nothing that registered. Move it with the
    # mosaic rather than leaving it at a stage position the rest abandoned.
    if pending:
        consensus = np.mean(
            np.stack([translations[i] for i in result.core if i in translations]),
            axis=0,
        )
        for index in pending:
            if index < len(result.params):
                result.params[index] = _with_translation(
                    result.params[index], consensus
                )
            result.orphans.append(index)
    return result


def tension_alerts(
    seams: Sequence,
    tiles: Sequence,
    tolerance_um: float = DEFAULT_TENSION_UM,
    label_fn: Optional[Callable[[object, int], str]] = None,
) -> List[str]:
    """One message per tile whose registered seams cannot all be satisfied.

    A tile with a single seam has nothing to disagree with, so it is never
    reported — the residual there is a property of that one measurement, not a
    conflict. Two or more, and a residual over tolerance means the solve had to
    choose: this is the L-shape case, where satisfying the left neighbour puts
    the bottom one out of register.

    Returns the messages rather than logging them, so the caller decides the
    level and the tests can read them.
    """
    label_fn = label_fn or tile_label
    per_tile: Dict[int, List[float]] = {}
    for seam in seams or []:
        if getattr(seam, "status", None) != "registered":
            continue
        residual = getattr(seam, "residual_px", None)  # micrometres, see module doc
        if residual is None:
            continue
        try:
            residual = float(residual)
        except (TypeError, ValueError):
            continue
        for index in (getattr(seam, "index_a", None), getattr(seam, "index_b", None)):
            if index is None:
                continue
            per_tile.setdefault(int(index), []).append(residual)

    messages: List[str] = []
    for index in sorted(per_tile):
        residuals = per_tile[index]
        if len(residuals) < 2:
            continue
        worst = max(residuals)
        if worst <= tolerance_um:
            continue
        tile = tiles[index] if 0 <= index < len(tiles) else None
        label = label_fn(tile, index) if tile is not None else f"tile {index}"
        messages.append(
            f"not able to resolve all overlaps for tile {label} — "
            f"{len(residuals)} registered seams, worst residual {worst:.1f} µm "
            f"(tolerance {tolerance_um:.1f} µm). Its neighbours disagree about "
            f"where it goes; the solve satisfied them as far as it could and "
            f"this tile carries the remainder."
        )
    return messages
