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
    # index -> (ring, [neighbour indices it was averaged from]). The ring says
    # how far from registered ground a tile sits, which is the honest measure of
    # how much its placement is worth.
    provenance: Dict[int, Tuple[int, List[int]]] = field(default_factory=dict)

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
            from_indices = sorted(adjacency[index] & placed)
            sources = [translations[j] for j in from_indices]
            updates[index] = np.mean(np.stack(sources, axis=0), axis=0)
            result.provenance[index] = (result.rounds, from_indices)
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
            result.provenance[index] = (0, [])
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
    """One message per tile whose overlaps could not all be honoured.

    The obvious test — "a registered seam with a big residual" — cannot fire.
    multiview-stitcher's global optimisation loops until
    ``max_residuals[-1] < abs_tol`` (global_optimization.py:423), dropping the
    worst edge each round, so EVERY surviving edge is under that tolerance by
    construction. Thresholding kept edges at abs_tol is thresholding at the
    number the solver already guaranteed. The first real run proved it: 11
    pruned seams, 1 implausible, and not one alert.

    Tension shows up as a seam that was MEASURED and then not used. A tile with
    at least one registered seam and at least one measured-but-unused seam is
    exactly "we know where this tile sits relative to that neighbour, and the
    placement we chose does not honour it" — the L-shape, in the data.

    The residual test is kept as a second trigger for solvers that do not prune
    to a tolerance (shortest_paths returns whatever the path gives), where it
    can still catch something.

    Returns the messages rather than logging them, so the caller picks the level
    and the tests can read them.
    """
    label_fn = label_fn or tile_label
    # Measured, then discarded by the solve. STATUS_PRUNED survived the quality
    # filter and lost to edge pruning; implausible_shift passed quality and was
    # refused on geometry. Both mean: this overlap was measurable and is not
    # reflected in where the tile ended up.
    unused_measured = ("pruned", "implausible_shift")

    # Counted separately from the residuals: a registered seam whose residual
    # the solver did not report still counts as an overlap that WAS honoured.
    kept: Dict[int, int] = {}
    residuals: Dict[int, List[float]] = {}
    dropped: Dict[int, int] = {}
    for seam in seams or []:
        status = getattr(seam, "status", None)
        indices = [
            index
            for index in (
                getattr(seam, "index_a", None), getattr(seam, "index_b", None)
            )
            if index is not None
        ]
        if status == "registered":
            residual = getattr(seam, "residual_px", None)  # µm, see module doc
            try:
                residual = float(residual) if residual is not None else None
            except (TypeError, ValueError):
                residual = None
            for index in indices:
                kept[int(index)] = kept.get(int(index), 0) + 1
                if residual is not None:
                    residuals.setdefault(int(index), []).append(residual)
        elif status in unused_measured:
            for index in indices:
                dropped[int(index)] = dropped.get(int(index), 0) + 1

    messages: List[str] = []
    for index in sorted(set(kept) | set(dropped)):
        n_kept = kept.get(index, 0)
        n_dropped = dropped.get(index, 0)
        seen = residuals.get(index, [])
        worst = max(seen) if seen else None

        by_drop = n_kept >= 1 and n_dropped >= 1
        by_residual = len(seen) >= 2 and worst is not None and worst > tolerance_um
        if not (by_drop or by_residual):
            continue

        tile = tiles[index] if 0 <= index < len(tiles) else None
        label = label_fn(tile, index) if tile is not None else f"tile {index}"
        detail = (
            f"{n_dropped} of its measured overlaps could not be honoured "
            f"alongside the {n_kept} that were"
            if by_drop
            else f"worst residual {worst:.1f} µm over {len(seen)} seams "
            f"(tolerance {tolerance_um:.1f} µm)"
        )
        messages.append(
            f"not able to resolve all overlaps for tile {label} — {detail}. "
            f"Its neighbours disagree about where it goes; the solve satisfied "
            f"them as far as it could and this tile carries the remainder."
        )
    return messages
