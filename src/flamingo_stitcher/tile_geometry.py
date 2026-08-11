"""How far apart the tiles are, and therefore how much they share.

Every stage-position question in this package reduces to the same two lines:
take the distinct positions along an axis, difference them, compare the step to
what one frame covers. That calculation had grown four separate homes --
``pipeline._clamp_registration_shifts`` (min gap, effective µm),
``pipeline._detect_tile_spacing_gaps`` (median step, raw mm),
``border_qc._overlap_px`` (per pair, downsampled px) and ``border_qc._min_pitch``
(grouping tolerance) -- each with its own rounding and its own choice of
statistic. Adding a fifth for the registration overlap gate is how a repo ends
up fixing the same bug three times and shipping it a fourth.

So the arithmetic lives here once, in µm, and each caller says which statistic
it wants rather than reimplementing one:

  * **median** pitch for "how are these tiles laid out" -- one stray or
    duplicated stage position cannot drag it.
  * **min** pitch for a grouping *tolerance*, where the tightest spacing is the
    thing that must not be split.

Deliberately free of I/O and of any package import: tiles are duck-typed on
``x_mm`` / ``y_mm`` / ``z_min_mm``, so tests pass ``SimpleNamespace``, and
``border_qc`` (which imports nothing from the package) can use it without
creating a cycle back through ``pipeline``.

Signs matter. ``overlap_um`` is ``extent - pitch`` and is returned **signed**: a
negative value is a gap between tiles -- missing acquired data -- which is a
different problem from a small overlap, and collapsing the two to zero would
hide it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

__all__ = [
    "AxisOverlap",
    "AXIS_ATTRS",
    "axis_overlap",
    "distinct_positions_um",
    "grid_overlap",
    "median_pitch_um",
    "min_pitch_um",
    "pair_overlap_um",
]

# Which tile attribute carries each axis's stage position. Z is the *start* of
# the sweep; a tile's Z extent comes from its plane count, not from this field.
AXIS_ATTRS = {"x": "x_mm", "y": "y_mm", "z": "z_min_mm"}

# Stage positions are quantized (folder names carry 0.01 mm), and floats
# accumulate noise through the mm->µm conversions. Fold anything below 0.1 µm
# together before differencing, or one tile's rounding becomes a spurious extra
# row with a ~0 pitch.
_POSITION_ROUND_MM = 4
_MIN_GAP_MM = 1e-6


@dataclass(frozen=True)
class AxisOverlap:
    """What one axis of a tile grid looks like, in µm.

    ``pitch_um``/``overlap_um``/``fraction`` are ``None`` -- not zero -- when the
    axis has fewer than two distinct positions, because a single column has no
    spacing to measure. A caller that needs a number must supply its own
    fallback and say so; silently substituting 0 reads as "the tiles are exactly
    on top of each other", which is the opposite of "unknown".
    """

    axis: str
    extent_um: float
    n_positions: int
    pitch_um: Optional[float] = None
    overlap_um: Optional[float] = None
    fraction: Optional[float] = None
    statistic: str = "median"

    @property
    def is_gapped(self) -> bool:
        """True when tiles are stepped farther apart than one frame covers."""
        return self.overlap_um is not None and self.overlap_um < 0.0

    def describe(self) -> str:
        """One short phrase for a log line or a report cell."""
        if self.fraction is None:
            return f"{self.axis}: single position"
        return (
            f"{self.axis}: pitch {self.pitch_um:.1f} µm, "
            f"overlap {self.overlap_um:.1f} µm ({self.fraction * 100:.1f}%)"
        )


def _positions_mm(tiles: Sequence, axis: str) -> List[float]:
    attr = AXIS_ATTRS[axis]
    out: List[float] = []
    for tile in tiles:
        value = getattr(tile, attr, None)
        if value is None:
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def distinct_positions_um(tiles: Sequence, axis: str) -> List[float]:
    """Sorted distinct stage positions along `axis`, in µm."""
    rounded = sorted({round(p, _POSITION_ROUND_MM) for p in _positions_mm(tiles, axis)})
    return [p * 1000.0 for p in rounded]


def _gaps_um(tiles: Sequence, axis: str) -> List[float]:
    uniq = sorted({round(p, _POSITION_ROUND_MM) for p in _positions_mm(tiles, axis)})
    return [
        (b - a) * 1000.0 for a, b in zip(uniq, uniq[1:]) if (b - a) > _MIN_GAP_MM
    ]


def min_pitch_um(tiles: Sequence, axis: str) -> Optional[float]:
    """Tightest step between distinct positions, µm. None if fewer than two.

    Use this for a grouping *tolerance* -- a fraction of the tightest spacing is
    the largest wobble that still cannot merge two genuine rows.
    """
    gaps = _gaps_um(tiles, axis)
    return min(gaps) if gaps else None


def median_pitch_um(tiles: Sequence, axis: str) -> Optional[float]:
    """Typical step between distinct positions, µm. None if fewer than two.

    Use this to characterise the layout. Unlike the min, a single duplicated or
    drifted stage position cannot drag it -- and a duplicated position is
    exactly what a partially re-acquired tile set produces.
    """
    gaps = _gaps_um(tiles, axis)
    if not gaps:
        return None
    gaps.sort()
    return gaps[len(gaps) // 2]


def axis_overlap(
    tiles: Sequence,
    axis: str,
    extent_um: float,
    *,
    statistic: str = "median",
) -> AxisOverlap:
    """Pitch, overlap and overlap fraction along one axis.

    Args:
        tiles: duck-typed on ``x_mm`` / ``y_mm`` / ``z_min_mm``.
        axis: ``"x"``, ``"y"`` or ``"z"``.
        extent_um: what ONE tile covers along this axis, in µm. The caller owns
            this because the right frame differs by use: raw camera pixels ×
            pixel size for an acquisition-geometry warning, processed shape ×
            effective voxel size for anything downstream of the downsample.
        statistic: ``"median"`` (default) or ``"min"``.
    """
    if axis not in AXIS_ATTRS:
        raise ValueError(f"axis must be one of {sorted(AXIS_ATTRS)}, got {axis!r}")
    if statistic not in ("median", "min"):
        raise ValueError(f"statistic must be 'median' or 'min', got {statistic!r}")

    extent_um = float(extent_um)
    n_positions = len(distinct_positions_um(tiles, axis))
    pitch = (
        median_pitch_um(tiles, axis)
        if statistic == "median"
        else min_pitch_um(tiles, axis)
    )
    if pitch is None or extent_um <= 0.0:
        return AxisOverlap(
            axis=axis,
            extent_um=extent_um,
            n_positions=n_positions,
            statistic=statistic,
        )
    overlap = extent_um - pitch
    return AxisOverlap(
        axis=axis,
        extent_um=extent_um,
        n_positions=n_positions,
        pitch_um=pitch,
        overlap_um=overlap,
        fraction=overlap / extent_um,
        statistic=statistic,
    )


def grid_overlap(
    tiles: Sequence,
    *,
    extent_x_um: float,
    extent_y_um: float,
    extent_z_um: Optional[float] = None,
    statistic: str = "median",
) -> Dict[str, AxisOverlap]:
    """`axis_overlap` for each axis, keyed ``"x"`` / ``"y"`` / ``"z"``.

    Z is included only when ``extent_z_um`` is given. On an XY mosaic every tile
    spans the same Z range, so the Z entry describes tile *depth staggering*
    (what Collect Tiles produces), not the shared slab an XY seam registers on.
    """
    out = {
        "x": axis_overlap(tiles, "x", extent_x_um, statistic=statistic),
        "y": axis_overlap(tiles, "y", extent_y_um, statistic=statistic),
    }
    if extent_z_um is not None:
        out["z"] = axis_overlap(tiles, "z", extent_z_um, statistic=statistic)
    return out


def pair_overlap_um(tile_a, tile_b, axis: str, extent_um: float) -> float:
    """Overlap between two specific tiles along `axis`, µm (signed).

    The per-pair counterpart of `axis_overlap`: uses these two tiles' actual
    separation rather than the grid's typical pitch, which is what a seam
    measurement needs when the grid is irregular or a tile is displaced.
    """
    if axis not in AXIS_ATTRS:
        raise ValueError(f"axis must be one of {sorted(AXIS_ATTRS)}, got {axis!r}")
    attr = AXIS_ATTRS[axis]
    pitch_um = abs(float(getattr(tile_b, attr)) - float(getattr(tile_a, attr))) * 1000.0
    return float(extent_um) - pitch_um
