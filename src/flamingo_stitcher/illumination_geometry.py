"""Measure which illumination side lights which end of the frame.

``split`` and ``blend`` illumination fusion each keep one sheet's voxels over
one half of the frame. Which half belongs to which sheet is, until now, a
number the user types into Options (``illumination_low_side``). Nothing checks
it, and getting it backwards is silent: the output stays smooth and plausible
because every voxel is still real data -- it is just the FAR half of each
sheet, the blurred, attenuated end that the split exists to discard. The
2026-09-18 7x7 run spent 9h21m producing exactly that, and the first sign of
trouble was the stitched image.

This module answers the question from the data instead. For each sheet it
measures in-focus detail as a function of position along the illumination
axis; the sheet that is sharper at the low end of the axis is the one lighting
it. The comparison is DIFFERENTIAL -- sheet A's tilt against sheet B's tilt --
so a sample that genuinely has more structure at one end biases both sheets
equally and cancels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Difference-of-Gaussians band used as the focus measure. The inner sigma
# suppresses pixel noise (a dim, far region is noisy, and raw high-frequency
# energy would read that as "sharp"); the outer one removes the illumination
# envelope and the sample's own large-scale shape. What is left is the
# mid-frequency detail that defocus actually destroys.
_DOG_INNER = 1.0
_DOG_OUTER = 4.0

# Bands along the illumination axis. The outermost band on each end is the
# comparison; the middle is ignored because that is where the two sheets cross
# over and neither is clearly better.
_N_BANDS = 6
_EDGE_BANDS = 2

# Below this separation between the two sheets' tilts the measurement is not
# saying anything: the sample is too uniform, or the sheets genuinely overlap.
_MIN_MARGIN = 0.04


@dataclass(frozen=True)
class IlluminationGeometry:
    """What the data says about the two sheets."""

    low_side: Optional[int]
    """Illumination side lighting the LOW end of the axis, or None if unclear."""

    margin: float
    """Separation between the sheets' focus tilts. Higher is more certain."""

    tilts: Dict[int, float]
    """Per-side focus tilt: positive means sharper at the low end."""

    axis: int

    @property
    def confident(self) -> bool:
        return self.low_side is not None and self.margin >= _MIN_MARGIN

    def describe(self) -> str:
        parts = ", ".join(
            f"side {s} tilt {t:+.3f}" for s, t in sorted(self.tilts.items())
        )
        if self.low_side is None:
            return (
                f"illumination geometry undetermined ({parts}; margin "
                f"{self.margin:.3f} < {_MIN_MARGIN})"
            )
        return (
            f"side {self.low_side} lights the LOW end of axis {self.axis} "
            f"({parts}; margin {self.margin:.3f})"
        )


def _focus_profile(data: np.ndarray, axis: int) -> Optional[np.ndarray]:
    """Mean in-focus detail per band along `axis`, normalised by brightness.

    Normalising by the band's own mean intensity matters: the near end of a
    sheet is both brighter AND sharper, and without the division the measure
    would mostly be reporting the illumination envelope, which is the one thing
    a flat-field correction may already have removed.
    """
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover - scipy is a hard dependency
        return None
    try:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim < 2 or arr.shape[axis] < _N_BANDS * 4:
            return None
        finite = np.isfinite(arr)
        if not finite.all():
            if not finite.any():
                return None
            arr = np.where(finite, arr, float(np.median(arr[finite])))

        # Filter only in the lateral plane; smoothing across Z would mix
        # planes that are physically far apart at a 10 um step.
        sigma = [0.0] * arr.ndim
        lateral = [d for d in range(arr.ndim) if d != 0] if arr.ndim == 3 else \
            list(range(arr.ndim))
        inner = list(sigma)
        outer = list(sigma)
        for d in lateral:
            inner[d] = _DOG_INNER
            outer[d] = _DOG_OUTER
        detail = ndimage.gaussian_filter(arr, inner) - ndimage.gaussian_filter(
            arr, outer
        )

        n = arr.shape[axis]
        edges = np.linspace(0, n, _N_BANDS + 1).astype(int)
        out = np.empty(_N_BANDS, dtype=np.float64)
        for b in range(_N_BANDS):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(edges[b], edges[b + 1])
            band_detail = np.abs(detail[tuple(sl)])
            band_level = float(np.mean(arr[tuple(sl)]))
            if not np.isfinite(band_level) or band_level <= 0.0:
                out[b] = 0.0
            else:
                out[b] = float(np.mean(band_detail)) / band_level
        if not np.isfinite(out).all():
            return None
        return out
    except Exception as exc:  # noqa: BLE001 - never fail a run over a heuristic
        logger.debug("Could not profile illumination focus: %s", exc)
        return None


def measure(
    volumes: Dict[int, np.ndarray], axis: int = -1
) -> Optional[IlluminationGeometry]:
    """Decide which of two illumination sides lights the low end of `axis`.

    `volumes` maps illumination side -> tile volume (already subsampled by the
    caller; this does no sampling of its own so it can be handed the same small
    array the content scorer already read).

    Returns None when there are not exactly two sides or the profiles could not
    be measured. A returned geometry may still be inconclusive -- check
    ``confident`` before acting on ``low_side``.
    """
    from flamingo_stitcher.tile_content import _resident_sample

    sides = sorted(volumes.keys())
    if len(sides) != 2:
        return None

    tilts: Dict[int, float] = {}
    for side in sides:
        sample = _resident_sample(volumes[side])
        if sample is None:
            return None
        profile = _focus_profile(sample, axis)
        if profile is None:
            return None
        low = float(np.mean(profile[:_EDGE_BANDS]))
        high = float(np.mean(profile[-_EDGE_BANDS:]))
        total = low + high
        if total <= 0.0:
            return None
        # Normalised so the tilt is a share of this sheet's own detail, not an
        # absolute that the brighter sheet would win on size alone.
        tilts[side] = (low - high) / total

    a, b = sides
    margin = abs(tilts[a] - tilts[b])
    low_side = a if tilts[a] > tilts[b] else b
    if margin < _MIN_MARGIN:
        low_side = None
    return IlluminationGeometry(
        low_side=low_side, margin=margin, tilts=tilts, axis=axis
    )


@dataclass(frozen=True)
class AcquisitionGeometry:
    """What a whole acquisition says, pooled over several tiles."""

    low_side: Optional[int]
    votes: Dict[int, int]
    rows: list
    """(tile name, channel, low_side or None, margin, tilts) per probed tile."""

    @property
    def agreed(self) -> int:
        return self.votes.get(self.low_side, 0) if self.low_side is not None else 0

    @property
    def total(self) -> int:
        return sum(self.votes.values())

    @property
    def split(self) -> bool:
        """True when probed tiles disagreed — which should not happen on one
        acquisition, and means the illumination axis or the tile orientation is
        wrong before split/blend can mean anything."""
        return len(self.votes) > 1

    def summary(self) -> str:
        if self.low_side is None:
            return (
                "No tile gave a confident answer. The sample may be too uniform, "
                "or the two sheets may genuinely look alike. Open one raw file "
                "from each side and look at which half of the frame is in focus."
            )
        text = (
            f"Side {self.low_side} lights the LOW end of the frame "
            f"({self.agreed}/{self.total} tiles agree)."
        )
        if self.split:
            text += (
                f"\n\nNOTE: tiles disagreed ({dict(self.votes)}). That should not "
                f"happen on one acquisition — check the tile orientation and the "
                f"illumination axis before trusting split or blend."
            )
        return text


def measure_acquisition(
    acquisition_dir, max_tiles: int = 6, axis: int = -1
) -> AcquisitionGeometry:
    """Probe up to `max_tiles` of an acquisition and pool the answers.

    Shared by the CLI and the Options tab's "Measure from data" button so the
    two cannot drift. Reads only a subsampled slice of each raw file.
    """
    from flamingo_stitcher.pipeline import discover_tiles, load_tile_volume

    tiles = discover_tiles(str(acquisition_dir))
    votes: Dict[int, int] = {}
    rows = []
    for tile in (tiles or [])[:max_tiles]:
        for ch_id, illum_files in sorted(tile.raw_files.items()):
            if len(illum_files) < 2:
                continue
            volumes = {
                side: load_tile_volume(
                    path, tile.n_planes, tile.frame_width, tile.frame_height
                )
                for side, path in sorted(illum_files.items())[:2]
            }
            g = measure(volumes, axis=axis)
            if g is None:
                continue
            name = getattr(tile.folder, "name", str(tile.folder))
            rows.append((name, ch_id, g.low_side, g.margin, dict(g.tilts)))
            if g.confident:
                votes[g.low_side] = votes.get(g.low_side, 0) + 1
    winner = max(votes, key=votes.get) if votes else None
    return AcquisitionGeometry(low_side=winner, votes=votes, rows=rows)


def main(argv=None) -> int:
    """``python -m flamingo_stitcher.illumination_geometry <acquisition_dir>``

    Reports, per tile, which illumination side lights the low end of the frame,
    so the Options setting can be checked in seconds instead of inferred from a
    finished stitch.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Measure which illumination side lights the low end of the frame. "
            "Use the answer for 'Illumination side lighting the low end of the "
            "frame' in the Options tab."
        )
    )
    parser.add_argument("acquisition_dir", help="Acquisition folder to probe")
    parser.add_argument(
        "--tiles", type=int, default=6, help="How many tiles to probe (default 6)"
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=-1,
        choices=(-1, -2),
        help="Illumination axis: -1 = across X/columns (default), -2 = across Y",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = measure_acquisition(
        args.acquisition_dir, max_tiles=args.tiles, axis=args.axis
    )
    if not result.rows:
        print(f"No two-sided tiles found in {args.acquisition_dir}")
        return 1

    print(f"probing up to {args.tiles} tiles on axis {args.axis}\n")
    print(f"{'tile':<34}{'ch':>4}{'says':>8}{'margin':>9}  tilts")
    print("-" * 78)
    for name, ch_id, low_side, margin, tilts in result.rows:
        says = "unclear" if low_side is None else str(low_side)
        tilt = " ".join(f"{k}:{v:+.3f}" for k, v in sorted(tilts.items()))
        print(f"{name[:33]:<34}{ch_id:>4}{says:>8}{margin:>9.3f}  {tilt}")

    print()
    print(result.summary())
    if result.low_side is None:
        return 2
    print(
        f"\nSet 'Illumination side lighting the LOW end of the frame' to "
        f"{result.low_side} in the Options tab."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
