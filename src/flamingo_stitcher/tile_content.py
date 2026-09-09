"""Does a tile contain anything phase correlation can lock onto?

Registration does not need a tile to be BRIGHT, it needs the tile to have
STRUCTURE. Those are different questions, and on a real light-sheet sample they
give opposite answers. A fish in agarose in an FEP tube has at least four
distinct "backgrounds": air outside the tube, the tube wall, the agarose, and
the mounting medium. The agarose is well above the noise floor and completely
featureless, so any threshold on intensity calls it content and hands the
registration a tile with nothing to align. That is how a mosaic ends up with
tiles registered against smooth gel.

So the measure here is texture relative to noise, which is independent of
absolute brightness by construction:

    structure = std(smoothed) / std(raw)

Smoothing destroys noise and preserves real structure. A featureless region --
agarose, air, medium -- is a constant plus noise, so smoothing collapses its
variance and the ratio falls toward zero however bright the constant is. A
region with actual structure keeps most of its variance and the ratio stays
high. Uniform scaling of the data cancels in the ratio, so gain, exposure and
bit depth do not move it.

**What this measure does NOT tell you** is whether the structure is *useful*.
The FEP tube wall is a strong straight edge running through many tiles: it
scores highly here and is still a poor thing to register on, because a straight
edge is free to slide along its own length without changing the correlation
(the aperture problem). Catching that is the geometric shift bound's job, not
this one -- these are two different failures and each needs its own guard.

Kept dependency-light and computed on a downsampled level: this runs before
registration to decide what to register, so it must be cheap relative to what it
saves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Smoothing width, in downsampled voxels. Wide enough to flatten shot noise,
# narrow enough to leave cellular-scale structure standing.
#
# The threshold below is calibrated against THIS value. Changing it moves the
# featureless floor, so re-derive the threshold if you touch it.
_SMOOTH_SIGMA = 1.5

# Below this, a tile is treated as having nothing to register against.
#
# Measured rather than guessed. On synthetic phantoms at sigma=1.5:
#
#   featureless (constant + noise)     0.087 - 0.090
#     ...and that range is over levels 20 to 60000, i.e. a 3000x change in
#     brightness moves the score by 0.003. The ratio really is independent of
#     how bright the thing is, which is the property the agarose case needs.
#
#   real structure, by contrast-to-noise ratio:
#     CNR 60 -> 0.86    CNR 10 -> 0.61    CNR 5 -> 0.39
#     CNR 2.5 -> 0.21   CNR 1.5 -> 0.18   CNR 0.8 -> 0.11   CNR 0.33 -> 0.09
#
# So ~0.088 is the floor, and real sample sinks into it at CNR ~1 -- which is
# also the point below which there is nothing for phase correlation to find, so
# excluding it there is correct rather than merely tolerable.
#
# 0.15 sits at CNR ~1.2: clear of the floor, and it still keeps dim, noisy
# sample. That direction matters -- excluding a tile that DID have content
# throws away a real measurement, while including one that did not is a case the
# rest of the registration guard already handles. An earlier 0.35 excluded a
# dim-but-real tile at CNR 5, which is exactly the failure to avoid.
DEFAULT_MIN_STRUCTURE = 0.15

# Cap on the voxels actually measured. A structure ratio is a global statistic;
# it does not get meaningfully better with more than a few million samples, and
# this runs once per tile before anything else happens.
_MAX_SAMPLE_VOXELS = 8_000_000

# Planes to keep when a stack is too big to measure whole. Deriving the Z
# stride from the voxel budget alone is what collapsed a real 643 x 2048 x 2048
# tile to plane 0 (see _subsample) -- one plane already fills any sane budget.
_MIN_SAMPLE_PLANES = 16


# Lateral stride target for the per-plane profile. Every plane is kept (that
# is the axis being profiled); only the in-plane sampling is reduced.
_PROFILE_LATERAL = 128

# Planes added either side of the detected content span. The threshold clips the
# faint edges of a sample, and a shift can only be found where both tiles have
# data, so a little slack costs almost nothing and protects against cropping
# into the structure being measured.
_Z_MARGIN_PLANES = 8

# Do not bother cropping unless it removes a real share of the stack. Below this
# the read costs more than the crop saves, and a near-full crop is also the case
# where the profile is most likely to be wrong.
_MIN_Z_CROP_SAVING = 0.20

# The per-plane profile must separate content from background by at least this
# much before a crop is anything but a guess. Below it the stack either has
# content throughout or none at all, and both are left uncropped.
_MIN_PROFILE_SEPARATION = 0.08

# Where between the stack's own floor and its own peak to put the cut.
_PROFILE_BAND_FRACTION = 0.25


# Width of the edge band scored for seam gating, as a fraction of the tile.
# Deliberately >= a typical tile overlap: the band must CONTAIN the shared
# strip, and being too wide only makes the gate more permissive, which is the
# safe direction (attempting a seam costs time, wrongly skipping one costs a
# measurement that cannot be recovered).
DEFAULT_EDGE_FRACTION = 0.2


@dataclass
class EdgeContent:
    """Structure in each lateral edge band of a tile.

    A tile can have plenty of structure somewhere and nothing at all in the
    strip it shares with one particular neighbour -- the sample ends partway
    across the mosaic. The whole-tile score cannot see that, so it lets a seam
    through whose overlap is pure medium on one or both sides. This is what the
    seam gate reads.

    `None` for a band means "could not measure", which is treated as content --
    the same convention as `TileContent`, and for the same reason.
    """

    x_low: Optional[float] = None
    x_high: Optional[float] = None
    y_low: Optional[float] = None
    y_high: Optional[float] = None

    def at(self, axis: str, high: bool) -> Optional[float]:
        """Score for the `axis` band on the high or low side."""
        return getattr(self, f"{axis}_{'high' if high else 'low'}", None)

    def has_content(
        self, axis: str, high: bool, min_structure: float = DEFAULT_MIN_STRUCTURE
    ) -> bool:
        score = self.at(axis, high)
        return True if score is None else score >= float(min_structure)


@dataclass
class TileContent:
    """One tile's structure score and the verdict drawn from it."""

    index: int
    structure: Optional[float] = None
    has_content: bool = True
    note: str = ""
    edges: Optional[EdgeContent] = None

    @property
    def measured(self) -> bool:
        return self.structure is not None


def _subsample(volume: np.ndarray) -> np.ndarray:
    """A small view of `volume`, sampled without materializing the whole thing.

    Z is sampled by COUNT, never by a stride derived from the voxel budget.
    That stride collapsed a real 643 x 2048 x 2048 tile to a SINGLE plane --
    plane 0, because one 2048^2 plane already fills the budget on its own. In a
    light-sheet stack plane 0 is empty medium, so every score described the
    medium instead of the tile: on the 2026-09-08 7x7 that put all 49 tiles in a
    0.20-0.89 band just above the featureless floor, and the content gate was
    deciding on a plane containing nothing. Keeping _MIN_SAMPLE_PLANES spread
    across the stack costs a coarser lateral stride and answers the question
    that was actually asked.

    Slicing happens BEFORE any materialization, so a dask/memmap-backed tile
    yields ~30 MB rather than a 5.4 GB resident copy per tile (times the
    preprocess workers).
    """
    try:
        size = int(volume.size)
    except Exception:
        return volume
    try:
        if size <= _MAX_SAMPLE_VOXELS or volume.ndim != 3:
            return volume
        n_planes = int(volume.shape[0])
    except Exception:
        return volume
    keep = min(n_planes, _MIN_SAMPLE_PLANES)
    if keep < n_planes:
        idx = np.unique(np.linspace(0, n_planes - 1, keep).astype(int))
        try:
            out = volume[idx]
        except Exception:
            # Anything that cannot take a fancy index falls back to a stride
            # that still spans the stack.
            out = volume[:: max(1, n_planes // keep)]
    else:
        out = volume
    if out.size > _MAX_SAMPLE_VOXELS:
        lateral = int(np.ceil(np.sqrt(out.size / _MAX_SAMPLE_VOXELS)))
        out = out[:, ::lateral, ::lateral]
    return out


def _structure_ratio(data: np.ndarray) -> Optional[float]:
    """The ratio itself, on an array that is already resident and sampled.

    Split out so one read of a tile can answer several questions -- the whole
    tile AND each of its edges -- instead of re-reading it per question.
    """
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover - scipy is a hard dependency
        return None
    try:
        if data.size < 8:
            return None
        finite = np.isfinite(data)
        if not finite.all():
            if not finite.any():
                return None
            data = np.where(finite, data, np.nanmedian(data[finite]))
        raw = float(data.std())
        if not np.isfinite(raw) or raw <= 0.0:
            # A perfectly constant region. Genuinely no structure, and the ratio
            # is 0/0 -- report the verdict directly rather than divide.
            return 0.0
        smoothed = ndimage.gaussian_filter(data, sigma=_SMOOTH_SIGMA)
        score = float(smoothed.std()) / raw
        if not np.isfinite(score):
            return None
        return max(0.0, min(1.0, score))
    except Exception as exc:  # noqa: BLE001 - never fail a run over a heuristic
        logger.debug("Could not score structure: %s", exc)
        return None


def _resident_sample(volume) -> Optional[np.ndarray]:
    """`volume` subsampled and materialized as float32, or None."""
    try:
        return np.asarray(_subsample(volume), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not sample volume: %s", exc)
        return None


def structure_score(volume) -> Optional[float]:
    """`std(smoothed) / std(raw)` in [0, 1], or None if it cannot be measured.

    Returns None rather than a number whenever the answer would be meaningless
    -- an empty array, a perfectly flat one, non-finite data. A caller must not
    be able to mistake "could not measure" for "measured, and it is low".
    """
    data = _resident_sample(volume)
    if data is None:
        return None
    return _structure_ratio(data)


def _edges_from_sample(
    data: np.ndarray, fraction: float = DEFAULT_EDGE_FRACTION
) -> Optional[EdgeContent]:
    """Score the four lateral edge bands of an already-resident sample.

    Costs four gaussian filters over ~a fifth of an already-read sample each,
    so this rides along with the whole-tile score for no extra I/O -- which is
    the point. Probing each seam's overlap separately would read a comparable
    volume to the registration it is meant to save.
    """
    try:
        if data.ndim != 3:
            return None
        _, height, width = data.shape
        frac = min(0.5, max(0.01, float(fraction)))
        n_y = max(2, int(round(height * frac)))
        n_x = max(2, int(round(width * frac)))
        return EdgeContent(
            x_low=_structure_ratio(data[:, :, :n_x]),
            x_high=_structure_ratio(data[:, :, -n_x:]),
            y_low=_structure_ratio(data[:, :n_y, :]),
            y_high=_structure_ratio(data[:, -n_y:, :]),
        )
    except Exception as exc:  # noqa: BLE001 - a heuristic must not fail a run
        logger.debug("Could not score tile edges: %s", exc)
        return None


def edge_structure(
    volume, *, fraction: float = DEFAULT_EDGE_FRACTION
) -> Optional[EdgeContent]:
    """Edge-band structure for one tile, or None if it cannot be measured."""
    data = _resident_sample(volume)
    if data is None:
        return None
    return _edges_from_sample(data, fraction)


def score_tiles(
    volumes: Sequence,
    *,
    min_structure: float = DEFAULT_MIN_STRUCTURE,
    edge_fraction: Optional[float] = None,
) -> List[TileContent]:
    """Score every tile and mark the ones with nothing to register against.

    A tile that could not be measured is treated as HAVING content: the whole
    point of this gate is to remove tiles we are confident about, and an
    unmeasurable tile is the opposite of that.
    """
    results: List[TileContent] = []
    for index, volume in enumerate(volumes):
        # ONE read per tile, then every measure comes off that resident sample.
        data = _resident_sample(volume)
        score = None if data is None else _structure_ratio(data)
        edges = (
            _edges_from_sample(data, edge_fraction)
            if (edge_fraction is not None and data is not None)
            else None
        )
        if score is None:
            results.append(
                TileContent(
                    index=index,
                    structure=None,
                    has_content=True,
                    note="structure could not be measured; kept in the registration",
                    edges=edges,
                )
            )
            continue
        has_content = score >= float(min_structure)
        results.append(
            TileContent(
                index=index,
                structure=score,
                has_content=has_content,
                edges=edges,
                note=(
                    ""
                    if has_content
                    else (
                        f"structure {score:.2f} is below {float(min_structure):.2f}: "
                        f"smooth or empty (medium, gel or air), nothing for phase "
                        f"correlation to lock onto"
                    )
                ),
            )
        )
    return results


def plane_structure_profile(volume) -> Optional[np.ndarray]:
    """Per-plane structure score down Z, or None if it cannot be measured.

    Same ratio as :func:`structure_score`, computed plane by plane, so a light
    sheet stack whose sample occupies a slab in the middle shows as a bump with
    a noise floor either side. Strided laterally rather than in Z -- Z is the
    axis being profiled, so it is the one axis that must keep every sample.
    """
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover - scipy is a hard dependency
        return None
    try:
        data = np.asarray(volume)
        if data.ndim != 3 or data.shape[0] < 3:
            return None
        stride_y = max(1, data.shape[1] // _PROFILE_LATERAL)
        stride_x = max(1, data.shape[2] // _PROFILE_LATERAL)
        data = np.asarray(data[:, ::stride_y, ::stride_x], dtype=np.float32)
        if data.shape[1] < 4 or data.shape[2] < 4:
            return None
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=float(np.nanmedian(data)))
        raw = data.std(axis=(1, 2))
        smoothed = ndimage.gaussian_filter(
            data, sigma=(0.0, _SMOOTH_SIGMA, _SMOOTH_SIGMA)
        ).std(axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            profile = np.where(raw > 0, smoothed / raw, 0.0)
        return np.clip(np.nan_to_num(profile, nan=0.0), 0.0, 1.0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not profile tile structure down Z: %s", exc)
        return None


def content_z_range(
    volume,
    *,
    min_structure: float = DEFAULT_MIN_STRUCTURE,
    margin_planes: int = _Z_MARGIN_PLANES,
) -> Optional[tuple]:
    """``(z0, z1)`` planes worth registering on, or None to use the whole stack.

    Returns None -- meaning "do not crop" -- whenever cropping would be a guess
    or would not pay: no measurable profile, no plane above the threshold, or a
    span that already covers most of the stack.

    **This does not make the measured shift better.** Measured on a two-tile
    phantom with independent per-tile noise, content occupying 25% / 10% / 5% of
    the stack: the recovered shift was exactly right in all three cases, and it
    was the seam QUALITY that fell (0.72 / 0.52 / 0.43) as empty planes diluted
    the rank correlation. At 5% that lands on a 0.4 threshold and the seam is
    thrown away for a reason unrelated to its alignment. Cropping restores it
    (0.99 / 0.91 / 0.76) and reads a fraction of the planes. Quality and cost,
    not accuracy -- do not re-justify it as accuracy later.
    """
    profile = plane_structure_profile(volume)
    if profile is None or not len(profile):
        return None

    # Calibrate against THIS stack rather than against a constant.
    #
    # `DEFAULT_MIN_STRUCTURE` is calibrated for the 3-D score, where smoothing
    # runs across Z as well and drives the featureless floor to ~0.088. The
    # per-plane profile smooths only in Y/X, so it removes less noise and its
    # floor sits around 0.195 — applying the 3-D number here passes every plane
    # and silently disables the crop. Rather than carry a second constant that
    # has to be re-derived whenever the kernel changes, take the floor from the
    # stack's own empty planes, which is what they are for.
    #
    # This is the same idea as ASHLAR's null distribution, done locally: measure
    # what "no information" scores on this data, then require better than that.
    floor = float(np.percentile(profile, 20))
    peak = float(np.percentile(profile, 99))
    separation = peak - floor
    if separation < _MIN_PROFILE_SEPARATION:
        # No distinct band: either the whole stack has content or none of it
        # does. Both are cases where cropping would be a guess.
        return None
    threshold = max(floor + _PROFILE_BAND_FRACTION * separation, float(min_structure))
    above = np.flatnonzero(profile >= threshold)
    if not len(above):
        return None
    n_planes = int(len(profile))
    z0 = max(0, int(above[0]) - int(margin_planes))
    z1 = min(n_planes, int(above[-1]) + 1 + int(margin_planes))
    if z1 - z0 < 4:
        return None
    if (z1 - z0) > (1.0 - _MIN_Z_CROP_SAVING) * n_planes:
        return None  # keeps nearly everything; not worth the bookkeeping
    return z0, z1


def describe_z_crop(ranges: Sequence, n_planes: int) -> str:
    """One line for the log about what the Z crop will do."""
    cropped = [r for r in ranges if r is not None]
    if not cropped:
        return f"no tile has a narrow enough content band to crop ({n_planes} planes)"
    spans = [z1 - z0 for z0, z1 in cropped]
    return (
        f"{len(cropped)} of {len(ranges)} tiles register on a Z sub-range: "
        f"{min(spans)}-{max(spans)} of {n_planes} planes "
        f"({100.0 * float(np.mean(spans)) / max(1, n_planes):.0f}% on average). "
        f"Empty planes dilute the seam quality score without improving the "
        f"shift, so this is about which seams survive the filter, not accuracy."
    )


def describe(results: Sequence[TileContent]) -> str:
    """One line for the log: how many, and the spread that produced the split."""
    if not results:
        return "no tiles to score"
    scored = [r.structure for r in results if r.measured]
    excluded = [r for r in results if not r.has_content]
    if not scored:
        return f"structure could not be measured on any of {len(results)} tiles"
    spread = (
        f"structure {min(scored):.2f}-{max(scored):.2f} "
        f"(median {float(np.median(scored)):.2f})"
    )
    if not excluded:
        return f"all {len(results)} tiles have structure to register on; {spread}"
    return (
        f"{len(excluded)} of {len(results)} tiles have nothing to register "
        f"against and were left at their stage position; {spread}"
    )
