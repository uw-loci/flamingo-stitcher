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
_MAX_SAMPLE_VOXELS = 4_000_000


@dataclass
class TileContent:
    """One tile's structure score and the verdict drawn from it."""

    index: int
    structure: Optional[float] = None
    has_content: bool = True
    note: str = ""

    @property
    def measured(self) -> bool:
        return self.structure is not None


def _subsample(volume: np.ndarray) -> np.ndarray:
    """A strided view small enough to measure quickly, without loading more."""
    try:
        size = int(volume.size)
    except Exception:
        return volume
    if size <= _MAX_SAMPLE_VOXELS or volume.ndim != 3:
        return volume
    # Stride Z first: planes are the cheapest axis to skip and the one with the
    # most redundancy in a light-sheet stack.
    step_z = max(1, int(np.ceil(size / _MAX_SAMPLE_VOXELS)))
    out = volume[::step_z]
    if out.size > _MAX_SAMPLE_VOXELS:
        lateral = int(np.ceil(np.sqrt(out.size / _MAX_SAMPLE_VOXELS)))
        out = out[:, ::lateral, ::lateral]
    return out


def structure_score(volume) -> Optional[float]:
    """`std(smoothed) / std(raw)` in [0, 1], or None if it cannot be measured.

    Returns None rather than a number whenever the answer would be meaningless
    -- an empty array, a perfectly flat one, non-finite data. A caller must not
    be able to mistake "could not measure" for "measured, and it is low".
    """
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover - scipy is a hard dependency
        return None
    try:
        data = np.asarray(_subsample(np.asarray(volume)), dtype=np.float32)
        if data.size < 8:
            return None
        finite = np.isfinite(data)
        if not finite.all():
            if not finite.any():
                return None
            data = np.where(finite, data, np.nanmedian(data[finite]))
        raw = float(data.std())
        if not np.isfinite(raw) or raw <= 0.0:
            # A perfectly constant tile. Genuinely no structure, and the ratio
            # is 0/0 -- report the verdict directly rather than divide.
            return 0.0
        smoothed = ndimage.gaussian_filter(data, sigma=_SMOOTH_SIGMA)
        score = float(smoothed.std()) / raw
        if not np.isfinite(score):
            return None
        return max(0.0, min(1.0, score))
    except Exception as exc:  # noqa: BLE001 - never fail a run over a heuristic
        logger.debug("Could not score tile structure: %s", exc)
        return None


def score_tiles(
    volumes: Sequence, *, min_structure: float = DEFAULT_MIN_STRUCTURE
) -> List[TileContent]:
    """Score every tile and mark the ones with nothing to register against.

    A tile that could not be measured is treated as HAVING content: the whole
    point of this gate is to remove tiles we are confident about, and an
    unmeasurable tile is the opposite of that.
    """
    results: List[TileContent] = []
    for index, volume in enumerate(volumes):
        score = structure_score(volume)
        if score is None:
            results.append(
                TileContent(
                    index=index,
                    structure=None,
                    has_content=True,
                    note="structure could not be measured; kept in the registration",
                )
            )
            continue
        has_content = score >= float(min_structure)
        results.append(
            TileContent(
                index=index,
                structure=score,
                has_content=has_content,
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
