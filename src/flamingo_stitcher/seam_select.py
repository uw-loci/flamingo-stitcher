"""Pick each seam's shift by whether the CELLS line up, not the brightness.

Phase correlation returns a peak position, and a peak position is ambiguous:
the shift could be +s or -s on each axis, or s minus the overlap width (the
correlation is circular). multiview-stitcher's ``phase_correlation_registration``
expands every peak into those candidates -- correctly -- and then picks one by
structural similarity (SSIM) over the overlap. SSIM is dominated by the smooth
brightness of the tissue, which barely changes between candidates, so on a
punctate sample it picks almost at random.

Measured on four Ket_Trial1 seam strips (2026-10-22, `tools/export_seam_strips.py`),
running multiview-stitcher's own function and logging every candidate:

  seam               MVS's pick (by SSIM)            right candidate
  X001 Y002|Y001     dx -111 um, cells r 0.00        dx +111 um, cells r 0.40
  X005 Y003|Y002     dy +167 um, cells r -0.04       dy -58 um, cells r 0.66
  X003 Y002|Y001     dy -56 um, cells r 0.23         dy -76 um, cells r 0.33
  X002|X001 Y003     right (control), cells r 0.67   same

The peak was right every time; the tie-break was wrong on three of four. The
dy +167 um pick is the wrap-around alias of -58 um (the two differ by exactly
the 102 px overlap) and is what opened a visible gap between two tiles.

"Cells r" is the Pearson correlation of the two overlaps after a band-pass
(difference of Gaussians) that keeps cell-sized detail and removes the smooth
brightness. Wrong candidates score ~0 on it; right ones 0.3-0.7.

This module keeps multiview-stitcher's candidate list exactly, scores each
candidate by cells r, and returns multiview-stitcher's own quality (Spearman)
for the chosen one -- so the quality threshold and the global solve's weights
mean what they always meant. A seam whose best candidate still does not line
the cells up is recorded so the caller can drop it before the solve.

Qt-free and multiview-stitcher-free at import, so it tests anywhere.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Band-pass sigmas in micrometres (Z, Y, X): keep detail between roughly a cell
# and a few cells across, drop the brightness gradient. Chosen on the
# Ket_Trial1 strips (2026-10-22): in voxels of 10 um Z x 2.1 um XY they are
# (0.5, 1, 1) and (1.5, 4, 4), which separated right from wrong candidates on
# all four seams. Converted to voxels per run from the registration voxel size.
BAND_INNER_UM: Tuple[float, float, float] = (5.0, 2.0, 2.0)
BAND_OUTER_UM: Tuple[float, float, float] = (15.0, 8.0, 8.0)

# Below this, the outer sigma is under 2 voxels in XY: cells are not resolved at
# the registration resolution and the band-pass measures noise, so selection
# falls back to multiview-stitcher's SSIM. 8 um / 2 voxels = 4 um voxels.
MIN_OUTER_SIGMA_PX = 2.0

# multiview-stitcher skips a candidate whose shifted image overlaps less than
# 10% of the moving image (registration.py, phase_correlation_registration).
MIN_CANDIDATE_OVERLAP = 0.1


def band_sigmas_px(voxel_um: Dict[str, float]) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    """Band-pass sigmas in registration voxels, or None when cells are not
    resolved at this voxel size (selection then stays with SSIM)."""
    try:
        vz, vy, vx = (float(voxel_um[k]) for k in ("z", "y", "x"))
    except (KeyError, TypeError, ValueError):
        return None
    if min(vz, vy, vx) <= 0:
        return None
    inner = tuple(u / v for u, v in zip(BAND_INNER_UM, (vz, vy, vx)))
    outer = tuple(u / v for u, v in zip(BAND_OUTER_UM, (vz, vy, vx)))
    if min(outer[1], outer[2]) < MIN_OUTER_SIGMA_PX:
        return None
    return inner, outer


@dataclass
class SeamChoice:
    """What the selection saw on one seam."""

    shift_px: Tuple[float, ...]  # MVS convention: moving(x + t) ~ fixed(x)
    quality: float  # multiview-stitcher's Spearman quality of the chosen shift
    cell_r: float  # band-pass Pearson r at the chosen shift
    n_candidates: int


@dataclass
class SeamChoiceLog:
    """Thread-safe record of every seam's choice, for the report and the gate.

    multiview-stitcher keeps only ``affine_matrix`` and ``quality`` from the
    pairwise function, and passes the pairwise function pixel-space crops with
    no tile identity. The quality it stores on the edge is the float returned
    here, unchanged, so the caller matches an edge to its choice by quality.
    """

    choices: List[SeamChoice] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, choice: SeamChoice) -> None:
        with self._lock:
            self.choices.append(choice)

    def by_quality(self, quality) -> Optional[SeamChoice]:
        try:
            q = float(np.asarray(quality).ravel()[0])
        except Exception:
            return None
        if not np.isfinite(q):
            return None
        with self._lock:
            hits = [c for c in self.choices if c.quality == q]
        return hits[0] if len(hits) == 1 else None


def _candidates(shift_candidates: Sequence[np.ndarray], shape: Sequence[int]) -> List[Tuple[float, ...]]:
    """multiview-stitcher's expansion of each peak into sign and wrap variants,
    kept verbatim (registration.py, phase_correlation_registration) so the
    candidate list is the same one SSIM chose from."""
    ndim = len(shape)
    max_shift = max(shape)
    out: List[Tuple[float, ...]] = []
    for sc in shift_candidates:
        for s in np.ndindex(tuple(1 if sc[d] == 0 else 4 for d in range(ndim))):
            t = []
            for d in range(ndim):
                if s[d] == 0:
                    t.append(sc[d])
                elif s[d] == 1:
                    t.append(-sc[d])
                elif s[d] == 2:
                    t.append(-(sc[d] - shape[d]))
                else:
                    t.append(-sc[d] - shape[d])
            if np.max(np.abs(t)) < max_shift:
                out.append(tuple(float(v) for v in t))
    return out


def _overlap_slices(t, shape):
    """Slices of fixed and moving that coincide when moving(x + t) ~ fixed(x),
    t rounded to whole voxels; None if the overlap is empty."""
    sf, sm = [], []
    for k, n in zip((int(round(v)) for v in t), shape):
        if abs(k) >= n:
            return None
        if k >= 0:
            sf.append(slice(0, n - k)); sm.append(slice(k, n))
        else:
            sf.append(slice(-k, n)); sm.append(slice(0, n + k))
    return tuple(sf), tuple(sm)


def cell_r(f0: np.ndarray, f1: np.ndarray, v0: np.ndarray, v1: np.ndarray, t) -> float:
    """Pearson r of band-passed fixed and moving where both are valid, after
    shifting moving by t. NaN when too little overlaps to judge."""
    sl = _overlap_slices(t, f0.shape)
    if sl is None:
        return float("nan")
    sf, sm = sl
    if np.prod([s.stop - s.start for s in sf]) < MIN_CANDIDATE_OVERLAP * f0.size:
        return float("nan")
    m = v0[sf] & v1[sm]
    n = int(m.sum())
    if n < 1000:
        return float("nan")
    a = f0[sf][m].astype(np.float64)
    b = f1[sm][m].astype(np.float64)
    a -= a.mean(); b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _band_pass(im: np.ndarray, valid: np.ndarray, inner, outer) -> np.ndarray:
    from scipy import ndimage

    x = np.where(valid, im, np.float32(np.nanmean(im) if valid.any() else 0)).astype(np.float32)
    out = ndimage.gaussian_filter(x, inner)
    out -= ndimage.gaussian_filter(x, outer)
    return out


def cell_matched_registration(
    fixed_data,
    moving_data,
    band_sigmas_px=None,
    choice_log: Optional[SeamChoiceLog] = None,
    **skimage_phase_corr_kwargs,
):
    """Drop-in replacement for multiview-stitcher's ``phase_correlation_registration``.

    Same peak search, same candidate expansion; the candidate is chosen by
    band-passed cell correlation instead of SSIM. ``band_sigmas_px`` is
    ``(inner, outer)`` in registration voxels (see :func:`band_sigmas_px`); None
    means cells are not resolved and the function defers to multiview-stitcher.
    """
    from multiview_stitcher import registration as mvs_reg

    if band_sigmas_px is None:
        return mvs_reg.phase_correlation_registration(
            fixed_data, moving_data, **skimage_phase_corr_kwargs
        )

    import skimage.registration
    from scipy import ndimage
    from skimage.exposure import rescale_intensity

    # Rescaled straight from the input, as multiview-stitcher does: converting
    # to float64 first would hold two copies of each overlap at once.
    im0 = fixed_data.data
    im1 = moving_data.data
    im0, im1 = (
        rescale_intensity(im, in_range=(np.nanmin(im), np.nanmax(im)), out_range=(0, 1))
        for im in (im0, im1)
    )
    nan0, nan1 = np.isnan(im0), np.isnan(im1)
    any_nan = bool(nan0.any() or nan1.any())
    im0nn = np.nan_to_num(im0) if any_nan else im0
    im1nn = np.nan_to_num(im1) if any_nan else im1
    kw = dict(skimage_phase_corr_kwargs)
    kw.setdefault("upsample_factor", 10 if im0.ndim == 2 else 2)

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        peaks = [
            skimage.registration.phase_cross_correlation(
                im0nn, im1nn, disambiguate=False, normalization=norm, **kw
            )[0]
            for norm in ("phase", None)
        ]
        if any_nan:
            peaks.append(
                skimage.registration.phase_cross_correlation(
                    im0, im1, reference_mask=nan0, moving_mask=nan1,
                    disambiguate=False, **kw,
                )[0]
            )
    del im0nn, im1nn
    cands = _candidates(peaks, im1.shape)
    ndim = im0.ndim
    if not cands:
        return {"affine_matrix": _translation_affine(np.zeros(ndim)), "quality": float("nan")}

    inner, outer = band_sigmas_px
    v0, v1 = ~nan0, ~nan1
    f0 = _band_pass(im0, v0, inner, outer)
    f1 = _band_pass(im1, v1, inner, outer)
    scores = [cell_r(f0, f1, v0, v1, t) for t in cands]
    del f0, f1
    best = int(np.nanargmax(np.nan_to_num(scores, nan=-np.inf)))
    t = np.asarray(cands[best])

    # multiview-stitcher's quality for this one candidate, computed its way.
    im1t = ndimage.affine_transform(
        im1, _translation_affine(t), order=1, mode="constant", cval=np.nan
    )
    mask = ~np.isnan(im1t) & v0
    if mask.sum() < MIN_CANDIDATE_OVERLAP * v1.sum():
        quality = -1.0
    else:
        quality = float(mvs_reg.link_quality_metric_func(im0[mask], im1t[mask] - 1))
    del im1t, mask

    if choice_log is not None:
        choice_log.add(
            SeamChoice(
                shift_px=tuple(float(v) for v in t),
                quality=quality,
                cell_r=float(scores[best]),
                n_candidates=len(cands),
            )
        )
    return {"affine_matrix": _translation_affine(t), "quality": quality}


def _translation_affine(t) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    m = np.eye(len(t) + 1)
    m[:-1, -1] = t
    return m
