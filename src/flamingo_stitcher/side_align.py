"""Line up a tile's two light paths before they are merged.

The server images each tile with the left light path as a full stack, then
the same stack again with the right path (``WorkflowBase.cpp``: per stack, per
laser, per path). The same plane in the two paths is therefore one stack-time
apart, and a sample that moves during the run (Ket_Trial1, 2026-10-27: up to
~110 um Z per row early on, attributed to laser heating) is imaged in two
places. Max, mean and content fusion then show it doubled; split shows a step
at the midline.

This module measures the shift of the second path against the first on the
middle of the frame, where both sheets light the sample, and applies it in
WHOLE planes and WHOLE camera pixels. Nothing is interpolated, so no plane is
rebuilt from two acquired planes (the reason registration snaps Z to whole
planes), and the merged tile stays in the first path's frame: tile placement,
registration and Z snapping downstream see the same geometry as before.
Planes or edge pixels the shifted path does not cover take the first path's
own voxels, so the shift never opens a black band.

A measured shift is applied only when the band-passed cells line up better
with it than without it and clear the same cells-r floor that seam selection
uses; an empty or featureless tile keeps the two paths as acquired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from flamingo_stitcher import seam_select

logger = logging.getLogger(__name__)

# Z window measured, in acquired planes. A shift up to a quarter of the window
# is measured reliably (the two windows still overlap by 75%), so 128 planes
# reaches 32 planes: 8x the ~4 planes estimated for the early Ket_Trial1 rows
# (row step ~110 um Z / 12 stacks per row / 2.5 um planes, 2026-10-29).
WINDOW_PLANES = 128

# Central share of the frame measured on each in-plane axis. Both sheets light
# the middle of the frame; the far half of each is blur and haze, which would
# correlate the haze instead of the cells.
CENTRE_FRACTION = 0.5

# XY binning for the measurement. Seam selection's band-pass needs its outer
# sigma (8 um) to be at least 2 voxels (MIN_OUTER_SIGMA_PX), so voxels up to
# 4 um work; 2 camera pixels keep ~2 um voxels, the size cells r was set on.
MEASURE_BIN_XY = 2

# Planes sampled to place the window on the tissue. Coarse is enough: the
# window only has to land where there is structure.
PLACEMENT_SAMPLES = 16


@dataclass
class SideShift:
    """Shift of the second light path onto the first, in acquired voxels.

    ``shift`` follows seam selection's convention: second(x + t) ~ first(x).
    ``applied`` is False when the gate refused it; ``shift`` then still says
    what was measured.
    """

    shift: Tuple[int, int, int]
    r_without: float
    r_with: float
    applied: bool
    reason: str
    window: Tuple[int, int]

    def as_applied(self) -> Optional[Tuple[int, int, int]]:
        if self.applied and any(self.shift):
            return self.shift
        return None


def _bin_xy(a: np.ndarray, b: int) -> np.ndarray:
    z, y, x = a.shape
    y2, x2 = (y // b) * b, (x // b) * b
    a = a[:, :y2, :x2].astype(np.float32)
    return a.reshape(z, y2 // b, b, x2 // b, b).mean(axis=(2, 4))


def _centre(n: int) -> slice:
    m = int(round(n * CENTRE_FRACTION))
    start = (n - m) // 2
    return slice(start, start + m)


def _window(first: np.ndarray) -> Tuple[int, int]:
    """[z0, z1) of WINDOW_PLANES planes centred where the first path shows the
    most structure (standard deviation of a coarse central crop)."""
    nz = first.shape[0]
    w = min(WINDOW_PLANES, nz)
    if w == nz:
        return 0, nz
    ys, xs = _centre(first.shape[1]), _centre(first.shape[2])
    step = max(1, 8 * MEASURE_BIN_XY)
    zs = np.linspace(w // 2, nz - 1 - w // 2, PLACEMENT_SAMPLES).astype(int)
    scores = [float(np.std(np.asarray(first[z, ys, xs][::step, ::step], dtype=np.float32)))
              for z in zs]
    c = int(zs[int(np.argmax(scores))])
    z0 = max(0, min(nz - w, c - w // 2))
    return z0, z0 + w


def measure(first, second, voxel_um: Tuple[float, float, float],
            min_cell_r: float) -> SideShift:
    """Shift of ``second`` onto ``first`` (both (Z, Y, X), same shape).

    ``voxel_um`` is the acquired voxel (z, y, x). ``min_cell_r`` is the floor a
    shifted match must clear, as for seams.
    """
    from skimage.registration import phase_cross_correlation

    z0, z1 = _window(first)
    ys, xs = _centre(first.shape[1]), _centre(first.shape[2])
    a = _bin_xy(np.asarray(first[z0:z1, ys, xs]), MEASURE_BIN_XY)
    b = _bin_xy(np.asarray(second[z0:z1, ys, xs]), MEASURE_BIN_XY)
    mvox = {"z": voxel_um[0], "y": voxel_um[1] * MEASURE_BIN_XY,
            "x": voxel_um[2] * MEASURE_BIN_XY}
    sig = seam_select.band_sigmas_px(mvox)
    if sig is None:
        return SideShift((0, 0, 0), float("nan"), float("nan"), False,
                         "cells not resolved at this pixel size", (z0, z1))
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return SideShift((0, 0, 0), float("nan"), float("nan"), False,
                         "no structure", (z0, z1))

    # skimage: second shifted by s matches first, i.e. second(x - s) ~ first(x).
    s, _err, _phase = phase_cross_correlation(a, b, upsample_factor=2,
                                              normalization="phase")
    t_measured = (-float(s[0]), -float(s[1]), -float(s[2]))
    # Whole planes; whole camera pixels (the measurement grid is binned).
    t = (int(round(t_measured[0])),
         int(round(t_measured[1] * MEASURE_BIN_XY)),
         int(round(t_measured[2] * MEASURE_BIN_XY)))

    valid = np.ones(a.shape, dtype=bool)
    fa = seam_select._band_pass(a, valid, *sig)
    fb = seam_select._band_pass(b, valid, *sig)
    r0 = seam_select.cell_r(fa, fb, valid, valid, (0.0, 0.0, 0.0))
    r1 = seam_select.cell_r(fa, fb, valid, valid, t_measured)

    limit = (a.shape[0] // 4, first.shape[1] // 4 * CENTRE_FRACTION,
             first.shape[2] // 4 * CENTRE_FRACTION)
    if not any(t):
        return SideShift(t, r0, r1, False, "already aligned", (z0, z1))
    if any(abs(v) > lim for v, lim in zip(t, limit)):
        return SideShift(t, r0, r1, False,
                         "beyond a quarter of the measured window", (z0, z1))
    if not np.isfinite(r1) or r1 < min_cell_r:
        return SideShift(t, r0, r1, False,
                         f"cells r {r1:.2f} below {min_cell_r:g}", (z0, z1))
    if np.isfinite(r0) and r1 <= r0:
        return SideShift(t, r0, r1, False,
                         "lines the cells up no better than no shift", (z0, z1))
    return SideShift(t, r0, r1, True, "", (z0, z1))


def shifted_plane(second, z: int, shift: Tuple[int, int, int],
                  fallback: np.ndarray) -> np.ndarray:
    """Plane ``z`` of ``second`` moved by ``shift`` (second(x + t) ~ first(x)),
    in whole voxels; anything it does not cover is taken from ``fallback``."""
    dz, dy, dx = shift
    zs = z + dz
    out = np.array(fallback, copy=True)
    if not 0 <= zs < second.shape[0]:
        return out
    ny, nx = out.shape
    if abs(dy) >= ny or abs(dx) >= nx:
        return out

    def _pair(d, n):
        return (slice(0, n - d), slice(d, n)) if d >= 0 else (slice(-d, n), slice(0, n + d))

    (ydst, ysrc), (xdst, xsrc) = _pair(dy, ny), _pair(dx, nx)
    out[ydst, xdst] = np.asarray(second[zs][ysrc, xsrc])
    return out
