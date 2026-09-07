"""Measure how the stage's idea of a tile step differs from the image's.

Registration corrections on the 2026-09-06 7x7 were not scattered — they grew
monotonically across the grid. Every Y-seam independently measured about
-32 um between vertically adjacent tiles, every X-seam about +12 um between
horizontally adjacent ones, and the per-tile corrections accumulated to 130 um
by the far edge. That is not stage jitter, which would be uncorrelated between
neighbours. It is a constant per-step disagreement between where the stage says
it went and where the image says it went.

If that holds up it matters twice over. It is a calibration fault worth fixing
at the source (a 1.8% Y error is ~400 px of drift across a 14x14 grid), and it
means most of what registration computes for 84 pairs is a handful of numbers.

**This module only measures.** It fits the corrections that registration already
produced and reports the fit; nothing is applied, no placement changes. One
acquisition of one sample is an observation, not a calibration — the fit has to
be reproduced across several acquisitions on the same microscope before anyone
should think about correcting with it. The report is deliberately shaped to make
that comparison easy, and to make a WEAK fit obvious rather than quietly
plausible.

The fit is per axis and includes the cross terms, because the 7x7 showed more
than scale: Y-seams also drifted in X (a shear), and X-seams drifted in Z (the
stage not level). A model that only fitted scale would report a good fit while
missing most of the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Below this R^2 the ramp explains too little of the correction to be called
# systematic; the report says so rather than quoting a slope nobody should use.
MIN_SYSTEMATIC_R2 = 0.5
# Fewer placed tiles than this and a slope is curve-fitting, not measurement.
MIN_TILES_FOR_FIT = 6


@dataclass
class AxisFit:
    """One correction axis regressed against one grid axis."""

    response: str          # "dz" | "dy" | "dx"
    against: str           # "col" | "row"
    n: int = 0
    slope_um_per_step: float = 0.0
    intercept_um: float = 0.0
    r2: float = 0.0
    residual_um: float = 0.0      # RMS about the fit
    pitch_um: Optional[float] = None

    @property
    def percent_of_pitch(self) -> Optional[float]:
        if not self.pitch_um:
            return None
        return 100.0 * self.slope_um_per_step / self.pitch_um

    @property
    def is_systematic(self) -> bool:
        return self.n >= MIN_TILES_FOR_FIT and self.r2 >= MIN_SYSTEMATIC_R2


@dataclass
class StageGeometry:
    """The whole measurement, and whether it is worth anything."""

    fits: List[AxisFit] = field(default_factory=list)
    n_tiles: int = 0
    reason: str = ""

    def get(self, response: str, against: str) -> Optional[AxisFit]:
        for fit in self.fits:
            if fit.response == response and fit.against == against:
                return fit
        return None


def _grid_indices(values: Sequence[float], tol: float) -> List[int]:
    """Rank each position among the distinct stage positions on that axis.

    Derived from the positions rather than from a filename index, so a
    multi-acquisition set with no X###_Y### naming still gets a grid.
    """
    distinct: List[float] = []
    for v in sorted(values):
        if not distinct or abs(v - distinct[-1]) > tol:
            distinct.append(v)
    out = []
    for v in values:
        best = min(range(len(distinct)), key=lambda i: abs(distinct[i] - v))
        out.append(best)
    return out


def _fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """(slope, intercept, r2, rms residual). Degenerate input gives a zero fit."""
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0, float(np.mean(y)) if len(y) else 0.0, 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = y - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    # A perfectly flat correction is a perfect fit with nothing to explain;
    # calling that r2=1 would advertise a systematic error of slope zero.
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2), float(np.sqrt(ss_res / len(x)))


def measure(
    tiles: Sequence,
    shifts: Sequence,
    placed: Optional[Sequence[int]] = None,
    pitch_um: Optional[Dict[str, float]] = None,
) -> StageGeometry:
    """Regress each tile's correction against its grid position.

    ``shifts`` are the per-tile corrections (anything with dz_um/dy_um/dx_um and
    an ``index``), ``placed`` restricts the fit to tiles registration actually
    measured — a carried tile's correction is its neighbours' average, so
    including one would be fitting the model to its own output.
    """
    out = StageGeometry()
    allowed = set(int(i) for i in placed) if placed is not None else None

    rows: List[Tuple[float, float, float, float, float]] = []
    xs, ys = [], []
    for shift in shifts or []:
        index = int(getattr(shift, "index", -1))
        if allowed is not None and index not in allowed:
            continue
        if not (0 <= index < len(tiles)):
            continue
        tile = tiles[index]
        try:
            xs.append(float(tile.x_mm) * 1000.0)
            ys.append(float(tile.y_mm) * 1000.0)
            rows.append(
                (
                    float(getattr(shift, "dz_um", 0.0) or 0.0),
                    float(getattr(shift, "dy_um", 0.0) or 0.0),
                    float(getattr(shift, "dx_um", 0.0) or 0.0),
                    0.0,
                    0.0,
                )
            )
        except (TypeError, ValueError, AttributeError):
            continue

    out.n_tiles = len(rows)
    if out.n_tiles < MIN_TILES_FOR_FIT:
        out.reason = (
            f"only {out.n_tiles} tiles were placed by registration; a slope "
            f"from fewer than {MIN_TILES_FOR_FIT} is curve-fitting"
        )
        return out

    x_um = np.asarray(xs, float)
    y_um = np.asarray(ys, float)
    # A quarter of the smallest gap separates columns without splitting one.
    def _tol(v):
        gaps = np.diff(np.unique(np.round(v, 3)))
        gaps = gaps[gaps > 1e-9]
        return float(gaps.min() * 0.25) if len(gaps) else 1.0

    cols = np.asarray(_grid_indices(x_um, _tol(x_um)), float)
    rws = np.asarray(_grid_indices(y_um, _tol(y_um)), float)
    data = np.asarray(rows, float)

    for response, column in (("dz", 0), ("dy", 1), ("dx", 2)):
        for against, index in (("col", cols), ("row", rws)):
            slope, intercept, r2, rms = _fit(index, data[:, column])
            out.fits.append(
                AxisFit(
                    response=response,
                    against=against,
                    n=out.n_tiles,
                    slope_um_per_step=slope,
                    intercept_um=intercept,
                    r2=r2,
                    residual_um=rms,
                    pitch_um=(pitch_um or {}).get("x" if against == "col" else "y"),
                )
            )
    return out


def format_report(
    geometry: StageGeometry,
    pixel_size_um: Optional[float] = None,
    frame_extent_um: Optional[Dict[str, float]] = None,
) -> List[str]:
    """The measurement, with enough context to compare it against another run."""
    lines = ["=" * 71, " STAGE GEOMETRY — measured, not applied", "=" * 71]
    if geometry.reason:
        lines.append(f"  Not measured: {geometry.reason}")
        lines.append("=" * 71)
        return lines

    lines.append(
        f"  Fitted to {geometry.n_tiles} tiles placed by registration. Each row "
        f"regresses one"
    )
    lines.append(
        "  correction axis against grid position; the slope is the per-step "
        "disagreement"
    )
    lines.append("  between where the stage went and where the image says it went.")
    lines.append("")
    lines.append(
        f"  {'correction':<12} {'vs':<5} {'µm/step':>9} {'% of pitch':>11} "
        f"{'R²':>6} {'RMS µm':>8}  verdict"
    )
    lines.append("  " + "-" * 69)

    primary = (("dx", "col"), ("dy", "row"), ("dx", "row"), ("dy", "col"),
               ("dz", "col"), ("dz", "row"))
    for response, against in primary:
        fit = geometry.get(response, against)
        if fit is None:
            continue
        pct = fit.percent_of_pitch
        pct_text = f"{pct:+10.2f}%" if pct is not None else "         -"
        verdict = (
            "systematic" if fit.is_systematic
            else ("scattered" if abs(fit.slope_um_per_step) > 0.5 else "flat")
        )
        lines.append(
            f"  {response:<12} {against:<5} {fit.slope_um_per_step:+9.2f} "
            f"{pct_text} {fit.r2:6.2f} {fit.residual_um:8.2f}  {verdict}"
        )

    if frame_extent_um:
        lines.append("")
        for axis, against, extent_key in (("x", "col", "x"), ("y", "row", "y")):
            fit = geometry.get("dx" if axis == "x" else "dy", against)
            extent = frame_extent_um.get(extent_key)
            if fit is None or not fit.pitch_um or not extent:
                continue
            nominal = 1.0 - fit.pitch_um / extent
            actual = 1.0 - (fit.pitch_um + fit.slope_um_per_step) / extent
            lines.append(
                f"  Implied {axis.upper()} overlap {actual*100:5.1f}% "
                f"(the layout assumes {nominal*100:.1f}%)"
            )

    lines.append("")
    lines.append(
        "  ONE acquisition is an observation, not a calibration. Compare these"
    )
    lines.append(
        "  numbers across several runs on the SAME microscope before treating"
    )
    lines.append(
        "  them as a stage property; a sample-shaped effect would not repeat."
    )
    lines.append("=" * 71)
    return lines
