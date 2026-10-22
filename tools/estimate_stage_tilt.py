#!/usr/bin/env python3
"""Estimate how far the stage's X/Y travel is tilted out of the imaging plane,
from the Z corrections registration made.

If moving the stage in X also carries the sample a little along the detection
axis, every tile's content turns up at a slightly different depth than the
stage reported, and registration corrects it: tile dz grows linearly with
stage position. The slope of that line is tan(tilt).

Two independent estimates are printed, because they fail differently:

* **Tile plane fit** -- dz = a*X + b*Y + c over the tiles registration placed
  (tiles with at least one registered seam; carried tiles are left out, their
  dz was copied from neighbours, not measured). Uses the global solve, so one
  bad join that shifts a whole block of rows biases it.
* **Seam slopes** -- each registered seam's own pairwise dz divided by the
  stage step across it, median per axis. Independent of the solve and of
  every other seam; quantised to one Z plane per seam (2.5 um over a 1.9 mm
  pitch is 0.07 deg), which the median averages out.

When the two agree, the number is worth believing. When they don't, a block
moved as a unit (look at the map's row joins) and the seam median is the safer
one.

What it cannot separate: a tilted stage axis, a tilted sample holder relative
to the axis it rides on (same thing, optically), and sample drift in Z over
the acquisition -- rows are acquired in order, so slow drift looks like a Y
tilt. A second acquisition in a different tile order, or rotated 90 deg,
separates them. One run is an observation, not a calibration.

Inputs (either):

    python tools/estimate_stage_tilt.py run.log
        The run log, with the VERBOSE ALIGNMENT tables (on by default).
    python tools/estimate_stage_tilt.py registration_report.csv registration_seams.csv
        The two CSVs written next to the stitched output.

Signs: dz is the correction applied to a tile (positive = its content sat
deeper than the stage said). "dz grows +20 um per mm of +X" means the sample
rises toward the detection objective... or falls, depending on your Z
convention; the magnitude and which stage axis carry the tilt are the
convention-free part.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Tile:
    key: str
    x_mm: float
    y_mm: float
    dz_um: float


@dataclass
class Seam:
    a: str
    b: str
    axis: str
    status: str
    dz_um: Optional[float]


@dataclass
class Data:
    tiles: Dict[str, Tile] = field(default_factory=dict)
    seams: List[Seam] = field(default_factory=list)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

_LABEL = r"X\d{3} Y\d{3}"
_NUM = r"-?\d+(?:\.\d+)?"
_TILE_ROW = re.compile(
    rf"\s({_LABEL})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s"
)
_SEAM_ROW = re.compile(rf"\s({_LABEL})\s+({_LABEL})\s+([xyz])\s+(\w+)\s+(.*)$")


def read_log(path: Path) -> Data:
    """TILE PLACEMENT and SEAM MEASUREMENTS from the LAST verbose block."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max((i for i, l in enumerate(lines) if "TILE PLACEMENT (" in l), default=None)
    if start is None:
        raise SystemExit(
            f"{path}: no TILE PLACEMENT table. It is printed when registration "
            f"finishes, with 'Verbose alignment log' on (the default)."
        )
    data = Data()
    in_seams = False
    for line in lines[start:]:
        if "SEAM MEASUREMENTS (" in line:
            in_seams = True
            continue
        if in_seams and "=====" in line:
            break
        if not in_seams:
            m = _TILE_ROW.search(line)
            if m:
                label, x, y, dz = m.group(1), *map(float, m.group(2, 3, 4))
                data.tiles[label] = Tile(label, x, y, dz)
            continue
        m = _SEAM_ROW.search(line)
        if not m:
            continue
        a, b, axis, status, rest = m.groups()
        nums = re.findall(rf"(?<![\w%]){_NUM}(?![\w%])", rest)
        # registered/pruned rows: qual dz dy dx resid; others carry qual only
        dz = float(nums[1]) if len(nums) >= 5 else None
        data.seams.append(Seam(a, b, axis, status, dz))
    return data


def read_csvs(tile_csv: Path, seam_csv: Optional[Path]) -> Data:
    data = Data()
    with tile_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dz_um") in (None, ""):
                continue
            if row.get("clamped_z", "").lower() in ("true", "1"):
                continue                       # not measured: kept stage Z
            k = row["tile_index"]
            data.tiles[k] = Tile(k, float(row["x_mm"]), float(row["y_mm"]), float(row["dz_um"]))
    if seam_csv is not None:
        with seam_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dz = row.get("dz_um")
                data.seams.append(Seam(
                    row["index_a"], row["index_b"], (row.get("axis") or "").lower()[:1],
                    row.get("status", ""), float(dz) if dz not in (None, "") else None,
                ))
    return data


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _deg(slope_um_per_mm: float) -> float:
    return math.degrees(math.atan(slope_um_per_mm / 1000.0))


def plane_fit(tiles: List[Tile]) -> Optional[dict]:
    """Least squares dz = a*X + b*Y + c (a, b in um per mm), with standard
    errors. Tiles are not independent (they share seams), so the errors are
    optimistic; read them as a floor."""
    if len(tiles) < 4:
        return None
    X = np.array([[t.x_mm, t.y_mm, 1.0] for t in tiles])
    z = np.array([t.dz_um for t in tiles])
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    resid = z - X @ coef
    dof = max(1, len(z) - 3)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    ss_tot = float(((z - z.mean()) ** 2).sum()) or 1e-12
    return {
        "a": float(coef[0]), "b": float(coef[1]),
        "se_a": float(math.sqrt(cov[0, 0])), "se_b": float(math.sqrt(cov[1, 1])),
        "r2": 1.0 - float(resid @ resid) / ss_tot,
        "rms": float(math.sqrt((resid ** 2).mean())),
        "n": len(z),
        "worst": sorted(zip(resid, (t.key for t in tiles)), key=lambda p: -abs(p[0]))[:3],
    }


def seam_slopes(data: Data, statuses: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    """Per axis: each seam's dz / stage step (um per mm). Seam dz is tile A's
    correction minus tile B's when the solve honours it, so the stage step is
    A minus B as well."""
    out: Dict[str, list] = {"x": [], "y": []}
    for s in data.seams:
        if s.status not in statuses or s.dz_um is None or s.axis not in out:
            continue
        ta, tb = data.tiles.get(s.a), data.tiles.get(s.b)
        if ta is None or tb is None:
            continue
        step = (ta.x_mm - tb.x_mm) if s.axis == "x" else (ta.y_mm - tb.y_mm)
        if abs(step) < 1e-6:
            continue
        out[s.axis].append(s.dz_um / step)
    return {k: np.array(v) for k, v in out.items()}


def join_slopes(data: Data, statuses: Tuple[str, ...]) -> Dict[str, List[tuple]]:
    """Per axis, per join (the line between two columns or rows): stage
    position of the join, seam count, median slope. A stage tilt gives the
    same slope at every join; a step that grows across the grid is not a
    tilt (drift over time, a sagging sample, or bad joins)."""
    groups: Dict[Tuple[str, float], list] = {}
    for s in data.seams:
        if s.status not in statuses or s.dz_um is None or s.axis not in ("x", "y"):
            continue
        ta, tb = data.tiles.get(s.a), data.tiles.get(s.b)
        if ta is None or tb is None:
            continue
        if s.axis == "x":
            step, mid = ta.x_mm - tb.x_mm, (ta.x_mm + tb.x_mm) / 2
        else:
            step, mid = ta.y_mm - tb.y_mm, (ta.y_mm + tb.y_mm) / 2
        if abs(step) < 1e-6:
            continue
        groups.setdefault((s.axis, round(mid, 2)), []).append(s.dz_um / step)
    out: Dict[str, List[tuple]] = {"x": [], "y": []}
    for (axis, mid), v in sorted(groups.items()):
        out[axis].append((mid, len(v), float(np.median(v)), float(min(v)), float(max(v))))
    return out


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="a run log, or registration_report.csv [registration_seams.csv]")
    ap.add_argument("--include-pruned", action="store_true",
                    help="also use seams the solve pruned (measured, but disagreed with the solve)")
    ap.add_argument("--all-tiles", action="store_true",
                    help="fit every tile, including carried ones (dz copied from neighbours)")
    args = ap.parse_args(argv)

    first = args.inputs[0]
    if first.suffix.lower() == ".csv":
        data = read_csvs(first, args.inputs[1] if len(args.inputs) > 1 else None)
    else:
        data = read_log(first)
    if not data.tiles:
        print("No tile rows found.", file=sys.stderr)
        return 1

    statuses = ("registered", "pruned") if args.include_pruned else ("registered",)
    placed = {k for s in data.seams if s.status == "registered" for k in (s.a, s.b)}
    if args.all_tiles or not data.seams:
        fit_tiles = list(data.tiles.values())
        which = "all tiles" if args.all_tiles else "all tiles (no seam table to tell placed from carried)"
    else:
        fit_tiles = [t for k, t in data.tiles.items() if k in placed]
        which = f"tiles with a registered seam ({len(fit_tiles)} of {len(data.tiles)})"

    print(f"Input: {', '.join(str(p) for p in args.inputs)}")
    print()
    print(f"1. Tile plane fit over {which}")
    p = plane_fit(fit_tiles)
    if p is None:
        print("   too few tiles to fit a plane")
    else:
        for axis, k in (("X", "a"), ("Y", "b")):
            sl, se = p[k], p["se_" + k]
            print(f"   along stage {axis}: dz {sl:+7.2f} +/- {se:5.2f} um per mm  ->  "
                  f"{_deg(sl):+.3f} deg (+/- {abs(_deg(sl + se) - _deg(sl)):.3f})")
        mag = math.hypot(p["a"], p["b"])
        direction = math.degrees(math.atan2(p["b"], p["a"]))
        print(f"   steepest: {_deg(mag):.3f} deg, toward {direction:+.0f} deg from +X (+90 = +Y)")
        print(f"   R^2 {p['r2']:.2f}, residual RMS {p['rms']:.1f} um over {p['n']} tiles; "
              f"worst: " + ", ".join(f"{k} {r:+.0f}" for r, k in p["worst"]))

    print()
    label = "registered + pruned" if args.include_pruned else "registered"
    print(f"2. Seam slopes ({label} seams; each seam's dz / stage step)")
    for axis, v in seam_slopes(data, statuses).items():
        if not len(v):
            print(f"   {axis.upper()}-seams: none")
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        print(f"   {axis.upper()}-seams (n={len(v):2d}): median dz {med:+7.2f} um per mm  ->  "
              f"{_deg(med):+.3f} deg   (middle half {_deg(q1):+.3f} to {_deg(q3):+.3f} deg)")

    print()
    print("3. Per join (median of its seams). A stage tilt is the same at every join.")
    for axis, rows in join_slopes(data, statuses).items():
        for mid, n, med, lo, hi in rows:
            print(f"   {axis.upper()} join at {axis.upper()}={mid:7.2f} mm  n={n}  "
                  f"{_deg(med):+.3f} deg  (seams {_deg(lo):+.2f} to {_deg(hi):+.2f})")

    print()
    print("A tilt seen along only one axis, and not repeated by a second acquisition in a\n"
          "different tile order, may be Z drift over time rather than stage geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
