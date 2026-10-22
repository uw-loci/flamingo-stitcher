#!/usr/bin/env python3
"""Explain registration's Z corrections: stage tilt, curvature, or Z drift
over time.

If moving the stage in X also carries the sample along the detection axis,
each tile's content turns up deeper or shallower than the stage reported, and
registration corrects it. A straight, tilted axis makes that correction grow
LINEARLY with stage position, at the same slope everywhere; the slope is
tan(tilt). A slope that changes across the grid is not a tilt.

**Fitted to seams, not tiles.** Each registered seam is one measured Z
difference between two neighbours. A tile's solved dz is the sum of every
seam on the path from the anchor, so one bad join shifts every tile beyond
it; a seam is only itself. Every model here is fitted to seam differences
(dz_A - dz_B = f(A) - f(B)), so dropping a bad join drops exactly that
evidence and nothing else.

What is used, by default: seams the solve REGISTERED (not pruned, not below
quality, not empty), between tiles registration placed (carried / no-
neighbour tiles have no measurement of their own). Drop more with
``--drop-join Y001-Y002`` (every seam across that join) and
``--drop-tile "X000 Y006"``.

Models, each with its residual RMS:

* **tilt**          dz = a*x + b*y
* **tilt + curve**  adds x^2, y^2, x*y -- a stage cannot plausibly bend this
                    much (see the printed sag), so a strong curve term points
                    at time, not geometry
* **tilt + drift**  with ``--acq-dir``: adds d*t (+ d2*t^2), t = each tile's
                    acquisition time from its raw files' modification times.
                    In a raster scan a STEADY drift is exactly a plane (t is
                    linear in row and column), so it cannot be told from a
                    tilt in one run; an accelerating drift, or a different
                    tile order in a second run, can.

    python tools/estimate_stage_tilt.py run.log --drop-join Y000-Y001 --drop-join Y001-Y002
    python tools/estimate_stage_tilt.py run.log --acq-dir D:/CTLSM1/2026...Ket_Trial1
    python tools/estimate_stage_tilt.py registration_report.csv registration_seams.csv

Signs: dz is the correction applied to a tile. Slopes are um of dz per mm of
stage travel; angle = atan(slope). Which way is "up" depends on the Z
convention; magnitude and axis do not.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Tile:
    key: str
    x_mm: float
    y_mm: float
    dz_um: float
    how: str = "registered"
    label: str = ""          # "X### Y###"
    t_s: Optional[float] = None


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
    rf"\s({_LABEL})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+"
    rf"{_NUM}\s+{_NUM}\s+{_NUM}\s+(\S+(?: r\d+| neighbour)?)"
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
                label = m.group(1)
                x, y, dz = map(float, m.group(2, 3, 4))
                data.tiles[label] = Tile(label, x, y, dz, how=m.group(7), label=label)
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


def _grid_labels(tiles: Dict[str, Tile]) -> None:
    """CSV rows carry no grid label: number columns and rows by ascending
    stage position. (The microscope's own X### may run the other way.)"""
    xs = sorted({round(t.x_mm, 3) for t in tiles.values()})
    ys = sorted({round(t.y_mm, 3) for t in tiles.values()})
    for t in tiles.values():
        t.label = f"X{xs.index(round(t.x_mm, 3)):03d} Y{ys.index(round(t.y_mm, 3)):03d}"


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
    _grid_labels(data.tiles)
    if seam_csv is not None:
        with seam_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dz = row.get("dz_um")
                data.seams.append(Seam(
                    row["index_a"], row["index_b"], (row.get("axis") or "").lower()[:1],
                    row.get("status", ""), float(dz) if dz not in (None, "") else None,
                ))
    return data


_FILE_LABEL = re.compile(r"_X(\d{3})_Y(\d{3})_")


def read_times(acq_dir: Path, data: Data) -> int:
    """Each tile's acquisition time: the latest modification time of its raw
    files (written as the stack finishes). Flat layout only. Returns how many
    tiles got a time. Copied data keeps these only if the copy preserved
    them; the printed span shows whether it did."""
    latest: Dict[str, float] = {}
    for f in acq_dir.iterdir():
        m = _FILE_LABEL.search(f.name)
        if not m or f.suffix.lower() not in (".raw", ".tif", ".tiff", ".btf"):
            continue
        label = f"X{m.group(1)} Y{m.group(2)}"
        latest[label] = max(latest.get(label, 0.0), f.stat().st_mtime)
    n = 0
    for t in data.tiles.values():
        if t.label in latest:
            t.t_s = latest[t.label]
            n += 1
    return n


# --------------------------------------------------------------------------
# Selecting the evidence
# --------------------------------------------------------------------------


def _join_key(label_a: str, label_b: str, axis: str) -> str:
    """'Y001-Y002' for a Y-seam between rows 1 and 2; 'X000-X001' for X."""
    i = 1 if axis == "y" else 0
    parts = sorted((label_a.split()[i], label_b.split()[i]))
    return f"{parts[0]}-{parts[1]}"


def select_seams(
    data: Data,
    statuses: Sequence[str] = ("registered",),
    drop_joins: Sequence[str] = (),
    drop_tiles: Sequence[str] = (),
) -> List[Tuple[Seam, Tile, Tile]]:
    drop_j = {"-".join(sorted(j.upper().replace(" ", "").split("-"))) for j in drop_joins}
    drop_t = {t.upper() for t in drop_tiles}
    out = []
    for s in data.seams:
        if s.status not in statuses or s.dz_um is None or s.axis not in ("x", "y"):
            continue
        ta, tb = data.tiles.get(s.a), data.tiles.get(s.b)
        if ta is None or tb is None:
            continue
        if ta.how != "registered" or tb.how != "registered":
            continue
        if ta.label.upper() in drop_t or tb.label.upper() in drop_t:
            continue
        if _join_key(ta.label, tb.label, s.axis) in drop_j:
            continue
        out.append((s, ta, tb))
    return out


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _deg(slope_um_per_mm: float) -> float:
    return math.degrees(math.atan(slope_um_per_mm / 1000.0))


def fit_seams(seams, terms: Dict[str, callable]) -> Optional[dict]:
    """Least squares on dz_A - dz_B = sum_k c_k (f_k(A) - f_k(B))."""
    names = list(terms)
    if len(seams) <= len(names):
        return None
    A = np.array([[terms[n](ta) - terms[n](tb) for n in names] for _s, ta, tb in seams])
    y = np.array([s.dz_um for s, _a, _b in seams])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(1, len(y) - len(names))
    cov = float(resid @ resid) / dof * np.linalg.pinv(A.T @ A)
    return {
        "coef": dict(zip(names, coef)),
        "se": dict(zip(names, np.sqrt(np.clip(np.diag(cov), 0, None)))),
        "rms": float(math.sqrt((resid ** 2).mean())),
        "n": len(y),
        "resid": sorted(
            ((float(r), f"{ta.label}|{tb.label}") for r, (_s, ta, tb) in zip(resid, seams)),
            key=lambda p: -abs(p[0]),
        ),
    }


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="a run log, or registration_report.csv registration_seams.csv")
    ap.add_argument("--drop-join", action="append", default=[], metavar="Y001-Y002",
                    help="ignore every seam across this row (Y) or column (X) join; repeatable")
    ap.add_argument("--drop-tile", action="append", default=[], metavar="'X000 Y006'",
                    help="ignore every seam touching this tile; repeatable")
    ap.add_argument("--include-pruned", action="store_true",
                    help="also use seams the solve pruned (measured, but disagreed with it)")
    ap.add_argument("--acq-dir", type=Path,
                    help="the acquisition folder: tile times from the raw files, for the drift model")
    args = ap.parse_args(argv)

    first = args.inputs[0]
    if first.suffix.lower() == ".csv":
        if len(args.inputs) < 2:
            raise SystemExit("CSV input needs registration_report.csv AND registration_seams.csv")
        data = read_csvs(first, args.inputs[1])
    else:
        data = read_log(first)

    statuses = ("registered", "pruned") if args.include_pruned else ("registered",)
    seams = select_seams(data, statuses, args.drop_join, args.drop_tile)
    n_skip = sum(1 for t in data.tiles.values() if t.how != "registered")
    print(f"Input: {', '.join(str(p) for p in args.inputs)}")
    print(f"Evidence: {len(seams)} {'/'.join(statuses)} seams between registered tiles"
          f" ({n_skip} carried / unplaced tiles left out"
          + (f"; dropped joins {', '.join(args.drop_join)}" if args.drop_join else "")
          + (f"; dropped tiles {', '.join(args.drop_tile)}" if args.drop_tile else "") + ")")
    if len(seams) < 4:
        print("Too few seams to fit anything.")
        return 1
    xs = [t.x_mm for _s, a, b in seams for t in (a, b)]
    ys = [t.y_mm for _s, a, b in seams for t in (a, b)]
    x0, y0 = float(np.mean(xs)), float(np.mean(ys))   # centre: curve terms stay interpretable
    X = lambda t: t.x_mm - x0          # noqa: E731
    Y = lambda t: t.y_mm - y0          # noqa: E731

    # --- 1. where the slope changes ----------------------------------------
    print()
    print("1. Slope at each seam, by row (X-seams) and column (Y-seams), deg.")
    print("   A tilted straight axis gives the same number everywhere.")
    for axis, idx, name in (("x", 1, "row"), ("y", 0, "column")):
        groups: Dict[str, List[float]] = {}
        for s, ta, tb in seams:
            if s.axis != axis:
                continue
            step = (ta.x_mm - tb.x_mm) if axis == "x" else (ta.y_mm - tb.y_mm)
            groups.setdefault(ta.label.split()[idx], []).append(_deg(s.dz_um / step))
        for g in sorted(groups):
            v = groups[g]
            print(f"   {axis.upper()}-seams in {name} {g}: median {np.median(v):+.2f}   "
                  + " ".join(f"{a:+.2f}" for a in v))

    # --- 2. models ------------------------------------------------------------
    print()
    print("2. Models fitted to the seams (slopes in um of dz per mm; angle = atan)")
    tilt = fit_seams(seams, {"x": X, "y": Y})
    curve = fit_seams(seams, {"x": X, "y": Y, "xx": lambda t: X(t) ** 2,
                              "yy": lambda t: Y(t) ** 2, "xy": lambda t: X(t) * Y(t)})
    if tilt:
        a, b = tilt["coef"]["x"], tilt["coef"]["y"]
        print(f"   tilt:          X {a:+6.2f} +/- {tilt['se']['x']:.2f} ({_deg(a):+.3f} deg)   "
              f"Y {b:+6.2f} +/- {tilt['se']['y']:.2f} ({_deg(b):+.3f} deg)   "
              f"combined {_deg(math.hypot(a, b)):.3f} deg   residual RMS {tilt['rms']:.1f} um")
    if curve:
        c = curve["coef"]
        span_x = (max(xs) - min(xs)) / 2
        span_y = (max(ys) - min(ys)) / 2
        print(f"   tilt + curve:  slope at the centre X {_deg(c['x']):+.3f}  Y {_deg(c['y']):+.3f} deg;"
              f"   residual RMS {curve['rms']:.1f} um")
        for corner, sx, sy in (("+X+Y", 1, 1), ("+X-Y", 1, -1), ("-X+Y", -1, 1), ("-X-Y", -1, -1)):
            gx = c["x"] + 2 * c["xx"] * sx * span_x + c["xy"] * sy * span_y
            gy = c["y"] + 2 * c["yy"] * sy * span_y + c["xy"] * sx * span_x
            print(f"      slope at the {corner} corner: X {_deg(gx):+.3f}  Y {_deg(gy):+.3f} deg")
        sag_x = abs(c["xx"]) * span_x ** 2
        sag_y = abs(c["yy"]) * span_y ** 2
        print(f"      departure from a straight line over the span: X {sag_x:.0f} um, Y {sag_y:.0f} um"
              f" (a positioning stage is flat to a few um)")

    if args.acq_dir is not None:
        n = read_times(args.acq_dir, data)
        timed = [(s, a, b) for s, a, b in seams if a.t_s is not None and b.t_s is not None]
        print()
        if len(timed) < 6:
            print(f"3. Drift: only {n} tiles got a time from {args.acq_dir}; not fitted.")
        else:
            t0 = min(t.t_s for t in data.tiles.values() if t.t_s is not None)
            T = lambda t: (t.t_s - t0) / 3600.0      # noqa: E731  hours
            span_h = max(T(t) for t in data.tiles.values() if t.t_s is not None)
            order = sorted((t for t in data.tiles.values() if t.t_s is not None), key=lambda t: t.t_s)
            print(f"3. Drift, from file times: {n} tiles over {span_h:.2f} h; first "
                  f"{order[0].label}, last {order[-1].label}"
                  + ("  (all within a minute: these are COPY times, not acquisition times)"
                     if span_h < 1 / 60 else ""))
            drift = fit_seams(timed, {"x": X, "y": Y, "t": T})
            accel = fit_seams(timed, {"x": X, "y": Y, "t": T, "tt": lambda t: T(t) ** 2})
            for name, f in (("tilt + steady drift", drift), ("tilt + accelerating drift", accel)):
                if not f:
                    continue
                c = f["coef"]
                extra = f"drift {c['t']:+.1f} um/h" + (f", {c['tt']:+.2f} um/h^2" if "tt" in c else "")
                print(f"   {name}: X {_deg(c['x']):+.3f}  Y {_deg(c['y']):+.3f} deg, {extra};"
                      f" residual RMS {f['rms']:.1f} um")
            print("   A steady drift in a raster scan is nearly a plane, so compare the RMS:"
                  " drift only earns its place if it beats 'tilt + curve' with fewer terms.")

    if tilt:
        print()
        print("Worst seams against the tilt model: "
              + ", ".join(f"{k} {r:+.0f} um" for r, k in tilt["resid"][:4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
