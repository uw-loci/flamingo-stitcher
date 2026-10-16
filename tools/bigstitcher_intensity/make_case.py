"""Write a synthetic intensity-adjustment case for BigStitcher and our port.

A 3x3 (default) mosaic cut from one smooth, textured field, each tile given a
known gain and offset, placed at fractional world positions (so no grid point
sits exactly on a view edge) with anisotropic Z. Writes one raw little-endian
float32 file per view (x fastest -- imglib2 order, i.e. numpy (Z, Y, X)
C-order) and ``case.txt`` for ``IntensityHarness``.

    python3 tools/bigstitcher_intensity/make_case.py OUT_DIR [--downsampling 2]
        [--max-matches 300] [--affine 1] [--grid 3x3] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_views(grid=(3, 3), tile=(12, 40, 48), overlap=(10, 12), z_scale=2.4, seed=0):
    rng = np.random.default_rng(seed)
    ny, nx = grid
    tz, ty, tx = tile
    step_y, step_x = ty - overlap[0], tx - overlap[1]
    H, W = step_y * (ny - 1) + ty + 4, step_x * (nx - 1) + tx + 4
    from scipy import ndimage as nd

    field = nd.gaussian_filter(rng.normal(0, 1, (tz + 2, H, W)), (1.0, 3.0, 3.0))
    field = 2000 + 1500 * (field - field.min()) / (field.max() - field.min())
    field += rng.normal(0, 20, field.shape)
    views, truth = [], []
    for iy in range(ny):
        for ix in range(nx):
            gain = float(rng.uniform(0.7, 1.3))
            off = float(rng.uniform(-150, 150))
            y0, x0 = iy * step_y + 2, ix * step_x + 2
            data = field[1 : 1 + tz, y0 : y0 + ty, x0 : x0 + tx] * gain + off
            # world px: x, y as cut (+ a fractional shift), z scaled by z_scale
            fx, fy, fz = 0.37 + 0.05 * ix, 0.21 + 0.03 * iy, 0.13
            aff = np.array(
                [
                    [1.0, 0.0, 0.0, x0 + fx],
                    [0.0, 1.0, 0.0, y0 + fy],
                    [0.0, 0.0, z_scale, fz],
                ]
            )
            views.append((data.astype(np.float32), aff))
            truth.append({"gain": gain, "offset": off})
    return views, truth


def write_case(out: Path, views, params: dict, truth=None):
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"{k} {v}" for k, v in params.items()]
    for i, (data, aff) in enumerate(views):
        f = out / f"view_{i}.f32"
        np.ascontiguousarray(data, dtype="<f4").tofile(f)
        z, y, x = data.shape
        m = " ".join(repr(float(v)) for v in aff.ravel())
        lines.append(f"view {i} {x} {y} {z} {f} {m}")
    (out / "case.txt").write_text("\n".join(lines) + "\n")
    if truth is not None:
        (out / "truth.json").write_text(json.dumps(truth, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--downsampling", type=float, default=2.0)
    ap.add_argument("--max-matches", type=int, default=300)
    ap.add_argument("--affine", type=int, default=1)
    ap.add_argument("--reg-trans", type=float, default=0.1)
    ap.add_argument("--reg-identity", type=float, default=0.1)
    ap.add_argument("--grid", default="3x3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver-iterations", type=int, default=0)
    a = ap.parse_args()
    ny, nx = (int(v) for v in a.grid.split("x"))
    views, truth = make_views(grid=(ny, nx), seed=a.seed)
    write_case(
        Path(a.out),
        views,
        {
            "downsampling": a.downsampling,
            "maxMatches": a.max_matches,
            "affine": a.affine,
            "regTrans": a.reg_trans,
            "regIdentity": a.reg_identity,
            "solverIterations": a.solver_iterations,
        },
        truth,
    )


if __name__ == "__main__":
    main()
