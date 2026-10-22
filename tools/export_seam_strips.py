#!/usr/bin/env python3
"""Cut the overlap strips of a few seams out of an acquisition, so registration
methods can be compared on real data in minutes instead of a 20-hour stitch.

Each tile goes through the stitcher's OWN preprocessing (light paths merged,
flat-field if asked, downsample, tile orientation), so a strip is exactly what
registration saw, in the stage-aligned frame. The edge facing the neighbour is
then chosen from the two tiles' stage positions, with the same reverse-X/Y
rules the stitcher applies -- no guessing about camera orientation.

    python tools/export_seam_strips.py D:\\CTLSM1\\20261006_..._Ket_Trial1 G:\\seam_strips
        (default seams: the two failing Y-seams X001/X003 Y002-Y001 and a
         good X-seam X002-X001 Y003)
    python tools/export_seam_strips.py ACQ OUT --seam "X001 Y002:X001 Y001" --seam ...

Writes, per seam, ``<A>__<B>/A.npy`` and ``B.npy`` (uint16, Z x Y x X) and a
``seam.json`` with voxel size, each strip's world origin in um, the stage
overlap, and which edge was cut. Size at 2x downsample with the default band
is ~1.3 GB per strip; tiles are processed one at a time (one tile's raw data
in RAM, ~48 GB for two light paths of 2999 x 2048 x 2048).

Settings to match the run: ``--downsample-xy`` (default 2), ``--illumination``
(default max), ``--flat-field`` (off by default: on Ket_Trial1 the correction
was 0.994-1.008, below the noise).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_SEAMS = (
    "X001 Y002:X001 Y001",   # bottom row join, measured dx -112, pruned
    "X003 Y002:X003 Y001",   # same join, measured dx +120, kept
    "X002 Y003:X001 Y003",   # within-row X-seam, q 0.94, the control
)


def _label(tile) -> str:
    ix, iy = tile.tile_index
    return f"X{int(ix):03d} Y{int(iy):03d}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("acquisition", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seam", action="append", default=None, metavar="'X001 Y002:X001 Y001'",
                    help="tile A and tile B, colon-separated; repeatable")
    ap.add_argument("--downsample-xy", type=int, default=2)
    ap.add_argument("--illumination", default="max", help="illumination fusion, as in the run")
    ap.add_argument("--flat-field", action="store_true", help="estimate and apply flat-field first")
    ap.add_argument("--margin-um", type=float, default=150.0,
                    help="extra width beyond the stage overlap, so shifts up to this are inside the strip")
    ap.add_argument("--pixel-um", type=float, default=None,
                    help="native XY pixel size; default: from the objective in ScopeSettings.txt")
    ap.add_argument("--tile-orientation", default=None,
                    help="override the per-microscope orientation (e.g. identity)")
    ap.add_argument("--reverse-x", choices=("auto", "yes", "no"), default="auto")
    ap.add_argument("--reverse-y", choices=("auto", "yes", "no"), default="auto")
    args = ap.parse_args(argv)

    from flamingo_stitcher.orientation import resolve_tile_orientation
    from flamingo_stitcher.pipeline import (
        StitchingConfig,
        StitchingPipeline,
        acquisition_is_flat,
        discover_flat_tiles,
        discover_tiles,
        suggested_pixel_size_um,
    )

    acq = args.acquisition
    seams = [tuple(s.strip() for s in spec.split(":")) for spec in (args.seam or DEFAULT_SEAMS)]
    found = discover_flat_tiles(acq) if acquisition_is_flat(acq) else discover_tiles(acq)
    if any(t.tile_index is None for t in found):
        # Folder-per-tile layout carries no X###/Y### in its names: number the
        # columns and rows by ascending stage position.
        xs = sorted({round(t.x_mm, 3) for t in found})
        ys = sorted({round(t.y_mm, 3) for t in found})
        for t in found:
            t.tile_index = (xs.index(round(t.x_mm, 3)), ys.index(round(t.y_mm, 3)))
    tiles = {_label(t): t for t in found}
    missing = sorted({n for pair in seams for n in pair} - set(tiles))
    if missing:
        raise SystemExit(f"Not found in {acq}: {', '.join(missing)} (have {len(tiles)} tiles)")

    cfg = StitchingConfig.with_yaml_defaults()
    cfg.downsample_xy = args.downsample_xy
    cfg.downsample_z = 1
    cfg.illumination_fusion = args.illumination
    cfg.destripe = False
    cfg.deconvolution_enabled = False
    cfg.depth_attenuation = False
    cfg.flat_field_correction = bool(args.flat_field)
    cfg.registration_map_enabled = False
    ori = resolve_tile_orientation(acq)
    if ori is not None and ori.name:
        cfg.tile_orientation = ori.name
        cfg.reverse_x_tiles = bool(ori.reverse_x)
        cfg.reverse_y_tiles = bool(ori.reverse_y)
    if args.tile_orientation:
        cfg.tile_orientation = args.tile_orientation
    if not cfg.tile_orientation:
        raise SystemExit(
            "No tile orientation saved for this microscope. Pass --tile-orientation "
            "(the one the stitch log names on its 'Tile orientation:' line)."
        )
    if (ori is None or not ori.name) and "auto" in (args.reverse_x, args.reverse_y):
        raise SystemExit(
            "No saved orientation preset for this microscope, so the tile order is "
            "unknown too. Pass --reverse-x and --reverse-y (yes/no) as the stitch log's "
            "'Tile orientation:' line says; for n7 Ket_Trial1: --reverse-x yes --reverse-y no."
        )
    if args.reverse_x != "auto":
        cfg.reverse_x_tiles = args.reverse_x == "yes"
    if args.reverse_y != "auto":
        cfg.reverse_y_tiles = args.reverse_y == "yes"
    px_native = args.pixel_um or suggested_pixel_size_um(acq)
    if not px_native:
        raise SystemExit("No objective in ScopeSettings.txt; pass --pixel-um")
    px = px_native * args.downsample_xy
    print(f"Orientation {cfg.tile_orientation or '(default)'}, reverse X {cfg.reverse_x_tiles}, "
          f"reverse Y {cfg.reverse_y_tiles}; pixel {px_native:.4f} um native, {px:.4f} um exported")

    pipe = StitchingPipeline(cfg)
    any_tile = next(iter(tiles.values()))
    ch = sorted(any_tile.raw_files)[0]
    if args.flat_field:
        pipe._flatfield_models = pipe._estimate_flatfield_models(list(tiles.values()), [ch])

    def world_um(t):
        x = (-t.x_mm if cfg.reverse_x_tiles else t.x_mm) * 1000.0
        y = (-t.y_mm if cfg.reverse_y_tiles else t.y_mm) * 1000.0
        return x, y

    # Which edge of which tile, for every seam; then load each tile once.
    cuts = {}       # label -> list of (seam_dir, role, axis, side, band)
    plans = []
    for a, b in seams:
        ta, tb = tiles[a], tiles[b]
        (xa, ya), (xb, yb) = world_um(ta), world_um(tb)
        axis = "x" if abs(xb - xa) > abs(yb - ya) else "y"
        d = (xb - xa) if axis == "x" else (yb - ya)
        frame_px = (ta.frame_width if axis == "x" else ta.frame_height) // args.downsample_xy
        overlap_px = frame_px - abs(d) / px
        band = int(min(frame_px, round(overlap_px + args.margin_um / px)))
        seam_dir = args.out / f"{a.replace(' ', '_')}__{b.replace(' ', '_')}"
        # B lies at +axis from A: A gives its high edge, B its low edge.
        side_a, side_b = ("high", "low") if d > 0 else ("low", "high")
        cuts.setdefault(a, []).append((seam_dir, "A", axis, side_a, band))
        cuts.setdefault(b, []).append((seam_dir, "B", axis, side_b, band))
        plans.append(dict(dir=seam_dir, a=a, b=b, axis=axis, stage_step_um=abs(d),
                          overlap_um=overlap_px * px, band_px=band))

    meta = {p["a"] + "|" + p["b"]: dict(p, dir=str(p["dir"]), strips={}) for p in plans}
    for label, jobs in cuts.items():
        t0 = time.time()
        print(f"{label}: preprocessing ...", flush=True)
        vol = pipe._preprocess_single_tile(tiles[label], ch)
        nz, ny, nx = vol.shape
        zx_um = (tiles[label].z_max_mm - tiles[label].z_min_mm) * 1000.0 / max(1, nz - 1)
        wx, wy = world_um(tiles[label])
        for seam_dir, role, axis, side, band in jobs:
            seam_dir.mkdir(parents=True, exist_ok=True)
            if axis == "x":
                sl = slice(nx - band, nx) if side == "high" else slice(0, band)
                strip = np.ascontiguousarray(vol[:, :, sl])
                origin = (wx + sl.start * px, wy)
            else:
                sl = slice(ny - band, ny) if side == "high" else slice(0, band)
                strip = np.ascontiguousarray(vol[:, sl, :])
                origin = (wx, wy + sl.start * px)
            np.save(seam_dir / f"{role}.npy", strip)
            key = next(k for k, m in meta.items() if m["dir"] == str(seam_dir))
            meta[key]["strips"][role] = dict(
                tile=label, edge=side, shape=list(strip.shape),
                origin_um={"z": tiles[label].z_min_mm * 1000.0, "x": origin[0], "y": origin[1]},
                voxel_um={"z": zx_um, "y": px, "x": px},
            )
            print(f"   {role} strip for {key}: {axis} {side} edge, {strip.shape}, "
                  f"{strip.nbytes / 1e9:.2f} GB", flush=True)
        del vol
        print(f"   done in {time.time() - t0:.0f} s", flush=True)

    for m in meta.values():
        (Path(m["dir"]) / "seam.json").write_text(json.dumps(
            dict(m, acquisition=str(acq), downsample_xy=args.downsample_xy,
                 illumination=args.illumination, flat_field=bool(args.flat_field),
                 reverse_x=cfg.reverse_x_tiles, reverse_y=cfg.reverse_y_tiles,
                 tile_orientation=cfg.tile_orientation), indent=2))
    print(f"Wrote {len(meta)} seams to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
