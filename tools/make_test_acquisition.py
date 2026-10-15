#!/usr/bin/env python3
"""Build a small stitching test acquisition from REAL collagen volumes.

Why this exists: every stitching question so far has been settled by staring at
a 9-hour run of a 2 TB acquisition, where "did registration work?" is answered
by eye and nobody knows the right answer. This writes a mosaic whose answer is
known -- each tile is cut from one continuous volume at a recorded offset -- so
registration can be SCORED instead of judged.

The pixels are real collagen (the curated phantom library), not band-limited
noise: synthetic texture registers far too easily and hides exactly the
failures that matter.

    python tools/make_test_acquisition.py --out /tmp/collagen3x3

Writes an acquisition folder the stitcher discovers normally, plus
``ground_truth.json`` giving every tile's true offset in planes and pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _synth_acq import write_synth_acquisition  # noqa: E402

DEFAULT_SOURCE = "/mnt/g/FlamingoCollagen/Phantoms/vol4d-aligned.tif"


def build_field(source: Path, planes: int, side: int) -> np.ndarray:
    """A (planes, side, side) volume of real collagen.

    The source is 8 DECORRELATED timepoints of 64 x 256 x 256. Laying
    timepoints out in XY gives a bigger field whose quadrants do not repeat --
    repeating content in XY would hand phase correlation two equally good
    answers and quietly invalidate the whole dataset.
    """
    import tifffile

    vol = tifffile.imread(str(source))  # (T, Z, Y, X)
    if vol.ndim == 5:  # TZCYX -> take the collagen channel
        vol = vol[:, :, 0]
    if vol.ndim != 4:
        raise SystemExit(f"expected a 4-D (T,Z,Y,X) source, got {vol.shape}")
    n_t, n_z, h, w = vol.shape

    per = -(-side // h)  # timepoints per row/col, rounded up
    need = per * per
    if need > n_t:
        raise SystemExit(
            f"need {need} decorrelated timepoints to fill {side}x{side} from "
            f"{h}x{w}, source has {n_t}. Use a smaller --side."
        )

    field = np.zeros((max(planes, n_z), per * h, per * w), dtype=vol.dtype)
    for i in range(per):
        for j in range(per):
            field[:n_z, i * h:(i + 1) * h, j * w:(j + 1) * w] = vol[i * per + j]
    # Deepen by mirroring whole slabs rather than repeating them, so there is
    # no abrupt Z seam. Z repeats are tolerable: a tile seam is measured across
    # XY, not across Z.
    while field.shape[0] < planes:
        field = np.concatenate([field, field[::-1]], axis=0)
    return field[:planes, :side, :side]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    ap.add_argument("--grid", type=int, default=3, help="N x N tiles (default 3)")
    ap.add_argument("--planes", type=int, default=128)
    ap.add_argument("--overlap", type=float, default=0.10)
    ap.add_argument("--frame", type=int, default=0,
                    help="tile side in px; 0 = largest that fits the source")
    ap.add_argument("--z-step-um", type=float, default=2.5,
                    help="matches the 4x-in-Z rig runs, where the Z shift bound bites")
    ap.add_argument("--z-drift-per-seam-percent", type=float, default=0.40,
                    help="ground-truth Z drift between ADJACENT tiles, as a "
                         "percent of stack depth. Default 0.40 is what the "
                         "2026-10-07 rig runs actually showed (~30 um per seam "
                         "on a 7497 um stack). Drift is physical, so expressing "
                         "it against depth keeps the dataset honest at any "
                         "--planes. Use --planes 3000 to reproduce the rig "
                         "condition where the Z shift bound rejects it.")
    ap.add_argument("--step-mm", type=float, default=0.10,
                    help="stage step between tiles, in mm. The pixel size is "
                         "DERIVED from this so the step lands exactly on the "
                         "0.01 mm grid the acquisition folder names use "
                         "(X{mm:.2f}). Get this wrong and the metadata says one "
                         "thing while the pixels say another: at a 0.066 mm step "
                         "the 0.01 mm rounding is 15%% of a step, and "
                         "registration correctly reports 4-6 um of lateral error "
                         "that the test author put there.")
    ap.add_argument("--xy-drift-per-tile-px", default="0,0",
                    help="DY,DX lateral drift added per tile in ACQUISITION "
                         "order (see --walk), so it accumulates like a real "
                         "stage/sample drift. 2026-10-14 rig run: ~2.5 px per "
                         "X step at 8x, so a raster row-to-row seam carries a "
                         "whole row of it.")
    ap.add_argument("--walk", choices=("serpentine", "raster"),
                    default="serpentine",
                    help="acquisition order the drift accumulates along. The "
                         "2026-10-14 rig log is raster: X ascending every row, "
                         "flying back between rows.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(f"source volume not found: {args.source}")

    # Ground-truth Z drift: accumulates across the grid like a real stage, so
    # neighbouring tiles differ by a little and far-apart tiles by a lot. The
    # metadata keeps one z_start for every tile, so the pixels and the metadata
    # disagree by exactly this -- which is the disagreement registration exists
    # to measure.
    n = args.grid
    depth_um = args.planes * args.z_step_um
    per_seam_um = depth_um * args.z_drift_per_seam_percent / 100.0
    per_seam_planes = per_seam_um / args.z_step_um
    xy_step = [float(v) for v in str(args.xy_drift_per_tile_px).split(",")]
    offsets = {}
    xy_offsets = {}
    for iy in range(n):
        for ix in range(n):
            # Accumulates along the order the stage walks, so adjacent tiles
            # differ by one step and opposite corners by many.
            if args.walk == "raster":
                step = iy * n + ix
            else:
                step = iy * n + (ix if iy % 2 == 0 else n - 1 - ix)
            offsets[(ix, iy)] = int(round(step * per_seam_planes))
            xy_offsets[(ix, iy)] = (
                int(round(step * xy_step[0])), int(round(step * xy_step[1]))
            )


    # Largest frame whose grid -- plus the padding a drifted tile needs to still
    # cut real content -- fits the 512 px field the source can supply without
    # repeating itself in XY.
    pad = max([0, *(max(abs(v[0]), abs(v[1])) for v in xy_offsets.values())])
    span = 1.0 + (n - 1) * (1.0 - args.overlap)
    side_guess = 512
    frame = args.frame or int((side_guess - 2 * pad) / span)
    step_px = max(1, int(round(frame * (1.0 - args.overlap))))
    side = step_px * (n - 1) + frame + 2 * pad

    # Derive the pixel size from the wanted stage step so the step is an exact
    # multiple of 0.01 mm, the resolution of the folder names this format uses.
    step_mm = round(args.step_mm, 2)
    fov_mm = step_mm / (1.0 - args.overlap)
    pixel_size_um = fov_mm * 1000.0 / frame
    quant_um = abs(round(step_mm, 2) - step_mm) * 1000.0
    overlap_um = frame * args.overlap * pixel_size_um

    field = build_field(args.source, args.planes, side)

    acq = write_synth_acquisition(
        args.out,
        grid=(n, n),
        overlap=args.overlap,
        n_planes=args.planes,
        frame_size=(frame, frame),
        z_step_um=args.z_step_um,
        pixel_size_um=pixel_size_um,
        tile_z_offset_planes=offsets,
        tile_xy_offset_px=xy_offsets,
        source_field=field,
        seed=args.seed,
    )

    truth = {
        "source": str(args.source),
        "grid": [n, n],
        "frame_px": frame,
        "planes": args.planes,
        "overlap": args.overlap,
        "overlap_px": int(round(frame * args.overlap)),
        "stage_step_mm": step_mm,
        "overlap_um": overlap_um,
        "z_step_um": args.z_step_um,
        "stack_depth_um": args.planes * args.z_step_um,
        "z_drift_per_seam_percent": args.z_drift_per_seam_percent,
        "z_drift_per_seam_um": per_seam_um,
        "z_drift_per_seam_planes": per_seam_planes,
        "tile_z_offset_planes": {f"{ix},{iy}": v for (ix, iy), v in offsets.items()},
        "walk": args.walk,
        "xy_drift_per_tile_px": xy_step,
        "tile_xy_offset_px": {
            f"{ix},{iy}": list(v) for (ix, iy), v in xy_offsets.items()
        },
        "pixel_size_um": pixel_size_um,
        "tile_z_offset_um": {
            f"{ix},{iy}": v * args.z_step_um for (ix, iy), v in offsets.items()
        },
        "note": (
            "tile_z_offset_* is the TRUE content offset. Registration should "
            "recover the DIFFERENCE between two tiles' offsets on each seam. "
            "Metadata says every tile starts at the same z_start_mm."
        ),
    }
    (Path(acq) / "ground_truth.json").write_text(json.dumps(truth, indent=2))

    print(f"acquisition: {acq}")
    print(f"  {n}x{n} tiles, frame {frame}x{frame} px, {args.planes} planes")
    print(f"  overlap {args.overlap:.0%} = {truth['overlap_px']} px "
          f"= {overlap_um:.1f} um")
    print(f"  pixel {pixel_size_um:.4f} um derived from a {step_mm:.2f} mm step "
          f"(exact on the 0.01 mm folder-name grid)")
    print(f"  field {side}x{side} px from real collagen")
    print(f"  stack depth {truth['stack_depth_um']:.0f} um at {args.z_step_um} um/plane")
    print(f"  ground-truth Z drift per seam: {per_seam_um:.1f} um "
          f"({per_seam_planes:.1f} planes), {args.z_drift_per_seam_percent:.2f}% of depth")
    print(f"  ground_truth.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
