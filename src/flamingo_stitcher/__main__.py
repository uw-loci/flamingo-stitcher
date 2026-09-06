"""CLI entry point for the stitching pipeline.

Usage:
    flamingo-stitch /path/to/acquisition -o /path/to/output

Examples:
    # Basic stitching with sharded OME-Zarr output (default)
    flamingo-stitch /data/20260310_acquisition

    # Custom pixel size and Z step
    flamingo-stitch /data/acq --pixel-size-um 0.812 --z-step-um 2.5

    # Full preprocessing pipeline
    flamingo-stitch /data/acq --destripe --illumination-fusion leonardo --deconvolution

    # Output as pyramidal OME-TIFF (single file)
    flamingo-stitch /data/acq --output-format ome-tiff

    # Write both OME-Zarr and OME-TIFF, plus package as .ozx
    flamingo-stitch /data/acq --output-format both --package-ozx

    # Dry run (discover tiles only, no processing)
    flamingo-stitch /data/acq --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from .orientation import OrientationUnknownError
from .pipeline import StitchingConfig, StitchingPipeline, discover_tiles


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, separated from running it so tests can inspect defaults.

    Split out because a flag's default is not a cosmetic choice here: anything
    passed into ``StitchingConfig`` unconditionally overrides the dataclass and
    the YAML, so a wrong default is a silent behaviour change on every run
    rather than a visible error.
    """
    parser = argparse.ArgumentParser(
        description="Stitch Flamingo T-SPIM raw acquisitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "acquisition_dir",
        type=Path,
        help="Root directory containing tile folders (X{x}_Y{y}/ subfolders)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: {acquisition_dir}/stitched)",
    )

    # Voxel size
    voxel_group = parser.add_argument_group("Voxel geometry")
    voxel_group.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="XY pixel size in micrometers (default: derived from the "
        "objective in ScopeSettings.txt, else 0.406)",
    )
    voxel_group.add_argument(
        "--z-step-um",
        type=float,
        default=None,
        help="Z step in micrometers (default: computed from Workflow.txt)",
    )
    voxel_group.add_argument(
        "--frame-width",
        type=int,
        default=None,
        help="Raw frame (camera AOI) width override in px "
        "(default: auto-detected from the file size / Workflow.txt AOI)",
    )
    voxel_group.add_argument(
        "--frame-height",
        type=int,
        default=None,
        help="Raw frame (camera AOI) height override in px (default: auto-detected)",
    )

    # Preprocessing
    preproc_group = parser.add_argument_group("Preprocessing")
    preproc_group.add_argument(
        "--illumination-fusion",
        choices=["max", "mean", "leonardo"],
        default="max",
        help="Dual-illumination fusion method (default: max)",
    )
    preproc_group.add_argument(
        "--split-illumination",
        action="store_true",
        help=(
            "Diagnostic: do NOT fuse the two light-sheet sides. Stitch each "
            "illumination path independently and write it as its own output "
            "channel (Channel_<ch>_I0, Channel_<ch>_I1), so a per-side artifact "
            "can be told apart from one introduced by fusing. Doubles the output "
            "channel count and forces streaming mode."
        ),
    )
    preproc_group.add_argument(
        "--tile-overlap-fusion",
        choices=["blend", "max", "brightest"],
        default="max",
        help=(
            "How overlapping tiles are combined: 'max' (pixel-wise maximum, "
            "default — best for sparse/sub-FOV samples where blending dilutes "
            "signal against background), 'blend' (weighted cosine), or "
            "'brightest' (winner-take-all — each overlap taken whole from the "
            "brighter tile by mean intensity; no per-pixel mixing)"
        ),
    )
    preproc_group.add_argument(
        "--flat-field",
        action="store_true",
        help="Apply BaSiC flat-field correction (requires basicpy)",
    )
    preproc_group.add_argument(
        "--destripe",
        action="store_true",
        help="Apply PyStripe destriping (requires pystripe)",
    )
    preproc_group.add_argument(
        "--destripe-fast",
        action="store_true",
        help="Destripe after downsample (faster, slightly lower quality)",
    )
    preproc_group.add_argument(
        "--destripe-workers",
        type=int,
        default=None,
        help="Max parallel destripe threads (default: auto based on available memory)",
    )
    preproc_group.add_argument(
        "--border-qc-max-shift",
        type=int,
        default=None,
        help=(
            "Half-width of the border-QC seam alignment search, in effective "
            "pixels (default 8). Raise it when the QC report says shifts are "
            "hitting the search limit — at the limit the reported offset is a "
            "floor, not a measurement."
        ),
    )
    preproc_group.add_argument(
        "--preprocess-workers",
        type=int,
        default=None,
        help=(
            "Tiles preprocessed concurrently (default: auto — 1 when destriping "
            "is on, since concurrent tiles measured 3x slower at ~4x peak RAM; "
            "otherwise up to 4). Raise only if preprocessing is I/O-starved."
        ),
    )
    preproc_group.add_argument(
        "--destripe-sigma",
        type=float,
        nargs=2,
        metavar=("FOREGROUND", "BACKGROUND"),
        default=None,
        help="Destripe filter bandwidth in px for the foreground and background "
        "bands (default: 128 256). The main strength lever — larger removes "
        "wider stripes; 0 disables that band.",
    )
    preproc_group.add_argument(
        "--destripe-level",
        type=int,
        default=None,
        help="Destripe wavelet decomposition depth (default: 7; 0 = auto/max)",
    )
    preproc_group.add_argument(
        "--destripe-wavelet",
        default=None,
        help="Destripe mother wavelet, any PyWavelets name (default: db2)",
    )
    preproc_group.add_argument(
        "--destripe-direction",
        choices=["auto", "horizontal", "vertical"],
        default="auto",
        help="Stripe orientation to remove (default: auto-detect per tile). The "
        "filter is axis-fixed and runs before the tile rot/flip, so the wrong "
        "axis removes nothing; use horizontal/vertical to force it.",
    )
    preproc_group.add_argument(
        "--depth-attenuation",
        action="store_true",
        help="Correct exponential Z-intensity falloff (Beer-Lambert model)",
    )
    preproc_group.add_argument(
        "--depth-attenuation-mu",
        type=float,
        default=None,
        help="Decay coefficient mu (1/um); omit for auto-fit from data",
    )
    preproc_group.add_argument(
        "--deconvolution",
        action="store_true",
        help="Apply GPU deconvolution (requires pycudadecon or RedLionfish)",
    )
    preproc_group.add_argument(
        "--deconv-engine",
        choices=["pycudadecon", "redlionfish"],
        default="pycudadecon",
        help="Deconvolution engine (default: pycudadecon)",
    )
    preproc_group.add_argument(
        "--deconv-iterations",
        type=int,
        default=10,
        help="Richardson-Lucy iterations (default: 10)",
    )

    # Registration
    reg_group = parser.add_argument_group("Registration")
    reg_group.add_argument(
        "--reg-channel",
        type=int,
        default=0,
        help="Channel to use for registration (default: 0)",
    )
    reg_group.add_argument(
        "--quality-threshold",
        type=float,
        # None, not a literal: a value here is passed into StitchingConfig on
        # every run, so a hardcoded default silently shadows the dataclass's.
        # This one said 0.2 while the dataclass says 0.4 — the exact value the
        # dataclass comment calls out as too permissive — so every CLI run
        # registered at the known-bad threshold. See the overlay below.
        default=None,
        help="Min phase correlation quality (default: 0.4)",
    )
    reg_group.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Channel indices to process (default: all)",
    )
    reg_group.add_argument(
        "--skip-registration",
        action="store_true",
        default=None,
        help="Place tiles by stage position only. Any tile-to-tile offset in "
        "the output is then stage placement error, uncorrected.",
    )
    reg_group.add_argument(
        "--registration-binning",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
        help="Phase-correlation binning per axis (default: 2 4 4)",
    )
    reg_group.add_argument(
        "--reg-binning-xy",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Lateral phase-correlation binning, applied to both X and Y "
            "(default: 4). Overrides the Y/X components of "
            "--registration-binning."
        ),
    )
    reg_group.add_argument(
        "--reg-binning-z",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Axial phase-correlation binning, independent of XY (default: 2). "
            "Sets the floor on Z precision -- a shift resolves to about one "
            "binned voxel -- so 1 is the cheapest way to improve Z alignment. "
            "Overrides the Z component of --registration-binning."
        ),
    )
    reg_group.add_argument(
        "--reg-upsample",
        type=int,
        default=None,
        help="Sub-pixel upsampling for phase correlation. multiview-stitcher "
        "defaults 3-D data to 2, i.e. about half a binned voxel; 4-10 is "
        "usually worth it, especially in Z. 0 = leave its default alone.",
    )
    reg_group.add_argument(
        "--max-reg-shift",
        type=float,
        default=None,
        metavar="UM",
        help="Max lateral (X/Y) move from the stage position, µm "
        "(0 = auto: one overlap width)",
    )
    reg_group.add_argument(
        "--max-reg-shift-z",
        type=float,
        default=None,
        metavar="UM",
        help="Max axial (Z) move from the stage position, µm (0 = auto). "
        "Separate from --max-reg-shift: a mosaic tiled only in X/Y has no Z "
        "overlap to derive a bound from.",
    )
    reg_group.add_argument(
        "--z-content-crop",
        dest="z_content_crop",
        action="store_true",
        default=None,
        help="Register on the Z planes that contain something rather than the "
        "whole stack (default on). Does not change the measured shift; it stops "
        "empty planes diluting the seam quality score until good seams are "
        "rejected, and reads far fewer planes.",
    )
    reg_group.add_argument(
        "--no-z-content-crop",
        dest="z_content_crop",
        action="store_false",
        help="Register on the whole Z stack.",
    )
    reg_group.add_argument(
        "--min-tile-structure",
        type=float,
        default=None,
        metavar="FRAC",
        help="Hold tiles with less structure than this out of the registration "
        "(default 0.15). Texture relative to noise, so it does not track "
        "brightness: bright featureless agarose scores the same as empty air. "
        "Featureless material sits near 0.09. 0 registers every tile.",
    )
    reg_group.add_argument(
        "--min-registered-seams",
        type=float,
        default=None,
        metavar="FRAC",
        help="Do not apply the registration unless at least this fraction of "
        "expected seams actually registered (default 0.5). Below it, tiles are "
        "placed by stage position: multiview-stitcher solves each connected "
        "group of tiles independently, so a partly-registered mosaic slides "
        "whole blocks past tiles that stayed put. Lower it for a sample that "
        "only covers part of the grid; 0 disables the check. Usually better set "
        "per microscope in the GUI's Options tab.",
    )
    reg_group.add_argument(
        "--min-reg-overlap",
        type=float,
        default=None,
        metavar="FRAC",
        help="Skip registration when measured tile overlap is below this "
        "fraction (default 0.05). Phase correlation on a sliver returns a "
        "confident wrong answer rather than failing.",
    )
    reg_group.add_argument(
        "--z-refine",
        dest="z_refine",
        action="store_true",
        default=None,
        help="Second registration pass at fine Z binning and sub-plane "
        "upsampling, to fix axial alignment the binned main pass cannot "
        "express. Roughly 2-3x the registration time.",
    )
    reg_group.add_argument(
        "--no-z-refine", dest="z_refine", action="store_false", help=argparse.SUPPRESS
    )
    reg_group.add_argument(
        "--z-refine-range-um",
        type=float,
        default=None,
        help="Half-width of the Z-refine search, µm (default 40). A correction "
        "returned at the limit is rejected, not applied — it is a floor.",
    )
    reg_group.add_argument(
        "--z-refine-upsample",
        type=int,
        default=None,
        help="Sub-plane upsampling for the Z-refine pass (default 10)",
    )
    reg_group.add_argument(
        "--approach",
        choices=("default", "center_xy"),
        default=None,
        help="How the mosaic is assembled. 'default' solves each connected "
        "group of tiles on its own and leaves an unregistered tile at its stage "
        "position. 'center_xy' anchors the solve at the centre-most tile and "
        "carries every tile outside the registered core to its neighbours' mean "
        "correction, so an empty rim keeps its overlap instead of tearing.",
    )
    reg_group.add_argument(
        "--no-z-snap",
        dest="z_snap_to_plane",
        action="store_false",
        default=None,
        help="Keep sub-plane Z shifts instead of snapping each tile to a whole "
        "plane. Fusion interpolates linearly, so a tile placed between two "
        "acquired planes is rebuilt from both across the whole XY field (a "
        "half-plane shift keeps 51%% of a single plane's peak). Snapping is on "
        "by default and costs at most half a plane of placement accuracy.",
    )
    reg_group.add_argument(
        "--z-snap", dest="z_snap_to_plane", action="store_true", help=argparse.SUPPRESS
    )

    # Multi-view (rotation)
    mv_group = parser.add_argument_group("Multi-view (rotation)")
    mv_group.add_argument(
        "--multiview",
        action="store_true",
        help="Fuse a multi-angle acquisition into one volume: place each view by "
        "a rotation about the vertical (Y) axis using its Workflow.txt angle. "
        "No effect on single-angle data.",
    )
    mv_group.add_argument(
        "--rotation-sign",
        type=float,
        choices=[1.0, -1.0],
        default=1.0,
        help="Handedness relating stage angle to the fusion rotation (default: 1). "
        "Flip to -1 if the fused views come out mirrored (RIG-VALIDATE).",
    )
    mv_group.add_argument(
        "--rotation-center",
        type=float,
        nargs=2,
        metavar=("X_MM", "Z_MM"),
        default=None,
        help="Rotation axis (X, Z) in mm (default: auto — centroid of tile "
        "positions). Only used with --multiview.",
    )

    # QC / diagnostics
    qc_group = parser.add_argument_group("QC & diagnostics")
    qc_group.add_argument(
        "--registration-report",
        dest="registration_report",
        action="store_true",
        default=None,
        help="Write registration_report.csv / registration_seams.csv / "
        "registration_report.txt into the output dir (default: on)",
    )
    qc_group.add_argument(
        "--no-registration-report",
        dest="registration_report",
        action="store_false",
        help="Suppress the registration report",
    )
    qc_group.add_argument(
        "--registration-report-json",
        action="store_true",
        default=None,
        help="Also write registration_report.json",
    )
    qc_group.add_argument(
        "--border-qc",
        action="store_true",
        help="Scan neighboring-tile seams for sharp intensity steps and write a "
        "text report next to the run log",
    )
    qc_group.add_argument(
        "--border-qc-mode",
        choices=("mip", "full", "pairs"),
        default="mip",
        help="Border-QC detail: mip=border length (fast), full=area+Z-range, "
        "pairs=offending pairs only (default: mip)",
    )
    qc_group.add_argument(
        "--preview-orientations",
        nargs="?",
        const="",
        default=None,
        metavar="PNG",
        help="Build a fast MIP-mosaic preview of all 8 whole-mosaic "
        "orientations and write a labelled contact sheet, then exit. Use it to "
        "pick the correct orientation for a system. Optional PNG path "
        "(default: orientation_preview.png next to the acquisition).",
    )
    qc_group.add_argument(
        "--preview-z-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="Restrict the orientation preview projection to a Z sub-range, as "
        "plane fractions in [0,1] (e.g. 0 0.25 = bottom quarter). Reveals "
        "structure the full-stack projection buries under scattered beads. "
        "Default: full stack.",
    )

    orient_group = parser.add_argument_group("Tile orientation")
    orient_group.add_argument(
        "--tile-orientation",
        default=None,
        metavar="NAME",
        help="Per-tile camera→stage orientation applied to EVERY tile so tiles "
        "connect. One of: identity, rot90, rot180, rot270, flip_h, flip_v, "
        "transpose, anti_transpose. Default: the saved per-microscope preset (by "
        "acquisition name), else the legacy X-flip. Choose it with "
        "--preview-orientations.",
    )
    orient_group.add_argument(
        "--reverse-x-tiles",
        action="store_true",
        help="Reverse the tile ORDER along stage X (X3 X2 X1 X0) without flipping "
        "the images — the stage-sign control, independent of --tile-orientation.",
    )
    orient_group.add_argument(
        "--reverse-y-tiles",
        action="store_true",
        help="Reverse the tile order along stage Y.",
    )

    # Output format
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-format",
        choices=["tiff", "ome-zarr", "ome-zarr-sharded", "ome-tiff", "both"],
        default="ome-zarr-sharded",
        help="Output format (default: ome-zarr-sharded)",
    )
    output_group.add_argument(
        "--package-ozx",
        action="store_true",
        help="Also create .ozx (single ZIP file) from OME-Zarr output",
    )
    output_group.add_argument(
        "--zarr-compression",
        choices=["zstd", "lz4", "blosc", "none"],
        default="zstd",
        help="Zarr compression codec (default: zstd)",
    )
    output_group.add_argument(
        "--tiff-compression",
        choices=["zlib", "lzw", "zstd", "none"],
        default="zlib",
        help="TIFF compression codec (default: zlib)",
    )
    output_group.add_argument(
        "--use-tensorstore",
        action="store_true",
        help="Use TensorStore backend for Zarr writes (faster for large data)",
    )
    output_group.add_argument(
        "--if-exists",
        choices=("overwrite", "skip", "unique"),
        default="overwrite",
        help="What to do when the output already exists (same acquisition + "
        "settings): overwrite (default; replace it), skip (leave it, don't "
        "re-stitch), or unique (write to a new numbered folder, preserving the "
        "old one).",
    )

    # Pyramid
    pyramid_group = parser.add_argument_group("Multi-resolution pyramid")
    pyramid_group.add_argument(
        "--pyramid-levels",
        type=int,
        default=None,
        help="Number of pyramid levels (default: auto)",
    )
    pyramid_group.add_argument(
        "--pyramid-method",
        choices=["itkwasm_bin_shrink", "itkwasm_gaussian", "dask_image_gaussian"],
        default="itkwasm_bin_shrink",
        help="Pyramid downsampling method (default: itkwasm_bin_shrink)",
    )

    # Utility
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and report tiles without processing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser


def resolve_registration_binning(args) -> Optional[Dict[str, int]]:
    """The per-axis binning the flags ask for, or None if none was given.

    XY and Z are chosen independently, as in the GUI, and layer OVER the
    three-axis ``--registration-binning`` so a script that already passes it
    keeps working and can still override one axis.

    None means "the user said nothing" and must stay None: passing a value in
    unconditionally is exactly how ``--quality-threshold`` spent its life
    overriding the dataclass default of 0.4 with 0.2 on every CLI run.

    Lives here rather than inline in ``main()`` so it can be tested. Inline, a
    test can only re-implement it — and a copy of the logic passes whatever the
    shipped logic does.
    """
    binning = None
    if getattr(args, "registration_binning", None) is not None:
        z, y, x = args.registration_binning
        binning = {"z": z, "y": y, "x": x}
    xy_arg = getattr(args, "reg_binning_xy", None)
    z_arg = getattr(args, "reg_binning_z", None)
    if xy_arg is not None or z_arg is not None:
        binning = dict(binning or {"z": 2, "y": 4, "x": 4})
        if xy_arg is not None:
            binning["y"] = binning["x"] = xy_arg
        if z_arg is not None:
            binning["z"] = z_arg
    return binning


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate input
    acq_dir = args.acquisition_dir.resolve()
    if not acq_dir.is_dir():
        print(f"Error: {acq_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Dry run: just discover and report tiles
    if args.dry_run:
        tiles = discover_tiles(acq_dir)
        if not tiles:
            from .pipeline import discover_flat_tiles

            tiles = discover_flat_tiles(acq_dir)
        if not tiles:
            print("No tiles found.")
            sys.exit(1)
        print(f"\nFound {len(tiles)} tiles:\n")
        for i, t in enumerate(tiles):
            flag = " ⚠" if getattr(t, "metadata_warning", None) else ""
            print(
                f"  {i + 1:3d}. {t.folder.name}  "
                f"X={t.x_mm:8.3f}  Y={t.y_mm:8.3f}  "
                f"Z=[{t.z_min_mm:.3f}, {t.z_max_mm:.3f}]  "
                f"planes={t.n_planes}  ch={t.channels}  illum={t.illumination_sides}"
                f"{flag}"
            )
        xs = sorted(set(t.x_mm for t in tiles))
        ys = sorted(set(t.y_mm for t in tiles))
        print(f"\nGrid: ~{len(xs)} x {len(ys)} tiles")
        print(
            f"Total raw files: {sum(sum(len(v) for v in t.raw_files.values()) for t in tiles)}"
        )
        # Prominent data-quality warning — corrupt/degraded tiles the run would
        # otherwise handle silently.
        warned = [t for t in tiles if getattr(t, "metadata_warning", None)]
        if warned:
            print(
                f"\n⚠  {len(warned)} of {len(tiles)} tiles had corrupt or "
                f"degraded data (stitching can still run, but review them):"
            )
            for t in warned:
                print(f"   • {t.metadata_warning}")
        sys.exit(0)

    # Orientation preview: build one mosaic per orientation, each re-orienting
    # EVERY TILE before placement, write a labelled contact sheet, and exit.
    # Lets a user pick the per-tile orientation that makes tiles CONNECT
    # (invaluable for beads) without running a full stitch.
    if args.preview_orientations is not None:
        from .orientation import (
            orientation_previews,
            read_microscope_name,
            render_contact_sheet,
            resolve_output_orientation,
        )

        z_range = tuple(args.preview_z_range) if args.preview_z_range else None
        previews = orientation_previews(
            acq_dir, pixel_size_um=args.pixel_size_um, z_range=z_range
        )
        if not previews:
            print("Could not build an orientation preview (no tiles/MIPs).")
            sys.exit(1)
        out_png = (
            Path(args.preview_orientations)
            if args.preview_orientations
            else acq_dir / "orientation_preview.png"
        )
        written = render_contact_sheet(previews, out_png)
        name = read_microscope_name(acq_dir)
        preset = resolve_output_orientation(acq_dir)
        print(f"\nMicroscope name: {name or '(none found)'}")
        print(f"Current preset : {preset or '(none)'}")
        if written:
            print(f"Wrote 8-orientation contact sheet: {written}")
            print(
                "Open it and note the panel (name) where the tissue is "
                "CONTINUOUS across the tile seams — that's this system's tile "
                "orientation."
            )
        else:
            print(
                "Pillow not available to render the contact sheet; install "
                "pillow, or use the GUI orientation preview."
            )
        sys.exit(0 if written else 2)

    # Resolve XY pixel size: explicit flag wins, else derive from the
    # acquisition's objective (ScopeSettings.txt), else the legacy default.
    pixel_size_um = args.pixel_size_um
    if pixel_size_um is None:
        from .pipeline import suggested_pixel_size_um

        suggested = suggested_pixel_size_um(acq_dir)
        if suggested:
            pixel_size_um = round(suggested, 4)
            print(
                f"Using objective-derived XY pixel size: {pixel_size_um} µm "
                f"(from ScopeSettings.txt). Override with --pixel-size-um."
            )
        else:
            pixel_size_um = 0.406
            print(
                "No objective found in ScopeSettings.txt; using default XY pixel "
                "size 0.406 µm. Set --pixel-size-um if this is wrong."
            )

    # Build config
    config = StitchingConfig(
        pixel_size_um=pixel_size_um,
        # No explicit --pixel-size-um → derive per-acquisition from its own
        # objective (the value above is only a fallback). An explicit flag is a
        # manual override applied as-is.
        auto_pixel_size=(args.pixel_size_um is None),
        z_step_um=args.z_step_um,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        illumination_fusion=args.illumination_fusion,
        split_illumination=args.split_illumination,
        tile_overlap_fusion=args.tile_overlap_fusion,
        flat_field_correction=args.flat_field,
        destripe=args.destripe,
        destripe_fast=args.destripe_fast,
        destripe_workers=args.destripe_workers,
        border_qc_max_shift_px=(
            0 if args.border_qc_max_shift is None else int(args.border_qc_max_shift)
        ),
        preprocess_workers=(
            0 if args.preprocess_workers is None else int(args.preprocess_workers)
        ),
        destripe_direction=args.destripe_direction,
        destripe_params={
            k: v
            for k, v in {
                "sigma_foreground": (
                    args.destripe_sigma[0] if args.destripe_sigma else None
                ),
                "sigma_background": (
                    args.destripe_sigma[1] if args.destripe_sigma else None
                ),
                "level": args.destripe_level,
                "wavelet": args.destripe_wavelet,
            }.items()
            if v is not None
        },
        depth_attenuation=args.depth_attenuation,
        depth_attenuation_mu=args.depth_attenuation_mu,
        reg_channel=args.reg_channel,
        output_format=args.output_format,
        package_ozx=args.package_ozx,
        zarr_compression=args.zarr_compression,
        zarr_use_tensorstore=args.use_tensorstore,
        tiff_compression=args.tiff_compression,
        pyramid_levels=args.pyramid_levels,
        pyramid_method=args.pyramid_method,
        deconvolution_enabled=args.deconvolution,
        deconvolution_engine=args.deconv_engine,
        deconvolution_iterations=args.deconv_iterations,
        border_qc_enabled=args.border_qc,
        border_qc_mode=args.border_qc_mode,
        multiview_fusion=args.multiview,
        rotation_sign=args.rotation_sign,
        rotation_center_um=(
            (args.rotation_center[0] * 1000.0, args.rotation_center[1] * 1000.0)
            if args.rotation_center is not None
            else None
        ),
    )

    # Value flags whose argparse default is None mean "the user didn't say", so
    # they must be applied as an overlay rather than passed into the constructor
    # above. Passing them in unconditionally is how --quality-threshold spent its
    # life overriding the dataclass default of 0.4 with 0.2 on every CLI run.
    _binning = resolve_registration_binning(args)
    for _attr, _value in (
        ("quality_threshold", args.quality_threshold),
        ("skip_registration", args.skip_registration),
        ("registration_binning", _binning),
        ("registration_upsample_factor", args.reg_upsample),
        ("max_registration_shift_um", args.max_reg_shift),
        ("max_registration_shift_z_um", args.max_reg_shift_z),
        ("min_registration_overlap_frac", args.min_reg_overlap),
        ("min_registered_seam_frac", args.min_registered_seams),
        ("min_tile_structure", args.min_tile_structure),
        ("registration_z_content_crop", args.z_content_crop),
        ("registration_z_refine", args.z_refine),
        ("registration_z_refine_range_um", args.z_refine_range_um),
        ("registration_z_refine_upsample", args.z_refine_upsample),
        ("registration_z_snap_to_plane", args.z_snap_to_plane),
        ("stitching_approach", args.approach),
        ("registration_report_enabled", args.registration_report),
        ("registration_report_json", args.registration_report_json),
    ):
        if _value is not None:
            setattr(config, _attr, _value)

    # Tile orientation: an explicit --tile-orientation flag wins (with any
    # --reverse-*-tiles). Otherwise resolve per-acquisition from the microscope's
    # preset AND require a known orientation — the pipeline refuses to guess.
    if args.tile_orientation:
        config.tile_orientation = args.tile_orientation.strip().lower()
        config.reverse_x_tiles = args.reverse_x_tiles
        config.reverse_y_tiles = args.reverse_y_tiles
    else:
        config.auto_tile_orientation = True
        if args.reverse_x_tiles or args.reverse_y_tiles:
            print(
                "Note: --reverse-x-tiles/--reverse-y-tiles are ignored without "
                "--tile-orientation; set them with the orientation, or in the "
                "per-microscope preset."
            )

    # Output path
    output_path = args.output or acq_dir / "stitched"

    # Run
    pipeline = StitchingPipeline(config)

    # Existing-output policy: the store name encodes the acquisition + settings,
    # so an existing one means "same run again".
    existing = pipeline.expected_output_path(acq_dir, output_path)
    if existing.exists():
        if args.if_exists == "skip":
            print(f"Output already exists, skipping: {existing}")
            sys.exit(0)
        if args.if_exists == "unique":
            base = Path(output_path)
            n = 2
            while pipeline.expected_output_path(acq_dir, base).exists():
                base = Path(str(output_path) + f"_{n}")
                n += 1
            output_path = base
            print(f"Output exists; writing to a new location: {output_path}")
        else:
            print(f"Overwriting existing output: {existing}")
    try:
        result = pipeline.run(acq_dir, output_path, channels=args.channels)
        print(f"\nStitched output: {result}")
    except ImportError as e:
        print(f"\nMissing dependency: {e}", file=sys.stderr)
        sys.exit(1)
    except OrientationUnknownError as e:
        print(f"\nTile orientation not set: {e}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print(f"\nNo data found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).exception("Pipeline failed")
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
