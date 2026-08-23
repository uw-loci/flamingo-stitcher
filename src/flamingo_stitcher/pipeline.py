"""
Stitching pipeline for Flamingo T-SPIM raw acquisitions.

Takes a raw acquisition directory and produces a stitched volume.
Reuses existing Flamingo parsers for filename/metadata extraction.

Usage:
    flamingo-stitch /path/to/acquisition --pixel-size-um 0.406
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from flamingo_stitcher import registration_report, tile_geometry

logger = logging.getLogger(__name__)

# Separator placed between a phase's own "this step:" ETA and the whole-run
# "overall:" ETA in a single status line. The GUI splits on this to colour the
# two segments distinctly; kept visually distinct (spaced middle dot) so the
# plain-text status (CLI/logs) also reads cleanly.
OVERALL_ETA_SEP = "   ·   "

# Fraction of expected seams that must register before the result is believed.
# A default, not a constant: how much of a mosaic contains registerable content
# depends on the sample and the scope, so this is overridable per microscope and
# objective (config `min_registered_seam_frac`). Half is where "most seams
# agree" stops being true.
DEFAULT_MIN_REGISTERED_SEAM_FRAC = 0.5


# StitchingConfig fields recorded into stitch_metadata.json's "stitching_config"
# block so a run's settings can be reloaded into the GUI ("Load Configuration",
# to share a setup that worked). The file-specific ones at the end are recorded
# for provenance; the GUI loader deliberately skips them (Discover re-derives
# them from the actual acquisition). Order is presentation-only.
SHAREABLE_CONFIG_FIELDS = (
    "illumination_fusion",
    "split_illumination",
    "tile_overlap_fusion",
    "output_format",
    "tiff_compression",
    "zarr_compression",
    "flat_field_correction",
    "destripe",
    "destripe_fast",
    "deconvolution_enabled",
    "content_based_fusion",
    "downsample_xy",
    "downsample_z",
    "skip_registration",
    "registration_binning",
    "quality_threshold",
    "max_registration_shift_um",
    "max_registration_shift_z_um",
    "min_registration_overlap_frac",
    "min_registered_seam_frac",
    "scope_profile_source",
    "registration_upsample_factor",
    "registration_z_refine",
    "registration_z_refine_range_um",
    "registration_z_refine_binning",
    "registration_z_refine_upsample",
    "registration_report_enabled",
    "registration_report_json",
    "streaming_mode",
    "output_chunksize",
    "package_ozx",
    "tiff_pyramids",
    "background_zero_enabled",
    "background_zero_thresholds",
    "border_qc_enabled",
    "border_qc_mode",
    "border_qc_all_channels",
    "border_qc_include_z_seams",
    "border_qc_json",
    "border_qc_alpha",
    "border_qc_beta",
    "border_qc_min_component_px",
    "border_qc_z_stride",
    # File-specific (recorded for provenance; GUI loader skips these):
    "pixel_size_um",
    "z_step_um",
    "frame_width",
    "frame_height",
)


def serialize_stitching_config(config) -> Dict[str, Any]:
    """Serialize the shareable subset of a StitchingConfig to JSON-safe types.

    Dict-valued fields (chunk sizes, per-channel thresholds/binning) have
    their keys stringified so ``json.dumps`` accepts them; the loader
    reverses that. Fields absent on the config object are simply omitted.
    """
    out: Dict[str, Any] = {}
    for name in SHAREABLE_CONFIG_FIELDS:
        if not hasattr(config, name):
            continue
        val = getattr(config, name)
        if isinstance(val, dict):
            val = {str(k): v for k, v in val.items()}
        out[name] = val
    return out


def _get_git_version() -> Optional[str]:
    """Get current git commit hash, or None if unavailable."""
    try:
        import subprocess

        repo_dir = Path(__file__).parent.parent.parent.parent
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def stitcher_provenance() -> Dict[str, Any]:
    """Software-provenance fields for embedding in stitch_metadata.json.

    Records the Flamingo Stitcher version, whether this was a frozen (installer)
    or source build, and — for source builds — the git describe string. This is
    what lets anyone reading a stitched output later know exactly which stitcher
    produced it. Every lookup is best-effort so it can never break a run.
    """
    import sys

    try:
        from flamingo_stitcher import __version__ as fs_version
    except Exception:
        fs_version = "?"

    frozen = bool(getattr(sys, "frozen", False))
    prov: Dict[str, Any] = {
        "stitcher_version": fs_version,
        "stitcher_build": "frozen" if frozen else "source",
    }
    git = _get_git_version() if not frozen else None
    if git:
        prov["stitcher_git"] = git
    return prov


def _pkg_version(dist_name: str) -> str:
    """Installed version of a distribution, or '?' if unavailable.

    Defensive on purpose: a missing/odd metadata entry must never break the
    run header. In a frozen build the metadata may be absent for some packages,
    in which case we fall back to the imported module's ``__version__``.
    """
    try:
        import importlib.metadata as _md

        return _md.version(dist_name)
    except Exception:
        # Frozen builds sometimes drop dist metadata; try the live module.
        mod_name = dist_name.replace("-", "_")
        try:
            import importlib

            return str(getattr(importlib.import_module(mod_name), "__version__", "?"))
        except Exception:
            return "?"


# multiview-stitcher below this version writes ZEROS at every fusion-block
# boundary whenever a tile's translation is not a whole number of output
# voxels — which is the normal case, since stage positions are arbitrary
# relative to the output grid. The result is a regular grid of black
# one-voxel lines through the fused volume that reads like a tile-seam
# artifact but tracks the dask chunk grid. Fixed upstream in 0.1.57.
# Verified by reproduction: 0.1.44/0.1.48/0.1.49/0.1.52/0.1.56 all produce
# the lines, 0.1.57/0.1.58/0.1.59 do not. Affects blend AND max fusion;
# content-based weighting happens to mask it (it requests a 2*sigma_2 halo).
MIN_SAFE_MVS_VERSION = (0, 1, 57)

# Fractional spread in per-tile Z step above which the steps are treated as
# genuinely different rather than as rounding in the acquisition's Z bounds.
# 1% is far above the observed noise floor (~0.05%) and far below any real
# step change (5 vs 8 µm = 60%).
Z_STEP_SPREAD_WARN_FRAC = 0.01


def _parse_version_tuple(text: str) -> Optional[Tuple[int, ...]]:
    """Leading numeric components of a version string, or None.

    ``0.1.57`` → ``(0, 1, 57)``; ``0.2.0rc1`` → ``(0, 2, 0)``. Returns None for
    anything unparseable (``"?"``, a git hash) so callers can stay silent
    rather than guess.
    """
    if not text:
        return None
    parts: List[int] = []
    for chunk in str(text).split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def _mvs_module_version() -> Tuple[Optional[str], Optional[str]]:
    """``(__version__, file path)`` of the imported multiview_stitcher.

    Deliberately separate from ``_pkg_version``. In a frozen bundle the two can
    disagree: the installer overwrites ``multiview_stitcher/*.py`` in place but
    never deletes the OLD release's ``*.dist-info`` directory, so several
    dist-infos pile up and ``importlib.metadata`` answers with whichever it
    finds first — a version that hasn't been installed for months. The module's
    own ``__version__`` comes from the code that will actually run, which is the
    thing the correctness guard cares about.
    """
    try:
        import importlib

        mod = importlib.import_module("multiview_stitcher")
        return (
            str(getattr(mod, "__version__", "") or "") or None,
            str(getattr(mod, "__file__", "") or "") or None,
        )
    except Exception:
        return None, None


def check_multiview_stitcher_version(
    version: Optional[str] = None,
    module_version: Optional[str] = None,
    module_path: Optional[str] = None,
) -> List[str]:
    """Warning lines when the multiview-stitcher that will RUN corrupts output.

    Returns [] when the version is new enough — or when it cannot be parsed,
    since a false alarm on every run is worse than a missed one. See
    ``MIN_SAFE_MVS_VERSION`` for what the old versions do.

    Judged on the imported module, not on package metadata: a stale leftover
    dist-info makes metadata report a version nobody is running (see
    ``_mvs_module_version``). When the two disagree we say so, because the
    disagreement is itself the bug worth fixing — and because otherwise the
    same guard would either cry wolf on a healthy install or stay silent on a
    broken one.
    """
    # Only probe the live environment when the caller named nothing. Passing a
    # version means "judge THIS one", so a caller's explicit value is never
    # silently overruled by whatever happens to be importable.
    raw_meta = version
    if version is None and module_version is None and module_path is None:
        raw_meta = _pkg_version("multiview-stitcher")
        module_version, module_path = _mvs_module_version()

    meta_parsed = _parse_version_tuple(raw_meta or "")
    mod_parsed = _parse_version_tuple(module_version or "")

    # The module is the code that runs, so it decides. Fall back to metadata
    # only when the module can't be read at all.
    effective_raw = module_version if mod_parsed is not None else raw_meta
    effective = mod_parsed if mod_parsed is not None else meta_parsed

    lines: List[str] = []
    if (
        mod_parsed is not None
        and meta_parsed is not None
        and mod_parsed != meta_parsed
    ):
        where = f" ({module_path})" if module_path else ""
        lines.append(
            f"⚠ multiview-stitcher metadata says {raw_meta} but the imported "
            f"module is {module_version}{where}. That mismatch means stale "
            f"*.dist-info left behind by an older install — the module version "
            f"is the one that counts. Reinstall into a clean directory to "
            f"clear it."
        )

    if effective is None or effective >= MIN_SAFE_MVS_VERSION:
        return lines

    want = ".".join(str(p) for p in MIN_SAFE_MVS_VERSION)
    lines.append(
        f"⚠ multiview-stitcher {effective_raw} is TOO OLD and will corrupt this "
        f"stitch: it writes black one-voxel lines across the fused volume at "
        f"every fusion-block boundary (a regular grid that looks like tile "
        f"seams). Upgrade to >= {want}:  "
        f"pip install -U 'multiview-stitcher>={want}'"
    )
    lines.append(
        f"  If this is a frozen build, it needs rebuilding — the installer "
        f"pins >= {want}, so a bundle running {effective_raw} was not built "
        f"from the current pin."
    )
    return lines


def environment_summary() -> List[str]:
    """Lines describing the running environment for the log header.

    Captures the bits that actually matter when reproducing or diagnosing a
    run from its log alone: the Flamingo Stitcher version, whether this is a
    frozen build, the Python/OS, and the versions of the dependencies that
    drive (and most often break) stitching. Every lookup is best-effort.
    """
    import platform
    import sys

    try:
        from flamingo_stitcher import __version__ as fs_version
    except Exception:
        fs_version = "?"

    frozen = bool(getattr(sys, "frozen", False))
    build = "frozen" if frozen else "source"
    git = _get_git_version() if not frozen else None

    lines = [
        f"Flamingo Stitcher: {fs_version} ({build})" + (f", git {git}" if git else ""),
        f"Python {platform.python_version()} on {platform.platform()}",
        "Key deps: "
        + ", ".join(
            f"{name}={_pkg_version(name)}"
            for name in (
                "multiview-stitcher",
                "ngff-zarr",
                "dask",
                "zarr",
                "numpy",
                "scipy",
                "tifffile",
            )
        ),
    ]
    return lines


# ---------------------------------------------------------------------------
# Constants (loaded from microscope_hardware.yaml if available)
# ---------------------------------------------------------------------------
try:
    from flamingo_stitcher.config_loader import get_hardware_config as _get_hw

    _hw = _get_hw()
    FRAME_WIDTH = _hw.sensor_width_px
    FRAME_HEIGHT = _hw.sensor_height_px
except Exception:
    FRAME_WIDTH = 2048
    FRAME_HEIGHT = 2048

# Dask chunk size for internal processing (tile loading, fusion).
# Loaded from stitching_config.yaml memory.dask_processing_chunks.
try:
    from flamingo_stitcher.config_loader import get_stitching_value as _get_sv

    _dpc = _get_sv("memory", "dask_processing_chunks", default=[64, 512, 512])
    _DASK_PROCESSING_CHUNKS = tuple(int(c) for c in _dpc)
except Exception:
    _DASK_PROCESSING_CHUNKS = (64, 512, 512)


# Rough cold-start time-estimate constants. Only used when the timing cache has
# no measured total for a config — deliberately approximate (order-of-magnitude)
# and superseded by real timings and by the live progress extrapolation as the
# run proceeds. The point of the cold-start prior is only to put the FIRST few
# estimates in the right ballpark; the live ETA self-corrects from actual pace.
#   _ROUGH_LOAD_MBPS   : effective throughput of the load + preprocess phase
#                          (raw read + illumination fuse + downsample), MB of
#                          INPUT per second. Input volume dominates this phase,
#                          so it's sized from the actual on-disk bytes rather
#                          than a flat per-tile guess (a 1662-plane 2-illumination
#                          tile costs far more than a 200-plane single-illum one).
#   _ROUGH_S_PER_TILE_CH : small fixed per-tile-per-channel overhead (open/seek/
#                          registration), on top of the volume term.
#   _ROUGH_S_PER_OUT_UNIT: fuse + write, per OUTPUT voxel-unit (tile × plane ×
#                          channel after XY/Z downsampling).
_ROUGH_LOAD_MBPS = 200.0
_ROUGH_S_PER_TILE_CH = 1.5
_ROUGH_S_PER_OUT_UNIT = 0.08


def _drive_root(path) -> str:
    """Stable drive identifier for a path, e.g. 'H:' on Windows, '' on POSIX.

    Used to bucket timing-cache entries by source/destination drive so the
    cache learns per-drive I/O speed. Falls back to the path anchor (mount
    point) when there's no drive letter; uppercased for case-insensitive match.
    """
    if not path:
        return ""
    try:
        p = Path(path)
        return (p.drive or p.anchor or "").upper().rstrip("\\/")
    except Exception:
        return ""


def build_timing_key(tiles, config, acquisition_dir=None, output_dir=None):
    """Build the StitchingTimingKey for a set of tiles + config.

    Single source of truth for the cache key, shared by the live estimator
    (``_build_estimator``) and the GUI's pre-run queue-time estimate so both
    look up the same cached totals. ``acquisition_dir`` / ``output_dir`` feed
    the source/destination drive axes (so I/O speed is bucketed per drive);
    when omitted, the source drive falls back to the first tile's folder.
    """
    from flamingo_stitcher.timing_cache import StitchingTimingKey

    n_tiles = len(tiles)
    n_channels = len(sorted({ch for t in tiles for ch in t.channels}))
    planes = max((t.n_planes for t in tiles), default=1)
    # pyramid_levels: None means auto — bucket "auto" (-1) separately from 0/N.
    pyramid_levels = -1 if config.pyramid_levels is None else int(config.pyramid_levels)
    fusion_method = "content_based" if config.content_based_fusion else "cosine"

    src = acquisition_dir
    if not src and tiles:
        src = getattr(tiles[0], "folder", None)
    return StitchingTimingKey(
        n_tiles=n_tiles,
        n_channels=n_channels,
        n_pyramid_levels=pyramid_levels,
        n_timepoints=1,  # multi-timepoint not yet a config axis
        output_format=config.output_format,
        fusion_method=fusion_method,
        skip_registration=bool(config.skip_registration),
        # A second pairwise pass is the same class of cost swing as
        # registration on/off, one level down.
        z_refine=bool(getattr(config, "registration_z_refine", False)),
        planes_per_tile=planes,
        downsample_xy=int(getattr(config, "downsample_xy", 1) or 1),
        downsample_z=int(getattr(config, "downsample_z", 1) or 1),
        # Per-tile preprocessing: these dominate the load+preprocess phase, so
        # they have to key separately or their costs get averaged together.
        destripe=bool(getattr(config, "destripe", False)),
        destripe_fast=bool(getattr(config, "destripe_fast", False)),
        deconvolution=bool(getattr(config, "deconvolution_enabled", False)),
        flat_field=bool(getattr(config, "flat_field_correction", False)),
        source_drive=_drive_root(src),
        dest_drive=_drive_root(output_dir),
    )


def rough_run_seconds(tiles, config) -> float:
    """Very rough cold-start wall-time estimate (seconds) for one acquisition.

    Used only when the timing cache has no measured total for this config.
    Order-of-magnitude at best — two terms: a per-tile-per-channel cost
    (load + preprocess + registration) plus a per-output-voxel cost (fuse +
    write) that shrinks with XY/Z downsampling. Real measured timings replace
    this for any config that has run before.
    """
    n_tiles = len(tiles)
    n_channels = max(1, len(sorted({ch for t in tiles for ch in t.channels})))
    planes = max((t.n_planes for t in tiles), default=1)
    ds_xy = max(1, getattr(config, "downsample_xy", 1) or 1)
    ds_z = max(1, getattr(config, "downsample_z", 1) or 1)

    # Load + preprocess is dominated by moving the raw input through
    # read → illumination-fuse → downsample, so size it from actual on-disk
    # bytes (planes × frame area × 2 bytes × illumination sides × channels).
    in_bytes = 0.0
    for t in tiles:
        fw = int(getattr(t, "frame_width", FRAME_WIDTH) or FRAME_WIDTH)
        fh = int(getattr(t, "frame_height", FRAME_HEIGHT) or FRAME_HEIGHT)
        n_illum = max(1, len(getattr(t, "illumination_sides", []) or [0]))
        in_bytes += float(t.n_planes) * fw * fh * 2.0 * n_illum * n_channels
    load_seconds = in_bytes / (_ROUGH_LOAD_MBPS * 1e6)
    overhead = _ROUGH_S_PER_TILE_CH * n_tiles * n_channels
    load_register = load_seconds + overhead

    out_units = (n_tiles * planes * n_channels) / (ds_xy * ds_xy * ds_z)
    fuse_write = _ROUGH_S_PER_OUT_UNIT * out_units
    return load_register + fuse_write


# Tile file extensions the discovery + loader understand. The acquisition can
# save tiles as raw uint16, TIFF, or BigTIFF (the microscope's output-format
# dropdown). `.btf` is the conventional BigTIFF extension; tifffile also reads
# BigTIFF transparently from a `.tif` name.
TILE_EXTENSIONS = (".raw", ".tif", ".tiff", ".btf")
_TIFF_EXTENSIONS = (".tif", ".tiff", ".btf")
# Regex fragment matching any supported tile extension (case-insensitive).
_TILE_EXT_RE = r"\.(?:raw|tiff?|btf)$"

# Tile filename pattern: S000_t000000_V000_R0000_X000_Y000_C{ch}_I{illum}_D{det}_P{planes}.<ext>
# Named groups so the view (V) and rotation-index (R) fields — previously parsed
# and discarded — can be captured for multi-view acquisitions without renumbering
# the positional groups their consumers rely on.
RAW_FILE_PATTERN = re.compile(
    r"S\d+_t\d+_V(?P<view>\d+)_R(?P<rot>\d+)_X\d+_Y\d+"
    r"_C(?P<ch>\d+)_I(?P<illum>\d+)_D(?P<det>\d+)_P(?P<planes>\d+)" + _TILE_EXT_RE,
    re.IGNORECASE,
)

# Flat-layout filename: captures X_idx, Y_idx, view, rotation, channel, illum, det, planes
FLAT_RAW_PATTERN = re.compile(
    r"S\d+_t\d+_V(?P<view>\d+)_R(?P<rot>\d+)_X(?P<xidx>\d+)_Y(?P<yidx>\d+)"
    r"_C(?P<ch>\d+)_I(?P<illum>\d+)_D(?P<det>\d+)_P(?P<planes>\d+)" + _TILE_EXT_RE,
    re.IGNORECASE,
)


def _glob_tile_files(directory: Path) -> List[Path]:
    """All tile files (raw/tif/tiff/btf) directly inside ``directory``."""
    files: List[Path] = []
    for ext in TILE_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))
    return files


# Folder coordinate pattern: X{float}_Y{float} anywhere in name
FOLDER_COORD_PATTERN = re.compile(r"X([-\d.]+)_Y([-\d.]+)")


def acquisition_is_flat(acquisition_dir: Path) -> bool:
    """Does this acquisition hold tile FILES directly, rather than tile folders?

    The two layouts this package reads, and the two dialogs that read them:

    * **Subfolder-per-tile** — ``<sample>/<date>/X4.00_Y12.00/*.raw``. The
      acquisition directory is a bare date, so its own name does not say which
      sample it is.
    * **Flat** (the C++ server's native output, Single Workflow) —
      ``<whatever>/<dataset>/*.raw``. The acquisition directory IS the dataset
      and its name is the descriptive one.

    Used for output naming, where the difference matters: a date needs its
    sample folder prepended to mean anything, and a dataset does not.

    Built from the same two primitives the discovery functions use rather than
    a third opinion about layout -- this package has shipped one bug three
    times by keeping copies of a single rule.
    """
    try:
        if _glob_tile_files(acquisition_dir):
            return True
        for sub in sorted(acquisition_dir.iterdir()):
            if not sub.is_dir():
                continue
            if FOLDER_COORD_PATTERN.search(sub.name):
                return False  # a tile folder: subfolder-per-tile layout
            if _glob_tile_files(sub):
                return True  # dated flat layout: <dataset>/<date>/*.raw
    except OSError:
        # Unreadable or gone. Naming must not be the thing that fails a run, and
        # the old behaviour is the safer default here: a redundant prefix is
        # ugly, a missing one can collide two samples' dates in one folder.
        return False
    return False


# ---------------------------------------------------------------------------
# Fused-volume dtype conversion (shared by every fuse path)
# ---------------------------------------------------------------------------
def lazy_uint16(darr):
    """Squeeze a fused SpatialImage's singleton dims and clamp it to uint16
    **inside the dask graph**.

    Every consumer of a fused array needs this, and it must stay lazy. Doing it
    eagerly on the already-materialized volume —

        vol = np.clip(vol, 0, 65535).astype(np.uint16)

    — allocates a full-size clip temporary AND a full-size astype result while
    the original is still referenced, so three copies of the fused volume are
    live at once. That was a real +2x-output spike in the in-memory path (53 GB
    on a 26 GB output) that no memory estimate modelled, and it bought nothing:
    ``multiview_stitcher.fusion`` returns the fused array at ``sims[0].dtype``,
    which is already uint16. Done lazily the same clamp costs one block.

    Kept as one shared function so the in-memory, streaming and preview paths
    cannot drift apart again.
    """
    import dask.array as _da

    while darr.ndim > 3:
        darr = darr[0]
    return _da.clip(darr, 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# Lazy memmap-backed dask array
# ---------------------------------------------------------------------------
def _read_memmap_slice(path, shape, dtype, slices):
    """Read a slice out of a numpy memmap on disk. Runs inside each dask
    task so the memmap is opened per-chunk, never materialized whole.
    Returns a *copy* of the slice so the memmap handle can be dropped
    immediately (otherwise Windows holds the file lock)."""
    mm = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    out = np.array(mm[slices])  # np.array forces a read-into-RAM for THIS slice only
    del mm
    return out


def _dask_array_from_memmap(path, shape, dtype, chunks):
    """Build a dask array backed by a memmap without ever calling
    ``da.from_array``. dask's ``from_array`` unconditionally does
    ``x.copy()`` on anything arraylike *before* the ``asarray`` flag is
    considered (dask/array/core.py L3657) — that materializes the full
    memmap into RAM and defeats spill-to-disk.

    We side-step by constructing the task graph ourselves: one node per
    chunk, each node calls :func:`_read_memmap_slice` with the chunk's
    slice ranges. Dask executes these lazily, chunk-at-a-time, during
    fusion.
    """
    from itertools import product

    import dask.array as _da
    from dask.base import tokenize

    norm_chunks = _da.core.normalize_chunks(chunks, shape=shape, dtype=dtype)
    name = f"memmap-{tokenize(str(path), shape, dtype, norm_chunks)}"

    locations = _da.core.slices_from_chunks(norm_chunks)
    keys = list(product([name], *(range(len(c)) for c in norm_chunks)))

    dsk = {
        key: (_read_memmap_slice, str(path), shape, dtype, loc)
        for key, loc in zip(keys, locations)
    }
    return _da.Array(dsk, name, norm_chunks, dtype=dtype)


# ---------------------------------------------------------------------------
# Dask progress callback (time-throttled so the log stays readable)
# ---------------------------------------------------------------------------
try:
    from dask.callbacks import Callback as _DaskCallback
except ImportError:  # pragma: no cover
    _DaskCallback = object


class _TimeThrottledProgress(_DaskCallback):
    """Log dask task progress at most once every ``interval_s`` seconds.

    Rationale: ``da.store`` on a content-based fusion graph can take
    hours with no intermediate output, making it impossible to tell
    whether the pipeline is healthy, stuck, or hopelessly slow. This
    callback prints a single line at a bounded cadence (default 30 s)
    so the log stays useful without spamming — at 30 s intervals a
    3-hour run emits ~360 lines, a 10-min run emits ~20.

    Each line reports % of tasks complete, a rate (tasks/s), and an
    ETA from the observed rate so the user can judge at a glance
    whether to wait it out or cancel. ``tasks`` here means dask graph
    nodes, not output chunks — content-based fusion has ~5–10 tasks
    per output chunk (read, gaussian filters, blend, store), which
    makes the percentage a decent but not exact fraction of output
    completion. Good enough for triage.
    """

    def __init__(
        self,
        logger,
        label: str = "store",
        interval_s: float = 30.0,
        progress_fn=None,
    ):
        """
        Args:
            logger: where to write the heartbeat log line.
            label: prefix shown in the log / status line.
            interval_s: minimum seconds between updates.
            progress_fn: optional ``(int_pct, status_str)`` callback,
                wired through to ``StitchingPipeline._progress_fn`` so
                the dialog status line shows the live fuse ETA. The
                clock-time projection is included in the status string.
                The int is whatever phase percentage the caller last
                used, since the dialog currently ignores it.
        """
        super().__init__()
        self._logger = logger
        self._label = label
        self._interval = interval_s
        self._progress_fn = progress_fn
        self._completed = 0
        self._total: Optional[int] = None
        self._start_t: Optional[float] = None
        self._last_log_t: float = 0.0

    def _start_state(self, dsk, state):
        # Called once before any task runs. Use it to capture the total
        # task count so we can emit a percentage.
        try:
            self._total = len(dsk)
        except Exception:
            self._total = None
        self._start_t = time.time()
        self._last_log_t = self._start_t

    def _emit_status(self, pct_local: float, eta_seconds: Optional[float]):
        """Push the live status to the dialog if a progress_fn is set."""
        if self._progress_fn is None:
            return
        if eta_seconds is not None and eta_seconds > 0:
            from datetime import datetime, timedelta

            seconds = int(round(eta_seconds))
            if seconds < 60:
                rem = f"{seconds}s"
            elif seconds < 3600:
                rem = f"{seconds // 60}:{seconds % 60:02d}"
            else:
                h, rest = divmod(seconds, 3600)
                rem = f"{h}:{rest // 60:02d}:{rest % 60:02d}"
            clock = datetime.now() + timedelta(seconds=eta_seconds)
            eta_str = (
                clock.strftime("%H:%M")
                if clock.date() == datetime.now().date()
                else clock.strftime("%a %H:%M")
            )
            tail = f" — this step: {rem} remaining (Done at ~{eta_str})"
        else:
            tail = ""
        msg = f"{self._label}: {pct_local:.0f}%{tail}"
        try:
            # Pct ignored at the receiver (stitching_dialog._on_progress
            # drops it and renders only the status string), so 0 is fine.
            self._progress_fn(0, msg)
        except Exception as e:
            # Never let a UI hiccup break the dask graph
            self._logger.debug(f"progress_fn raised: {e}")

    def _posttask(self, key, result, dsk, state, worker_id):
        self._completed += 1
        now = time.time()
        if now - self._last_log_t < self._interval:
            return
        elapsed = max(now - (self._start_t or now), 0.001)
        rate = self._completed / elapsed
        if self._total and rate > 0:
            pct = 100.0 * self._completed / self._total
            eta_seconds = (self._total - self._completed) / rate
            self._logger.info(
                f"    {self._label} progress: {pct:.1f}% "
                f"({self._completed}/{self._total} tasks, "
                f"{rate:.0f} tasks/s, ETA {eta_seconds / 60.0:.1f} min)"
            )
            self._emit_status(pct, eta_seconds)
        else:
            self._logger.info(
                f"    {self._label} progress: {self._completed} tasks "
                f"({rate:.0f}/s, {elapsed / 60:.1f} min elapsed)"
            )
            self._emit_status(0.0, None)
        self._last_log_t = now

    def _finish(self, dsk, state, errored):
        if self._start_t is None:
            return
        elapsed = max(time.time() - self._start_t, 0.001)
        rate = self._completed / elapsed
        status = "errored" if errored else "done"
        self._logger.info(
            f"    {self._label} {status}: {self._completed} tasks "
            f"in {elapsed:.1f}s ({rate:.0f} tasks/s)"
        )


class PipelineCancelled(Exception):
    """Raised inside a dask compute to abort it promptly on user cancel.

    The pipeline polls the cancel flag between stages, but a single big
    fuse/write ``compute()`` / ``da.store()`` can run for minutes-to-hours
    with no stage boundary in between — so cancelling "requested" but the run
    kept going. This exception, raised from :class:`_CancelCallback` before
    each dask task, tears the running compute down at the next task boundary
    (≈ one chunk), turning cancel into a near-immediate stop.
    """


class _CancelCallback(_DaskCallback):
    """Dask callback that aborts a running compute/store when the user cancels.

    Registered (as a context manager) around the whole pipeline body, so EVERY
    dask compute — in-memory fuse, streaming ``da.store``, the various writers —
    becomes cancellable at chunk granularity without wrapping each call site.

    Note: dask callbacks register process-globally for the duration of the
    ``with`` block, so any *other* dask compute running concurrently in this
    process would also honour this cancel flag. During a stitch run (which owns
    the machine) that is not a concern in practice.
    """

    def __init__(self, cancelled_fn):
        super().__init__()
        self._cancelled_fn = cancelled_fn

    def _pretask(self, key, dsk, state):
        # Called by the scheduler before each task starts; raising here stops
        # the compute at the next task boundary.
        if self._cancelled_fn():
            raise PipelineCancelled()


def _scratch_base_dir(config, output_path) -> Path:
    """Directory that holds the ``.stitch_tmp`` scratch folder.

    ``config.scratch_dir`` when set (put temp on a fast local disk), else the
    output directory (default — ``<output_dir>/.stitch_tmp``, unchanged).
    """
    scratch = getattr(config, "scratch_dir", None)
    return Path(scratch) if scratch else Path(output_path)


def _same_volume(a, b) -> Optional[bool]:
    """True if ``a`` and ``b`` live on the same physical volume/drive.

    Returns None when it can't be determined. Walks up to the nearest existing
    ancestor so it works before the scratch dir is created.
    """
    import os

    def _dev(p):
        p = Path(p)
        while not p.exists() and p != p.parent:
            p = p.parent
        return os.stat(p).st_dev

    try:
        return _dev(a) == _dev(b)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class StitchingConfig:
    """Configuration for the stitching pipeline."""

    # Voxel size
    pixel_size_um: float = 0.406  # XY pixel size in micrometers
    # When True, ignore ``pixel_size_um`` as anything but a fallback and derive
    # the XY pixel size per-acquisition from that acquisition's own recorded
    # objective (ScopeSettings.txt). Lets a batch that mixes objectives stitch
    # each item at its own scale instead of one value applied to all. A manual
    # value (auto_pixel_size=False) is an explicit override applied to every item.
    auto_pixel_size: bool = False
    # Z step: computed from data if None, otherwise override
    z_step_um: Optional[float] = None
    # Raw frame (camera AOI) override in pixels. None = auto-detect per
    # acquisition from Workflow.txt `AOI width`/`AOI height`, cross-checked
    # against the actual file size. Set both to force a frame size (e.g. when
    # metadata is missing or wrong).
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None

    # Registration
    skip_registration: bool = False  # Use stage positions only (no phase correlation)
    reg_channel: int = 0  # Channel index to use for registration
    registration_binning: Dict[str, int] = field(
        default_factory=lambda: {"z": 2, "y": 4, "x": 4}
    )
    quality_threshold: float = 0.4  # Min phase-correlation quality (Spearman).
    # Was 0.2 — too permissive: low-content tiles (background / featureless
    # bright blur) clear a 0.2 correlation with a garbage shift. 0.4 makes those
    # pairs fall back to the stage position instead of being trusted.
    # Max distance (µm) a tile may be moved from its stage position by
    # registration. multiview-stitcher's phase correlation bounds the shift to
    # the tile SIZE, not the overlap, so a low-content tile can be flung ~a full
    # tile away, opening gaps. After registration, any tile whose net lateral
    # correction exceeds this bound is reverted to its stage position (see
    # _register_tiles). 0.0 = auto: use the smaller of the X/Y overlap widths, so
    # a tile can never move more than one overlap (gaps become impossible).
    max_registration_shift_um: float = 0.0
    # Max distance (µm) registration may move a tile along Z. Separate from
    # max_registration_shift_um, which bounds X/Y, because the two cannot share
    # a number: the lateral bound is the tile OVERLAP width, and a mosaic that
    # tiles only in X/Y has no Z overlap to derive a bound from. Reusing the
    # lateral value in Z is not conservative, it is arbitrary — and being large,
    # it effectively switched the Z guard off. 0.0 = auto: enough to admit the
    # few-frame stage/focus error we actually see (>= 8 Z steps, never under
    # 25 µm), capped at a quarter of the stack so a garbage correlation peak
    # halfway down the volume is still rejected.
    max_registration_shift_z_um: float = 0.0
    # Global optimization residual thresholds — inspired by BigStitcher's
    # iterative edge-pruning algorithm (Hörl et al., Nature Methods 2019).
    # Edges with residuals exceeding abs_tol are removed (if graph stays
    # connected) and the optimization re-runs, preventing bad pairwise
    # registrations from corrupting the global solution.
    global_opt_abs_tol: float = 3.5  # Max acceptable residual (pixels)
    global_opt_rel_tol: float = 0.01  # Convergence threshold

    # Minimum measured tile overlap fraction before registration is ATTEMPTED.
    # Phase correlation on a sliver of shared content does not fail loudly — it
    # returns a confident garbage peak, which is the failure the shift clamp
    # exists to mop up afterwards. Below this, place by stage position and say
    # so in the report rather than clamp half the tiles and call it alignment.
    min_registration_overlap_frac: float = 0.05

    # Fraction of expected seams that must register before the result is used
    # at all. Registration can succeed on a handful of seams and fail on the
    # rest — a sample covering part of the grid does exactly that — and
    # multiview-stitcher then solves each connected group of tiles independently
    # with nothing tying the groups together. Below this fraction the run is
    # placed by stage position and the report says why. 0 disables the check.
    min_registered_seam_frac: float = DEFAULT_MIN_REGISTERED_SEAM_FRAC

    # Which per-microscope/objective profile shaped this run ("liara|17.0x"),
    # or "" if none did. Set by whoever applied it, and read by the pipeline so
    # a GUI run that already resolved a profile — and let the user override a
    # control on top of it — is not silently re-overridden here. Also the
    # provenance answer to "why did this run use that threshold?".
    scope_profile_source: str = ""

    # Sub-pixel upsampling for phase correlation. multiview-stitcher silently
    # picks 10 for 2-D data and 2 for 3-D (registration.py:412), so every 3-D
    # registration this pipeline has ever run resolved shifts to about half a
    # BINNED voxel — with registration_binning z=2 that is one raw plane, on the
    # axis where misalignment matters most. 0 = leave MVS's default alone (so a
    # run is byte-identical to before); 4-10 costs a slightly larger correlation
    # matrix per pair and nothing else.
    registration_upsample_factor: int = 0

    # Dedicated second registration pass whose only job is Z. The main pass is
    # binned z=2 for speed, which is the wrong trade for the axis with the
    # coarsest voxel: it resolves Z to ~one raw plane while resolving XY to a
    # fraction of a pixel. The second pass re-registers from the first pass's
    # result at full Z resolution and contributes ONLY its Z component. Off by
    # default — it is a second pairwise pass, roughly doubling registration
    # time. Turn it on when the seam report shows Z disagreement.
    registration_z_refine: bool = False
    # Half-width of the Z search, µm. A correction that comes back at the limit
    # is a floor rather than a measurement, so it is rejected (the tile keeps
    # its pass-1 Z) and counted separately in the report.
    registration_z_refine_range_um: float = 40.0
    # Finer Z, same XY as the main pass: the point is Z resolution, and
    # unbinning XY as well would multiply the cost for nothing.
    registration_z_refine_binning: Dict[str, int] = field(
        default_factory=lambda: {"z": 1, "y": 4, "x": 4}
    )
    registration_z_refine_upsample: int = 10  # sub-plane; skimage's 2-D default

    # Write registration_report.csv / registration_seams.csv /
    # registration_report.txt into the output directory. ON by default, unlike
    # border QC: registration is the one step whose effect is invisible in the
    # output, and the cost is a few kB and milliseconds.
    registration_report_enabled: bool = True
    registration_report_json: bool = False  # machine-readable twin

    # Illumination fusion
    illumination_fusion: str = "max"  # "max", "mean", or "leonardo"
    # Diagnostic: when True, do NOT fuse the two light-sheet illumination sides.
    # Each side is stitched independently and written as its own output channel
    # (e.g. Channel_3_I0, Channel_3_I1), so a per-side artifact (a seam/step that
    # only appears after fusing the sides, vs. one already present within a single
    # light path) can be told apart in one viewer. Forces streaming mode (the
    # split doubles the output channel count). Single-side tiles are unaffected.
    split_illumination: bool = False

    # Output format:
    #   ome-zarr-sharded — Zarr v3, OME-NGFF v0.5, sharded (napari only)
    #   ome-zarr-v2      — Zarr v2, OME-NGFF v0.4 (Fiji/QuPath/BDV/napari)
    #   ome-tiff         — Pyramidal OME-TIFF BigTIFF (single file, universal)
    #   both             — Write ome-zarr-sharded + ome-tiff
    #   tiff             — Flat TIFF (legacy, no pyramid)
    #   ome-zarr         — OME-Zarr via multiview-stitcher (legacy)
    output_format: str = "ome-zarr-sharded"
    output_chunksize: Dict[str, int] = field(
        default_factory=lambda: {"z": 128, "y": 256, "x": 256}
    )
    # Size the output chunks against the FINAL (post-downsample) grid rather
    # than using output_chunksize verbatim. A fixed 256-px chunk covers 2x the
    # sample area at each downsample step, so more tiles overlap every fused
    # block and the float64 fusion working set climbs even as the output
    # shrinks — heavy downsample stopped saving memory. See
    # resolve_output_chunksize. Set False to use output_chunksize as written.
    auto_output_chunksize: bool = True
    # Fusion super-block batching (item E): fuse the output in spatial regions
    # of this many output-chunks per axis, instead of building ONE dask graph
    # for the whole output. Bounds fusion graph memory to O(region) rather than
    # O(total output blocks) ≈ O(mosaic area) — the term that grows ~0.4 MB/tile
    # and becomes GBs on huge mosaics. Regions are chunk-aligned, which makes
    # the result BIT-IDENTICAL to whole-output fusion for max/blend/content
    # (see tests/test_superblock_fusion.py). 0 = disabled (whole-output, the
    # historical path); a value like 8 fuses 8×8×8-chunk regions at a time.
    fusion_superblock_chunks: int = 0
    # Auto super-block: when STREAMING a large output and fusion_superblock_chunks
    # is left at 0, regions are auto-sized to ~this many GB of output each so the
    # streaming fuse graph memory is bounded (the whole-output da.store lets the
    # threaded scheduler hold far more than n_workers blocks — e.g. 127 GB on a
    # 623 GB output — while the estimate assumes bounded per-block fusion). An
    # explicit fusion_superblock_chunks always wins; 0 here disables auto.
    fusion_superblock_target_gb: float = 4.0
    # Cosine fade-out blending widths (µm) — controls the smooth transition
    # zone at tile boundaries.  Inspired by BigStitcher's cosine-weighted
    # blending (Hörl et al., Nature Methods 2019).  multiview-stitcher
    # implements the same algorithm via its weights.get_blending_weights().
    blending_widths: Dict[str, int] = field(
        default_factory=lambda: {"z": 50, "y": 100, "x": 100}
    )
    # Content-based tile-overlap weighting — uses Preibisch's local-variance
    # algorithm (bandpass-filtered intensity variance) to weight each tile's
    # contribution in overlap regions by local sharpness.  This concept
    # originates from BigStitcher (Preibisch et al.) and is implemented in
    # multiview-stitcher's weights.content_based().  Increases computation
    # time but improves fusion quality in overlap regions with uneven content.
    content_based_fusion: bool = False

    # Tile-overlap fusion method — how overlapping tiles are combined.
    #   "blend" : weighted-average (cosine) blending — seamless on dense,
    #             well-flat-fielded samples, but in the overlap it averages
    #             each tile against its neighbour, so a sparse sample on a
    #             mostly-empty FOV (or any slight stage-only misregistration)
    #             gets diluted against background → a dark dip at the seam.
    #   "max"   : pixel-wise maximum (np.nanmax) across tiles — keeps the
    #             brighter tile in the overlap, so signal can't be diluted by
    #             a neighbour's background. Best for sparse / sub-FOV samples.
    # NOTE: distinct from `illumination_fusion`, which combines left/right
    # light-sheet illumination sides, not adjacent tiles.
    # Default "max": on this hardware's light-sheet data (structured tissue,
    # stage-only registration) weighted blending produced visible dark seam
    # dips; max holds up and is also lighter on memory than blend.
    tile_overlap_fusion: str = "max"

    # --- Multi-view (rotation) fusion ---
    # When on, tiles carrying a non-zero rotation-stage angle (RawTileInfo.
    # angle_deg, read from Workflow.txt at discovery) are placed by a rotation
    # about the vertical Y axis before fusion, so several angles fuse into one
    # common frame. Off by default → single-angle runs are byte-identical
    # (_tile_metadata_affine returns None for every tile).
    multiview_fusion: bool = False
    # Rotation axis (x, z) in world µm. None → derived from the tile positions
    # (centroid of x, z-midpoint) at fuse time.
    rotation_center_um: Optional[Tuple[float, float]] = None
    # Physical handedness relating stage angle to the fusion rotation (+1/−1).
    # Validated on synthetic data; RIG-VALIDATE the sign/center on a real
    # two-angle acquisition (flip to -1 if the views come out mirrored).
    rotation_sign: float = 1.0

    # --- Tile-border artifact QC (diagnostic; see border_qc.py) ---
    # When enabled, after preprocessing the reference channel is scanned for
    # sharp intensity steps along neighboring-tile seams; a plain-text report is
    # written next to the run log. Off by default (opt-in diagnostic).
    border_qc_enabled: bool = False
    border_qc_mode: str = "mip"  # "mip" (length) | "full" (area+Z) | "pairs"
    border_qc_all_channels: bool = False  # else reference channel only
    border_qc_include_z_seams: bool = False  # niche: Z-tiled mosaics
    border_qc_json: bool = False  # also emit a machine-readable JSON twin
    border_qc_alpha: float = 4.0  # step vs local-gradient ratio
    # Half-width of the seam alignment search, in EFFECTIVE (post-downsample)
    # pixels. 0 = auto (the historical 8 px floor, widened for coarse pixels).
    # Raise it when the report says shifts are hitting the limit — at 8 px a
    # badly placed mosaic reports a tidy "ds=8" that is really ">=8, unknown".
    border_qc_max_shift_px: int = 0
    border_qc_beta: float = 3.0  # step vs noise floor
    border_qc_min_component_px: int = 10  # drop flagged blobs smaller than this
    border_qc_z_stride: int = 1  # subsample Z in full mode when huge

    # OME-Zarr sharding options
    zarr_chunks: Tuple = (32, 256, 256)  # Inner chunk shape (~4 MB per chunk)
    zarr_shard_chunks: Tuple = (4, 4, 4)  # Chunks per shard per axis
    zarr_compression: str = "zstd"  # Compression codec
    zarr_compression_level: int = 3  # Compression level
    zarr_use_tensorstore: bool = False  # TensorStore writing backend

    # Pyramid options
    pyramid_levels: Optional[int] = None  # None = auto
    pyramid_method: str = "itkwasm_bin_shrink"  # Anti-alias downsampling

    # TIFF options
    tiff_compression: str = "zlib"
    tiff_tile_size: Tuple = (256, 256)
    # Pyramid SubIFDs help napari/QuPath viewing but break older OME-TIFF
    # readers (notably ImarisFileConverter) which may read each SubIFD as
    # a separate Z plane. Set False for maximum compatibility.
    tiff_pyramids: bool = True

    # Package as single file after writing
    package_ozx: bool = False  # Create .ozx (ZIP) from OME-Zarr output

    # Camera orientation — flip tile data in X before stitching if camera
    # X axis is inverted relative to stage X (common in lightsheet systems).
    camera_x_inverted: bool = True

    # Per-tile camera→stage orientation. The camera can be mounted mirrored
    # and/or rotated relative to the stage, so each tile's PIXELS must be
    # reoriented before placement or adjacent tiles won't connect (a 90°-rotated
    # camera needs each tile transposed, etc.). One of the 8 dihedral names
    # (see flamingo_stitcher.orientation.MosaicOrientation.NAMES). Empty ""
    # derives the legacy behaviour from camera_x_inverted (True→"flip_h",
    # False→"identity"), so existing single-camera systems are unchanged.
    tile_orientation: str = ""

    # Reverse the tile ORDER along a stage axis (place tiles X3 X2 X1 X0 instead
    # of X0 X1 X2 X3) WITHOUT flipping tile pixels — the stage-sign degree of
    # freedom, independent of tile_orientation. Needed when a stage axis runs
    # opposite to the camera content direction (a system can need a per-tile
    # flip in X but an order reversal in Y). Applied to the placement
    # translation in _register_tiles.
    reverse_x_tiles: bool = False
    reverse_y_tiles: bool = False
    # When True, resolve tile_orientation / reverse_x_tiles / reverse_y_tiles
    # PER acquisition from that acquisition's own microscope name (user preset >
    # bundled YAML > the camera_x_inverted default), instead of using one choice
    # for a whole batch. Mirrors auto_pixel_size: a batch mixing systems orients
    # each correctly, and a stale/other-scope choice can't contaminate a run.
    auto_tile_orientation: bool = False

    # Processing
    flat_field_correction: bool = False  # BaSiCPy flat-field correction
    destripe: bool = False  # Run PyStripe destriping
    destripe_fast: bool = False  # Destripe after downsample (faster, lower quality)
    destripe_workers: Optional[int] = None  # Max parallel threads; None = auto
    # Stripe orientation to remove, in the RAW CAMERA FRAME: "auto",
    # "horizontal", or "vertical". The filter is axis-fixed and destriping runs
    # in the raw camera frame (before the per-tile rot/flip), so the wrong axis
    # silently removes nothing. "auto" DERIVES the axis from
    # `destripe_output_axis` + the tile orientation — it does not inspect pixels.
    destripe_direction: str = "auto"
    # Stripe orientation in the STITCHED OUTPUT frame. The output is
    # stage-aligned and the light sheet propagates in a fixed direction, so this
    # is a known constant per microscope rather than something to measure —
    # horizontal on the Flamingo. Everything else is derived from it.
    destripe_output_axis: str = "horizontal"
    # Destripe filter tuning. Empty dict = use the YAML/built-in defaults.
    # Recognised keys: sigma_foreground, sigma_background (filter bandwidth in px
    # — the main strength lever), level (wavelet depth), wavelet (mother wavelet),
    # crossover (fg/bg blend width), threshold (fg/bg split; None = Otsu auto).
    destripe_params: Dict[str, Any] = field(default_factory=dict)
    # Depth-dependent attenuation correction (Beer-Lambert Z-falloff)
    depth_attenuation: bool = False
    depth_attenuation_mu: Optional[float] = None  # 1/µm; None = auto-fit
    downsample_xy: int = 1  # XY downsample factor (1, 2, 4, 8; -1 = iso)
    downsample_z: int = 1  # Z downsample factor (1, 2, 4; -1 = iso)

    # Deconvolution
    deconvolution_enabled: bool = False
    # Run deconvolution AFTER downsample (fewer voxels, smaller PSF → much
    # faster, lower quality). Mirrors destripe_fast. Off = native-res (default).
    deconvolution_fast: bool = False
    deconvolution_engine: str = "pycudadecon"  # "pycudadecon" or "redlionfish"
    deconvolution_iterations: int = 10
    deconvolution_na: float = 0.4
    deconvolution_wavelength_nm: float = 488.0
    deconvolution_n_immersion: float = 1.33
    deconvolution_psf_path: Optional[str] = None

    # Background zeroing (post-fusion threshold).
    # Voxels at or below the per-channel threshold are set to 0 in the
    # fused dask graph immediately before write. Zeroed regions compress
    # to nearly nothing under blosc/zstd, giving large disk savings on
    # cleared-tissue acquisitions with empty space around the sample.
    # Applied once per channel, after illumination fusion and tile
    # blending, so a single threshold acts on the actual stored
    # intensities. The user must inspect a downsampled preview before
    # enabling — see run_preview() and the dialog's Preview button.
    background_zero_enabled: bool = False
    # Map of channel id -> threshold (uint16 intensity, inclusive lower bound
    # to zero). Channels not in the map are written verbatim.
    background_zero_thresholds: Dict[int, int] = field(default_factory=dict)
    # Hard safety cap: abort write if the chosen threshold would zero
    # more than this fraction of voxels in any channel. Protects against
    # picking a threshold above the entire signal range.
    background_zero_sanity_cap_fraction: float = 0.99
    # Downsample factors used by run_preview() — independent of the main
    # downsample so users can inspect at coarse resolution quickly without
    # changing the full-res output.
    background_zero_preview_downsample_xy: int = 8
    background_zero_preview_downsample_z: int = 4

    # Resource constraints
    max_memory_gb: Optional[float] = None  # None = auto (50% of system RAM)

    # Hard resource guard — abort BEFORE processing an item whose projected
    # peak RAM (for the chosen mode, incl. writer overhead) or output+spill
    # disk footprint would exceed a safety fraction of what's available. This
    # turns a machine-killing mid-run OOM / disk-full into a clean, explanatory
    # failure for that item. Set ``resource_guard_enabled=False`` (or raise the
    # fractions) to override for an intentionally tight run.
    resource_guard_enabled: bool = True
    resource_guard_ram_fraction: float = 0.95  # fraction of AVAILABLE RAM
    resource_guard_disk_fraction: float = 0.95  # fraction of free output-drive

    # Streaming mode — writes fused output chunk-by-chunk instead of
    # materializing the full volume into RAM. Required for TB-scale datasets.
    #   None  = auto-detect based on estimated output size vs available RAM
    #   True  = force streaming (low memory, may be slower)
    #   False = force in-memory (fast, requires all data to fit in RAM)
    streaming_mode: Optional[bool] = None

    # Per-tile preprocess parallelism in streaming mode.
    # 0 = auto: 1 when destriping is on (it saturates the machine by itself and
    #     concurrent tiles measured 3x SLOWER at ~4x the peak RAM), otherwise
    #     picked from available RAM vs tile size, clamped to [1, 4];
    # otherwise the requested worker count, clamped to [1, 8].
    preprocess_workers: int = 0

    # dask thread-pool size for the fused-memmap `da.store` compute.
    # Content-based fusion does two NaN Gaussian filters per chunk on
    # float32 and is CPU-bound but trivially parallel across chunks.
    # 0 = auto (picked from available RAM, clamped to [1, 4] — 1
    # degenerates to the old synchronous scheduler);
    # >0 = honor the request, clamped to [1, 8].
    fuse_workers: int = 0

    # Scratch directory for the streaming-mode temp files (per-tile spill
    # memmaps + the fused `.dat` memmap), i.e. the `.stitch_tmp` folder.
    #   None (default) = alongside the OUTPUT, i.e. `<output_dir>/.stitch_tmp`
    #     (unchanged behaviour).
    #   a path        = put `.stitch_tmp` under this directory instead.
    # Point this at a FAST LOCAL disk (NVMe/SSD) when the input/output live on a
    # slow or network drive: streaming is I/O-bound on the temp spill, so moving
    # it off the busy drive can cut wall time substantially. Notes:
    #   * No benefit if it resolves to the same physical drive as input/output
    #     (the pipeline warns in that case).
    #   * Needs room for the peak temp (~tile spill + fused memmap); the run
    #     aborts up front if the scratch drive lacks the space.
    #   * The final output is still written from the fused memmap to output_dir,
    #     so the fused data crosses drives once (scratch -> output). That is the
    #     normal fuse->write pass, NOT an extra copy of the finished result;
    #     the whole `.stitch_tmp` is deleted at the end.
    scratch_dir: Optional[str] = None

    @classmethod
    def with_yaml_defaults(cls) -> "StitchingConfig":
        """Create a StitchingConfig with defaults loaded from stitching_config.yaml.

        YAML values override the hardcoded dataclass defaults. Returns a
        plain StitchingConfig if the YAML file is not found.
        """
        config = cls()
        try:
            from flamingo_stitcher.config_loader import apply_stitching_yaml_to_config

            apply_stitching_yaml_to_config(config)
        except Exception:
            logger.debug("Could not load YAML defaults, using built-in defaults")
        return config


# Sentinel stored in StitchingConfig.downsample_xy/_z to request
# automatic isotropic factor selection. Resolved at run time against
# the acquisition's actual z_step, so batch queues with different
# Z steps each get their own resolution.
ISO_DOWNSAMPLE = -1


#: Never chunk finer than this (px). Below it, zarr shard/chunk bookkeeping and
#: per-block dask overhead cost more than the memory saved.
_MIN_CHUNK_PX = 64


def resolve_output_chunksize(
    config,
    tiles: Optional[List["RawTileInfo"]] = None,
    out_shape: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, int]:
    """Output chunk size for the FINAL (post-downsample) grid.

    ``output_chunksize`` is expressed in output pixels, so at 8x downsample a
    256-px chunk spans 8x the sample area it did at 1x. Fusion costs
    ``views_overlapping_the_block x block_voxels x 8 bytes``, so that growth
    pulls more tiles into every block and the fusion working set RISES with
    downsample even though the output shrinks — the "heavy downsample stops
    helping" effect.

    Two adjustments, both bounded by ``_MIN_CHUNK_PX``:

    * **XY** is capped near one tile PITCH in output pixels, keeping roughly a
      2x2 neighbourhood per block at any downsample factor.
    * **Every axis** is clamped to the output extent, so a small output never
      declares chunks larger than itself.

    Returns the configured chunks unchanged when ``auto_output_chunksize`` is
    off, or when there is nothing to size against.
    """
    base = dict(getattr(config, "output_chunksize", None) or {})
    chunks = {
        "z": max(1, int(base.get("z", 128))),
        "y": max(1, int(base.get("y", 256))),
        "x": max(1, int(base.get("x", 256))),
    }
    if not getattr(config, "auto_output_chunksize", True):
        return chunks

    ds_xy = int(getattr(config, "downsample_xy", 1) or 1)
    if ds_xy == ISO_DOWNSAMPLE:
        z_step = getattr(config, "z_step_um", 0) or 0
        ds_xy = (
            compute_iso_downsample(config.pixel_size_um, z_step)[0] if z_step else 1
        )
    ds_xy = max(1, ds_xy)

    # Cap XY at ~one tile pitch in OUTPUT pixels.
    pitch_mm = _min_tile_pitch_mm(tiles) if tiles else None
    pixel_um = float(getattr(config, "pixel_size_um", 0) or 0)
    if pitch_mm and pixel_um > 0:
        pitch_px = int(pitch_mm * 1000.0 / (pixel_um * ds_xy))
        cap = max(_MIN_CHUNK_PX, pitch_px)
        chunks["y"] = min(chunks["y"], cap)
        chunks["x"] = min(chunks["x"], cap)

    # Never chunk larger than the output itself.
    if out_shape is not None and len(out_shape) == 3:
        for axis, extent in zip(("z", "y", "x"), out_shape):
            if extent and extent > 0:
                chunks[axis] = max(1, min(chunks[axis], int(extent)))

    return chunks


def _min_tile_pitch_mm(tiles: List["RawTileInfo"]) -> Optional[float]:
    """Smallest spacing between adjacent tile centres in X or Y (mm).

    The densest spacing is the conservative choice: it's where the most tiles
    land in one block.
    """
    best = None
    for values in (
        [t.x_mm for t in tiles or []],
        [t.y_mm for t in tiles or []],
    ):
        uniq = sorted({round(v, 4) for v in values})
        gaps = [b - a for a, b in zip(uniq, uniq[1:]) if b - a > 1e-6]
        if gaps:
            gap = min(gaps)
            best = gap if best is None else min(best, gap)
    return best


def _avail_worker_cap(per_worker_bytes: int) -> int:
    """How many preprocess workers currently-available RAM allows.

    Mirrors ``StitchingPipeline._pick_preprocess_workers``' RAM cap (keep total
    preprocessing RAM under ~50% of available) so the pre-run estimate and the
    run itself can't disagree about how many native-resolution tiles are held
    at once.
    """
    try:
        import psutil

        avail_bytes = psutil.virtual_memory().available
    except Exception:
        avail_bytes = 8 * 1024**3  # conservative 8 GB fallback
    return max(1, int(avail_bytes // (max(1, int(per_worker_bytes)) * 2)))


def _preprocess_peak_bytes(config, native_vox: int, plane_vox: int = 0) -> int:
    """Peak RAM one preprocess worker holds for a single tile, at NATIVE
    resolution.

    Preprocessing (illumination fusion / destripe / deconvolution / the
    downsample float32 upcast) all run BEFORE the tile is downsampled, so the
    working set scales with the *native* tile size, not the downsampled one.
    Modelling it at downsampled size is what let both the memory estimate and
    the worker-picker under-count by up to ~30-120x with downsampling on
    ("projected safe, then OOM"). Kept as a single function so those two callers
    can never diverge again.

    Reflects the (post-optimisation) per-step behaviour: max/mean illumination
    fusion and XY-only downsample are per-plane (cheap); the whole-volume float32
    terms below are added only for the steps that still take them.
    """
    native_vox = max(1, int(native_vox))
    # A native uint16 volume is held through the chain, plus clip/cast transients.
    peak = native_vox * 2 * 1.5
    if getattr(config, "deconvolution_enabled", False):
        # input_float + decon output + PSF/FFT scratch (RedLionfish/pycudadecon
        # hold ~4 float32 working arrays); model 4x native float32.
        peak += 4 * native_vox * 4
    if getattr(config, "illumination_fusion", "max") == "leonardo":
        peak += 2 * native_vox * 4  # two float32 illumination sides
    ds_z = int(getattr(config, "downsample_z", 1) or 1)
    if ds_z > 1:
        # Z downsample averages consecutive slabs of ds_z planes, so it holds
        # one float32 slab (+ its reduced plane), NOT a float32 copy of the
        # whole tile. Without a plane size, fall back to a conservative slab
        # guess of 1/8 the tile so this can never under-count.
        slab_vox = (plane_vox * ds_z) if plane_vox > 0 else (native_vox // 8)
        peak += 2 * max(1, int(slab_vox)) * 4
    # Non-fast destripe and flat-field each allocate a second whole-volume buffer
    # (uint16) while the input volume is still live — add one native uint16.
    if getattr(config, "destripe", False) and not getattr(
        config, "destripe_fast", False
    ):
        peak += native_vox * 2
    if getattr(config, "flat_field_correction", False):
        peak += native_vox * 2
    return int(peak)


def compute_iso_downsample(
    xy_pixel_um: float,
    z_step_um: float,
    xy_choices: Sequence[int] = (1, 2, 4, 8, 16, 32),
    z_choices: Sequence[int] = (1, 2, 4, 8, 16),
) -> Tuple[int, int]:
    """Pick (downsample_xy, downsample_z) that make output voxels closest to cubic.

    Searches the Cartesian product of allowed XY and Z factors, minimising
    post-downsample anisotropy ``max(out_xy, out_z) / min(out_xy, out_z)``.
    Ties break toward less data loss (smaller ``dxy * dz``).

    Args:
        xy_pixel_um: Native XY pixel size in micrometres.
        z_step_um: Native Z step in micrometres.
        xy_choices: Allowed XY downsample factors (must match UI).
        z_choices: Allowed Z downsample factors (must match UI).

    Returns:
        (downsample_xy, downsample_z) integers from the allowed sets.
    """
    if xy_pixel_um <= 0 or z_step_um <= 0:
        return 1, 1

    best: Tuple[int, int] = (1, 1)
    best_score: Tuple[float, int] = (float("inf"), 0)
    for dz in z_choices:
        for dxy in xy_choices:
            out_xy = xy_pixel_um * dxy
            out_z = z_step_um * dz
            anis = max(out_xy, out_z) / min(out_xy, out_z)
            score = (anis, dxy * dz)
            if score < best_score:
                best_score = score
                best = (dxy, dz)
    return best


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def resolve_superblock_chunks(
    config: "StitchingConfig",
    out_z_px: int,
    out_y_px: int,
    out_x_px: int,
    use_streaming: bool,
    bpv: int = 2,
) -> int:
    """Region size (output-chunks per axis) for super-block fusion; 0 = whole
    output.

    An explicit ``config.fusion_superblock_chunks`` always wins. Otherwise, for
    STREAMING of a large output, auto-size cubic regions to ~
    ``config.fusion_superblock_target_gb`` of output each, so the fuse graph is
    bounded to O(region) instead of the unbounded whole-output ``da.store``
    (which the estimate already assumes). In-memory mode and small outputs keep
    the whole-output path (nothing to bound / no benefit). Shared by the runtime
    and estimate so they never disagree.
    """
    import math

    explicit = int(getattr(config, "fusion_superblock_chunks", 0) or 0)
    if explicit > 0:
        return explicit
    target_gb = float(getattr(config, "fusion_superblock_target_gb", 0.0) or 0.0)
    if not use_streaming or target_gb <= 0:
        return 0

    cs = getattr(config, "output_chunksize", {}) or {}
    cz = max(1, int(cs.get("z", 128)))
    cy = max(1, int(cs.get("y", 256)))
    cx = max(1, int(cs.get("x", 256)))
    n_ch_z = math.ceil(max(1, out_z_px) / cz)
    n_ch_y = math.ceil(max(1, out_y_px) / cy)
    n_ch_x = math.ceil(max(1, out_x_px) / cx)

    total_out_gb = (out_z_px * out_y_px * out_x_px * bpv) / (1024**3)
    # Small outputs: the whole-output graph is already bounded — don't pay the
    # per-region re-fuse overhead.
    if total_out_gb <= 2.0 * target_gb:
        return 0

    chunk_gb = (cz * cy * cx * bpv) / (1024**3)
    region_blocks = max(1.0, target_gb / max(chunk_gb, 1e-9))
    n = max(1, int(round(region_blocks ** (1.0 / 3.0))))
    # A region that already spans the whole output in every axis == whole-output.
    if n >= n_ch_z and n >= n_ch_y and n >= n_ch_x:
        return 0
    return n


def estimate_memory_usage(
    tiles: List["RawTileInfo"],
    channels: List[int],
    config: "StitchingConfig",
) -> Dict[str, float]:
    """Estimate peak memory for in-memory vs streaming stitching modes.

    Returns:
        Dict with keys:
            in_memory_gb: estimated peak RAM for in-memory mode
            streaming_gb: estimated peak RAM for streaming mode
            output_gb: estimated uncompressed output size
            auto_streaming: whether auto-detect would choose streaming
    """
    if not tiles:
        return {
            "in_memory_gb": 0.0,
            "streaming_gb": 0.0,
            "output_gb": 0.0,
            "auto_streaming": False,
            "fusion_gb": 0.0,
            "views_per_block": 0,
        }

    n_channels = len(channels)
    n_planes = max(t.n_planes for t in tiles)
    ds_xy = config.downsample_xy
    ds_z = config.downsample_z
    # Resolve the "iso" sentinel (-1) to concrete factors. Otherwise the
    # `// ds_xy` divisions below run on -1, which silently cancels in the X*Y
    # product and reports the full-resolution size. Needs the Z step; if it's
    # unknown here, fall back to no downsample (a conservative over-estimate).
    if ds_xy == ISO_DOWNSAMPLE or ds_z == ISO_DOWNSAMPLE:
        if config.z_step_um:
            ds_xy, ds_z = compute_iso_downsample(config.pixel_size_um, config.z_step_um)
        else:
            ds_xy, ds_z = 1, 1
    ds_xy = max(1, int(ds_xy))
    ds_z = max(1, int(ds_z))
    frame_w, frame_h = _resolve_frame_size(tiles, config)

    # Estimate output spatial extent from tile positions
    x_vals = [t.x_mm for t in tiles]
    y_vals = [t.y_mm for t in tiles]
    x_range_mm = max(x_vals) - min(x_vals) if len(x_vals) > 1 else 0
    y_range_mm = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 0

    # FOV per tile in mm (approx: pixel_size * frame_width)
    fov_mm = config.pixel_size_um * frame_w / 1000.0

    # Output dimensions in pixels (downsampled)
    out_x_px = int((x_range_mm + fov_mm) / config.pixel_size_um * 1000.0) // ds_xy
    out_y_px = int((y_range_mm + fov_mm) / config.pixel_size_um * 1000.0) // ds_xy
    # Output Z extent must account for Z-TILING (tiles stacked in Z), not just
    # one tile's plane count — otherwise a Z-tiled mosaic under-counts output_gb
    # (and the fused-memmap + disk guard) by ~1/N_z-tiers. Mirror the X/Y range
    # logic: span from the lowest tile start to the highest tile end.
    z_starts = [t.z_min_mm for t in tiles]
    z_ends = [t.z_max_mm for t in tiles]
    z_span_mm = (max(z_ends) - min(z_starts)) if len(tiles) > 1 else 0.0
    per_plane_mm = tiles[0].z_step_mm if tiles else 0.0
    if per_plane_mm > 0 and z_span_mm > 0:
        out_z_full = max(n_planes, int(round(z_span_mm / per_plane_mm)) + 1)
    else:
        out_z_full = n_planes
    out_z_px = out_z_full // ds_z if ds_z > 1 else out_z_full

    # Bytes per voxel
    bpv = 2  # uint16
    output_bytes = n_channels * out_z_px * out_y_px * out_x_px * bpv
    output_gb = output_bytes / (1024**3)

    # Load memory estimation tunables from YAML config
    try:
        from flamingo_stitcher.config_loader import get_stitching_value

        _mem_multiplier = float(
            get_stitching_value("memory", "in_memory_multiplier", default=2.5)
        )
        _fallback_ram = float(
            get_stitching_value("memory", "fallback_system_ram_gb", default=64.0)
        )
        _streaming_threshold = float(
            get_stitching_value("memory", "auto_streaming_threshold", default=0.6)
        )
        _streaming_workers = int(
            get_stitching_value("memory", "streaming_workers", default=4)
        )
        # Fusion working-set model tunables (see the derived per-block cost
        # below). Bytes-per-voxel is 8 because multiview_stitcher.fusion.fuse_np
        # does ``sim.astype(float)`` (float64) on every overlapping view before
        # resampling it onto the output block grid.
        _fusion_float_bytes = float(
            get_stitching_value("memory", "fusion_float_bytes", default=8.0)
        )
        # Coexistence factor: within one block fuse_np holds the per-view list
        # AND the np.array-stacked copy AND (for weighted/content) the blending
        # /fusion weight arrays simultaneously. ~2.5× the raw transformed stack.
        _fusion_coexist = float(
            get_stitching_value("memory", "fusion_coexist_factor", default=2.5)
        )
        # Content-based blending keeps a longer chain of per-block buffers alive
        # than plain blending (see the YAML for the term-by-term count against
        # multiview_stitcher.weights.content_based), so it gets its own factor.
        _fusion_coexist_cb = float(
            get_stitching_value(
                "memory", "fusion_coexist_factor_content_based", default=5.5
            )
        )
        # How many output blocks fuse concurrently (dask thread pool). Matches
        # the _pick_fuse_workers auto cap.
        _fusion_concurrency = int(
            get_stitching_value("memory", "fusion_concurrency", default=4)
        )
        # Deconvolution per-tile float working set as a multiple of the tile's
        # voxel count (float32 in/out + FFT scratch). 0 disables the term.
        _deconv_working_factor = float(
            get_stitching_value("memory", "deconv_working_factor", default=4.0)
        )
    except Exception:
        _mem_multiplier = 2.5
        _fallback_ram = 64.0
        _streaming_threshold = 0.6
        _streaming_workers = 4
        _fusion_float_bytes = 8.0
        _fusion_coexist = 2.5
        _fusion_coexist_cb = 5.5
        _fusion_concurrency = 4
        _deconv_working_factor = 4.0

    import math

    pyramid_overhead_gb = output_gb * 0.33
    per_channel_gb = output_gb / max(n_channels, 1)

    ds_any = ds_xy > 1 or ds_z > 1
    ds_tile_planes = n_planes // ds_z if ds_z > 1 else n_planes
    ds_tile_w = frame_w // ds_xy if ds_xy > 1 else frame_w
    ds_tile_h = frame_h // ds_xy if ds_xy > 1 else frame_h
    ds_tile_vox = ds_tile_planes * ds_tile_w * ds_tile_h
    n_tiles = len(tiles)

    # --- Derived fusion working set (dominant, and applies to BOTH modes) ---
    # multiview_stitcher.fusion.fuse_np processes ONE output block at a time:
    # for every source view overlapping that block it does
    #   transform_sim(sim.astype(float), ...)  -> a float64 array at block size
    # then np.array()-stacks them. Peak per block is therefore
    #   coexist × (views overlapping the block) × (block voxels incl. halo) × 8
    # The dask thread pool runs several blocks at once. This term — NOT the sum
    # of tile sizes — is what made an 8× job estimated at ~10 GB peak ~190 GB.
    # Same chunk resolution the run will use, so the estimate can't disagree
    # with what fusion actually allocates.
    _chunks = resolve_output_chunksize(
        config, tiles, out_shape=(max(out_z_px, 1), max(out_y_px, 1), max(out_x_px, 1))
    )
    chunk_z = _chunks["z"]
    chunk_y = _chunks["y"]
    chunk_x = _chunks["x"]

    # Content-based blending expands each block by a halo of 2×sigma_2 pixels
    # (multiview_stitcher.weights.calculate_required_overlap); plain blend/max
    # use no halo. The halo is resampled at float64 too, so it inflates the
    # per-block cost super-linearly for small chunks.
    content_based = bool(
        getattr(config, "content_based_fusion", False)
        and getattr(config, "tile_overlap_fusion", "max") not in ("max", "brightest")
    )
    halo = 2 * 11 if content_based else 0  # 2 × default sigma_2
    block_vox = (chunk_z + 2 * halo) * (chunk_y + 2 * halo) * (chunk_x + 2 * halo)

    # Views overlapping one output block, from the ACTUAL tile grid pitch
    # (median spacing between adjacent tile centres) rather than a tile-size
    # heuristic — overlapping tiles have pitch < tile size, so more of them
    # land in each block than "chunk / tile_size" would suggest.
    def _pitch_px(values, px_per_unit):
        uniq = sorted({round(v, 4) for v in values})
        if len(uniq) < 2:
            return None
        gaps = [b - a for a, b in zip(uniq, uniq[1:]) if b - a > 1e-6]
        if not gaps:
            return None
        # Use the SMALLEST spacing between adjacent tiles, not the median. For a
        # regular grid they're equal; for irregular/clustered montages the min
        # gap is the densest region, so more tiles land in a block there — the
        # conservative (over-count) direction. A guard should err toward
        # false-abort, never toward under-count → OOM.
        return min(gaps) * px_per_unit, len(uniq)

    px_per_mm_xy = 1000.0 / (config.pixel_size_um * ds_xy)
    px_per_mm_z = 1000.0 / (config.pixel_size_um * ds_z)  # z pitch in xy-px units

    def _views_along(vals, chunk_px, px_per_unit):
        info = _pitch_px(vals, px_per_unit)
        if info is None:
            return 1
        pitch, n_distinct = info
        if pitch <= 0:
            return 1
        return min(n_distinct, int(math.ceil(chunk_px / pitch)) + 1)

    vx = _views_along([t.x_mm for t in tiles], chunk_x, px_per_mm_xy)
    vy = _views_along([t.y_mm for t in tiles], chunk_y, px_per_mm_xy)
    # Z pitch: tiles may share one Z tier (→ 1) or be Z-tiled. z_step converts
    # the block's Z pixels back to the pitch units used for x/y comparison.
    z_vals = [getattr(t, "z_min_mm", 0.0) for t in tiles]
    vz = _views_along(z_vals, chunk_z, px_per_mm_z)
    views_per_block = min(n_tiles, max(1, vx) * max(1, vy) * max(1, vz))

    coexist = _fusion_coexist_cb if content_based else _fusion_coexist
    block_float_gb = (coexist * views_per_block * block_vox * _fusion_float_bytes) / (
        1024**3
    )
    # Use the EFFECTIVE concurrent-block count: if the user pinned fuse_workers
    # (a first-class config field, clamped to 8 by _pick_fuse_workers), the
    # executor runs that many concurrent float64 blocks — model the same number
    # so a fuse_workers override can't under-count the dominant fusion term.
    _eff_fuse = int(getattr(config, "fuse_workers", 0) or 0)
    _eff_fuse = min(_eff_fuse, 8) if _eff_fuse > 0 else _fusion_concurrency
    fusion_gb = _eff_fuse * block_float_gb

    # --- Preprocessing (materialize) working set, both modes ---
    # Sized at NATIVE resolution (preprocessing runs before downsample) via the
    # shared _preprocess_peak_bytes, so this can't diverge from the worker-picker.
    native_vox = int(n_planes) * int(frame_w) * int(frame_h)
    per_worker_pp = _preprocess_peak_bytes(
        config, native_vox, plane_vox=int(frame_w) * int(frame_h)
    )
    # Model the SAME worker count _pick_preprocess_workers will use: honor a
    # preprocess_workers override up to 8, else the auto ceiling of 4. Modelling
    # a flat 4 while the executor runs up to 8 under-counts the preprocess peak
    # by up to 2x (the "safe-then-OOM" bug this whole path exists to prevent —
    # already fixed for fuse_workers/D4, mirrored here).
    _req_pp = int(getattr(config, "preprocess_workers", 0) or 0)
    if _req_pp > 0:
        pp_workers = min(_req_pp, 8)
    elif getattr(config, "destripe", False):
        # Destripe forces sequential preprocessing (see
        # _pick_preprocess_workers). Modelling 4 here would over-state the
        # preprocess floor ~4x and push the run into streaming mode — and
        # inflate the ETA — for memory it will never use.
        pp_workers = 1
    else:
        # Auto mode: _pick_preprocess_workers also caps by AVAILABLE RAM
        # (avail // (per_worker * 2)), so on a box that can't hold 4 native
        # tiles the run uses fewer. Modelling a flat 4 here reported a
        # preprocess floor the run would never reach — and because that floor
        # then clamped BOTH peaks via the max() below, the displayed estimate
        # stopped responding to downsample and in-memory/streaming showed the
        # same number. Mirror the picker exactly so the two can't diverge.
        pp_workers = min(4, _avail_worker_cap(per_worker_pp))
    pp_workers = max(1, min(pp_workers, max(1, n_tiles)))
    materialize_gb = pp_workers * per_worker_pp / (1024**3)

    # --- In-memory peak ---
    # All channels' preprocessed tiles stay resident, plus the fusion working
    # set, plus the stacked output array and pyramid overhead during write.
    held_tiles_gb = (n_tiles * n_channels * ds_tile_vox * bpv) / (1024**3)
    # Materialising one channel: dask's ``.compute()`` keeps every chunk result
    # alive AND builds the concatenated full-size array from them, so the fuse
    # phase peaks at 2x the channel, not 1x. With one channel that result IS the
    # stacked output (``stacked = vol``, no copy); with several, the
    # pre-allocated stacked array coexists with the channel being computed.
    # Writing this out explicitly replaces a `max(pyramid, per_channel)` term
    # that happened to equal 2x output at n_channels == 1 and silently
    # under-counted every multi-channel run.
    fused_materialize_gb = 2.0 * per_channel_gb
    if n_channels > 1:
        fused_materialize_gb += output_gb
    # The write phase holds the stacked array plus the pyramid buffers instead.
    write_phase_gb = output_gb + pyramid_overhead_gb
    in_memory_resident_gb = (
        held_tiles_gb + fusion_gb + max(fused_materialize_gb, write_phase_gb)
    )
    in_memory_gb = max(in_memory_resident_gb, materialize_gb)

    # --- Streaming peak ---
    # Tiles spill to an on-disk memmap and the fused output goes to an on-disk
    # memmap (see _run_streaming), so neither is counted as hard RAM. The
    # pipeline runs materialize THEN fuse per channel, so the peak is the max
    # of the two phases, not their sum. A couple of chunk buffers ride along
    # during the zarr/TIFF write.
    chunk_buffer_gb = _streaming_workers * chunk_z * chunk_y * chunk_x * bpv / (1024**3)
    streaming_gb = max(materialize_gb, fusion_gb + chunk_buffer_gb)

    # Which term the peak is pinned to. Preprocessing runs at NATIVE resolution
    # (before downsample), so once it dominates, turning downsample up stops
    # moving the estimate and both modes report the same number -- surfacing
    # that here is what makes an unresponsive figure explainable instead of
    # looking broken.
    if materialize_gb >= in_memory_resident_gb:
        limited_by = "preprocess"
    elif fusion_gb >= max(output_gb, held_tiles_gb):
        limited_by = "fusion"
    elif output_gb >= held_tiles_gb:
        limited_by = "output"
    else:
        limited_by = "tiles"

    # Auto-detect: stream if in-memory estimate exceeds available RAM
    try:
        import psutil

        system_ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # D2: psutil is a hard dependency; if it's missing (e.g. a stripped
        # frozen build) the memory guards below silently disable themselves and
        # the mode auto-select assumes a 64 GB box — a real OOM risk on a
        # smaller machine. Make it loud rather than silent.
        system_ram_gb = _fallback_ram
        logger.warning(
            "psutil not available — memory estimate assumes a %.0f GB box and "
            "the runtime memory guards/watchdog are DISABLED. Install psutil, "
            "or force Streaming mode + downsample on low-RAM machines.",
            _fallback_ram,
        )
    auto_streaming = in_memory_gb > system_ram_gb * _streaming_threshold

    return {
        "in_memory_gb": round(in_memory_gb, 1),
        "streaming_gb": round(streaming_gb, 1),
        "output_gb": round(output_gb, 1),
        "auto_streaming": auto_streaming,
        "fusion_gb": round(fusion_gb, 1),
        "views_per_block": int(views_per_block),
        "preprocess_gb": round(materialize_gb, 1),
        "preprocess_workers": int(pp_workers),
        "limited_by": limited_by,
        # Peak of the in-memory fuse phase: the pre-allocated stacked array (if
        # more than one channel) plus the two copies of the channel being
        # computed that dask's .compute() holds at once. Returned so the figure
        # is inspectable rather than buried in in_memory_gb.
        "materialize_fused_gb": round(fused_materialize_gb, 2),
        "held_tiles_gb": round(held_tiles_gb, 2),
    }


# ---------------------------------------------------------------------------
# Tile metadata
# ---------------------------------------------------------------------------
@dataclass
class RawTileInfo:
    """Parsed metadata for a single tile's raw files."""

    folder: Path
    x_mm: float  # Stage X position in mm
    y_mm: float  # Stage Y position in mm
    z_min_mm: float  # Z sweep start in mm
    z_max_mm: float  # Z sweep end in mm
    n_planes: int
    # channel_id -> {illumination_side -> raw_file_path}
    raw_files: Dict[int, Dict[int, Path]] = field(default_factory=dict)
    channels: List[int] = field(default_factory=list)
    illumination_sides: List[int] = field(default_factory=list)
    # Multi-view acquisition provenance. ``view``/``rotation_index`` come from the
    # filename V###/R#### fields; ``angle_deg`` is the physical rotation-stage angle
    # read from Workflow.txt (<Start Position> Angle (degrees)). All default to the
    # single-angle case (0) so existing acquisitions are unaffected.
    view: int = 0
    rotation_index: int = 0
    angle_deg: float = 0.0
    # Raw frame (camera AOI) dimensions in pixels. Resolved per-acquisition
    # from the Workflow.txt `AOI width`/`AOI height`, cross-checked against the
    # actual on-disk file size (the file always wins — see _resolve_tile_frame_dims).
    # Defaulting to the module FRAME_WIDTH/HEIGHT keeps older call sites valid.
    frame_width: int = FRAME_WIDTH
    frame_height: int = FRAME_HEIGHT
    # Set when discovery had to degrade or infer this tile — e.g. a corrupt /
    # unreadable _Settings.txt (position taken from the Workflow.txt grid) or a
    # raw file whose byte size doesn't match its plane/frame geometry (possibly
    # truncated). Kept so the caller (GUI/CLI) can WARN the user visibly rather
    # than only logging it: the run can continue, but the data may be off.
    metadata_warning: Optional[str] = None
    # Integer grid index (x_idx, y_idx) from the flat filename (X###_Y###), when
    # available. Used for user-facing tile labels (e.g. the orientation preview);
    # None for folder-layout acquisitions whose folders carry mm, not indices.
    tile_index: Optional[Tuple[int, int]] = None

    @property
    def z_step_mm(self) -> float:
        if self.n_planes <= 1:
            return 0.0
        return (self.z_max_mm - self.z_min_mm) / (self.n_planes - 1)


# ---------------------------------------------------------------------------
# Frame (camera AOI) size resolution
# ---------------------------------------------------------------------------
def _read_aoi_from_workflow(workflow_file: Path) -> Optional[Tuple[int, int]]:
    """Read `AOI width`/`AOI height` (camera sensor crop) from a Workflow.txt.

    Returns (width, height) in pixels, or None if the file/fields are absent.
    """
    try:
        if not workflow_file.exists():
            return None
        content = _read_text_resilient(workflow_file)
    except OSError:
        return None
    w = re.search(r"AOI width\s*=\s*(\d+)", content)
    h = re.search(r"AOI height\s*=\s*(\d+)", content)
    if w and h:
        return (int(w.group(1)), int(h.group(1)))
    return None


def _read_tiff_shape(path: Path) -> Tuple[int, int, int]:
    """Return ``(n_planes, height, width)`` of a (Big)TIFF tile stack from its
    header — no pixel data is read. Page count is the plane count; the first
    page's shape gives Y, X."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        n_planes = len(tf.pages)
        shape = tf.pages[0].shape  # (Y, X) for a 2-D page
        height, width = int(shape[0]), int(shape[1])
    return n_planes, height, width


def _resolve_tile_geometry(
    sample_file: Path, n_planes: int, aoi: Optional[Tuple[int, int]]
) -> Tuple[int, int, int]:
    """Resolve ``(n_planes, frame_width, frame_height)`` for a tile.

    For TIFF / BigTIFF everything is authoritative from the file header (page
    count + page shape), so the AOI-from-file-size guesswork is not needed. For
    raw, the plane count stays as parsed from the filename and the frame dims
    come from the file size (the data wins over stale AOI metadata).
    """
    if sample_file.suffix.lower() in _TIFF_EXTENSIONS:
        try:
            tif_planes, height, width = _read_tiff_shape(sample_file)
            return (tif_planes or n_planes), width, height
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"{sample_file.name}: could not read TIFF header ({e}); "
                f"falling back to filename planes={n_planes} and AOI metadata."
            )
            fw, fh = aoi or (FRAME_WIDTH, FRAME_HEIGHT)
            return n_planes, fw, fh
    fw, fh = _resolve_tile_frame_dims(sample_file, n_planes, aoi)
    return n_planes, fw, fh


def _resolve_tile_frame_dims(
    raw_file: Path, n_planes: int, aoi: Optional[Tuple[int, int]]
) -> Tuple[int, int]:
    """Determine the (width, height) of the raw frames for one tile.

    The on-disk file size is the ground truth: ``bytes / (n_planes * 2)`` is the
    exact pixel count per plane (uint16). The AOI metadata only disambiguates
    non-square frames. This catches the case where the camera AOI was cropped
    (e.g. 1024×1024) but the metadata or a stale hardware config still says
    2048×2048 — the data wins, so cropped acquisitions load correctly.
    """
    import math

    fallback = aoi or (FRAME_WIDTH, FRAME_HEIGHT)
    try:
        nbytes = raw_file.stat().st_size
    except OSError:
        return fallback
    if n_planes <= 0 or nbytes <= 0:
        return fallback

    px_per_plane = nbytes // (n_planes * 2)
    # If the AOI metadata reproduces the file size exactly, trust it (handles
    # non-square frames that the square-inference branch below cannot).
    if aoi and aoi[0] * aoi[1] == px_per_plane:
        return aoi

    # Flamingo AOIs are square; infer the side from the file.
    side = math.isqrt(px_per_plane)
    if side > 0 and side * side == px_per_plane:
        if aoi and (aoi[0], aoi[1]) != (side, side):
            logger.warning(
                f"{raw_file.name}: AOI metadata {aoi[0]}×{aoi[1]} disagrees with "
                f"file size ({side}×{side} for {n_planes} planes) — using "
                f"file-derived {side}×{side}."
            )
        return (side, side)

    logger.warning(
        f"{raw_file.name}: cannot derive a square frame size from "
        f"{nbytes} bytes / {n_planes} planes; using {fallback[0]}×{fallback[1]}."
    )
    return fallback


def _resolve_frame_size(
    tiles: List["RawTileInfo"], config: Optional["StitchingConfig"] = None
) -> Tuple[int, int]:
    """Frame (AOI) width/height to use for geometry and memory math.

    Precedence: explicit config override → first tile's resolved dims →
    module default.
    """
    if config is not None:
        cw = getattr(config, "frame_width", None)
        ch = getattr(config, "frame_height", None)
        if cw and ch:
            return int(cw), int(ch)
    for t in tiles or []:
        if getattr(t, "frame_width", None) and getattr(t, "frame_height", None):
            return int(t.frame_width), int(t.frame_height)
    return FRAME_WIDTH, FRAME_HEIGHT


# Below this fraction of a frame, "overlap" is not usable: registration has no
# shared content to correlate and the frame-edge falloff has nowhere to hide.
MIN_USEFUL_TILE_OVERLAP = 0.03

# Below this fraction, registration should not be ATTEMPTED. Distinct from the
# 3% above, which is "the tiling itself is broken": phase correlation on a
# sliver of shared content does not fail loudly, it returns a confident garbage
# peak — which is exactly the failure the shift clamp exists to mop up after the
# fact. Better to place by stage position and say so than to clamp half the
# tiles and call the result alignment.
MIN_REGISTRATION_OVERLAP = 0.05


# Auto Z shift bound (see StitchingPipeline._axial_shift_bound). The reported
# real-world error is 3-6 frames between neighbours, so 8 steps clears it with
# margin; the 25 µm floor covers finer Z steps where 8 steps is still tiny; the
# quarter-stack cap keeps a shallow stack from admitting a peak found halfway
# down the volume.
# Fewest tiles for which "what did the mosaic agree on" means anything. With
# two disagreeing tiles there is no majority and no way to say which one moved.
_MIN_TILES_FOR_CONSENSUS = 3

_Z_CLAMP_MIN_STEPS = 8
_Z_CLAMP_FLOOR_UM = 25.0
_Z_CLAMP_STACK_FRACTION = 0.25


class ClampRecord(NamedTuple):
    """What the shift clamp saw and did for one tile.

    ``dz_um``/``dy_um``/``dx_um`` are the shifts registration proposed, BEFORE
    any clamping — the report needs the rejected value, not the zero that
    replaced it, because "we measured +97 µm and did not believe it" and "we
    measured nothing" are different findings.

    ``rel_*`` are those shifts minus the neighbours' consensus, and are the
    values the bound was tested against.
    """

    index: int
    dz_um: float
    dy_um: float
    dx_um: float
    # The same correction with the neighbours' consensus removed. This is what
    # the bound is applied to: a mosaic-wide offset or a slow drift moves every
    # tile together and opens no seam, so only the disagreement can be judged.
    rel_z_um: float = 0.0
    rel_y_um: float = 0.0
    rel_x_um: float = 0.0
    clamped_xy: bool = False
    clamped_z: bool = False
    whole_matrix: bool = False


class ClampResult(NamedTuple):
    """Clamped params plus the evidence of what was clamped and why."""

    params: list
    records: List[ClampRecord]
    bound_xy_um: Optional[float] = None
    bound_z_um: Optional[float] = None
    source_xy: str = ""
    source_z: str = ""

    def summary_line(self) -> str:
        n = len(self.records)
        n_xy = sum(1 for r in self.records if r.clamped_xy and not r.whole_matrix)
        n_z = sum(1 for r in self.records if r.clamped_z and not r.whole_matrix)
        n_whole = sum(1 for r in self.records if r.whole_matrix)
        seen_z = max((abs(r.dz_um) for r in self.records), default=0.0)
        seen_y = max((abs(r.dy_um) for r in self.records), default=0.0)
        seen_x = max((abs(r.dx_um) for r in self.records), default=0.0)
        return (
            f"  Registration shift clamp: bounds "
            f"z={self.bound_z_um:.1f} µm ({self.source_z}), "
            f"xy={self.bound_xy_um:.1f} µm ({self.source_xy}); "
            f"clamped z on {n_z}/{n} tiles, xy on {n_xy}/{n}, "
            f"whole-matrix reverts {n_whole}/{n} "
            f"(max shift seen z={seen_z:.1f} y={seen_y:.1f} x={seen_x:.1f} µm). "
            f"A clamped axis was NOT measured — it kept its stage position."
        )


def _detect_tile_spacing_gaps(
    tiles: List["RawTileInfo"],
    frame_width: int,
    frame_height: int,
    pixel_size_um: float,
) -> List[str]:
    """Flag axes where tiles are stepped farther apart than one frame covers.

    Tiles are placed at their *acquired* stage positions and drawn at
    ``frame_dim × pixel_size``. If the median step along an axis exceeds that
    coverage, the mosaic will have BLANK gaps between tiles — missing acquired
    data, not a stitching error. The classic cause is a non-square AOI whose
    tile step was computed from a square (width-only) field of view during
    acquisition, so the short axis is stepped as if it were the long one
    (produces full-width tiles with regular black bands between rows).

    Returns one human-readable warning string per gapped axis (empty when tiles
    touch or overlap on both axes). Reads only ``x_mm``/``y_mm`` off each tile
    (duck-typed) and does no I/O, so it is cheap and unit-testable.
    """
    if not tiles or not pixel_size_um or pixel_size_um <= 0:
        return []
    cov_mm = {
        "X": frame_width * pixel_size_um / 1000.0,
        "Y": frame_height * pixel_size_um / 1000.0,
    }
    # RAW camera frame, not the processed shape: this warning is about how the
    # acquisition was set up, so it must not move when someone downsamples.
    layout = tile_geometry.grid_overlap(
        tiles,
        extent_x_um=cov_mm["X"] * 1000.0,
        extent_y_um=cov_mm["Y"] * 1000.0,
    )
    warnings: List[str] = []
    for axis in ("X", "Y"):
        coverage = cov_mm[axis]
        if coverage <= 0:
            continue
        pitch_um = layout[axis.lower()].pitch_um
        if pitch_um is None:
            continue
        median_step = pitch_um / 1000.0
        # Overlapping/touching tiles have step <= coverage. Only warn on a clear
        # gap (>2% of a frame) so float noise / exact-touch don't false-positive.
        if median_step <= coverage * 1.02:
            # ...but "not a gap" is not the same as "enough overlap". Tiles
            # stepped at ~exactly one frame abut without overlapping, which
            # leaves nothing to register on (phase correlation needs shared
            # content) and no margin to hide each frame's edge falloff, so
            # every seam shows the vignette step. It also means any stage
            # drift becomes a real gap. This sits in the old check's blind
            # spot: a 0.01% overlap passes the gap test silently.
            overlap_frac = (coverage - median_step) / coverage
            if overlap_frac < MIN_USEFUL_TILE_OVERLAP:
                overlap_px = (coverage - median_step) * 1000.0 / pixel_size_um
                warnings.append(
                    f"{axis} tiles barely overlap: median {axis} step "
                    f"{median_step:.4f} mm vs one frame's {axis} coverage "
                    f"{coverage:.4f} mm — only {overlap_frac * 100:.2f}% "
                    f"({overlap_px:.1f} px) of overlap. Tile registration "
                    f"cannot work without shared content (the run will fall "
                    f"back to stage positions), and with no overlap each "
                    f"frame's edge falloff lands directly on the seam — "
                    f"visible as a brightness step at every tile boundary. "
                    f"Acquire with ~10% tile overlap."
                )
            continue
        gap_mm = median_step - coverage
        gap_px = gap_mm * 1000.0 / pixel_size_um
        frame_dim = frame_width if axis == "X" else frame_height
        other_axis = "Y" if axis == "X" else "X"
        other_cov = cov_mm[other_axis]
        msg = (
            f"{axis} tiles do not overlap: median {axis} step "
            f"{median_step:.3f} mm exceeds one frame's {axis} coverage "
            f"{coverage:.3f} mm ({frame_dim} px × {pixel_size_um:.3f} µm). The "
            f"mosaic will have ~{gap_mm:.3f} mm ({gap_px:.0f} px) blank gaps "
            f"between {'columns' if axis == 'X' else 'rows'} — this is missing "
            f"acquired data, not a stitching error."
        )
        # Strong hint at the square-FOV cause: the step matches the OTHER axis's
        # frame coverage, i.e. this axis was stepped as if the frame were square.
        if (
            frame_width != frame_height
            and other_cov > 0
            and abs(median_step - other_cov) < abs(median_step - coverage)
        ):
            other_dim = frame_height if axis == "X" else frame_width
            msg += (
                f" The step ≈ the {other_axis}-frame coverage "
                f"({other_cov:.3f} mm), i.e. {axis} was stepped as if the frame "
                f"were square ({other_dim} px): a non-square AOI "
                f"({frame_width}×{frame_height}) whose tile step was computed "
                f"from a square field of view during acquisition. Re-acquire with "
                f"a square AOI or explicit per-tile geometry."
            )
        warnings.append(msg)
    return warnings


# ---------------------------------------------------------------------------
# Optics / acquisition-flag parsing (objective, capture mode, angles)
# ---------------------------------------------------------------------------
def _sensor_pixel_size_um() -> float:
    """Physical camera sensor pixel pitch in µm (objective-independent)."""
    try:
        from flamingo_stitcher.config_loader import get_hardware_config

        return float(get_hardware_config().sensor_pixel_size_um)
    except Exception:
        return 6.5


def _find_acquisition_file(
    acquisition_dir: Path, name: str, max_depth: int = 4
) -> Optional[Path]:
    """Locate a metadata file (e.g. ScopeSettings.txt / Workflow.txt) near an
    acquisition.

    Searches, nearest first: the dir itself, its parent, then DESCENDANTS
    breadth-first down to ``max_depth`` levels (shallowest match wins). The
    descent matters because a user often selects a name-level folder
    (``BrainSingleChannel2/``) whose ScopeSettings.txt actually lives a couple
    levels down under a date-stamped subfolder — tile discovery recurses and
    finds the tiles there, so this must reach just as deep or the objective /
    pixel size silently falls back to a wrong default.
    """
    acq = Path(acquisition_dir)
    # Nearest-first exact locations.
    for c in (acq / name, acq.parent / name):
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    # Breadth-first descent into subdirectories, shallowest match first.
    frontier = [acq]
    depth = 0
    while frontier and depth < max_depth:
        next_frontier = []
        for d in frontier:
            try:
                children = sorted(d.iterdir())
            except OSError:
                continue
            for child in children:
                try:
                    if not child.is_dir():
                        continue
                except OSError:
                    continue
                cand = child / name
                try:
                    if cand.is_file():
                        return cand
                except OSError:
                    pass
                next_frontier.append(child)
        frontier = next_frontier
        depth += 1
    return None


def read_objective_magnification(acquisition_dir: Path) -> Optional[float]:
    """Read `Objective lens magnification` from the acquisition's ScopeSettings.txt.

    This is the system magnification the microscope recorded at capture time, so
    it tracks objective swaps that the static hardware config does not.
    """
    f = _find_acquisition_file(acquisition_dir, "ScopeSettings.txt")
    if f is None:
        return None
    try:
        content = _read_text_resilient(f)
    except OSError:
        return None
    m = re.search(r"Objective lens magnification\s*=\s*([\d.]+)", content)
    if m:
        try:
            mag = float(m.group(1))
            return mag if mag > 0 else None
        except ValueError:
            return None
    return None


def read_objective_magnification_metadata(
    acquisition_dir: Path,
) -> Optional[float]:
    """Read `Objective lens magnification` from FlamingoMetaData*.txt.

    This is the objective the microscope recorded in the *acquisition* metadata
    (captured at run time), which is a second, independent source from
    ScopeSettings.txt. When the two disagree it usually means ScopeSettings.txt
    holds a stale objective while FlamingoMetaData reflects what was actually
    used — a mismatch that silently rescales every tile. Returns None when no
    metadata file / value is found. Never raises.
    """
    acq = Path(acquisition_dir)
    bases = [acq, acq.parent]
    try:
        bases.extend(c for c in sorted(acq.iterdir()) if c.is_dir())
    except OSError:
        pass
    for base in bases:
        try:
            matches = sorted(base.glob("FlamingoMetaData*.txt"))
        except OSError:
            continue
        for f in matches:
            try:
                content = _read_text_resilient(f)
            except OSError:
                continue
            m = re.search(
                r"Objective lens magnification\s*=\s*([\d.]+)", content
            )
            if m:
                try:
                    mag = float(m.group(1))
                    if mag > 0:
                        return mag
                except ValueError:
                    continue
    return None


def _config_objective_for_microscope(name: Optional[str]) -> Optional[float]:
    """Per-microscope ``objective_magnification`` from microscope_hardware.yaml.

    Reads the optional ``microscopes:`` block, keyed by microscope name
    (case-insensitive). Lets a system whose acquisitions don't record the
    objective in ScopeSettings.txt (e.g. Liara) still resolve the right pixel
    size. Never raises.
    """
    if not name:
        return None
    try:
        import yaml

        from flamingo_stitcher.config_loader import _CONFIGS_DIR

        yaml_path = _CONFIGS_DIR / "microscope_hardware.yaml"
        if not yaml_path.is_file():
            return None
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        scopes = (raw.get("microscopes") or {}) if isinstance(raw, dict) else {}
        entry = scopes.get(str(name).strip().lower()) or {}
        mag = entry.get("objective_magnification") if isinstance(entry, dict) else None
        return float(mag) if mag and float(mag) > 0 else None
    except Exception:  # noqa: BLE001 - config lookup is best-effort
        return None


def suggested_pixel_size_um(acquisition_dir: Path) -> Optional[float]:
    """Effective XY pixel size for an acquisition.

    pixel = sensor_pixel_size / objective_magnification. Prefers the objective
    the microscope recorded in ScopeSettings.txt; falls back to a per-microscope
    ``objective_magnification`` from microscope_hardware.yaml (keyed by
    "Microscope name") for systems that don't record it. Returns None when
    neither is available.
    """
    mag = read_objective_magnification(acquisition_dir)
    if not mag:
        from flamingo_stitcher.orientation import read_microscope_name

        mag = _config_objective_for_microscope(read_microscope_name(acquisition_dir))
    if not mag:
        return None
    return _sensor_pixel_size_um() / mag


def _read_capture_and_angles(workflow_file: Optional[Path]) -> Dict[str, object]:
    """Read partial-capture and multi-angle flags from a Workflow.txt.

    Returns a dict with `capture_modes` (list of int per camera),
    `capture_percents` (list of int), and `n_angles` (int).
    """
    out: Dict[str, object] = {
        "capture_modes": [],
        "capture_percents": [],
        "n_angles": 1,
    }
    if workflow_file is None or not workflow_file.exists():
        return out
    try:
        content = _read_text_resilient(workflow_file)
    except OSError:
        return out
    modes = [
        int(m) for m in re.findall(r"Camera \d+ capture mode[^=]*=\s*(\d+)", content)
    ]
    pcts = [
        int(p)
        for p in re.findall(r"Camera \d+ capture percentage\s*=\s*(\d+)", content)
    ]
    out["capture_modes"] = modes
    out["capture_percents"] = pcts
    na = re.search(r"Number of angles\s*=\s*(\d+)", content)
    if na:
        try:
            out["n_angles"] = max(1, int(na.group(1)))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------------
# Parsing (self-contained, no GUI dependencies)
# ---------------------------------------------------------------------------
def discover_tiles(acquisition_dir: Path) -> List[RawTileInfo]:
    """Discover all tile folders in an acquisition directory.

    Handles two layouts:
      1. Flat: acquisition_dir/ contains X{x}_Y{y}/ folders directly
      2. Dated: acquisition_dir/{date}/ contains X{x}_Y{y}/ folders

    Returns list of RawTileInfo sorted by (Y, X).
    """
    tiles = []
    candidates = []

    # Check for tile folders directly
    for d in sorted(acquisition_dir.iterdir()):
        if not d.is_dir():
            continue
        if FOLDER_COORD_PATTERN.search(d.name):
            candidates.append(d)

    # If none found, look one level deeper (date subdirectories)
    if not candidates:
        for sub in sorted(acquisition_dir.iterdir()):
            if not sub.is_dir():
                continue
            for d in sorted(sub.iterdir()):
                if d.is_dir() and FOLDER_COORD_PATTERN.search(d.name):
                    candidates.append(d)

    for folder in candidates:
        try:
            tile = _parse_tile_folder(folder)
            if tile:
                tiles.append(tile)
        except Exception as e:
            logger.warning(f"Skipping {folder.name}: {e}")

    # Sort by Y then X for predictable ordering
    tiles.sort(key=lambda t: (t.y_mm, t.x_mm))
    logger.info(f"Discovered {len(tiles)} tiles in {acquisition_dir}")
    return tiles


def _parse_tile_folder(folder: Path) -> Optional[RawTileInfo]:
    """Parse a single tile folder for raw files and metadata."""
    # Extract coordinates from folder name
    match = FOLDER_COORD_PATTERN.search(folder.name)
    if not match:
        return None
    x_mm = float(match.group(1))
    y_mm = float(match.group(2))

    # Parse Workflow.txt for Z range
    z_min, z_max = _read_z_range(folder)

    # Discover raw files
    raw_files: Dict[int, Dict[int, Path]] = {}
    channels = set()
    illum_sides = set()
    views = set()
    rot_indices = set()
    n_planes = 0

    for f in sorted(folder.iterdir()):
        m = RAW_FILE_PATTERN.match(f.name)
        if m:
            ch = int(m.group("ch"))
            illum = int(m.group("illum"))
            planes = int(m.group("planes"))

            channels.add(ch)
            illum_sides.add(illum)
            views.add(int(m.group("view")))
            rot_indices.add(int(m.group("rot")))
            n_planes = max(n_planes, planes)

            if ch not in raw_files:
                raw_files[ch] = {}
            raw_files[ch][illum] = f

    if not raw_files:
        logger.warning(f"No .raw files in {folder.name}")
        return None

    angle_deg = _read_start_angle(folder / "Workflow.txt")

    # Resolve the raw frame (AOI) size: prefer Workflow.txt metadata, but let
    # the actual file size override it (handles cropped-AOI acquisitions).
    aoi = _read_aoi_from_workflow(folder / "Workflow.txt")
    _sample = next(iter(next(iter(raw_files.values())).values()))
    n_planes, frame_w, frame_h = _resolve_tile_geometry(_sample, n_planes, aoi)

    return RawTileInfo(
        folder=folder,
        x_mm=x_mm,
        y_mm=y_mm,
        z_min_mm=z_min,
        z_max_mm=z_max,
        n_planes=n_planes,
        raw_files=raw_files,
        channels=sorted(channels),
        illumination_sides=sorted(illum_sides),
        view=min(views) if views else 0,
        rotation_index=min(rot_indices) if rot_indices else 0,
        angle_deg=angle_deg,
        frame_width=frame_w,
        frame_height=frame_h,
    )


def _read_text_resilient(
    path: Path,
    *,
    retries: int = 4,
    backoff_s: float = 0.25,
) -> str:
    """Read a small text metadata file, retrying transient OS read errors.

    External drives — USB-C enclosures in particular — intermittently raise an
    ``OSError`` (on Windows, ``[WinError 1392] The file or directory is
    corrupted and unreadable`` / ``ERROR_FILE_CORRUPT``) on a read that
    succeeds a moment later; the file itself is fine and opens normally in
    another program. These ``*_Settings.txt`` / ``Workflow.txt`` companions are
    tiny, so a short bounded retry with linear backoff recovers the read
    without meaningfully slowing discovery. Re-raises the last ``OSError`` only
    if every attempt fails, so callers can degrade gracefully.
    """
    last_err: Optional[OSError] = None
    for attempt in range(1, retries + 1):
        try:
            return path.read_text(errors="replace")
        except OSError as e:
            last_err = e
            logger.warning(
                "Transient read error on %s (attempt %d/%d): %s",
                path.name,
                attempt,
                retries,
                e,
            )
            if attempt < retries:
                time.sleep(backoff_s * attempt)
    assert last_err is not None  # loop ran at least once
    raise last_err


def _read_z_range(folder: Path) -> Tuple[float, float]:
    """Read Z range from Workflow.txt in folder. Falls back to defaults."""
    wf = folder / "Workflow.txt"
    if not wf.exists():
        logger.warning(f"No Workflow.txt in {folder.name}, using default Z range")
        return (0.0, 1.0)

    content = _read_text_resilient(wf)

    z_min = 0.0
    z_max = 1.0

    start = re.search(r"<Start Position>.*?Z \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if start:
        z_min = float(start.group(1))

    end = re.search(r"<End Position>.*?Z \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if end:
        z_max = float(end.group(1))

    return (z_min, z_max)


def _read_start_angle(workflow_file: Path) -> float:
    """Read the rotation-stage angle (degrees) from a Workflow.txt.

    Uses ``<Start Position> ... Angle (degrees)`` — the physical stage angle for
    the tile, which is the authoritative source (the filename R#### field is an
    index, not necessarily degrees). Returns 0.0 when absent (single-angle case).
    """
    try:
        if not workflow_file.exists():
            return 0.0
        content = _read_text_resilient(workflow_file)
    except OSError:
        return 0.0
    m = re.search(
        r"<Start Position>.*?Angle \(degrees\) = ([-\d.]+)", content, re.DOTALL
    )
    return float(m.group(1)) if m else 0.0


def _read_position_from_settings(settings_file: Path) -> Dict[str, float]:
    """Read stage position from a _Settings.txt companion file.

    Parses <Start Position> for X, Y, Z and <End Position> for Z end.

    Returns:
        Dict with keys: x_mm, y_mm, z_min_mm, z_max_mm
    """
    content = _read_text_resilient(settings_file)

    result = {"x_mm": 0.0, "y_mm": 0.0, "z_min_mm": 0.0, "z_max_mm": 1.0}

    start = re.search(r"<Start Position>.*?X \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if start:
        result["x_mm"] = float(start.group(1))

    start_y = re.search(r"<Start Position>.*?Y \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if start_y:
        result["y_mm"] = float(start_y.group(1))

    start_z = re.search(r"<Start Position>.*?Z \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if start_z:
        result["z_min_mm"] = float(start_z.group(1))

    end_z = re.search(r"<End Position>.*?Z \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if end_z:
        result["z_max_mm"] = float(end_z.group(1))

    return result


def _read_plane_spacing(workflow_file: Path) -> Optional[float]:
    """Read plane spacing from a Workflow.txt file.

    Returns:
        Plane spacing in µm, or None if not found.
    """
    if not workflow_file.exists():
        return None

    content = _read_text_resilient(workflow_file)
    match = re.search(r"Plane spacing \(um\) = ([\d.]+)", content)
    if match:
        return float(match.group(1))
    return None


def discover_flat_tiles(acquisition_dir: Path) -> List[RawTileInfo]:
    """Discover tiles in a flat-layout acquisition (C++ server native format).

    In flat layout, all .raw files live in a single directory with integer
    tile indices (X000_Y000, X001_Y000, etc.) rather than subfolder-per-tile.
    Each .raw file may have a companion _Settings.txt with stage positions.

    Handles two layouts:
      1. Flat: acquisition_dir/ contains .raw files directly
      2. Dated: acquisition_dir/{date}/ contains .raw files

    Returns list of RawTileInfo sorted by (Y, X).
    """
    # Find the directory containing tile files (raw / tif / tiff / btf)
    raw_dir = None
    raw_files_found = _glob_tile_files(acquisition_dir)
    if raw_files_found:
        raw_dir = acquisition_dir
    else:
        # Check one level deeper (date subdirectories)
        for sub in sorted(acquisition_dir.iterdir()):
            if sub.is_dir() and _glob_tile_files(sub):
                raw_dir = sub
                break

    if raw_dir is None:
        logger.warning(f"No .raw files found in {acquisition_dir}")
        return []

    # Group raw files by (X_idx, Y_idx)
    tile_groups: Dict[Tuple[int, int], List[Path]] = {}
    for f in sorted(raw_dir.iterdir()):
        m = FLAT_RAW_PATTERN.search(f.name)
        if m:
            x_idx, y_idx = int(m.group("xidx")), int(m.group("yidx"))
            tile_groups.setdefault((x_idx, y_idx), []).append(f)

    if not tile_groups:
        logger.warning(f"No files matching flat raw pattern in {raw_dir}")
        return []

    # Try to read plane spacing from root Workflow.txt for fallback position calc
    root_wf = acquisition_dir / "Workflow.txt"
    if not root_wf.exists():
        root_wf = raw_dir / "Workflow.txt"

    # Grid extent for the position fallback, derived from the discovered tile
    # indices themselves rather than an ambiguous Workflow.txt field.
    n_tiles_x = max(x for x, _ in tile_groups) + 1
    n_tiles_y = max(y for _, y in tile_groups) + 1

    # AOI metadata is shared across the flat acquisition (one camera setting).
    flat_aoi = _read_aoi_from_workflow(root_wf)

    tiles = []
    for (x_idx, y_idx), files in sorted(tile_groups.items()):
        # Find a _Settings.txt companion (from first raw file in group)
        settings_file = None
        for f in files:
            candidate = f.with_name(f.stem + "_Settings.txt")
            if candidate.exists():
                settings_file = candidate
                break

        # Parse position from _Settings.txt or fall back to root Workflow.txt.
        # A tile whose _Settings.txt stays unreadable after the retries in
        # _read_text_resilient (e.g. a persistent USB-C WinError 1392) must NOT
        # abort discovery of the whole acquisition — degrade to the same
        # grid-computed position used when the companion file is simply absent.
        pos = None
        tile_warning: Optional[str] = None
        if settings_file:
            try:
                pos = _read_position_from_settings(settings_file)
            except OSError as e:
                logger.warning(
                    "Unreadable settings file %s (%s); falling back to grid "
                    "position for tile X%d_Y%d",
                    settings_file.name,
                    e,
                    x_idx,
                    y_idx,
                )
                tile_warning = (
                    f"corrupt/unreadable metadata ({settings_file.name}); "
                    f"position estimated from the acquisition grid"
                )
                settings_file = None  # also route the angle read to the fallback
        if pos is None:
            if root_wf.exists():
                # Fallback: compute from root Workflow.txt grid
                pos = _compute_grid_position(
                    root_wf, x_idx, y_idx, n_tiles_x, n_tiles_y
                )
            else:
                logger.warning(
                    f"No readable _Settings.txt or Workflow.txt for tile "
                    f"X{x_idx}_Y{y_idx}, using default positions"
                )
                pos = {"x_mm": 0.0, "y_mm": 0.0, "z_min_mm": 0.0, "z_max_mm": 1.0}
                tile_warning = (
                    "no readable position metadata (_Settings.txt / "
                    "Workflow.txt); using default (0, 0) — tile placement "
                    "is unreliable"
                )

        # Parse raw files for channel/illumination/planes metadata
        raw_files_dict: Dict[int, Dict[int, Path]] = {}
        channels = set()
        illum_sides = set()
        views = set()
        rot_indices = set()
        n_planes = 0

        for f in files:
            m = FLAT_RAW_PATTERN.search(f.name)
            if m:
                ch = int(m.group("ch"))
                illum = int(m.group("illum"))
                planes = int(m.group("planes"))

                channels.add(ch)
                illum_sides.add(illum)
                views.add(int(m.group("view")))
                rot_indices.add(int(m.group("rot")))
                n_planes = max(n_planes, planes)

                if ch not in raw_files_dict:
                    raw_files_dict[ch] = {}
                raw_files_dict[ch][illum] = f

        if not raw_files_dict:
            continue

        _sample = next(iter(next(iter(raw_files_dict.values())).values()))
        n_planes, frame_w, frame_h = _resolve_tile_geometry(_sample, n_planes, flat_aoi)

        # Image-data sanity: a raw file whose byte size doesn't match its
        # plane/frame geometry is likely truncated or corrupt. Flag it (the run
        # can still proceed) so the caller can warn visibly.
        img_warn = _raw_size_warning(_sample, n_planes, frame_w, frame_h)
        tile_warning = "; ".join(w for w in (tile_warning, img_warn) if w) or None

        # Rotation-stage angle: prefer the tile's _Settings.txt, else root Workflow.txt.
        angle_deg = _read_start_angle(settings_file) if settings_file else 0.0
        if angle_deg == 0.0 and root_wf.exists():
            angle_deg = _read_start_angle(root_wf)

        tiles.append(
            RawTileInfo(
                folder=raw_dir,
                x_mm=pos["x_mm"],
                y_mm=pos["y_mm"],
                z_min_mm=pos["z_min_mm"],
                z_max_mm=pos["z_max_mm"],
                n_planes=n_planes,
                raw_files=raw_files_dict,
                channels=sorted(channels),
                illumination_sides=sorted(illum_sides),
                view=min(views) if views else 0,
                rotation_index=min(rot_indices) if rot_indices else 0,
                angle_deg=angle_deg,
                frame_width=frame_w,
                frame_height=frame_h,
                metadata_warning=tile_warning,
                tile_index=(x_idx, y_idx),
            )
        )

    tiles.sort(key=lambda t: (t.y_mm, t.x_mm))
    logger.info(f"Discovered {len(tiles)} flat-layout tiles in {acquisition_dir}")
    return tiles


def _raw_size_warning(
    path: Path, n_planes: int, frame_w: int, frame_h: int
) -> Optional[str]:
    """Return a warning if a ``.raw`` file's byte size is inconsistent.

    A uint16 raw stack should be exactly ``n_planes * frame_w * frame_h * 2``
    bytes. A smaller file is truncated; a mismatch either way suggests corrupt
    or wrongly-sized data. Only checked for ``.raw`` (TIFF sizes vary with
    compression). Read errors are themselves reported as a corruption signal.
    """
    try:
        if path.suffix.lower() != ".raw":
            return None
        if n_planes <= 0 or frame_w <= 0 or frame_h <= 0:
            return None
        expected = int(n_planes) * int(frame_w) * int(frame_h) * 2
        actual = path.stat().st_size
        if actual == expected:
            return None
        if actual < expected:
            pct = 100.0 * actual / expected if expected else 0.0
            return (
                f"image file {path.name} looks truncated "
                f"({actual:,} of {expected:,} bytes, {pct:.0f}%)"
            )
        return (
            f"image file {path.name} is larger than its geometry implies "
            f"({actual:,} vs {expected:,} bytes)"
        )
    except OSError as e:
        return f"image file {path.name} is unreadable ({e})"


def _compute_grid_position(
    workflow_file: Path,
    x_idx: int,
    y_idx: int,
    n_tiles_x: int = 1,
    n_tiles_y: int = 1,
) -> Dict[str, float]:
    """Compute tile position from root Workflow.txt grid parameters.

    Uses Start/End Position to interpolate the position of tile
    (x_idx, y_idx) in an ``n_tiles_x`` × ``n_tiles_y`` grid. The grid extent
    is passed in (derived from the discovered tile indices) rather than parsed
    from an ambiguous Workflow.txt field.
    """
    content = _read_text_resilient(workflow_file)

    # Read start position
    start_x = start_y = start_z = 0.0
    end_z = 1.0

    sx = re.search(r"<Start Position>.*?X \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if sx:
        start_x = float(sx.group(1))
    sy = re.search(r"<Start Position>.*?Y \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if sy:
        start_y = float(sy.group(1))
    sz = re.search(r"<Start Position>.*?Z \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if sz:
        start_z = float(sz.group(1))
    ez = re.search(r"<End Position>.*?Z \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if ez:
        end_z = float(ez.group(1))

    # Read end position for X/Y extent to compute tile step
    end_x = start_x
    end_y = start_y
    ex = re.search(r"<End Position>.*?X \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if ex:
        end_x = float(ex.group(1))
    ey = re.search(r"<End Position>.*?Y \(mm\) = ([-\d.]+)", content, re.DOTALL)
    if ey:
        end_y = float(ey.group(1))

    # Grid extent comes from the caller (derived from discovered tile indices).
    n_tiles_x = max(1, int(n_tiles_x))
    n_tiles_y = max(1, int(n_tiles_y))

    # Compute step per tile
    step_x = (end_x - start_x) / max(1, n_tiles_x - 1) if n_tiles_x > 1 else 0.0
    step_y = (end_y - start_y) / max(1, n_tiles_y - 1) if n_tiles_y > 1 else 0.0

    return {
        "x_mm": start_x + x_idx * step_x,
        "y_mm": start_y + y_idx * step_y,
        "z_min_mm": start_z,
        "z_max_mm": end_z,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_raw_volume(
    path: Path,
    n_planes: int,
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
) -> np.ndarray:
    """Memory-map a raw uint16 file as (Z, Y, X) array.

    Does NOT load data into RAM — returns a read-only memmap. ``frame_width``/
    ``frame_height`` are the camera AOI dimensions for this acquisition; pass
    the per-tile resolved values so cropped-AOI data is not misread as a
    truncated full-frame stack.
    """
    plane_bytes = frame_height * frame_width * 2
    expected_bytes = n_planes * plane_bytes
    actual_bytes = path.stat().st_size

    if actual_bytes != expected_bytes:
        # Recompute planes from actual file size (frame size is authoritative)
        actual_planes = actual_bytes // plane_bytes
        logger.warning(
            f"{path.name}: expected {n_planes} planes "
            f"({expected_bytes} bytes @ {frame_width}×{frame_height}), "
            f"got {actual_bytes} bytes → using {actual_planes} planes"
        )
        n_planes = actual_planes

    return np.memmap(
        path, dtype=np.uint16, mode="r", shape=(n_planes, frame_height, frame_width)
    )


def _load_tiff_volume(path: Path) -> np.ndarray:
    """Load a (Big)TIFF tile stack as a (Z, Y, X) array.

    Prefers a memory-map (no RAM cost, like the raw path) when the TIFF is
    uncompressed and contiguous; falls back to a full read for compressed or
    non-contiguous files. A single-page 2-D TIFF is promoted to a 1-plane stack.
    """
    import tifffile

    try:
        vol = tifffile.memmap(str(path), mode="r")
    except (ValueError, MemoryError, NotImplementedError, OSError):
        # Compressed / non-contiguous TIFF can't be memmapped — read it.
        vol = np.asarray(tifffile.imread(str(path)))
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    elif vol.ndim > 3:
        # Collapse any leading singleton/sample axes to a plain (Z, Y, X).
        vol = vol.reshape((-1,) + vol.shape[-2:])
    return vol


def load_tile_volume(
    path: Path,
    n_planes: int,
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
) -> np.ndarray:
    """Load one tile's (Z, Y, X) volume, dispatching on file type.

    ``.raw`` → raw uint16 memmap using the supplied per-tile frame dims.
    ``.tif`` / ``.tiff`` / ``.btf`` → tifffile, with the shape taken from the
    file header (the ``frame_*``/``n_planes`` args are ignored for TIFF since
    the container is self-describing).
    """
    if path.suffix.lower() in _TIFF_EXTENSIONS:
        return _load_tiff_volume(path)
    return load_raw_volume(path, n_planes, frame_width, frame_height)


def fuse_illumination_sides(
    volumes: Dict[int, np.ndarray],
    method: str = "max",
) -> np.ndarray:
    """Fuse left (I0) and right (I1) illumination volumes.

    Args:
        volumes: {illumination_side: volume_array}
        method: "max" (naive, same as FlamingoConverter),
                "mean" (simple average), or "leonardo" (Leonardo FUSE)

    Returns:
        Fused volume (Z, Y, X)
    """
    sides = sorted(volumes.keys())

    if len(sides) == 1:
        return np.asarray(volumes[sides[0]])

    left = np.asarray(volumes[sides[0]])
    right = np.asarray(volumes[sides[1]])

    if method == "max":
        return np.maximum(left, right)
    elif method == "mean":
        # Per-plane so the float32 working set is one plane, not two whole-volume
        # float32 copies (~2x the native tile) plus their sum. Bit-identical to
        # the whole-volume expression sliced per Z. left/right are memmaps, so
        # only one plane of each is paged in at a time.
        out = np.empty(left.shape, dtype=np.uint16)
        for z in range(left.shape[0]):
            out[z] = (
                (left[z].astype(np.float32) + right[z].astype(np.float32)) / 2
            ).astype(np.uint16)
        return out
    elif method == "leonardo":
        return _fuse_leonardo(left, right)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def _fuse_leonardo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Dual-illumination fusion using Leonardo FUSE_illu.

    Handles ghost artifacts from tissue refraction that naive max/min misses.
    Tries direct import first, then isolated environment, then falls back to max.
    """
    # Try direct import (if user installed leonardo-toolset locally)
    try:
        from leonardo_toolset.fusion.fuse_illu import FUSE_illu

        logger.info("Using Leonardo FUSE_illu for dual-illumination fusion")
        fuser = FUSE_illu()
        fused = fuser.fuse(
            left.astype(np.float32),
            right.astype(np.float32),
        )
        return np.clip(fused, 0, 65535).astype(np.uint16)
    except ImportError:
        pass

    # Try isolated environment
    try:
        from .isolated_service import IsolatedPreprocessingService

        service = IsolatedPreprocessingService()
        if service.has_leonardo():
            return service.fuse_illumination_leonardo(left, right)
    except Exception as e:
        logger.warning(f"Leonardo fusion via isolated env failed: {e}")

    logger.warning(
        "leonardo-toolset not available, falling back to max fusion. "
        "Use 'Setup Preprocessing...' in the stitching dialog to install."
    )
    return np.maximum(left, right)


# Stride used to subsample each axis when ranking tiles by brightness for the
# "brightest" tile-overlap mode. A coarse 1/stride^3 sample is plenty to *rank*
# tiles; a full-volume mean would force a compute of every (possibly memmapped)
# tile just to sort them.
_BRIGHTNESS_STRIDE = 4


def _priority_coalesce_fusion(transformed_views):
    """Winner-take-all tile-overlap fusion: each output pixel is taken whole from
    the highest-priority view that covers it (no per-pixel mixing).

    ``multiview_stitcher.fusion.fuse_np`` calls this per output block with
    ``transformed_views`` shaped ``(n_views, *block)`` — float, ``NaN`` where a
    view doesn't cover a pixel. Views are pre-sorted brightest→dimmest by the
    caller (``_build_fusion_inputs``), so axis-0 order *is* the priority: fill
    from view 0, then patch any still-uncovered pixels from view 1, and so on.

    Because the priority order is a single global ranking (not per-block), a
    given world pixel always resolves to the same tile regardless of which
    output block contains it — so there are no seams *within* the data and the
    result is bit-identical under chunk-aligned super-block batching, exactly
    like ``max_fusion``.
    """
    tv = np.asarray(transformed_views)
    if tv.shape[0] == 0:
        return tv
    result = tv[0].copy()
    for i in range(1, tv.shape[0]):
        missing = np.isnan(result)
        if not missing.any():
            break
        result[missing] = tv[i][missing]
    return result


def _tile_brightness(volume) -> float:
    """Mean intensity of a tile, used to rank tiles for the "brightest" overlap
    mode. Cheap by design — a strided subsample (see ``_BRIGHTNESS_STRIDE``) is
    enough to *order* tiles and avoids computing every memmapped volume in full.
    Returns ``-inf`` on failure so an unreadable tile sorts last rather than
    crashing fusion.
    """
    try:
        s = _BRIGHTNESS_STRIDE
        sub = volume[::s, ::s, ::s] if volume.ndim == 3 else volume
        return float(np.asarray(sub).mean())
    except Exception:
        return float("-inf")


def _rotation_affine_zyx(
    angle_deg: float,
    center_x_um: float,
    center_z_um: float,
    rotation_sign: float = 1.0,
) -> np.ndarray:
    """4×4 affine (in z, y, x, 1 order) rotating a view about the vertical Y axis.

    Maps a view acquired at rotation-stage angle ``angle_deg`` into the common
    (unrotated) frame: a rotation in the X–Z plane about
    (``center_x_um``, ``center_z_um``), Y unchanged. ``rotation_sign`` (+1/−1)
    selects the physical handedness — validated on synthetic data; the sign and
    center must still be confirmed on the instrument with a two-angle test
    (RIG-VALIDATE). The matrix multiplies homogeneous world coordinates in the
    (z, y, x, 1) order multiview-stitcher expects.
    """
    import math

    a = math.radians(rotation_sign * angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    cx, cz = center_x_um, center_z_um
    # Rotation about the point (cx, cz) → translation terms carry the center.
    tz = cz * (1.0 - ca) + cx * sa
    tx = cx * (1.0 - ca) - cz * sa
    m = np.array(
        [
            [ca, 0.0, -sa, tz],
            [0.0, 1.0, 0.0, 0.0],
            [sa, 0.0, ca, tx],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # Snap entries that are within rounding error of an integer (e.g. the
    # sin(180°)=1.2e-16 residual) so axis-aligned rotations (90°/180°) stay exact
    # and don't spuriously grow the fused union by a voxel.
    near = np.abs(m - np.round(m)) < 1e-9
    m[near] = np.round(m[near])
    return m


def _estimate_destripe_workers(
    plane_shape: tuple, max_workers: Optional[int] = None
) -> int:
    """Estimate safe number of parallel destripe threads based on available RAM."""
    import os

    import psutil

    # Per-worker memory: float32 copy + wavelet decomposition buffers (~4x uint16 plane)
    plane_bytes = plane_shape[0] * plane_shape[1] * 2  # uint16
    working_mem_per_worker = plane_bytes * 4

    available = psutil.virtual_memory().available
    try:
        from flamingo_stitcher.config_loader import get_stitching_value

        reserved = int(
            get_stitching_value(
                "destripe", "reserved_memory_bytes", default=2 * 1024**3
            )
        )
    except Exception:
        reserved = 2 * 1024**3  # 2 GB for OS + app headroom
    usable = max(available - reserved, working_mem_per_worker)

    max_by_memory = max(1, int(usable / working_mem_per_worker))
    max_by_cpu = os.cpu_count() or 4

    n = min(max_by_memory, max_by_cpu)
    if max_workers is not None:
        n = min(n, max_workers)
    return max(1, n)


def _stripe_axis_scores(volume: np.ndarray) -> Tuple[float, float]:
    """Return (horizontal_score, vertical_score) "stripiness" for one tile.

    Stripes are a coherent, higher-frequency pattern running along one image
    axis. Averaging *perpendicular* to a stripe reinforces it (constant along the
    stripe) while averaging real anatomy down, so the mean profiles isolate the
    stripe: the row-mean profile (mean over X) captures HORIZONTAL stripes, the
    column-mean profile (mean over Y) captures VERTICAL ones.

    Each score is the FRACTION of that profile's power sitting above the lowest
    frequencies, not its absolute power. That normalization matters: the two
    profiles have different lengths (Y vs X) and different variances, so
    comparing raw power sums let the merely *brighter* axis win instead of the
    stripier one — anatomy with strong banding along one axis could outvote the
    actual stripes. A fraction is scale-free, so the comparison is like-for-like.
    """
    z = volume.shape[0]
    step = max(1, z // 16)
    proj = np.asarray(volume[::step]).astype(np.float32).mean(axis=0)  # (Y, X)

    def _hf_fraction(profile: np.ndarray) -> float:
        p = profile - profile.mean()
        f = np.abs(np.fft.rfft(p)) ** 2
        total = float(f.sum())
        if not np.isfinite(total) or total <= 0.0:
            return 0.0
        lo = max(1, len(f) // 8)  # drop the lowest freqs (broadband illumination)
        return float(f[lo:].sum() / total)

    horiz = _hf_fraction(proj.mean(axis=1))  # row profile → horizontal stripes
    vert = _hf_fraction(proj.mean(axis=0))   # column profile → vertical stripes
    return horiz, vert


def _detect_stripe_axis(volume: np.ndarray) -> str:
    """Auto-detect stripe orientation: ``"horizontal"`` or ``"vertical"``.

    (pystripe's filter removes horizontal stripes; a vertical result means the
    caller must transpose before filtering.)
    """
    horiz, vert = _stripe_axis_scores(volume)
    return "vertical" if vert > horiz else "horizontal"


def _stripe_axis_vote(volume: np.ndarray) -> Tuple[str, float]:
    """Detected axis plus a 0..1 confidence for weighting across tiles.

    Confidence is the normalized margin between the two scores, so a tile whose
    profiles look equally stripy in both directions — background-dominated
    tiles, or tiles where anatomy mimics a stripe pattern — contributes almost
    nothing to the acquisition-wide decision instead of casting a full vote.
    """
    horiz, vert = _stripe_axis_scores(volume)
    total = horiz + vert
    confidence = abs(vert - horiz) / total if total > 0 else 0.0
    return ("vertical" if vert > horiz else "horizontal"), confidence


def channel_wavelength_nm(ch_id: int) -> Optional[float]:
    """Laser wavelength for a channel index, or None if unknown."""
    try:
        from flamingo_stitcher.config_loader import get_hardware_config

        return get_hardware_config().channel_wavelengths_nm.get(int(ch_id))
    except Exception:  # noqa: BLE001 - labelling must never break a run
        return None


def describe_channel(ch_id) -> str:
    """A channel rendered so it cannot be misread as a count.

    Channel numbers here are zero-based HARDWARE indices — the 4th laser is
    channel 3. Logging a bare "channel 3" for a single-channel acquisition
    reads as "three channels" or "the third one of several", when in fact only
    one was acquired. Naming the laser removes the ambiguity.

    Also handles the ``"{ch}_I{side}"`` labels that ``split_illumination``
    produces, where a one-laser acquisition legitimately yields two output
    channels — one per light path — and "channel 3_I0" is no clearer than
    "channel 3" was.
    """
    text = str(ch_id)
    if "_I" in text:
        base, _, side = text.partition("_I")
        try:
            nm = channel_wavelength_nm(int(base))
        except (TypeError, ValueError):
            nm = None
        laser = f"channel {base} ({nm:g} nm)" if nm else f"channel {base}"
        return f"{laser} illumination side {side}"
    nm = channel_wavelength_nm(ch_id)
    return f"channel {text} ({nm:g} nm)" if nm else f"channel {text}"


def describe_channel_set(
    selected: List[int], available: Optional[List[int]] = None
) -> str:
    """Count first, identity second — e.g. ``1 (channel 3, 640 nm)``.

    The count leads because that is what a reader checks at a glance; the
    index follows so it is still possible to tell WHICH laser was used.
    """
    if not selected:
        return "0 (none)"
    label = f"{len(selected)} — {', '.join(describe_channel(c) for c in selected)}"
    if available and len(available) > len(selected):
        label += f", of {len(available)} acquired"
    return label


class _DestripeMeter:
    """Machine-wide destripe throughput, so concurrency costs are visible.

    Each ``destripe_volume`` call reports its own planes/s, which looks
    healthy in isolation. When several tiles destripe at once, what matters is
    the SUM across them — and that is where the damage showed: a run logging a
    plausible-looking 2.2 planes/s per tile was doing 8.4 planes/s in total
    against a 25.6 planes/s single-tile ceiling measured earlier in the very
    same run. Per-tile rates hid a 3x regression for twelve hours.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self._peak_concurrent = 0
        self._planes = 0
        self._t0: Optional[float] = None

    def start(self) -> int:
        with self._lock:
            if self._active == 0:
                self._t0 = time.time()
                self._planes = 0
                self._peak_concurrent = 0
            self._active += 1
            self._peak_concurrent = max(self._peak_concurrent, self._active)
            return self._active

    def add_planes(self, n: int) -> None:
        with self._lock:
            self._planes += n

    def finish(self) -> Optional[Tuple[float, int, int]]:
        """On the last concurrent call, ``(aggregate_rate, planes, peak)``."""
        with self._lock:
            self._active -= 1
            if self._active > 0 or self._t0 is None:
                return None
            elapsed = time.time() - self._t0
            rate = self._planes / elapsed if elapsed > 0 else 0.0
            return rate, self._planes, self._peak_concurrent


_destripe_meter = _DestripeMeter()


def destripe_volume(
    volume: np.ndarray,
    max_workers: Optional[int] = None,
    direction: str = "auto",
    params: Optional[Dict[str, Any]] = None,
    in_place: bool = False,
    cancel_fn: Optional[Any] = None,
) -> np.ndarray:
    """Apply PyStripe destriping to each Z-plane using parallel threads.

    Uses ThreadPoolExecutor for parallelism. Note that this scales POORLY —
    measured ~1.5x on 1024x1024 planes and flat beyond two workers, with
    processes no better than threads, so the ceiling is not the GIL but the
    filter's allocation/memory traffic. (``pywt`` does hold the GIL for roughly
    half the runtime, contrary to what this docstring used to claim, but
    removing that would not lift the ceiling.) The practical consequence:
    throwing more workers at destriping does not help, and running several
    tiles through it CONCURRENTLY actively hurts — see :class:`_DestripeMeter`
    and ``_pick_preprocess_workers``.

    ``in_place=True`` writes filtered planes back into ``volume`` instead of
    allocating a second full-volume array, saving one tile-sized allocation.
    Each plane is read into a float32 working copy before its slot is
    overwritten and planes are independent, so this is safe wherever the
    caller owns ``volume``.

    It is IGNORED for a read-only array, which is the common case on the
    per-illumination-side path: ``.raw`` tiles are memory-mapped read-only, so
    the output array is not a duplicate of resident memory at all — the memmap
    costs page cache, not anonymous RAM, and the result array is the only
    tile-sized thing actually in RAM. The saving is real only where the input
    is already a writable in-memory array, e.g. the ``destripe_fast`` path,
    which filters the downsampled tile this method produced.

    ``cancel_fn`` is polled per plane. Without it, Cancel did nothing until
    the whole tile finished: all N planes are submitted to the pool up front,
    so a 1600-plane tile held the user for a full destripe pass (minutes), and
    with several tiles in flight, for one pass each. Queued planes now return
    immediately once it trips, so the pool drains in moments.

    Falls back to fewer workers on MemoryError, and to sequential processing
    as a last resort.

    ``direction`` selects the stripe orientation to remove:
      * ``"horizontal"`` — pystripe's native axis (removes horizontal stripes),
      * ``"vertical"`` — transpose each plane so vertical stripes are removed,
      * ``"auto"`` (default) — detect per volume via :func:`_detect_stripe_axis`.
    This matters because the underlying filter is axis-fixed AND destriping runs
    in the raw camera frame, *before* the per-tile rot/flip orientation — a 90°
    rotation swaps which way the stripes run, so the naive default can silently
    filter the wrong axis and remove nothing (the v0.9.5 symptom).

    Uses the vendored pystripe stripe filter (``_pystripe_core``), which needs
    only numpy / scipy / pywt / scikit-image — the stack the app already ships.
    Raises ``RuntimeError`` if that backend can't load (e.g. pywt missing).
    Destriping is an explicit, opt-in request (``config.destripe``); on the ASLM
    scope it is essential (no beam-oscillation shadow reduction), so a broken
    backend must FAIL LOUDLY rather than silently return un-destriped data.
    """
    try:
        from flamingo_stitcher._pystripe_core import filter_streaks
    except Exception as exc:
        raise RuntimeError(
            "Destriping was requested (Destripe is ON) but its backend failed to "
            "load — it needs pywt / scipy / scikit-image. Underlying error: "
            f"{exc!r}. Refusing to continue and silently write un-destriped tiles."
        ) from exc

    from concurrent.futures import ThreadPoolExecutor, as_completed

    resolved = (direction or "auto").lower()
    if resolved == "auto":
        resolved = _detect_stripe_axis(volume)
        logger.info(f"Destripe direction: {resolved} stripes (auto-detected)")
    else:
        logger.info(f"Destripe direction: {resolved} stripes (configured)")
    # pystripe removes horizontal stripes; for vertical stripes we filter the
    # transpose (and transpose the result back).
    transpose = resolved == "vertical"

    n_planes = volume.shape[0]
    # A read-only input (memory-mapped .raw) can't be written back into, so
    # in_place silently degrades rather than dying mid-tile.
    writable = bool(getattr(volume, "flags", None) and volume.flags.writeable)
    if in_place and not writable:
        logger.debug("Destripe in_place requested but input is read-only; copying")
    result = volume if (in_place and writable) else np.empty_like(volume)

    # Resolve filter parameters: explicit `params` (GUI/CLI) wins, else the YAML
    # config, else the built-in defaults. Missing/None keys fall through, so a
    # partial override only changes what it names.
    p = dict(params or {})
    try:
        from flamingo_stitcher.config_loader import get_stitching_value

        _yaml_sigma = get_stitching_value("destripe", "sigma", default=[128, 256])
        _yaml_level = get_stitching_value("destripe", "level", default=7)
        _yaml_wavelet = get_stitching_value("destripe", "wavelet", default="db2")
    except Exception:
        _yaml_sigma, _yaml_level, _yaml_wavelet = [128, 256], 7, "db2"

    def _pick(key, fallback):
        v = p.get(key)
        return fallback if v is None else v

    _ds_sigma = [
        float(_pick("sigma_foreground", _yaml_sigma[0])),
        float(_pick("sigma_background", _yaml_sigma[1])),
    ]
    _ds_level = int(_pick("level", _yaml_level))
    _ds_wavelet = str(_pick("wavelet", _yaml_wavelet))
    _ds_crossover = float(_pick("crossover", 10.0))
    # threshold: None/absent → -1, which makes filter_streaks pick it via Otsu.
    _ds_threshold = float(_pick("threshold", -1.0))

    # Clamp the wavelet depth to what this frame size can actually support.
    # `level` is a fixed config value (default 7) but the usable depth is set by
    # the image: pywt warns "Level value of N is too high: all coefficients will
    # experience boundary effects" and the filter then barely works. Measured on
    # a synthetic stripe pattern with db2/level=7: 91% of the stripe removed at
    # 2048 px, but only 10.8% at 256 px -- rising back to 57% once clamped to
    # that frame's max level of 6. Low-resolution frames (small sensor AOI, or
    # the "fast" variant which destripes AFTER downsample) are exactly where
    # this bites, and it degraded silently apart from a pywt warning.
    _plane_min_dim = int(min(volume.shape[1], volume.shape[2]))
    if _ds_level > 0:
        try:
            from flamingo_stitcher._pystripe_core import max_level as _max_level

            _usable = int(_max_level(_plane_min_dim, _ds_wavelet))
            if _usable >= 1 and _ds_level > _usable:
                logger.warning(
                    f"Destripe level {_ds_level} exceeds what a "
                    f"{_plane_min_dim}px frame supports for wavelet "
                    f"{_ds_wavelet}; clamping to {_usable}. Deeper levels are "
                    "all boundary effects and remove far less striping."
                )
                _ds_level = _usable
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Could not clamp destripe level: {exc!r}")

    logger.info(
        f"Destripe params: sigma(fg/bg)={_ds_sigma[0]:g}/{_ds_sigma[1]:g} "
        f"level={_ds_level} wavelet={_ds_wavelet} crossover={_ds_crossover:g} "
        f"threshold={'Otsu (auto)' if _ds_threshold == -1 else f'{_ds_threshold:g}'}"
    )

    def _process_plane(z: int) -> int:
        # Queued planes turn into no-ops the moment Cancel is pressed, so the
        # already-submitted backlog drains instead of running to completion.
        if cancel_fn is not None and cancel_fn():
            return z
        plane = volume[z].astype(np.float32)
        if transpose:
            plane = plane.T
        filtered = filter_streaks(
            plane,
            sigma=_ds_sigma,
            level=_ds_level,
            wavelet=_ds_wavelet,
            crossover=_ds_crossover,
            threshold=_ds_threshold,
        )
        if transpose:
            filtered = filtered.T
        result[z] = filtered.astype(np.uint16)
        return z

    n_workers = _estimate_destripe_workers(volume.shape[1:], max_workers)
    concurrent = _destripe_meter.start()
    logger.info(
        f"Destriping {n_planes} planes ({volume.shape[1]}x{volume.shape[2]}) "
        f"with {n_workers} threads"
        + (f" ({concurrent} tiles destriping concurrently)" if concurrent > 1 else "")
        + "..."
    )

    remaining = set(range(n_planes))
    done: set = set()
    t0 = time.time()
    milestone = max(1, n_planes // 10)
    completed = 0

    cancelled = False
    while remaining and n_workers >= 1 and not cancelled:
        batch = list(remaining)
        failed: list = []
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_plane, z): z for z in batch}
                for future in as_completed(futures):
                    if cancel_fn is not None and cancel_fn():
                        cancelled = True
                        for f in futures:
                            f.cancel()
                        break
                    z = futures[future]
                    try:
                        future.result()
                        done.add(z)
                        completed += 1
                        if completed % milestone == 0 or completed == n_planes:
                            elapsed = time.time() - t0
                            rate = completed / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"  Destripe progress: {completed}/{n_planes} "
                                f"({100 * completed // n_planes}%, "
                                f"{rate:.1f} planes/s)"
                            )
                    except MemoryError:
                        failed.append(z)
        except MemoryError:
            # Pool-level OOM — retry everything not yet done
            failed = [z for z in batch if z not in done]

        remaining = set(failed)
        if not remaining:
            break

        # Reduce workers and retry failed planes
        old_workers = n_workers
        n_workers = max(1, n_workers // 2)
        logger.warning(
            f"Memory pressure during destripe — reducing threads from "
            f"{old_workers} to {n_workers}, retrying {len(remaining)} planes"
        )

        if n_workers == 1:
            # Last resort: sequential, one plane at a time
            logger.warning("Falling back to sequential destriping")
            for z in sorted(remaining):
                if cancel_fn is not None and cancel_fn():
                    cancelled = True
                    break
                _process_plane(z)
                completed += 1
            remaining = set()
            break

    elapsed = time.time() - t0
    rate = n_planes / elapsed if elapsed > 0 else 0
    if cancelled:
        logger.info(f"Destripe cancelled after {completed}/{n_planes} planes")
    else:
        logger.info(
            f"Destripe complete: {n_planes} planes in {elapsed:.1f}s "
            f"({rate:.1f} planes/s)"
        )
    _destripe_meter.add_planes(n_planes)
    summary = _destripe_meter.finish()
    if summary is not None and summary[2] > 1:
        agg_rate, agg_planes, peak = summary
        logger.info(
            f"Destripe THROUGHPUT (all tiles): {agg_planes} planes at "
            f"{agg_rate:.1f} planes/s with up to {peak} tiles at once. "
            "Per-tile rates above are each a fraction of this — compare the "
            "aggregate, not the per-tile figure, when judging speed."
        )
    return result


def downsample_volume(
    volume: np.ndarray, factor_xy: int = 1, factor_z: int = 1
) -> np.ndarray:
    """Downsample a volume with separate Z and XY factors.

    Both axes are streamed, so the working set is a plane (XY) or a slab of
    ``factor_z`` planes (Z) — never the whole volume:

    * **XY** uses ``scipy.ndimage.zoom`` (order=1) per plane. Planes don't
      couple under an XY-only zoom, so this is bit-identical to zooming the
      whole volume at once.
    * **Z** averages each consecutive group of ``factor_z`` planes (a block
      mean — the "average a small stack" reduction). This is a proper
      anti-aliased reduction AND costs one slab of float32 instead of upcasting
      the entire tile, which is what a whole-volume ``zoom`` used to do:
      2 x native float32 (~27 GB on an 800-plane 2048² tile) just to shrink it.

    Args:
        volume: (Z, Y, X) array
        factor_xy: XY downsample factor
        factor_z: Z downsample factor

    Returns:
        Downsampled volume
    """
    if factor_xy <= 1 and factor_z <= 1:
        return volume

    from scipy.ndimage import zoom

    label = f"Z{factor_z}x/XY{factor_xy}x" if factor_z != factor_xy else f"{factor_xy}x"
    logger.info(f"Downsampling volume {volume.shape} by {label}...")

    # XY-only downsample (factor_z == 1) is plane-independent, so stream it one
    # Z-plane at a time: the float32 working set is a single 2048x2048 plane
    # (~16 MB) instead of a full-volume float32 upcast (~12 GB for a native
    # tile) held alongside zoom's output. Bit-identical: order-1 zoom with a
    # z-factor of 1 does not mix planes.
    if factor_z == 1 and factor_xy > 1:
        z = volume.shape[0]
        xy_zoom = (1.0 / factor_xy, 1.0 / factor_xy)
        # Derive the output plane shape from zoom itself (it rounds), so this is
        # identical to the 3D zoom's per-plane result.
        first = zoom(volume[0].astype(np.float32), xy_zoom, order=1)
        out = np.empty((z, *first.shape), dtype=np.uint16)
        out[0] = np.clip(first, 0, 65535).astype(np.uint16)
        for zi in range(1, z):
            plane = zoom(volume[zi].astype(np.float32), xy_zoom, order=1)
            out[zi] = np.clip(plane, 0, 65535).astype(np.uint16)
        return out

    # Z is downsampled too. Reduce Z slab-by-slab (block mean over factor_z
    # planes) and XY-zoom each reduced plane immediately, so the peak working
    # set is one slab + one plane rather than a float32 copy of the whole tile.
    z_in = volume.shape[0]
    z_out = max(1, int(round(z_in / factor_z)))
    xy_zoom = (1.0 / factor_xy, 1.0 / factor_xy) if factor_xy > 1 else None

    def _reduced_plane(zi: int) -> np.ndarray:
        lo = min(zi * factor_z, max(0, z_in - 1))
        hi = min(z_in, lo + factor_z)
        # float32 only for the slab being averaged (factor_z planes).
        plane = volume[lo:hi].astype(np.float32).mean(axis=0)
        if xy_zoom is not None:
            plane = zoom(plane, xy_zoom, order=1)
        return np.clip(plane, 0, 65535).astype(np.uint16)

    first = _reduced_plane(0)
    out = np.empty((z_out, *first.shape), dtype=np.uint16)
    out[0] = first
    for zi in range(1, z_out):
        out[zi] = _reduced_plane(zi)
    return out


def _lazy_stack_channels(channel_arrays: list) -> "dask.array.Array":
    """Lazily stack per-channel dask arrays into (C, Z, Y, X).

    Pads shapes if they differ slightly (rounding differences between channels).
    Returns the single array if only one channel.
    """
    import dask.array as da

    if len(channel_arrays) == 1:
        return channel_arrays[0]

    # Pad to uniform shape
    max_shape = tuple(max(a.shape[d] for a in channel_arrays) for d in range(3))
    padded = []
    for a in channel_arrays:
        if a.shape != max_shape:
            pad_widths = [(0, max_shape[d] - a.shape[d]) for d in range(3)]
            a = da.pad(a, pad_widths, mode="constant", constant_values=0)
        padded.append(a)

    return da.stack(padded, axis=0)  # (C, Z, Y, X), still lazy


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class StitchingPipeline:
    """End-to-end stitching pipeline for Flamingo T-SPIM data.

    Usage:
        config = StitchingConfig(pixel_size_um=0.406)
        pipeline = StitchingPipeline(config)
        pipeline.run(
            acquisition_dir=Path("/data/20260310_acquisition"),
            output_path=Path("/data/20260310_acquisition_stitched"),
        )
    """

    def __init__(
        self,
        config: Optional[StitchingConfig] = None,
        cancelled_fn=None,
        progress_fn=None,
        memory_warning_fn=None,
    ):
        self.config = config or StitchingConfig()
        self.logger = logging.getLogger(__name__)
        self._cancelled_fn = cancelled_fn or (lambda: False)
        self._raw_progress_fn = progress_fn or (lambda pct, msg: None)
        # Called (from a background thread) the first time live private memory
        # crosses the projected bound. The GUI wires this to a non-blocking
        # popup; warn-only — the run is never aborted. Signature:
        #   memory_warning_fn(info: dict)  with keys used_gb, projected_gb,
        #   phase, mode.
        self._memory_warning_fn = memory_warning_fn
        self._memory_monitor = None
        self._estimator = None  # built once tile count is known
        # Per-run registration evidence. Stashed on self, like _estimator,
        # because _register_tiles is called from three places (in-memory,
        # preview, streaming) while the report is written at the metadata step,
        # long after the params have been consumed by fusion.
        self._registration_report = None
        # Streaming-mode flat-field models {ch_id: model}. Populated by
        # _run_streaming when flat_field_correction is on, and consumed inside
        # _preprocess_single_tile. Empty in the in-memory path (which applies
        # flat-field via its own estimate/apply step), so no double-application.
        self._flatfield_models: Dict[int, Any] = {}

        # Stripe axis is a property of the camera frame, so it is decided once
        # per run and reused for every tile (see _resolve_destripe_direction).
        self._reset_destripe_axis()

        # Wrap caller's progress_fn so we (a) classify the phase from
        # the status message and update the estimator's phase clock,
        # and (b) append the live ETA to the status string the dialog
        # sees. Pipeline internal calls keep using `_progress_fn` so
        # all of them flow through the same hook without per-call-site
        # changes.
        def _hooked_progress(pct, msg):
            self._on_progress_emit(pct, msg)

        self._progress_fn = _hooked_progress

    # ------------------------------------------------------------------
    # ETA / phase tracking
    # ------------------------------------------------------------------

    # Map status-message substrings to estimator phase names. Mirrors
    # the dialog's _STATUS_TO_STEP but uses our six-phase taxonomy
    # (discover, register, preprocess, fuse, write, metadata). First
    # match wins; order follows pipeline execution.
    _STATUS_TO_PHASE = [
        ("discover", "discover"),
        ("loading reference", "register"),
        ("registering", "register"),
        ("skip registration", "register"),
        ("loading and preprocessing", "preprocess"),
        ("applying flat-field", "preprocess"),
        ("materializing", "preprocess"),
        ("preprocess", "preprocess"),
        ("fusing", "fuse"),
        ("computing channel", "fuse"),
        ("storing channel", "fuse"),
        ("channel ", "fuse"),
        ("writing", "write"),
        ("finalizing", "write"),
        ("metadata", "metadata"),
    ]

    def _classify_phase(self, msg: str) -> Optional[str]:
        s = (msg or "").lower()
        for needle, phase in self._STATUS_TO_PHASE:
            if needle in s:
                return phase
        return None

    def _on_progress_emit(self, pct, msg):
        """Hook called for every internal progress emit. Drives the
        estimator's phase transitions, the memory watchdog's phase
        attribution, and appends a live ETA tail."""
        phase = self._classify_phase(msg)
        if phase and self._memory_monitor is not None:
            self._memory_monitor.set_phase(phase)
        if self._estimator is not None:
            if phase and phase != getattr(self._estimator, "_current_phase", None):
                self._estimator.start_phase(phase)
            # Anchor the whole-run ETA to the pipeline's monotone global percent
            # (per-tile in preprocess, per-region in fusion). This is what makes
            # the estimate stable and self-correcting.
            try:
                self._estimator.update_fraction((pct or 0) / 100.0)
            except Exception:
                pass
            tail = self._estimator.format_label()
            if tail and tail != "estimating...":
                # Separator + explicit "overall:" label so this whole-run ETA
                # is not confused with the per-step "this step:" ETA that the
                # dask callback already appended (both read "... remaining
                # (Done at ~...)"). The GUI splits on OVERALL_ETA_SEP to colour
                # the two segments differently; plain-text/CLI/logs stay legible.
                msg = f"{msg}{OVERALL_ETA_SEP}overall: {tail}"
        self._raw_progress_fn(pct, msg)

    # ------------------------------------------------------------------
    # Memory watchdog (warn-only)
    # ------------------------------------------------------------------
    def _start_memory_watchdog(self, mem_est: Dict[str, float], use_streaming: bool):
        """Start a background monitor that WARNS (never aborts) if live private
        memory exceeds the projected peak by a margin.

        This is the runtime half of "independent checking": the pre-flight guard
        trusts the a-priori estimate; this catches what the estimate MISSED, in
        flight, and surfaces it loud + attributable (which phase, how far over)
        instead of an opaque OS OOM. Uses USS (private/committed memory).
        """
        try:
            from flamingo_stitcher.memory_monitor import MemoryMonitor
        except Exception:
            return  # monitor unavailable (e.g. no psutil) — skip silently

        projected_gb = float(
            mem_est.get("streaming_gb" if use_streaming else "in_memory_gb", 0.0)
        )
        try:
            margin = float(
                get_stitching_value("memory", "watchdog_margin", default=1.5)
            )
        except Exception:
            margin = 1.5
        if projected_gb <= 0:
            return  # nothing meaningful to bound (e.g. tiny/among test runs)

        # Cap the threshold at ~90% of AVAILABLE RAM (D1): the pre-flight guard
        # permits peaks up to 95% of available, so an uncapped 1.5x margin can
        # land the watchdog threshold ABOVE physical RAM — the process OOMs
        # before USS ever reaches it, on exactly the near-limit jobs most likely
        # to OOM. Capping keeps the threshold reachable so the warning fires.
        try:
            import psutil

            avail_gb = psutil.virtual_memory().available / (1024**3)
        except Exception:
            avail_gb = 0.0

        raw_gb = projected_gb * margin
        threshold_gb = min(raw_gb, avail_gb * 0.9) if avail_gb > 0 else raw_gb
        threshold_bytes = int(threshold_gb * (1024**3))  # a working-set DELTA
        mode = "streaming" if use_streaming else "in-memory"
        # Mode-aware remedy: don't tell a user who is already streaming to
        # "switch to Streaming". In streaming mode the fuse working set is
        # dominated by per-block cost — content-based blending and worker count
        # inflate it well beyond the estimate — so point at those levers.
        if use_streaming:
            remedy = (
                "already in Streaming mode — raise the XY/Z downsample factor, "
                "turn off content-based blending, lower the fuse worker count, "
                "or move the scratch dir to a fast local disk"
            )
        else:
            remedy = "switch to Streaming mode and/or raise the downsample factor"

        def _on_exceed(used_bytes: int, phase):
            base = getattr(self._memory_monitor, "baseline_bytes", 0) or 0
            delta_gb = max(0, used_bytes - base) / (1024**3)
            mapped_gb = (
                getattr(self._memory_monitor, "peak_mapped_bytes", 0) or 0
            ) / (1024**3)
            # Report the mapped set alongside, so the number is interpretable.
            # It is deliberately NOT in the threshold: those are pages of the
            # tile spill and fused.dat, disk-backed and reclaimed under
            # pressure. Counting them (as USS did) compared a projection that
            # excludes the memmaps against a measurement dominated by them.
            mapped_note = (
                f" Separately, {mapped_gb:.1f} GB of memory-mapped scratch "
                f"(tile spill + fused output) is resident; that is disk-backed "
                f"and reclaimable, and is not part of this threshold."
                if mapped_gb >= 1.0
                else ""
            )
            self.logger.warning(
                f"  [memory watchdog] private allocation {delta_gb:.1f} GB "
                f"exceeded projected {projected_gb:.1f} GB × {margin:g} "
                f"(threshold {threshold_gb:.1f} GB) during phase "
                f"'{phase or '?'}' ({mode}). The run continues; if it OOMs, "
                f"{remedy}.{mapped_note}"
            )
            if self._memory_warning_fn is not None:
                try:
                    self._memory_warning_fn(
                        {
                            "used_gb": round(delta_gb, 1),
                            "projected_gb": round(projected_gb, 1),
                            "margin": margin,
                            "phase": phase or "?",
                            "mode": mode,
                        }
                    )
                except Exception:
                    pass  # a broken callback must never crash the run

        self._memory_monitor = MemoryMonitor(
            interval_s=0.25,
            threshold_bytes=threshold_bytes,
            on_exceed=_on_exceed,
            # Private commit, NOT uss: the projection above excludes the tile
            # spill and the fused memmap by construction, so the measurement
            # has to exclude them too or the comparison is meaningless. On
            # Windows a single-process memmap reads as private under USS, which
            # is how a 97-tile run reported 127.7 GB against a 9.4 GB
            # projection while completing comfortably.
            metric="private",
        )
        self._memory_monitor.start()
        self.logger.info(
            f"Memory watchdog armed: warn if working-set memory exceeds "
            f"{threshold_gb:.1f} GB (projected {projected_gb:.1f} × {margin:g}"
            f"{', capped to 90% avail' if threshold_gb < raw_gb else ''}, {mode})"
        )

    def _stop_memory_watchdog(self):
        if self._memory_monitor is not None:
            try:
                self._memory_monitor.stop()
            except Exception:
                pass
            self._memory_monitor = None

    def _build_estimator(self, tiles, acquisition_dir=None, output_dir=None):
        """Construct the multi-phase estimator from tiles + config.

        Imported lazily so the module is importable in environments
        without the stitching subpackage's runtime deps.
        """
        from flamingo_stitcher.multi_phase_estimator import (
            MultiPhaseEstimator,
        )
        from flamingo_stitcher.timing_cache import StitchingTimingCache

        key = build_timing_key(
            tiles, self.config, acquisition_dir=acquisition_dir, output_dir=output_dir
        )
        self.logger.info(f"Stitching ETA key: {key.serialize()}")
        # Rough cold-start prior (order-of-magnitude); the estimator prefers a
        # cached measured total when this config has run before, and refines
        # both live from the global progress fraction.
        prior = rough_run_seconds(tiles, self.config)
        return MultiPhaseEstimator(StitchingTimingCache(), key, prior_total_s=prior)

    # ------------------------------------------------------------------
    # Tile-border artifact QC (diagnostic)
    # ------------------------------------------------------------------
    def _qc_report_dir(self, output_path=None) -> Path:
        """Directory for the border-QC text report: next to the run log.

        Prefers the active file-log handler's directory (where GUI runs write
        flamingo-stitcher_*.log); falls back to the output folder, then cwd.
        """
        import logging as _logging

        for h in _logging.getLogger().handlers:
            base = getattr(h, "baseFilename", None)
            if base:
                d = Path(base).parent
                if d.exists():
                    return d
        if output_path is not None:
            try:
                p = Path(output_path)
                d = p if p.is_dir() else p.parent
                d.mkdir(parents=True, exist_ok=True)
                return d
            except Exception:
                pass
        return Path.cwd()

    def _border_qc_params(self):
        from flamingo_stitcher import border_qc

        mode = getattr(self.config, "border_qc_mode", "mip")
        if mode not in border_qc._VALID_MODES:
            mode = "mip"
        return border_qc.BorderQCParams(
            mode=mode,
            alpha=float(self.config.border_qc_alpha),
            **(
                {"max_refine_shift_px": int(self.config.border_qc_max_shift_px)}
                if int(getattr(self.config, "border_qc_max_shift_px", 0) or 0) > 0
                else {}
            ),
            beta=float(self.config.border_qc_beta),
            min_component_px=int(self.config.border_qc_min_component_px),
            z_stride=max(1, int(self.config.border_qc_z_stride)),
            include_z_seams=bool(self.config.border_qc_include_z_seams),
        )

    def _write_registration_report(self, output_path, acquisition_dir):
        """Write the registration evidence next to the stitched output.

        Into the OUTPUT directory, unlike border QC, which puts its report
        beside the log. These numbers describe the store sitting next to them
        and have to travel with it when the result is copied off the rig.

        A report is written even when registration did not run — the file says
        so and names the reason. A missing file is indistinguishable from a
        feature nobody implemented, which is exactly the ambiguity that let
        'Skip registration' stay on unnoticed.
        """
        report = self._registration_report
        if report is None:
            report = registration_report.skipped_report(
                "no registration stage ran for this output"
            )
        acq_name = Path(acquisition_dir).name or "acquisition"
        written = registration_report.write_report(
            output_path,
            report,
            acquisition=acq_name,
            write_json=bool(self.config.registration_report_json),
            logger=self.logger,
        )
        for line in registration_report.format_report_text(
            report, acquisition=acq_name
        ).splitlines():
            self.logger.info(line)
        for path in written.values():
            self.logger.info(f"  Wrote {path}")

    def _run_border_qc(
        self, channel_tile_data, tiles, acquisition_dir, voxel_size_um, output_path
    ):
        """Scan neighbor-tile seams for sharp steps; write a report by the log."""
        import json as _json

        from flamingo_stitcher import border_qc

        params = self._border_qc_params()
        self._progress_fn(50, "Running tile-border QC...")
        self.logger.info(f"Running tile-border artifact QC (mode={params.mode})...")
        report = border_qc.run_border_qc(
            channel_tile_data,
            tiles,
            pixel_size_um=float(voxel_size_um["x"]),
            ds_xy=int(self.config.downsample_xy or 1),
            ds_z=int(self.config.downsample_z or 1),
            z_step_um=float(voxel_size_um["z"]),
            reg_channel=int(self.config.reg_channel),
            params=params,
            logger=self.logger,
            cancelled_fn=self._cancelled_fn,
        )
        acq_name = Path(acquisition_dir).name or "acquisition"
        text = border_qc.format_report_text(report, acquisition=acq_name)
        out_dir = self._qc_report_dir(output_path)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = out_dir / f"border_qc_{acq_name}_{stamp}.txt"
        try:
            report_path.write_text(text)
        except Exception as e:
            self.logger.warning(f"Could not write border-QC report: {e}")
            report_path = None
        if self.config.border_qc_json and report_path is not None:
            try:
                (report_path.with_suffix(".json")).write_text(
                    _json.dumps(border_qc.report_to_json(report, acquisition=acq_name), indent=2)
                )
            except Exception:
                pass
        self.logger.info(
            f"Border QC: {report.n_pairs_flagged}/{report.n_pairs_checked} seams flagged"
            + (f" — report at {report_path}" if report_path else "")
        )
        # Echo the report into the run log too (flagged pairs only — concise).
        for line in text.splitlines():
            self.logger.info(line)
        return report

    def _run_border_qc_streaming(
        self, tiles, process_channels, voxel_size_um, output_path,
        acquisition_dir, reg_reuse_ch, reg_reuse_data, reuse_side=None,
    ):
        """Streaming border QC. Reuses the registered ref-channel spill when
        available; otherwise materializes the ref channel just for QC.

        Returns ``(ref_ch, tile_data, tmp_dir, side)`` for a spill this method
        materialized itself (registration was skipped, so there was none to
        reuse) so the caller can hand it to the fusion loop instead of
        preprocessing + spilling every tile a *second* time; returns
        ``(None, None, None, None)`` when it reused the registration spill.

        ``reuse_side`` is the illumination side the FIRST output unit will
        want. It matters because the fusion loop can only reuse a spill whose
        side matches: with ``split_illumination`` on, a QC spill built from
        FUSED sides matches nothing, so QC silently became a whole extra
        preprocess of every tile — on a 98-tile run that was 5.2 hours of
        destriping thrown away, on top of the two passes the split itself
        needs.
        """
        if reg_reuse_data is not None:
            # Reuse the registration spill; we own nothing to hand back.
            self._run_border_qc(
                {reg_reuse_ch: reg_reuse_data}, tiles,
                acquisition_dir, voxel_size_um, output_path,
            )
            return None, None, None, None

        ref_ch = self.config.reg_channel
        if ref_ch not in process_channels:
            ref_ch = process_channels[0]
        side_tag = "" if reuse_side is None else f"_I{reuse_side}"
        if reuse_side is not None:
            self.logger.info(
                f"  Border QC will measure illumination side {reuse_side} only "
                "(sides are being kept separate, so this spill is also the "
                "first output channel — one preprocess pass instead of two). "
                "Seam steps are therefore per-light-path, not post-fusion."
            )
        own_tmp = (
            _scratch_base_dir(self.config, output_path)
            / ".stitch_tmp"
            / f"qc_ch{ref_ch:02d}{side_tag}"
        )
        # Materialize the reference channel once. A hard failure here leaves a
        # useless partial spill — drop it and re-raise (the caller swallows).
        try:
            probe = self._preprocess_single_tile(
                tiles[0], ref_ch, illum_side=reuse_side
            )
            shape = probe.shape
            del probe
            data = self._materialize_tiles_to_disk(
                tiles, ref_ch, shape, own_tmp, illum_side=reuse_side
            )
        except BaseException:
            import shutil as _shutil

            _shutil.rmtree(own_tmp, ignore_errors=True)
            raise

        # The report itself is best-effort: a failure must not discard the
        # freshly-materialized spill, which the fusion loop will reuse.
        try:
            self._run_border_qc(
                {ref_ch: data}, tiles,
                acquisition_dir, voxel_size_um, output_path,
            )
        except Exception as e:
            self.logger.warning(f"Border QC report failed (skipped): {e}")

        # Hand the spill back so the fusion loop reuses it instead of doing a
        # full second preprocess pass of every tile (the redundant pass this
        # method used to force when registration was skipped).
        return ref_ch, data, own_tmp, reuse_side

    def _build_output_basename(self, acquisition_dir: Path) -> str:
        """Build a descriptive base filename from acquisition path and settings.

        Short tags for the enabled preprocessing steps are appended either way,
        so runs with different settings produce distinct filenames.

        What goes in front of them depends on the layout, because the two
        layouts put the descriptive name in different places:

        * **Subfolder-per-tile** — ``.../OrganoidV2/2026-04-05``. The
          acquisition folder is a bare date, so the sample folder is prepended:
          ``2026-04-05.ome.zarr`` on its own would not say which sample it is,
          and two samples stitched to one output folder would collide.
        * **Flat** (Single Workflow) — ``.../whatever/MyDataset``. The
          acquisition folder IS the dataset. Prepending its parent produced
          ``whatever_MyDataset`` next to a ``MyDataset_stitched`` folder: a
          name taken from a directory that says nothing about the data, and
          disagreeing with the folder holding it.

        Examples:
            OrganoidV2_2026-04-05                    (subfolder-per-tile)
            OrganoidV2_2026-04-05_destripe
            MyDataset                                (flat)
            MyDataset_destripe
        """
        acq_name = acquisition_dir.name

        if acquisition_is_flat(acquisition_dir):
            base = acq_name
        else:
            parent_name = acquisition_dir.parent.name
            # Avoid redundancy if parent is a drive root or generic name
            if parent_name and parent_name not in (".", "/", "\\"):
                base = f"{parent_name}_{acq_name}"
            else:
                base = acq_name

        # Append short preprocessing tags
        tags = []
        if self.config.illumination_fusion != "max":
            tags.append(self.config.illumination_fusion)
        if self.config.flat_field_correction:
            tags.append("flatfield")
        if self.config.destripe:
            tags.append("destripe-fast" if self.config.destripe_fast else "destripe")
        if self.config.depth_attenuation:
            tags.append("atten")
        if self.config.deconvolution_enabled:
            tags.append("deconv")
        if self.config.downsample_xy > 1 or self.config.downsample_z > 1:
            if self.config.downsample_xy == self.config.downsample_z:
                tags.append(f"{self.config.downsample_xy}x")
            else:
                tags.append(
                    f"xy{self.config.downsample_xy}x_z{self.config.downsample_z}x"
                )
        if self.config.content_based_fusion:
            tags.append("cbf")

        if tags:
            base = base + "_" + "_".join(tags)

        return base

    # Primary output store extension by format (the one a re-run would clobber).
    _PRIMARY_EXT = {
        "ome-zarr-sharded": ".ome.zarr",
        "ome-zarr-v2": ".ome.zarr",
        "ome-tiff": ".ome.tif",
        "imaris": ".ims",
        "both": ".ome.zarr",  # zarr is the primary; .ome.tif is also written
    }

    def expected_output_path(
        self, acquisition_dir: Path, output_dir: Path
    ) -> Path:
        """Path of the primary output store this config would write.

        Lets the GUI/CLI detect an existing result (same acquisition + settings)
        before running, so a re-run can prompt / skip / rename instead of
        silently overwriting. The name encodes the enabled preprocessing tags,
        so runs with different settings resolve to different files.
        """
        basename = self._build_output_basename(Path(acquisition_dir))
        ext = self._PRIMARY_EXT.get(self.config.output_format, ".ome.zarr")
        return Path(output_dir) / f"{basename}{ext}"

    def run(
        self,
        acquisition_dir: Path,
        output_path: Path,
        channels: Optional[List[int]] = None,
        tiles: Optional[List[RawTileInfo]] = None,
    ) -> Path:
        """Run the full stitching pipeline.

        Thin wrapper that owns lifecycle of the ETA estimator so
        ``finalize()`` runs on every exit path -- success, user
        cancellation, or raised exception. The pipeline body lives
        in :meth:`_run_impl`.
        """
        try:
            # Register a dask cancel callback for the whole run so a long
            # fuse/write compute aborts promptly (≈ one chunk) instead of
            # running to completion after the user hits Cancel.
            with _CancelCallback(self._cancelled_fn):
                result = self._run_impl(acquisition_dir, output_path, channels, tiles)
            cancelled = bool(self._cancelled_fn())
            if self._estimator is not None:
                self._estimator.finalize(success=not cancelled)
            return result
        except PipelineCancelled:
            # A dask compute was torn down by the user cancel. Treat as a clean
            # cancellation (not an error): the worker checks _cancelled_fn().
            self.logger.info("Pipeline cancelled by user (compute aborted)")
            if self._estimator is not None:
                self._estimator.finalize(success=False)
            return output_path
        except BaseException:
            if self._estimator is not None:
                self._estimator.finalize(success=False)
            raise
        finally:
            self._stop_memory_watchdog()
            # Per-run time breakdown to the log, on every exit path. finalize()
            # above has flushed the in-progress phase in each branch, so the
            # durations are complete (partial when cancelled/errored, which is
            # itself useful -- it shows where the time went before the stop).
            if self._estimator is not None:
                try:
                    for _line in self._estimator.format_breakdown():
                        self.logger.info(_line)
                except Exception as _e:  # never let logging sink a run
                    self.logger.debug(f"Time-breakdown log failed: {_e}")

    def _run_impl(
        self,
        acquisition_dir: Path,
        output_path: Path,
        channels: Optional[List[int]] = None,
        tiles: Optional[List[RawTileInfo]] = None,
    ) -> Path:
        """Stitching pipeline body.

        Produces a single multi-channel (C,Z,Y,X) OME-Zarr/TIFF store
        with shared registration across channels.

        Args:
            acquisition_dir: Root directory containing tile folders
            output_path: Where to write the stitched result
            channels: Which channels to process (None = all found)
            tiles: Pre-discovered tiles (skips discover_tiles if provided)

        Returns:
            Path to the stitched output
        """
        t0 = time.time()
        # A reused pipeline object must re-vote: a different acquisition can
        # have a different camera orientation.
        self._reset_destripe_axis()
        self.logger.info(f"=== Stitching Pipeline Start ===")
        for _line in environment_summary():
            self.logger.info(_line)
        # Loud, before anything expensive: an old multiview-stitcher silently
        # writes black lines through the output, and finding that out after a
        # 10-hour fuse is the worst possible time.
        for _line in check_multiview_stitcher_version():
            self.logger.warning(_line)
        self.logger.info(f"Input:  {acquisition_dir}")
        self.logger.info(f"Output: {output_path}")
        self._apply_scope_profile(acquisition_dir)

        # --- Step 1: Discover tiles ---
        self._progress_fn(2, "Discovering tiles...")
        if tiles is None:
            self.logger.info("Step 1: Discovering tiles...")
            tiles = discover_tiles(acquisition_dir)
        else:
            self.logger.info(f"Step 1: Using {len(tiles)} pre-discovered tiles")
        if not tiles:
            raise FileNotFoundError(f"No tile folders found in {acquisition_dir}")

        self._log_tile_summary(tiles)

        # --- Resolve / verify acquisition geometry (frame size, optics, flags) ---
        self._apply_and_log_geometry(tiles, acquisition_dir)

        # --- Build the multi-phase ETA estimator ---
        # Built after discover so we know tile count + planes. The
        # progress hook already started the "discover" phase when the
        # first emit went through; we'll start it here too once the
        # estimator exists, then transition on subsequent emits.
        try:
            self._estimator = self._build_estimator(
                tiles, acquisition_dir=acquisition_dir, output_dir=output_path
            )
            self._estimator.start_phase("discover")
        except Exception as e:
            self.logger.debug(f"ETA estimator unavailable: {e}")
            self._estimator = None

        # Determine which channels to process
        all_channels = sorted(set(ch for t in tiles for ch in t.channels))
        if channels is not None:
            process_channels = [ch for ch in channels if ch in all_channels]
        else:
            process_channels = all_channels
        self.logger.info(
            f"Channels: {describe_channel_set(process_channels, all_channels)}"
        )

        # Determine Z step
        z_step_um = self.config.z_step_um
        if z_step_um is None:
            z_step_um = tiles[0].z_step_mm * 1000.0
            self.logger.info(f"Z step from data: {z_step_um:.3f} µm")

        # Resolve iso sentinel (either axis set to ISO_DOWNSAMPLE triggers
        # isotropic auto-selection for both). Writes resolved ints back to
        # the config so tags, preprocessing, and metadata see real factors.
        if (
            self.config.downsample_xy == ISO_DOWNSAMPLE
            or self.config.downsample_z == ISO_DOWNSAMPLE
        ):
            iso_xy, iso_z = compute_iso_downsample(self.config.pixel_size_um, z_step_um)
            self.logger.info(
                f"Iso downsample: native XY={self.config.pixel_size_um:.3f} µm "
                f"Z={z_step_um:.3f} µm → XY={iso_xy}x Z={iso_z}x "
                f"(output {self.config.pixel_size_um * iso_xy:.3f} × "
                f"{z_step_um * iso_z:.3f} µm)"
            )
            self.config.downsample_xy = iso_xy
            self.config.downsample_z = iso_z

        # Apply downsample factors to voxel sizes
        ds_xy = self.config.downsample_xy
        ds_z = self.config.downsample_z
        voxel_size_um = {
            "z": z_step_um * ds_z,
            "y": self.config.pixel_size_um * ds_xy,
            "x": self.config.pixel_size_um * ds_xy,
        }
        if ds_xy > 1 or ds_z > 1:
            self.logger.info(f"Downsample: XY={ds_xy}x Z={ds_z}x")
        self.logger.info(
            f"Voxel size: Z={voxel_size_um['z']:.3f} "
            f"Y={voxel_size_um['y']:.3f} X={voxel_size_um['x']:.3f} µm"
        )

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Determine streaming vs in-memory mode (before loading) ---
        # When splitting illumination, each side becomes its own output channel,
        # so size the output (disk guard, streaming vs in-memory) on the EXPANDED
        # channel count, not the raw one.
        output_units = self._output_channel_units(tiles, process_channels)
        _est_channels = (
            list(range(len(output_units)))
            if len(output_units) != len(process_channels)
            else process_channels
        )
        mem_est = estimate_memory_usage(tiles, _est_channels, self.config)
        use_streaming = self.config.streaming_mode
        if getattr(self.config, "split_illumination", False) and len(
            output_units
        ) != len(process_channels):
            # The split path is implemented only in the streaming fuse loop (it
            # materializes single-side tiles per output channel). Force streaming
            # so a small dataset that would otherwise go in-memory still splits.
            if not use_streaming:
                self.logger.info(
                    "Split illumination: forcing streaming mode "
                    f"({len(output_units)} output channels from "
                    f"{len(process_channels)} acquired channels)"
                )
            use_streaming = True
        if use_streaming is None:
            use_streaming = mem_est["auto_streaming"]
            self.logger.info(
                f"Memory estimate: in-memory ~{mem_est['in_memory_gb']:.0f} GB, "
                f"streaming ~{mem_est['streaming_gb']:.1f} GB, "
                f"output ~{mem_est['output_gb']:.0f} GB \u2192 "
                f"{'streaming' if use_streaming else 'in-memory'} mode"
            )
        else:
            self.logger.info(
                f"Mode: {'streaming' if use_streaming else 'in-memory'} (user-selected)"
            )

        # D3: if AUTO-mode picked in-memory but it would trip the resource guard
        # (which keys off AVAILABLE RAM, whereas auto-select keys off TOTAL RAM),
        # and streaming WOULD fit, fall back to streaming rather than hard-abort
        # a job that could have run. (Respect an explicit user mode choice.)
        if self.config.streaming_mode is None and not use_streaming:
            try:
                import psutil

                _avail_gb = psutil.virtual_memory().available / (1024**3)
            except Exception:
                _avail_gb = 0.0
            _ram_frac = float(getattr(self.config, "resource_guard_ram_fraction", 0.95))
            # Match _enforce_resource_limits, which adds PyImarisWriter scratch
            # (~0.25x output) to the peak for imaris output — otherwise the
            # fallback can pick streaming believing it fits, then the guard
            # aborts anyway.
            _imaris_extra = (
                0.25 * float(mem_est.get("output_gb", 0.0))
                if self.config.output_format == "imaris"
                else 0.0
            )
            if (
                _avail_gb > 0
                and mem_est["in_memory_gb"] + _imaris_extra > _avail_gb * _ram_frac
                and mem_est["streaming_gb"] + _imaris_extra <= _avail_gb * _ram_frac
            ):
                self.logger.info(
                    f"Auto-mode: in-memory ~{mem_est['in_memory_gb']:.0f} GB "
                    f"exceeds {_ram_frac * 100:.0f}% of {_avail_gb:.0f} GB "
                    f"available; falling back to streaming "
                    f"(~{mem_est['streaming_gb']:.1f} GB) instead of aborting."
                )
                use_streaming = True

        self._log_preflight(
            tiles, process_channels, output_path, mem_est, use_streaming
        )

        # Hard abort before committing any resources if this item would blow
        # past available RAM or disk (prevents machine-killing OOM/ENOSPC).
        self._enforce_resource_limits(tiles, output_path, mem_est, use_streaming)

        # Arm the warn-only watchdog for the rest of the run (stopped in run()).
        self._start_memory_watchdog(mem_est, use_streaming)

        if use_streaming:
            # ============================================================
            # STREAMING PATH: load one channel at a time to minimize RAM.
            # Only the reference channel is loaded for registration;
            # subsequent channels are loaded, fused, computed into the
            # output array, then freed before the next channel loads.
            # Peak RAM = output array + one channel's tiles.
            # ============================================================
            return self._run_streaming(
                process_channels,
                voxel_size_um,
                output_path,
                tiles,
                acquisition_dir,
                t0,
            )

        # ============================================================
        # IN-MEMORY PATH: load all channels, register, fuse, write.
        # ============================================================

        # --- Step 2: Load + preprocess tiles ---
        if self._cancelled_fn():
            self.logger.info("Pipeline cancelled by user")
            return output_path

        # Flat-field: estimate models UP FRONT (before load) so they're applied
        # inside _preprocess_single_tile during the load — the same single
        # flat-field path as the streaming mode (early, native, per plane). This
        # replaces the old post-load estimate-then-apply-whole-volume step, so
        # both modes now flat-field identically.
        if self.config.flat_field_correction:
            self._progress_fn(4, "Estimating flat-field profiles (BaSiCPy)...")
            self.logger.info("Step 2b: Estimating flat-field profiles (BaSiCPy)...")
            self._flatfield_models = self._estimate_flatfield_models(
                tiles, process_channels
            )
            if self._cancelled_fn():
                self.logger.info("Pipeline cancelled by user")
                return output_path
        else:
            self._flatfield_models = {}

        self._progress_fn(5, "Loading and preprocessing tiles...")
        self.logger.info("Step 2: Loading and preprocessing tiles...")
        channel_tile_data = self._load_and_preprocess(tiles, process_channels)

        if self._cancelled_fn():
            self.logger.info("Pipeline cancelled by user")
            return output_path

        # --- Step 3: Register using reference channel ---
        if self.config.skip_registration:
            self._progress_fn(45, "Skipping registration (using stage positions)...")
            self.logger.info(
                "Step 3: Skipping registration — using stage positions only"
            )
            self._registration_report = registration_report.skipped_report(
                "'Skip registration' was on: tiles were placed by stage position "
                "only, so any tile-to-tile offset in the output is stage placement "
                "error and no correction was even attempted",
                tiles=tiles,
            )
            reg_params = []
            try:
                from multiview_stitcher import io as mvs_io

                transform_key = mvs_io.METADATA_TRANSFORM_KEY
            except ImportError:
                transform_key = "affine_metadata"
        else:
            ref_ch = self.config.reg_channel
            if ref_ch not in channel_tile_data or not channel_tile_data[ref_ch]:
                ref_ch = process_channels[0]
            ref_tile_data = channel_tile_data[ref_ch]

            self._progress_fn(45, f"Registering tiles (channel {ref_ch})...")
            self.logger.info(
                f"Step 3: Registering on reference channel {ref_ch} "
                f"({len(ref_tile_data)} tiles)..."
            )
            reg_params, transform_key = self._register_tiles(
                ref_tile_data, voxel_size_um
            )

        if self._cancelled_fn():
            self.logger.info("Pipeline cancelled by user")
            return output_path

        # --- Optional: tile-border artifact QC (diagnostic) ---
        if self.config.border_qc_enabled and not self._cancelled_fn():
            try:
                self._run_border_qc(
                    channel_tile_data, tiles, acquisition_dir,
                    voxel_size_um, output_path,
                )
            except Exception as e:  # QC must never fail a run
                self.logger.warning(f"Border QC pass failed (skipped): {e}")

        # ============================================================
        # IN-MEMORY PATH (original)
        # ============================================================

        # --- Step 4+5: Fuse each channel and build stacked (C,Z,Y,X) ---
        # Memory-efficient approach: fuse the first channel to learn the
        # output shape, pre-allocate the full stacked array, copy ch0 into
        # it (then free ch0), and compute remaining channels directly into
        # their slice of the stacked array. Peak RAM = stacked + 1 channel
        # working set, NOT stacked + all channels.
        import dask.array as da
        import dask.diagnostics

        channel_origins = []
        fused_channel_ids = []
        stacked = None  # Will be allocated after first channel is fused

        for ch_idx, ch_id in enumerate(process_channels):
            if self._cancelled_fn():
                self.logger.info("Pipeline cancelled by user")
                return output_path

            tile_data = channel_tile_data.get(ch_id, [])
            if not tile_data:
                self.logger.warning(f"No data for {describe_channel(ch_id)}, skipping")
                continue

            fuse_pct = 55 + int(15 * ch_idx / max(len(process_channels), 1))
            self._progress_fn(
                fuse_pct, f"Fusing {describe_channel(ch_id)} ({len(tile_data)} tiles)..."
            )
            self.logger.info(
                f"Step 4: Fusing {describe_channel(ch_id)} ({len(tile_data)} tiles)..."
            )
            fused_sim, origin_um = self._fuse_channel(
                tile_data, voxel_size_um, reg_params, transform_key
            )

            self._progress_fn(
                fuse_pct + 5,
                f"Computing {describe_channel(ch_id)} into memory...",
            )
            # Convert IN the graph (see lazy_uint16) — doing it after .compute()
            # held three full-size copies of the fused volume at once.
            darr = lazy_uint16(fused_sim.data)
            # Background zeroing, in the graph, per chunk — the streaming path
            # has always done this in its _finalize; the in-memory path silently
            # ignored the setting, so the same acquisition came out different
            # depending on which mode auto-select happened to pick.
            if self.config.background_zero_enabled:
                _bg = int(self.config.background_zero_thresholds.get(ch_id, 0))
                if _bg > 0:
                    self.logger.info(
                        f"  {describe_channel(ch_id).capitalize()}: background "
                        f"zeroing below {_bg}"
                    )
                    darr = da.where(darr > np.uint16(_bg), darr, np.uint16(0))

            # Bound the dask scheduler the same way the streaming path does.
            # multiview-stitcher fuses block-by-block, stacking every source
            # tile that overlaps a block as a float64 array at the block's
            # size. With the default (unbounded) threaded scheduler, many such
            # blocks materialise at once — on a heavily-downsampled grid (small
            # tiles, many overlapping one block) that turned a "~10 GB" job into
            # a 190 GB OOM. Capping concurrency keeps peak ≈ workers × one block.
            fuse_workers = self._pick_fuse_workers(darr)
            if fuse_workers <= 1:
                scheduler_cfg: Dict[str, Any] = {"scheduler": "synchronous"}
                scheduler_name = "synchronous"
            else:
                scheduler_cfg = {"scheduler": "threads", "num_workers": fuse_workers}
                scheduler_name = f"threads×{fuse_workers}"
            self.logger.info(
                f"  Computing {describe_channel(ch_id)} into memory "
                f"(scheduler={scheduler_name})..."
            )
            with dask.config.set(**scheduler_cfg):
                with dask.diagnostics.ProgressBar():
                    vol = np.asarray(darr.compute())
            del darr

            if stacked is None:
                # First channel — allocate the full stacked array
                n_total = sum(1 for c in process_channels if channel_tile_data.get(c))
                if n_total > 1:
                    stacked = np.zeros((n_total, *vol.shape), dtype=np.uint16)
                    stacked[0] = vol
                    self.logger.info(
                        f"  Pre-allocated stacked array: "
                        f"{stacked.shape} "
                        f"({stacked.nbytes / (1024**3):.1f} GB)"
                    )
                else:
                    stacked = vol
            else:
                # Subsequent channels — copy into pre-allocated slice
                dest_idx = len(fused_channel_ids)
                sz, sy, sx = vol.shape
                stacked[dest_idx, :sz, :sy, :sx] = vol

            # Free the per-channel array
            del vol

            channel_origins.append(origin_um)
            fused_channel_ids.append(ch_id)

            self.logger.info(
                f"  {describe_channel(ch_id).capitalize()}: shape={stacked.shape[-3:] if stacked.ndim == 4 else stacked.shape}, "
                f"origin Z={origin_um['z']:.1f} Y={origin_um['y']:.1f} "
                f"X={origin_um['x']:.1f} µm"
            )

        if stacked is None:
            self.logger.error("No channels were fused successfully")
            return output_path

        self.logger.info(
            f"Step 5: Stacked {len(fused_channel_ids)} channels → "
            f"shape={stacked.shape}"
        )

        # --- Step 6: Write ---
        if self._cancelled_fn():
            self.logger.info("Pipeline cancelled by user")
            return output_path

        self._progress_fn(75, "Writing multi-channel output...")
        self.logger.info("Step 6: Writing multi-channel output...")
        basename = self._build_output_basename(acquisition_dir)
        self.logger.info(f"  Output basename: {basename}")
        channel_names = [f"Channel_{ch_id}" for ch_id in fused_channel_ids]
        self._write_multichannel_output(
            stacked, channel_names, voxel_size_um, output_path, basename
        )

        # --- Step 7: Write metadata ---
        self._progress_fn(95, "Writing metadata...")
        origin_um = channel_origins[0]
        self._write_stitch_metadata_v2(
            output_path,
            fused_channel_ids,
            origin_um,
            tiles,
            voxel_size_um,
            acquisition_dir,
            basename,
        )

        if self.config.registration_report_enabled:
            try:
                self._write_registration_report(output_path, acquisition_dir)
            except Exception as exc:  # evidence must never fail a run
                self.logger.warning(f"Registration report skipped: {exc}")

        elapsed = time.time() - t0
        self.logger.info(
            f"=== Pipeline complete in {elapsed:.1f}s === Output: {output_path}"
        )
        # Repeat the version warning at the end: on a multi-hour run the header
        # has long scrolled away, and this one means the output is corrupt.
        for _line in check_multiview_stitcher_version():
            self.logger.warning(_line)
        return output_path

    def run_preview(
        self,
        acquisition_dir: Path,
        channels: Optional[List[int]] = None,
        tiles: Optional[List[RawTileInfo]] = None,
    ) -> Dict[int, np.ndarray]:
        """Run preprocessing + registration + fusion at preview downsample.

        Returns one fused numpy array per channel WITHOUT writing any
        output files and WITHOUT applying background zeroing — the user
        will pick a threshold from these volumes. Background zeroing is
        intentionally skipped here so the preview shows the un-masked
        intensities the threshold will act on.

        Output factors are temporarily overridden with
        ``background_zero_preview_downsample_xy/_z`` so the preview is
        cheap regardless of the configured full-res downsample. The
        original config values are restored before return.
        """
        import shutil
        import tempfile

        import dask
        import dask.array as da

        # Snapshot fields we mutate so this method has no side effects
        # on the parent config.
        saved = {
            "downsample_xy": self.config.downsample_xy,
            "downsample_z": self.config.downsample_z,
            "background_zero_enabled": self.config.background_zero_enabled,
            "streaming_mode": self.config.streaming_mode,
        }
        self.config.downsample_xy = int(
            self.config.background_zero_preview_downsample_xy
        )
        self.config.downsample_z = int(self.config.background_zero_preview_downsample_z)
        # Preview shows intensities BEFORE thresholding so the user can
        # pick a threshold; never apply background-zero in preview path.
        self.config.background_zero_enabled = False

        tmp_dir = Path(tempfile.mkdtemp(prefix="stitch_preview_"))
        try:
            self.logger.info(
                f"=== Background-zero preview "
                f"(downsample XY={self.config.downsample_xy}x "
                f"Z={self.config.downsample_z}x) ==="
            )

            # Discover or accept tiles
            if tiles is None:
                tiles = discover_tiles(acquisition_dir)
            if not tiles:
                raise FileNotFoundError(f"No tiles in {acquisition_dir}")

            all_channels = sorted(set(ch for t in tiles for ch in t.channels))
            process_channels = (
                [ch for ch in channels if ch in all_channels]
                if channels is not None
                else all_channels
            )

            z_step_um = self.config.z_step_um
            if z_step_um is None:
                z_step_um = tiles[0].z_step_mm * 1000.0

            # Resolve any iso sentinel for the preview run too.
            if (
                self.config.downsample_xy == ISO_DOWNSAMPLE
                or self.config.downsample_z == ISO_DOWNSAMPLE
            ):
                iso_xy, iso_z = compute_iso_downsample(
                    self.config.pixel_size_um, z_step_um
                )
                self.config.downsample_xy = iso_xy
                self.config.downsample_z = iso_z

            ds_xy = self.config.downsample_xy
            ds_z = self.config.downsample_z
            voxel_size_um = {
                "z": z_step_um * ds_z,
                "y": self.config.pixel_size_um * ds_xy,
                "x": self.config.pixel_size_um * ds_xy,
            }

            # Register on reference channel (or skip)
            if self.config.skip_registration:
                reg_params = []
                try:
                    from multiview_stitcher import io as mvs_io

                    transform_key = mvs_io.METADATA_TRANSFORM_KEY
                except ImportError:
                    transform_key = "affine_metadata"
            else:
                ref_ch = self.config.reg_channel
                if ref_ch not in process_channels:
                    ref_ch = process_channels[0]
                ref_data = self._load_and_preprocess(tiles, [ref_ch])
                ref_tile_data = ref_data.get(ref_ch, [])
                if not ref_tile_data:
                    raise RuntimeError(f"No tiles for reference channel {ref_ch}")
                reg_params, transform_key = self._register_tiles(
                    ref_tile_data, voxel_size_um
                )
                del ref_data, ref_tile_data
                gc.collect()

            # Probe one tile for output shape
            probe_vol = self._preprocess_single_tile(tiles[0], process_channels[0])
            expected_tile_shape = probe_vol.shape
            del probe_vol
            gc.collect()

            preview: Dict[int, np.ndarray] = {}
            for ch_id in process_channels:
                if self._cancelled_fn():
                    break

                ch_tmp_dir = tmp_dir / f"ch{ch_id:02d}"
                tile_data = self._materialize_tiles_to_disk(
                    tiles, ch_id, expected_tile_shape, ch_tmp_dir
                )
                if not tile_data:
                    self.logger.warning(
                        f"Preview: no data for {describe_channel(ch_id)}, skipping"
                    )
                    shutil.rmtree(ch_tmp_dir, ignore_errors=True)
                    continue

                fused_sim, _origin = self._fuse_channel(
                    tile_data, voxel_size_um, reg_params, transform_key
                )
                darr = lazy_uint16(fused_sim.data)

                fuse_workers = self._pick_fuse_workers(darr)
                if fuse_workers <= 1:
                    scheduler_cfg: Dict[str, Any] = {"scheduler": "synchronous"}
                else:
                    scheduler_cfg = {
                        "scheduler": "threads",
                        "num_workers": fuse_workers,
                    }
                with dask.config.set(**scheduler_cfg):
                    arr = darr.compute()
                preview[ch_id] = np.asarray(arr, dtype=np.uint16)
                self.logger.info(
                    f"  Preview {describe_channel(ch_id)}: shape={preview[ch_id].shape} "
                    f"min={preview[ch_id].min()} max={preview[ch_id].max()}"
                )

                del darr, fused_sim, tile_data
                gc.collect()
                shutil.rmtree(ch_tmp_dir, ignore_errors=True)

            return preview
        finally:
            self.config.downsample_xy = saved["downsample_xy"]
            self.config.downsample_z = saved["downsample_z"]
            self.config.background_zero_enabled = saved["background_zero_enabled"]
            self.config.streaming_mode = saved["streaming_mode"]
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _run_streaming(
        self,
        process_channels: List[int],
        voxel_size_um: Dict[str, float],
        output_path: Path,
        tiles: List[RawTileInfo],
        acquisition_dir: Path,
        t0: float,
    ) -> Path:
        """Streaming pipeline path: load one channel at a time.

        For each channel: load tiles from disk \u2192 fuse \u2192 compute into
        a pre-allocated output array \u2192 free tiles before the next channel.
        Peak RAM = output array + one channel's tile data, instead of
        all channels' tile data simultaneously.
        """
        import gc
        import shutil

        import dask.diagnostics

        # Estimate flat-field models UP FRONT (before registration) so every
        # channel is corrected consistently — including the reference channel,
        # whose tiles are materialised for registration and reused for fusion
        # (C1). This matches the in-memory path, which flat-fields before
        # registering. Previously the streaming path skipped flat-field entirely.
        if self.config.flat_field_correction:
            self._progress_fn(4, "Estimating flat-field (streaming)...")
            self._flatfield_models = self._estimate_flatfield_models(
                tiles, process_channels
            )
        else:
            self._flatfield_models = {}

        # --- Step 2+3: Load reference channel + register ---
        # If registration runs, it materialises the reference channel's tiles to
        # disk; these are kept and reused for that channel's fusion pass instead
        # of preprocessing + spilling them a second time (C1).
        reg_reuse_ch = None
        reg_reuse_data = None
        reg_reuse_dir = None
        # Which illumination side the reusable spill was built from.
        # None = sides fused, which is what the registration path always
        # produces (it materialises the reference channel without a side).
        reg_reuse_side = None
        if self.config.skip_registration:
            self._progress_fn(45, "Skipping registration (using stage positions)...")
            self.logger.info(
                "Step 3: Skipping registration \u2014 using stage positions only"
            )
            self._registration_report = registration_report.skipped_report(
                "'Skip registration' was on: tiles were placed by stage position "
                "only, so any tile-to-tile offset in the output is stage placement "
                "error and no correction was even attempted",
                tiles=tiles,
            )
            reg_params = []
            try:
                from multiview_stitcher import io as mvs_io

                transform_key = mvs_io.METADATA_TRANSFORM_KEY
            except ImportError:
                transform_key = "affine_metadata"
        else:
            ref_ch = self.config.reg_channel
            if ref_ch not in process_channels:
                ref_ch = process_channels[0]

            self._progress_fn(
                5, f"Loading reference channel {ref_ch} for registration..."
            )
            self.logger.info(
                f"Step 2: Loading reference channel {ref_ch} "
                f"({len(tiles)} tiles) [streaming, spill to disk]..."
            )
            # Spill the reference channel to per-tile on-disk memmaps instead
            # of holding it in RAM. _load_and_preprocess would return one full
            # uint16 array per tile (~tile_bytes each) and keep ALL of them
            # resident — e.g. 176 × 3.25 GiB ≈ 570 GB, which overruns the
            # process commit limit and dies with "Unable to allocate 3.25 GiB"
            # even on a ~900 GB box. Registration only reads the tiles lazily
            # (and binned), so memmap-back them and let dask page in chunks.
            reg_tmp_dir = (
                _scratch_base_dir(self.config, output_path)
                / ".stitch_tmp"
                / f"reg_ch{ref_ch:02d}"
            )
            probe = self._preprocess_single_tile(tiles[0], ref_ch)
            ref_shape = probe.shape
            del probe
            gc.collect()
            ref_tile_data = self._materialize_tiles_to_disk(
                tiles, ref_ch, ref_shape, reg_tmp_dir
            )

            if not ref_tile_data:
                self.logger.error(f"No tiles loaded for reference channel {ref_ch}")
                shutil.rmtree(reg_tmp_dir, ignore_errors=True)
                return output_path

            if self._cancelled_fn():
                shutil.rmtree(reg_tmp_dir, ignore_errors=True)
                self.logger.info("Pipeline cancelled by user")
                return output_path

            self._progress_fn(45, f"Registering tiles (channel {ref_ch})...")
            self.logger.info(
                f"Step 3: Registering on reference channel {ref_ch} "
                f"({len(ref_tile_data)} tiles)..."
            )
            try:
                reg_params, transform_key = self._register_tiles(
                    ref_tile_data, voxel_size_um
                )
            except BaseException:
                # On failure/cancel drop the spill. (On success we KEEP it — see
                # below — so the fusion loop can reuse the ref channel's
                # already-materialised tiles.)
                del ref_tile_data
                gc.collect()
                shutil.rmtree(reg_tmp_dir, ignore_errors=True)
                raise
            # Keep the ref-channel spill for reuse in the fusion loop (C1):
            # re-materialising it there would preprocess + write N tiles a second
            # time (a full redundant pass, worst with deconvolution on).
            reg_reuse_ch = ref_ch
            reg_reuse_data = ref_tile_data
            reg_reuse_dir = reg_tmp_dir
            self.logger.info(
                "  Registration complete; reusing ref-channel spill for fusion"
            )

        if self._cancelled_fn():
            if reg_reuse_dir is not None:
                shutil.rmtree(reg_reuse_dir, ignore_errors=True)
            self.logger.info("Pipeline cancelled by user")
            return output_path

        # --- Optional: tile-border artifact QC (diagnostic, streaming) ---
        if self.config.border_qc_enabled and not self._cancelled_fn():
            try:
                # Build QC's spill for the side the FIRST output unit wants, so
                # the fusion loop can actually reuse it. Under
                # split_illumination a FUSED spill matches no unit, and QC
                # became a silent extra full preprocess of every tile.
                _first_side = (
                    self._output_channel_units(tiles, process_channels) or [(None,) * 3]
                )[0][2]
                qc_ch, qc_data, qc_dir, qc_side = self._run_border_qc_streaming(
                    tiles, process_channels, voxel_size_um, output_path,
                    acquisition_dir, reg_reuse_ch, reg_reuse_data,
                    reuse_side=_first_side,
                )
                # If QC materialized its own ref-channel spill (registration was
                # skipped, so there was none to reuse), keep it and hand it to
                # the fusion loop below — otherwise QC silently doubles the
                # preprocess (a full extra spill of every tile).
                if qc_data is not None and reg_reuse_data is None:
                    reg_reuse_ch = qc_ch
                    reg_reuse_data = qc_data
                    reg_reuse_dir = qc_dir
                    reg_reuse_side = qc_side
            except Exception as e:  # QC must never fail a run
                self.logger.warning(f"Border QC pass failed (skipped): {e}")

        # --- Step 4+5: Fuse each channel, compute, accumulate ---
        # Tile spill-to-disk: each tile is preprocessed exactly once and
        # written to a per-tile memmap under a temp dir. Fusion reads
        # chunks directly from the flat files, so output-chunk computes
        # never retrigger the full preprocess chain.
        import dask
        import dask.array as da

        # Determine tile output shape (load one tile, measure, free)
        probe_ch = process_channels[0]
        probe_vol = self._preprocess_single_tile(tiles[0], probe_ch)
        expected_tile_shape = probe_vol.shape
        del probe_vol
        gc.collect()
        self.logger.info(
            f"  Tile output shape: {expected_tile_shape} "
            f"({np.prod(expected_tile_shape) * 2 / (1024**3):.2f} GB uint16)"
        )

        channel_origins = []
        fused_channel_ids = []
        stacked = None
        fused_memmap_path = None

        # Expand real channels into output channels. Normally 1:1 (side=None →
        # fuse illumination); with split_illumination each two-sided channel
        # becomes one output channel per light path (side set → no fusion), so
        # the loop below writes e.g. "3_I0" and "3_I1" as separate channels.
        output_units = self._output_channel_units(tiles, process_channels)
        n_out = max(len(output_units), 1)

        tmp_root = _scratch_base_dir(self.config, output_path) / ".stitch_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        imaris_mode = self.config.output_format == "imaris"
        try:
            for ch_idx, (ch_id, src_ch, side) in enumerate(output_units):
                if self._cancelled_fn():
                    self.logger.info("Pipeline cancelled by user")
                    # This early return isn't inside the write-phase finally, so
                    # clean the spill ourselves (it now includes the reused
                    # ref-channel spill). Drop the memmap ref first (Windows lock).
                    stacked = None
                    gc.collect()
                    shutil.rmtree(tmp_root, ignore_errors=True)
                    return output_path

                fuse_pct = 50 + int(35 * (ch_idx + 0.5) / n_out)
                self._progress_fn(
                    fuse_pct,
                    f"Fusing channel {ch_id} "
                    f"({ch_idx + 1}/{n_out}) "
                    f"[materializing tiles]...",
                )
                self.logger.info(
                    f"Step 4: Fusing {describe_channel(ch_id)} ({len(tiles)} tiles) "
                    f"[streaming, one-shot tile preprocess → memmap]..."
                )

                # Preprocess each tile once, spill to memmap on disk — unless
                # this is the reference channel, whose tiles were already
                # materialised for registration or border QC (C1): reuse that
                # spill instead of preprocessing + writing every tile a second
                # time. The spill was built for ONE illumination side (None =
                # sides fused), so it only stands in for a unit wanting that
                # same side — otherwise its pixels are not what this unit needs.
                if (
                    side == reg_reuse_side
                    and src_ch == reg_reuse_ch
                    and reg_reuse_data is not None
                ):
                    ch_tmp_dir = reg_reuse_dir
                    tile_data = reg_reuse_data
                    reg_reuse_data = None  # consumed
                    self.logger.info(
                        f"  Reusing pre-registered spill for {describe_channel(ch_id)} "
                        f"(skipping re-materialize)"
                    )
                else:
                    ch_tmp_dir = tmp_root / f"ch{ch_idx:02d}"
                    tile_data = self._materialize_tiles_to_disk(
                        tiles, src_ch, expected_tile_shape, ch_tmp_dir,
                        illum_side=side,
                    )
                if not tile_data:
                    self.logger.warning(f"No data for {describe_channel(ch_id)}, skipping")
                    if ch_tmp_dir.exists():
                        shutil.rmtree(ch_tmp_dir, ignore_errors=True)
                    continue

                compute_pct = 50 + int(35 * (ch_idx + 0.8) / n_out)
                self._progress_fn(
                    compute_pct,
                    f"Computing channel {ch_id} "
                    f"({ch_idx + 1}/{n_out})...",
                )

                # Per-channel background-zero threshold, applied in the graph
                # (per-chunk, no extra peak). Shared by both fusion paths.
                # Thresholds are keyed by the real acquired channel, so both
                # split sides inherit their parent channel's threshold.
                bg_threshold = 0
                if self.config.background_zero_enabled:
                    bg_threshold = int(
                        self.config.background_zero_thresholds.get(src_ch, 0)
                    )
                    if bg_threshold > 0:
                        self.logger.info(
                            f"  Channel {ch_id}: background zeroing "
                            f"threshold={bg_threshold}"
                        )

                def _finalize(dk):
                    # Clip + cast to uint16 in the graph so float64 intermediates
                    # convert per-chunk; apply background zeroing if set.
                    dk = lazy_uint16(dk)
                    if bg_threshold > 0:
                        dk = da.where(dk > np.uint16(bg_threshold), dk, np.uint16(0))
                    return dk

                def _alloc_stacked(shape_zyx):
                    # Allocate the (C,Z,Y,X) on-disk fused memmap on first use so
                    # `stacked` never lives in RAM and writers stream from it.
                    nonlocal stacked, fused_memmap_path
                    if stacked is None:
                        fused_shape = (n_out, *shape_zyx)
                        fused_memmap_path = tmp_root / "fused.dat"
                        stacked = np.memmap(
                            fused_memmap_path,
                            dtype=np.uint16,
                            mode="w+",
                            shape=fused_shape,
                        )
                        self.logger.info(
                            f"  Fused output memmap: {fused_shape} "
                            f"({np.prod(fused_shape) * 2 / (1024**3):.1f} GB) "
                            f"-> {fused_memmap_path}"
                        )

                def _scheduler_for(dk):
                    fw = self._pick_fuse_workers(dk)
                    if fw <= 1:
                        return "synchronous", {"scheduler": "synchronous"}
                    return f"threads x{fw}", {
                        "scheduler": "threads",
                        "num_workers": fw,
                    }

                # Build fusion inputs once up front so the output shape is known
                # BEFORE committing to a fuse graph — that lets us auto-size
                # super-block regions to bound streaming fuse memory. The
                # whole-output da.store otherwise lets the threaded scheduler
                # hold far more than n_workers blocks (see
                # resolve_superblock_chunks); the estimate already assumes the
                # bounded, per-block model.
                sims, fuse_kwargs = self._build_fusion_inputs(
                    tile_data, voxel_size_um, reg_params, transform_key
                )
                full_props = self._full_stack_properties(sims, fuse_kwargs)
                _shp0 = full_props["shape"]
                _explicit_sb = int(
                    getattr(self.config, "fusion_superblock_chunks", 0) or 0
                )
                superblock = resolve_superblock_chunks(
                    self.config,
                    int(_shp0["z"]),
                    int(_shp0["y"]),
                    int(_shp0["x"]),
                    use_streaming=True,
                )
                if superblock > 0 and not _explicit_sb:
                    self.logger.info(
                        f"  Auto super-block: fusing in {superblock}×{superblock}"
                        f"×{superblock}-chunk regions to bound streaming fuse "
                        f"memory (target ~"
                        f"{float(self.config.fusion_superblock_target_gb):g} "
                        f"GB/region). Set 'Fusion super-block chunks' to override."
                    )

                # Both names always exist so the shared cleanup below can `del`
                # them regardless of which path ran (super-block leaves them None
                # and releases its region refs each iteration).
                darr = None
                fused_sim = None

                if superblock > 0:
                    # ---- Super-block batched fusion (item E): bound the dask
                    # graph to O(region) by fusing chunk-aligned sub-regions and
                    # storing each into the memmap slice, instead of one graph
                    # over the whole output. Chunk alignment makes this
                    # bit-identical to whole-output fusion for every overlap mode.
                    # (sims / fuse_kwargs / full_props already built above.)
                    org = full_props["origin"]
                    shp = full_props["shape"]
                    origin_um = {
                        "z": float(org["z"]),
                        "y": float(org["y"]),
                        "x": float(org["x"]),
                    }
                    full_shape = (int(shp["z"]), int(shp["y"]), int(shp["x"]))
                    _alloc_stacked(full_shape)
                    dest_idx = len(fused_channel_ids)
                    regions = list(
                        self._iter_superblock_regions(full_props, superblock)
                    )
                    self.logger.info(
                        f"  {describe_channel(ch_id).capitalize()}: shape={full_shape} "
                        f"origin Z={origin_um['z']:.1f} Y={origin_um['y']:.1f} "
                        f"X={origin_um['x']:.1f} um -- {len(regions)} "
                        f"super-block region(s) of {superblock} chunks/axis"
                    )
                    from multiview_stitcher import fusion as _fusion

                    for ridx, (rprops, (z0, z1, y0, y1, x0, x1)) in enumerate(regions):
                        if self._cancelled_fn():
                            self.logger.info("Pipeline cancelled by user")
                            break
                        region_kwargs = dict(fuse_kwargs)
                        region_kwargs["output_stack_properties"] = rprops
                        region_sim = self._fuse_with_fallback(
                            _fusion.fuse, sims, region_kwargs
                        )
                        rdarr = _finalize(region_sim.data)
                        _sname, scfg = _scheduler_for(rdarr)
                        with dask.config.set(**scfg):
                            # lock=False: regions/chunks write disjoint memmap
                            # ranges, so the default per-target write lock only
                            # serializes the block memcpys for no reason.
                            da.store(
                                rdarr,
                                stacked[dest_idx][z0:z1, y0:y1, x0:x1],
                                compute=True,
                                lock=False,
                            )
                        # Flush this region's writes so dirty pages of the
                        # (multi-hundred-GB) fused memmap don't accumulate across
                        # all regions. Without this the OS defers write-back and
                        # the process working set balloons far past the modeled
                        # fuse allocation (the on-disk fused output is meant to be
                        # off-RAM) — which on Windows (USS counts modified
                        # file-mapped pages) tripped the memory watchdog with a
                        # figure that is write-back lag, not real allocation. The
                        # msync only writes the ~region-sized dirty set, so it's
                        # cheap next to the region's fuse compute.
                        stacked.flush()
                        # Advance the global percent across regions so the
                        # (often long) fuse phase drives a live, moving ETA
                        # instead of sitting at a flat per-channel value. Fuse
                        # spans ~50–85%; give each channel an equal slice and
                        # interpolate by completed regions within it.
                        ch_frac = (ridx + 1) / max(len(regions), 1)
                        region_pct = 50 + int(35 * (ch_idx + ch_frac) / n_out)
                        self._progress_fn(
                            region_pct,
                            f"Channel {ch_id}: fused region "
                            f"{ridx + 1}/{len(regions)}",
                        )
                    del sims
                else:
                    # ---- Whole-output fusion (historical path) ----
                    # The up-front inputs were only needed to size super-block
                    # regions; the historical path rebuilds them in _fuse_channel.
                    del sims, fuse_kwargs, full_props
                    fused_sim, origin_um = self._fuse_channel(
                        tile_data, voxel_size_um, reg_params, transform_key
                    )
                    darr = _finalize(fused_sim.data)
                    self.logger.info(
                        f"  {describe_channel(ch_id).capitalize()}: shape={darr.shape} "
                        f"origin Z={origin_um['z']:.1f} Y={origin_um['y']:.1f} "
                        f"X={origin_um['x']:.1f} um"
                    )
                    _alloc_stacked(tuple(darr.shape))
                    dest_idx = len(fused_channel_ids)
                    scheduler_name, scheduler_cfg = _scheduler_for(darr)
                    self.logger.info(
                        f"  Storing channel {ch_id} into fused memmap "
                        f"(scheduler={scheduler_name}, memmap-backed tiles)..."
                    )
                    progress_cb = _TimeThrottledProgress(
                        self.logger,
                        label=f"channel {ch_id} store",
                        interval_s=30.0,
                        progress_fn=self._progress_fn,
                    )
                    with dask.config.set(**scheduler_cfg):
                        with progress_cb:
                            # lock=False: each channel writes a disjoint memmap
                            # slot; the default write lock only serializes memcpys.
                            da.store(darr, stacked[dest_idx], compute=True, lock=False)

                # Safety cap: if background zeroing was active, verify
                # the fraction of zeroed voxels stayed below the cap.
                # Reads the freshly-written memmap (one streaming pass,
                # cheap) rather than recomputing the dask graph.
                if bg_threshold > 0:
                    written = stacked[dest_idx]
                    n_total = int(written.size)
                    # Count zeros in Z-slabs. `written == 0` on the whole channel
                    # would allocate a full-size boolean array in RAM (~500 GB
                    # for a 1 TB channel) — an OOM at the exact scale this
                    # pipeline targets. One slab's worth of bool is bounded.
                    z_slab = max(1, int(self.config.output_chunksize.get("z", 128)))
                    n_zero = 0
                    for z0 in range(0, written.shape[0], z_slab):
                        n_zero += int(np.count_nonzero(written[z0 : z0 + z_slab] == 0))
                    zero_frac = n_zero / max(n_total, 1)
                    cap = float(self.config.background_zero_sanity_cap_fraction)
                    self.logger.info(
                        f"  Channel {ch_id}: background-zeroed "
                        f"{zero_frac * 100:.2f}% of voxels "
                        f"(threshold={bg_threshold}, cap={cap * 100:.0f}%)"
                    )
                    if zero_frac > cap:
                        raise RuntimeError(
                            f"Background-zero safety cap exceeded for "
                            f"channel {ch_id}: threshold {bg_threshold} "
                            f"would zero {zero_frac * 100:.2f}% of voxels "
                            f"(cap {cap * 100:.0f}%). Lower the threshold "
                            f"or disable Background zeroing."
                        )

                # Drop references to per-tile memmaps so Windows releases
                # the file locks, then delete the per-channel spill dir.
                del darr, fused_sim, tile_data
                gc.collect()
                shutil.rmtree(ch_tmp_dir, ignore_errors=True)

                channel_origins.append(origin_um)
                fused_channel_ids.append(ch_id)

            # Drop any never-reused reference spill (its fused tiles don't stand
            # in for a split side channel) so its memmaps are released before the
            # tmp_root cleanup below removes the backing files.
            if reg_reuse_data is not None:
                reg_reuse_data = None
                gc.collect()
        except Exception:
            # Release the memmap before bubbling so the file can be removed.
            if stacked is not None:
                stacked.flush()
                del stacked
                stacked = None
            gc.collect()
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
            raise

        if not fused_channel_ids:
            self.logger.error("No channels were fused successfully")
            if stacked is not None:
                del stacked
                stacked = None
            gc.collect()
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
            return output_path

        # Trim the memmap view to only the channels that actually fused
        # (in case a channel had no tile data and was skipped).
        if len(fused_channel_ids) < stacked.shape[0]:
            stacked = stacked[: len(fused_channel_ids)]

        # --- Step 6: Write ---
        if self._cancelled_fn():
            self.logger.info("Pipeline cancelled by user")
            if stacked is not None:
                del stacked
                stacked = None
            gc.collect()
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
            return output_path

        self._progress_fn(85, "Writing multi-channel output...")
        basename = self._build_output_basename(acquisition_dir)
        self.logger.info(f"  Output basename: {basename}")
        channel_names = [f"Channel_{ch_id}" for ch_id in fused_channel_ids]

        try:
            if imaris_mode:
                self.logger.info(
                    "Step 6: Writing Imaris .ims from fused memmap "
                    "(block reads are file I/O, not dask recompute)..."
                )
                ims_path = output_path / f"{basename}.ims"
                try:
                    from flamingo_stitcher.writers import imaris_writer

                    if not imaris_writer.is_available():
                        self.logger.error(
                            "Imaris writer unavailable: "
                            f"{imaris_writer.unavailable_reason()}"
                        )
                    else:
                        imaris_writer.write_imaris_streaming(
                            data=stacked,
                            output_path=ims_path,
                            voxel_size_um=voxel_size_um,
                            channel_names=channel_names,
                            progress_callback=self._progress_fn,
                        )
                except Exception as e:
                    self.logger.error(f"Imaris .ims write failed: {e}", exc_info=True)
            else:
                self.logger.info(
                    f"Step 5: Computed {len(fused_channel_ids)} channels \u2192 "
                    f"shape={stacked.shape}"
                )
                self.logger.info(
                    "Step 6: Writing multi-channel output from fused memmap..."
                )
                self._write_multichannel_output(
                    stacked, channel_names, voxel_size_um, output_path, basename
                )
        finally:
            # Release the memmap reference before deleting the backing file
            # (Windows holds locks until every numpy.memmap is gone).
            if stacked is not None:
                del stacked
                stacked = None
            gc.collect()
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)

        # --- Step 7: Write metadata ---
        self._progress_fn(95, "Writing metadata...")
        origin_um = channel_origins[0]
        self._write_stitch_metadata_v2(
            output_path,
            fused_channel_ids,
            origin_um,
            tiles,
            voxel_size_um,
            acquisition_dir,
            basename,
        )

        if self.config.registration_report_enabled:
            try:
                self._write_registration_report(output_path, acquisition_dir)
            except Exception as exc:  # evidence must never fail a run
                self.logger.warning(f"Registration report skipped: {exc}")

        elapsed = time.time() - t0
        self.logger.info(
            f"=== Pipeline complete (streaming) in {elapsed:.1f}s === "
            f"Output: {output_path}"
        )
        return output_path

    def _load_and_preprocess(
        self,
        tiles: List[RawTileInfo],
        channels: List[int],
    ) -> Dict[int, List[Tuple[Any, RawTileInfo]]]:
        """Load raw volumes, fuse illumination sides, optionally destripe.

        Returns {channel_id: [(volume_array, tile_info), ...]}.
        """
        # In-memory preprocessing is sequential (one tile at a time), so a nested
        # destripe pool may use the full machine budget.
        self._active_preprocess_workers = 1
        result: Dict[int, List[Tuple[Any, RawTileInfo]]] = {ch: [] for ch in channels}

        for i, tile in enumerate(tiles):
            if self._cancelled_fn():
                self.logger.info("Pipeline cancelled by user")
                return result

            # Progress: tiles span 5%–45% of total pipeline
            tile_pct = 5 + int(40 * i / max(len(tiles), 1))
            self._progress_fn(
                tile_pct,
                f"Loading tile {i + 1}/{len(tiles)} "
                f"(X={tile.x_mm:.2f} Y={tile.y_mm:.2f})",
            )

            self.logger.info(
                f"  Tile {i + 1}/{len(tiles)}: {tile.folder.name} "
                f"({tile.n_planes} planes, X={tile.x_mm:.2f} Y={tile.y_mm:.2f})"
            )

            for ch_id in channels:
                if ch_id not in tile.raw_files:
                    continue

                # Single shared preprocess path (load → illumination fuse →
                # depth attenuation → destripe → deconvolution → downsample →
                # camera-X flip). Both the in-memory and streaming modes go
                # through _preprocess_single_tile so the two can never diverge
                # again — they once did: the streaming copy dropped the frame
                # (AOI) size args and loaded cropped tiles at the wrong 2048²
                # default, scrambling the data.
                volume = self._preprocess_single_tile(tile, ch_id)
                result[ch_id].append((volume, tile))

        return result

    def _estimate_flatfield_models(
        self, tiles: List[RawTileInfo], channels: List[int]
    ) -> Dict[int, Any]:
        """Estimate a flat-field model per channel (both pipeline modes).

        Loads only the illumination-fused MIDDLE plane of each tile (one plane
        per tile per channel — bounded RAM, ~N_tiles × H × W), then fits BaSiC
        via the shared estimate_flat_fields. The returned models are consumed by
        _preprocess_single_tile, which applies flat-field per plane right after
        illumination fusion (native resolution, early in the chain) — the single
        flat-field path for streaming AND in-memory. Returns {} (with a loud
        warning) if basicpy is unavailable.
        """
        from .flat_field import estimate_flat_fields, is_available

        if not is_available():
            self.logger.warning(
                "Flat-field correction requested but basicpy is unavailable "
                "(direct or isolated env) — output will NOT be flat-fielded. "
                "Use 'Setup Preprocessing…' in the dialog to install."
            )
            return {}

        ch_data: Dict[int, List[Tuple[Any, RawTileInfo]]] = {ch: [] for ch in channels}
        for tile in tiles:
            for ch in channels:
                illum = tile.raw_files.get(ch, {})
                if not illum:
                    continue
                planes: Dict[int, np.ndarray] = {}
                for side, path in illum.items():
                    vol = load_tile_volume(
                        path, tile.n_planes, tile.frame_width, tile.frame_height
                    )
                    mid = vol.shape[0] // 2
                    # np.array (copy), not np.asarray (view): a view keeps the
                    # whole tile `vol` alive. For single-side tiles `fused` stays
                    # this array, so a view would pin every tile's full native
                    # volume until the final np.stack (OOM on large single-side
                    # TIFF mosaics — defeats the "one plane per tile" bound).
                    planes[side] = np.array(vol[mid])[None]  # (1, H, W) copy
                if len(planes) > 1:
                    fused = fuse_illumination_sides(
                        planes, method=self.config.illumination_fusion
                    )
                else:
                    fused = list(planes.values())[0]
                ch_data[ch].append((fused, tile))

        return estimate_flat_fields(
            ch_data, progress_fn=lambda m: self._progress_fn(45, m)
        )

    def _apply_flatfield_volume(self, volume: np.ndarray, model: Any) -> np.ndarray:
        """Apply a flat-field model to a native-resolution volume, one plane at
        a time (bounded to one plane of float32). Bit-for-bit the same math as
        flat_field.apply_flat_field."""
        ff = model.flatfield.astype(np.float32)
        ff = np.where(ff > 0.001, ff, 1.0)
        dark = model.darkfield.astype(np.float32)
        out = np.empty(volume.shape, dtype=np.uint16)
        for z in range(volume.shape[0]):
            plane = (volume[z].astype(np.float32) - dark) / ff
            out[z] = np.clip(plane, 0, 65535).astype(np.uint16)
        return out

    def _preprocess_single_tile(
        self, tile: RawTileInfo, ch_id: int, illum_side: Optional[int] = None
    ) -> np.ndarray:
        """Load and preprocess a single tile for one channel.

        Applies the full preprocessing chain: illumination fusion,
        depth attenuation, destripe, deconvolution, downsample, camera flip.
        Returns a uint16 numpy array ready for fusion.

        ``illum_side`` selects a single light path instead of fusing the two
        (used by the ``split_illumination`` diagnostic). When set and present on
        the tile, only that side is loaded and illumination fusion is skipped; if
        the requested side is absent the tile's available side(s) are used.
        """
        illum_files = tile.raw_files.get(ch_id, {})
        if not illum_files:
            raise ValueError(f"No raw files for channel {ch_id} in {tile.folder}")
        if illum_side is not None and illum_side in illum_files:
            illum_files = {illum_side: illum_files[illum_side]}

        # Destripe runs PER ILLUMINATION SIDE, BEFORE fusion (below): stripe
        # artifacts originate independently in each light-sheet path, so fusing
        # first and destriping the combined tile mixes two different stripe
        # patterns. (The "fast" variant is the exception — it destripes the
        # downsampled, already-fused tile as a speed/quality trade-off; see
        # below.) Single-side acquisitions (e.g. ASLM) have one entry, so this is
        # just "destripe the tile".
        _destripe_per_side = self.config.destripe and not self.config.destripe_fast

        illum_volumes = {}
        for illum_side, raw_path in illum_files.items():
            # Dispatch on file type (.raw vs .tif/.tiff/.btf). For raw we pass
            # the per-tile frame (camera AOI) dims — omitting them falls back to
            # the 2048×2048 module default, which reinterprets a cropped
            # 1024×1024 acquisition with the wrong stride (scrambled data, wrong
            # plane count — the streaming-mode corruption that produced blank
            # Imaris output). TIFF is self-describing, so the dims are ignored.
            vol = load_tile_volume(
                raw_path, tile.n_planes, tile.frame_width, tile.frame_height
            )
            if _destripe_per_side:
                vol = destripe_volume(
                    vol,
                    max_workers=self._destripe_worker_budget(),
                    direction=self._resolve_destripe_direction(vol),
                    params=self.config.destripe_params,
                    # Honoured only if `vol` is writable. It usually is NOT —
                    # .raw tiles are read-only memmaps — in which case the
                    # output array is the only tile-sized RAM allocation
                    # anyway, not a duplicate. Kept for the TIFF path, whose
                    # loader returns a real in-memory array.
                    in_place=True,
                    cancel_fn=self._cancelled_fn,
                )
            illum_volumes[illum_side] = vol

        if len(illum_volumes) > 1:
            self.logger.info(
                f"    Ch{ch_id}: fusing {len(illum_volumes)} illumination "
                f"sides ({self.config.illumination_fusion})"
            )
            volume = fuse_illumination_sides(
                illum_volumes, method=self.config.illumination_fusion
            )
        else:
            volume = np.asarray(list(illum_volumes.values())[0])

        # Flat-field correction (streaming path). In-memory mode applies its own
        # flat-field step and leaves _flatfield_models empty, so this is a no-op
        # there (no double application). Applied here — right after illumination
        # fusion, at native resolution, per plane — so the streaming path no
        # longer silently skips a requested flat-field correction.
        _ff_model = self._flatfield_models.get(ch_id)
        if _ff_model is not None:
            volume = self._apply_flatfield_volume(volume, _ff_model)

        if self.config.depth_attenuation:
            from .depth_attenuation import correct_depth_attenuation

            z_step = self.config.z_step_um
            if z_step is None:
                z_step = tile.z_step_mm * 1000.0 if tile.z_step_mm else 10.0
            volume = correct_depth_attenuation(
                volume, mu=self.config.depth_attenuation_mu, z_step_um=z_step
            )

        # (Non-fast destripe already ran per illumination side, before fusion.)

        _deconv_fast = bool(getattr(self.config, "deconvolution_fast", False))
        if self.config.deconvolution_enabled and not _deconv_fast:
            volume = self._deconvolve_tile(volume, tile)

        if self.config.downsample_xy > 1 or self.config.downsample_z > 1:
            volume = downsample_volume(
                volume, self.config.downsample_xy, self.config.downsample_z
            )

        if self.config.destripe and self.config.destripe_fast:
            volume = destripe_volume(
                volume,
                max_workers=self._destripe_worker_budget(),
                direction=self._resolve_destripe_direction(volume),
                params=self.config.destripe_params,
                # `volume` is this method's own downsampled array — writable,
                # so this genuinely avoids an allocation.
                in_place=True,
                cancel_fn=self._cancelled_fn,
            )

        # "Fast" deconvolution runs AFTER downsample (like destripe_fast): far
        # cheaper (fewer voxels + smaller PSF) but lower quality, since the PSF
        # blur is estimated/removed at reduced resolution. Off by default.
        if self.config.deconvolution_enabled and _deconv_fast:
            volume = self._deconvolve_tile(volume, tile)

        # Per-tile camera→stage orientation. Reorient each tile's pixels so its
        # content axes align with the stage axes BEFORE placement — otherwise
        # adjacent tiles don't connect (rotating the finished mosaic can't fix
        # that). Empty tile_orientation preserves the legacy X-flip behaviour.
        #
        # These are stride tricks (transpose = view, [::-1] = reversed view), NOT
        # contiguous copies. _materialize_tiles_to_disk writes into a memmap via
        # `mm[:] = vol`, which handles non-contiguous sources without a
        # full-volume scratch buffer (an np.ascontiguousarray() here cost ~5.7 GB
        # extra per tile and OOM'd multi-channel runs). The per-channel
        # reference_shape is derived from a preprocessed probe tile, so a
        # transpose that swaps the frame dims stays consistent automatically.
        from flamingo_stitcher.orientation import MosaicOrientation

        ori_name = self.config.tile_orientation or (
            "flip_h" if self.config.camera_x_inverted else "identity"
        )
        volume = MosaicOrientation.from_name(ori_name).apply_volume_xy(volume)

        return volume

    def _output_channel_units(
        self, tiles: List[RawTileInfo], process_channels: List[int]
    ) -> List[Tuple[Any, int, Optional[int]]]:
        """Expand each real channel into the output channels to fuse+write.

        Returns ``[(label, real_ch, side), ...]``. Normally one unit per real
        channel with ``label == real_ch`` (int) and ``side=None`` (illumination
        fused as configured) — byte-for-byte the historical behaviour. When
        ``split_illumination`` is on, a channel with >1 illumination side is
        expanded into one unit per side (``side`` set, so no fusion) with a
        ``"{ch}_I{side}"`` string label; single-side channels are left as-is.
        Order follows ``process_channels`` so channel slots stay deterministic.
        """
        split = bool(getattr(self.config, "split_illumination", False))
        units: List[Tuple[Any, int, Optional[int]]] = []
        for ch in process_channels:
            sides = sorted(
                {s for t in tiles for s in (t.raw_files.get(ch, {}) or {}).keys()}
            )
            if split and len(sides) > 1:
                for s in sides:
                    units.append((f"{ch}_I{s}", ch, s))
            else:
                units.append((ch, ch, None))
        return units

    def _materialize_tiles_to_disk(
        self,
        tiles: List[RawTileInfo],
        ch_id: int,
        reference_shape: tuple,
        tmp_dir: Path,
        illum_side: Optional[int] = None,
    ) -> List[Tuple[Any, RawTileInfo]]:
        """Preprocess each tile exactly once and spill to a memmap file.

        Returns [(dask_array, tile_info), ...] where each dask_array is
        backed by a flat on-disk memmap. Fusion chunk-reads become cheap
        file reads — no re-running of the preprocess chain per chunk.

        ``reference_shape`` is measured from a probe of ``tiles[0]`` and fixes
        the FRAME size only. **Tiles may legitimately differ in depth**: an
        acquisition with per-tile Z ranges (what Collect Tiles produces, so the
        scope only images the Z span where the sample actually is) gives every
        tile its own plane count. Fusion already handles that — each tile is
        placed at its own ``z_min_mm`` and multiview-stitcher accepts views of
        differing shape — so depth is read per tile rather than assumed.

        Frame dims are a different matter: Y/X disagreeing across tiles means
        the AOI, downsample, or orientation changed mid-acquisition, which
        nothing downstream can reconcile. That still raises.

        Caller owns ``tmp_dir`` and must remove it once the returned
        dask arrays are no longer referenced.
        """
        import time as _time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tmp_dir.mkdir(parents=True, exist_ok=True)
        frame_shape = tuple(reference_shape[1:])
        frame_bytes = int(np.prod(frame_shape) * 2) if frame_shape else 0
        # Depth varies per tile, so the probe's own size is only a reference.
        # Scale each tile's native plane count by the probe's native→output
        # ratio (downsample, ISO resolution, whatever preprocessing applied)
        # rather than re-deriving the factor from config.
        probe_planes = int(tiles[0].n_planes) if tiles else 0
        z_ratio = (
            (int(reference_shape[0]) / probe_planes) if probe_planes > 0 else 1.0
        )

        def _out_planes(tile: RawTileInfo) -> int:
            return max(1, int(round(int(tile.n_planes) * z_ratio)))

        # Filter to tiles that actually contain this channel, preserving
        # their original indices so memmap filenames stay stable across
        # channels (useful when debugging).
        active = [(i, t) for i, t in enumerate(tiles) if ch_id in t.raw_files]

        # Size the worker pool on the NATIVE per-tile peak — preprocessing runs
        # at native resolution before downsample (see _preprocess_peak_bytes).
        # With per-tile Z ranges the DEEPEST tile sets the peak, so size against
        # that rather than whichever tile happens to come first: on this
        # acquisition tile 0 was 1287 planes and others 1502+, and sizing on
        # tile 0 would hand every worker a budget the real tiles overrun.
        _geo_tile = active[0][1] if active else (tiles[0] if tiles else None)
        if _geo_tile is not None:
            plane_vox = int(_geo_tile.frame_width) * int(_geo_tile.frame_height)
            deepest = max(int(t.n_planes) for _, t in active) if active else 0
            native_vox = max(deepest, int(_geo_tile.n_planes)) * plane_vox
        else:
            native_vox = int(np.prod(reference_shape))
            plane_vox = (
                int(np.prod(reference_shape[1:])) if len(reference_shape) > 1 else 0
            )
        per_worker_bytes = _preprocess_peak_bytes(
            self.config, native_vox, plane_vox=plane_vox
        )
        n_workers = self._pick_preprocess_workers(per_worker_bytes, len(active))
        # Record concurrency so nested per-plane destripe pools inside
        # _preprocess_single_tile can size against it (avoid oversubscription).
        self._active_preprocess_workers = n_workers
        spill_bytes = sum(frame_bytes * _out_planes(t) for _, t in active)
        self.logger.info(
            f"  Materializing {len(active)} tiles for channel {ch_id} "
            f"→ {tmp_dir} "
            f"({spill_bytes / (1024**3):.1f} GB temp on disk, "
            f"{n_workers} worker{'s' if n_workers != 1 else ''})"
        )

        def process_one(i: int, tile: RawTileInfo) -> Tuple[Any, RawTileInfo]:
            # Preprocess (holds ~1 tile in RAM) → write memmap (stream) →
            # return a lazy dask wrapper backed by the file. All per-tile
            # preprocessing (flat-field/illum/depth atten/destripe/deconv/
            # downsample/X-flip) happens on this thread; numpy releases
            # the GIL for the heavy ops so multiple workers overlap cleanly.
            vol = self._preprocess_single_tile(tile, ch_id, illum_side=illum_side)
            if tuple(vol.shape[1:]) != frame_shape:
                raise RuntimeError(
                    f"Tile {i} frame is {tuple(vol.shape[1:])} but tile 0 is "
                    f"{frame_shape}. Every tile must share one frame size — a "
                    f"mid-acquisition AOI/binning change cannot be stitched. "
                    f"(Depth may vary per tile; frame size may not.)"
                )
            # Depth is whatever this tile actually has — see the docstring.
            tile_shape = tuple(vol.shape)
            mm_path = tmp_dir / f"tile_{i:04d}.dat"
            mm = np.memmap(mm_path, dtype=np.uint16, mode="w+", shape=tile_shape)
            mm[:] = vol
            mm.flush()
            del mm, vol
            # _dask_array_from_memmap builds the task graph by hand so
            # dask.from_array's unconditional x.copy() cannot materialize
            # the file back into RAM (see commit 88f88c2).
            lazy = _dask_array_from_memmap(
                mm_path, tile_shape, np.uint16, _DASK_PROCESSING_CHUNKS
            )
            return lazy, tile

        result_by_idx: Dict[int, Tuple[Any, RawTileInfo]] = {}
        completed = 0
        t_started = _time.time()

        if n_workers == 1:
            # Serial path — identical to the old loop so single-worker
            # runs behave exactly as they did before.
            for i, tile in active:
                if self._cancelled_fn():
                    self.logger.info("Pipeline cancelled by user")
                    break
                t0 = _time.time()
                self._progress_fn(
                    50,
                    f"Preprocessing tile {completed + 1}/{len(active)} (ch {ch_id})",
                )
                self.logger.info(
                    f"    Tile {completed + 1}/{len(active)}: {tile.folder.name} "
                    f"(X={tile.x_mm:.2f} Y={tile.y_mm:.2f})"
                )
                result_by_idx[i] = process_one(i, tile)
                completed += 1
                self.logger.info(
                    f"      (tile {completed}/{len(active)} took "
                    f"{_time.time() - t0:.1f}s)"
                )
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = {
                    ex.submit(process_one, i, tile): (i, tile) for i, tile in active
                }
                try:
                    for fut in as_completed(futures):
                        if self._cancelled_fn():
                            for f in futures:
                                f.cancel()
                            self.logger.info("Pipeline cancelled by user")
                            break
                        i, tile = futures[fut]
                        result_by_idx[i] = fut.result()
                        completed += 1
                        self._progress_fn(
                            50,
                            f"Preprocessed {completed}/{len(active)} (ch {ch_id})",
                        )
                        self.logger.info(
                            f"    Tile {completed}/{len(active)}: {tile.folder.name} "
                            f"(X={tile.x_mm:.2f} Y={tile.y_mm:.2f})"
                        )
                except Exception:
                    for f in futures:
                        f.cancel()
                    raise

        total = _time.time() - t_started
        if completed:
            self.logger.info(
                f"  Preprocessed {completed} tiles in {total:.1f}s "
                f"({total / completed:.1f}s/tile avg, "
                f"{spill_bytes / (1024**3) / max(total, 0.001):.1f} GB/s)"
            )

        # Return in original tile order so fusion sees tiles in the same
        # spatial sequence regardless of completion order from the pool.
        return [result_by_idx[i] for i in sorted(result_by_idx)]

    # How many tiles vote before the stripe axis is locked for the whole run.
    _DESTRIPE_AXIS_VOTES = 8

    def _effective_tile_orientation_name(self) -> str:
        """The orientation actually applied to tiles (mirrors _preprocess)."""
        return self.config.tile_orientation or (
            "flip_h" if self.config.camera_x_inverted else "identity"
        )

    def _resolve_destripe_direction(self, volume: np.ndarray) -> str:
        """Decide the stripe axis ONCE per acquisition, by DERIVING it.

        Stripe direction is set by the light-sheet propagation direction, and
        the stitched output is stage-aligned, so the axis in the OUTPUT frame is
        a known constant per microscope (``config.destripe_output_axis``,
        horizontal for the Flamingo). Destriping runs in the raw camera frame,
        before the per-tile rotation/flip, so the answer is just that constant
        mapped back through the tile orientation — no image content involved.

        This replaced a per-tile detector. Detecting was wrong twice over: the
        axis cannot vary per tile (it is fixed by the optics), and a detector
        comparing mean-profile spectra has nothing to work with on a
        background-dominated tile — or on the outright blank tiles a blind
        acquisition routinely produces, where the "measurement" is pure noise.

        Order of authority: an explicit ``destripe_direction`` wins; otherwise
        derive from the orientation; only if the orientation is genuinely
        unknown do we fall back to content detection, and say so loudly.
        """
        configured = (self.config.destripe_direction or "auto").lower()
        if configured != "auto":
            return configured

        with self._destripe_axis_lock:
            if self._destripe_axis_locked is not None:
                return self._destripe_axis_locked

            ori_name = self._effective_tile_orientation_name()
            try:
                from flamingo_stitcher.orientation import stripe_axis_in_camera_frame

                derived = stripe_axis_in_camera_frame(
                    ori_name, self.config.destripe_output_axis
                )
            except Exception as exc:  # noqa: BLE001 - fall through to detection
                self.logger.debug(f"Could not derive stripe axis: {exc!r}")
                derived = None

            if derived is not None:
                self._destripe_axis_locked = derived
                self.logger.info(
                    f"  Destripe direction: {derived} stripes in the camera frame "
                    f"— derived from tile orientation '{ori_name}' and a "
                    f"{self.config.destripe_output_axis} stripe axis in the "
                    "stitched output. Not detected from image content."
                )
                return derived

        self.logger.warning(
            "  Destripe direction could not be derived from the tile "
            "orientation; falling back to detecting it from image content, "
            "which is unreliable on dim or blank tiles. Set Dir: explicitly."
        )
        return self._detect_destripe_direction_from_content(volume)

    def _detect_destripe_direction_from_content(self, volume: np.ndarray) -> str:
        """Last-resort content-based axis guess, when derivation is impossible.

        Confidence-weighted votes across the first tiles, so a near-featureless
        tile barely counts. Still a guess — prefer the derivation above.
        """
        with self._destripe_axis_lock:
            if self._destripe_axis_locked is not None:
                return self._destripe_axis_locked

            axis, confidence = _stripe_axis_vote(volume)
            self._destripe_axis_votes[axis] += confidence
            self._destripe_axis_n += 1

            horiz = self._destripe_axis_votes["horizontal"]
            vert = self._destripe_axis_votes["vertical"]
            winner = "vertical" if vert > horiz else "horizontal"

            if self._destripe_axis_n >= self._DESTRIPE_AXIS_VOTES:
                self._destripe_axis_locked = winner
                self.logger.info(
                    f"  Destripe direction LOCKED to {winner} stripes for the "
                    f"rest of this run (votes from {self._destripe_axis_n} "
                    f"tiles: horizontal={horiz:.2f}, vertical={vert:.2f}). "
                    "Set Dir: explicitly in Destripe settings to override."
                )
            else:
                self.logger.info(
                    f"  Destripe direction vote {self._destripe_axis_n}/"
                    f"{self._DESTRIPE_AXIS_VOTES}: this tile says {axis} "
                    f"(confidence {confidence:.2f}); using {winner} so far"
                )
            return winner

    def _reset_destripe_axis(self) -> None:
        """Clear the locked stripe axis so a new run re-resolves it."""
        global _destripe_meter
        _destripe_meter = _DestripeMeter()
        self._destripe_axis_lock = threading.Lock()
        self._destripe_axis_votes = {"horizontal": 0.0, "vertical": 0.0}
        self._destripe_axis_n = 0
        self._destripe_axis_locked: Optional[str] = None

    def _destripe_worker_budget(self) -> Optional[int]:
        """Per-tile destripe thread cap that accounts for concurrent preprocess
        workers, so nested thread pools don't oversubscribe CPU/RAM.

        ``destripe_volume`` spawns its own per-plane ``ThreadPoolExecutor``. When
        it runs INSIDE the per-tile preprocess pool (streaming mode), the naive
        product ``preprocess_workers × destripe_workers`` oversubscribes cores and
        multiplies peak RAM (a documented OOM cause). Divide the machine's thread
        budget by the number of tiles processed concurrently. An explicit
        ``config.destripe_workers`` still caps the result; ``None`` = let
        ``destripe_volume`` auto-size (memory + cores) when there is no nesting.
        """
        explicit = self.config.destripe_workers  # None → auto
        pp = int(getattr(self, "_active_preprocess_workers", 1) or 1)
        if pp <= 1:
            return explicit  # sequential preprocess → no nesting to guard against
        import os

        budget = max(1, (os.cpu_count() or 1) // pp)
        if explicit:
            return max(1, min(int(explicit), budget))
        return budget

    def _pick_preprocess_workers(self, per_worker_bytes: int, n_tiles: int) -> int:
        """Choose a safe ThreadPool size for per-tile preprocessing.

        ``per_worker_bytes`` is the NATIVE-resolution peak one worker holds for a
        tile (from :func:`_preprocess_peak_bytes`, accounting for the enabled
        preprocessing steps). Preprocessing runs BEFORE downsample, so sizing on
        the downsampled tile — as this used to — picked far too many workers and
        could OOM. We keep total preprocessing RAM under ~50% of available.

        Config override:
          * ``preprocess_workers = 0``: auto, clamped to [1, 4].
          * ``preprocess_workers > 0``: honor the user's request,
            clamped to [1, 8] — the disk saturates well before 8 so
            larger values do nothing useful.
        """
        requested = int(getattr(self.config, "preprocess_workers", 0) or 0)

        try:
            import psutil

            avail_bytes = psutil.virtual_memory().available
        except Exception:
            avail_bytes = 8 * 1024**3  # conservative 8 GB fallback

        per_worker = max(1, int(per_worker_bytes))
        ram_cap = max(1, avail_bytes // (per_worker * 2))

        if requested > 0:
            return max(1, min(requested, 8, n_tiles or 1))

        # Destriping saturates the machine on its own and does NOT benefit from
        # being run on several tiles at once — it benefits from NOT being. On a
        # 98-tile 24-core run, one tile at 24 threads did 25.6 planes/s, while
        # four tiles at 6 threads each did ~2.1 planes/s apiece — 8.4 in total,
        # a 3x LOSS. Each concurrent tile also holds a full ~3.35 GB volume, so
        # four in flight is ~27 GB of buffers and the whole chain starts
        # thrashing: destripe alone slowed 11.6x, which no thread arithmetic
        # explains. Per-plane parallelism inside destripe_volume tops out at
        # ~1.5x anyway (measured, threads and processes alike), so splitting
        # the machine across tiles buys nothing and costs a great deal.
        if getattr(self.config, "destripe", False):
            self.logger.info(
                "Preprocess workers: 1 (destriping is on — it already uses "
                "every core and running tiles concurrently measured 3x slower "
                "and ~4x the peak RAM). Override with preprocess_workers > 0."
            )
            return 1

        return max(1, min(4, int(ram_cap), n_tiles or 1))

    def _pick_fuse_workers(self, darr) -> int:
        """Choose a safe dask thread-pool size for the fused-memmap store.

        Each worker's peak per-chunk footprint is dominated by
        content-based Gaussian weighting — float32 upcast of the output
        chunk plus overlapping tile chunks plus gaussian filter scratch.
        Empirically ~6× the uint16 chunk size per worker, so a default
        16 MB chunk (64×512×512) needs ~100 MB per worker with cosine
        blending, ~400 MB with content-based on top. We budget 1 GB per
        worker as a safe overestimate.

        On a 64 GB RAM box with ~40 GB free this picks 4 workers
        (worst case ~4 GB in-flight), well under the tile-memmap page
        cache that lives alongside it. On an 8 GB low-end box it drops
        to 1 (synchronous) automatically.

        Config override:
          * ``fuse_workers = 0``: auto, clamped to [1, 4].
          * ``fuse_workers > 0``: honor the request, clamped to [1, 8].
        """
        requested = int(getattr(self.config, "fuse_workers", 0) or 0)

        try:
            import psutil

            avail_bytes = psutil.virtual_memory().available
        except Exception:
            avail_bytes = 8 * 1024**3  # conservative 8 GB fallback

        # Reserve 25% for fused-memmap page cache + tile memmap pages + OS.
        # Budget per worker = the actual per-block float64 fusion working set,
        # derived from the real output chunk size and fusion mode (matches
        # estimate_memory_usage). The old flat 1 GB over-counted the default
        # `max` path on small chunks — collapsing to a single core on tight-RAM
        # boxes for no reason — and under-counted large chunks.
        budget_bytes = max(0, int(avail_bytes * 0.75))
        try:
            chunk_vox = 1
            for c in darr.chunksize:
                chunk_vox *= int(c)
        except Exception:
            chunk_vox = 64 * 256 * 256
        content = bool(
            getattr(self.config, "content_based_fusion", False)
            and getattr(self.config, "tile_overlap_fusion", "max")
            not in ("max", "brightest")
        )
        # ~9-tile local overlap; content-based adds a halo + extra weight buffers.
        coexist = 3.0 if content else 2.0
        halo_factor = 1.8 if content else 1.0
        per_worker_bytes = int(coexist * 9 * chunk_vox * halo_factor * 8)
        per_worker_bytes = max(per_worker_bytes, 256 * 1024**2)  # 256 MB floor
        ram_cap = max(1, budget_bytes // per_worker_bytes)

        # Cap by the number of output chunks so the pool never outsizes
        # the work (tiny runs / huge chunks).
        try:
            n_chunks = max(1, int(np.prod([len(c) for c in darr.chunks])))
        except Exception:
            n_chunks = 64

        if requested > 0:
            chosen = min(requested, 8, n_chunks)
        else:
            chosen = min(4, int(ram_cap), n_chunks)
        return max(1, chosen)

    def _resolve_rotation_center_um(
        self, tiles: List[RawTileInfo]
    ) -> Tuple[float, float]:
        """Rotation-axis (x, z) in world µm. Config value wins; else tile centroid."""
        if self.config.rotation_center_um is not None:
            cx, cz = self.config.rotation_center_um
            return float(cx), float(cz)
        if not tiles:
            return 0.0, 0.0
        xs = [t.x_mm * 1000.0 for t in tiles]
        zs = [(t.z_min_mm + t.z_max_mm) * 0.5 * 1000.0 for t in tiles]
        return sum(xs) / len(xs), sum(zs) / len(zs)

    def _tile_metadata_affine(
        self, tile_info: RawTileInfo, rot_center_um: Tuple[float, float]
    ) -> Optional[np.ndarray]:
        """Rotation affine for a tile, or None (identity) for single-angle tiles.

        Baked into the metadata transform at sim-construction so registration
        refines on top of the already-rotated placement. Returns None (so no
        affine is applied) whenever multi-view fusion is off or the tile sits at
        angle 0 — keeping single-angle runs byte-identical.
        """
        if not self.config.multiview_fusion or abs(tile_info.angle_deg) < 1e-9:
            return None
        return _rotation_affine_zyx(
            tile_info.angle_deg,
            rot_center_um[0],
            rot_center_um[1],
            self.config.rotation_sign,
        )

    def _register_tiles(
        self,
        tile_data: List[Tuple[Any, RawTileInfo]],
        voxel_size_um: Dict[str, float],
    ) -> Tuple[list, str]:
        """Register tiles using the reference channel's data.

        Builds multiscale spatial images, runs phase-correlation registration,
        and returns the affine parameters + transform key.

        Args:
            tile_data: [(volume, tile_info), ...] for the reference channel
            voxel_size_um: Voxel sizes dict

        Returns:
            (reg_params, transform_key) — reg_params is a list of affine params
            (one per tile), transform_key is the key to use for fusion.
        """
        try:
            from multiview_stitcher import io as mvs_io
            from multiview_stitcher import (
                msi_utils,
                registration,
            )
            from multiview_stitcher import spatial_image_utils as si_utils
        except ImportError as exc:
            # Surface the REAL missing module (e.g. a dependency stripped from a
            # frozen build) instead of hiding it behind a generic message — the
            # underlying ImportError names exactly what's absent.
            raise ImportError(
                "multiview-stitcher (or one of its dependencies) failed to import. "
                f"Underlying error: {exc}. "
                "If running from source, install with: pip install multiview-stitcher"
            ) from exc

        import dask.array as da

        # Build SpatialImages with stage positions
        self.logger.info("  Building tile spatial images for registration...")
        # Multi-view: rotation axis for placing angled views (None-op when off).
        rot_center_um = self._resolve_rotation_center_um(
            [ti for _, ti in tile_data]
        )
        msims = []
        for volume, tile_info in tile_data:
            translation_um = {
                "z": tile_info.z_min_mm * 1000.0,
                "y": (
                    -tile_info.y_mm if self.config.reverse_y_tiles else tile_info.y_mm
                )
                * 1000.0,
                "x": (
                    -tile_info.x_mm if self.config.reverse_x_tiles else tile_info.x_mm
                )
                * 1000.0,
            }
            if not isinstance(volume, da.Array):
                volume = da.from_array(volume, chunks=_DASK_PROCESSING_CHUNKS)

            sim = si_utils.get_sim_from_array(
                volume,
                dims=["z", "y", "x"],
                scale=voxel_size_um,
                translation=translation_um,
                affine=self._tile_metadata_affine(tile_info, rot_center_um),
                transform_key=mvs_io.METADATA_TRANSFORM_KEY,
            )
            msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
            msims.append(msim)

        self.logger.info(f"  Built {len(msims)} multiscale spatial images")

        tiles = [ti for _v, ti in tile_data]
        extent_um = self._frame_extent_um(tile_data, voxel_size_um)

        if len(msims) <= 1:
            self._registration_report = registration_report.skipped_report(
                "single tile — nothing to register against",
                tiles=tiles,
                voxel_size_um=voxel_size_um,
                frame_extent_um=extent_um,
            )
            self.logger.info("  Single tile — skipping registration")
            return [], mvs_io.METADATA_TRANSFORM_KEY

        gate_reason = self._registration_overlap_gate(tiles, extent_um)
        if gate_reason:
            self.logger.warning(f"  Registration SKIPPED: {gate_reason}")
            self._registration_report = registration_report.skipped_report(
                gate_reason,
                tiles=tiles,
                voxel_size_um=voxel_size_um,
                frame_extent_um=extent_um,
            )
            return [], mvs_io.METADATA_TRANSFORM_KEY

        # Run registration
        self.logger.info(
            f"  Running phase correlation registration "
            f"(quality threshold={self.config.quality_threshold})..."
        )
        started = time.time()
        # Assigned before the try so the finally block cannot raise NameError
        # over the top of the real failure if an import inside it blows up.
        reg_logger = logging.getLogger("multiview_stitcher.registration")
        saved_level = reg_logger.level
        try:
            import dask.diagnostics

            # Suppress per-tile-pair registration spam from multiview_stitcher
            reg_logger.setLevel(logging.WARNING)

            sink: Dict[str, Any] = {}
            reject = self._pairwise_shift_reject(tile_data, voxel_size_um)
            with dask.diagnostics.ProgressBar():
                with registration_report.capture_prefilter_graph(sink, reject):
                    result = registration.register(
                        msims,
                        reg_channel_index=0,
                        transform_key=mvs_io.METADATA_TRANSFORM_KEY,
                        new_transform_key="registered",
                        registration_binning=self.config.registration_binning,
                        post_registration_do_quality_filter=True,
                        post_registration_quality_threshold=(
                            self.config.quality_threshold
                        ),
                        pairwise_reg_func_kwargs=self._pairwise_reg_kwargs(
                            self.config.registration_upsample_factor
                        ),
                        groupwise_resolution_kwargs={
                            "abs_tol": self.config.global_opt_abs_tol,
                            "rel_tol": self.config.global_opt_rel_tol,
                        },
                        return_dict=True,
                    )
            params = list(result["params"])

            # Extract the seams once, here: the guard below and the report have
            # to be reading the same rows or the report cannot explain the
            # decision.
            seams = registration_report.extract_seams(
                tiles=tiles,
                voxel_size_um=voxel_size_um,
                reg_dict=result,
                prefilter_graph=sink.get("prefilter"),
                rejected_edges=sink.get("rejected"),
                quality_threshold=self.config.quality_threshold,
                frame_extent_um=extent_um,
            )
            n_rejected = len(sink.get("rejected") or {})
            if n_rejected:
                self.logger.warning(
                    f"  {n_rejected} seams passed the quality threshold but "
                    f"proposed a shift the geometry forbids, and were dropped "
                    f"before the global solve. A confident wrong correlation "
                    f"peak scores WELL, so lowering the quality threshold makes "
                    f"this worse, not better — see registration_seams.csv."
                )
            coverage = registration_report.mosaic_coverage(len(tiles), seams)
            self.logger.info(f"  Registration coverage: {coverage.describe()}")

            # Two independent questions, in order. First: is there enough
            # agreement here to believe any of it? Second: does what we believe
            # place every adjacent pair relative to its neighbour?
            reason = self._untrustworthy_registration_reason(coverage)
            if reason is None and not coverage.is_safe_to_apply:
                # Geometry, not trust: bind the loose tiles to the mosaic so no
                # seam tears, and carry on with the measurements that worked.
                params, bound = self._bind_unconstrained_tiles(params, coverage)
                if bound:
                    coverage.bound_tiles = list(bound)
                else:
                    reason = self._unconstrained_coverage_reason(coverage, params)

            if reason is not None:
                self.logger.warning(f"  Registration NOT APPLIED: {reason}")
                self._registration_report = registration_report.build_report(
                    tiles=tiles,
                    params=params,
                    voxel_size_um=voxel_size_um,
                    transform_key=mvs_io.METADATA_TRANSFORM_KEY,
                    seams=seams,
                    quality_threshold=self.config.quality_threshold,
                    frame_extent_um=extent_um,
                    settings=self._registration_settings(
                        None, voxel_size_um, coverage
                    ),
                    applied=False,
                    reason=reason,
                    elapsed_s=time.time() - started,
                )
                return [], mvs_io.METADATA_TRANSFORM_KEY

            clamp = self._clamp_registration_shifts(params, tile_data, voxel_size_um)
            params = clamp.params

            # multiview-stitcher wrote the UNCLAMPED params into the msims under
            # "registered" before returning, so a second pass starting from that
            # key would build on exactly the shifts the clamp just rejected.
            # Overwrite with the clamped ones. set_affine_transform re-derives
            # from the untouched metadata base, so this is idempotent.
            for msim, param in zip(msims, params):
                msi_utils.set_affine_transform(
                    msim,
                    param,
                    transform_key="registered",
                    base_transform_key=mvs_io.METADATA_TRANSFORM_KEY,
                )

            params, z_summary = self._refine_z_shifts(
                msims, params, tile_data, voxel_size_um, registration, clamp
            )

            self._registration_report = registration_report.build_report(
                tiles=tiles,
                params=params,
                voxel_size_um=voxel_size_um,
                transform_key="registered",
                clamp_records=clamp.records,
                seams=seams,
                quality_threshold=self.config.quality_threshold,
                frame_extent_um=extent_um,
                settings=self._registration_settings(
                    clamp, voxel_size_um, coverage
                ),
                z_refine=z_summary,
                elapsed_s=time.time() - started,
            )
            self.logger.info("  Registration complete")
            return params, "registered"

        except Exception as e:
            self.logger.error(f"  Registration failed: {e}")
            self.logger.info("  Falling back to metadata positions only")
            self._registration_report = registration_report.skipped_report(
                f"registration raised and the run fell back to stage positions: {e}",
                tiles=tiles,
                voxel_size_um=voxel_size_um,
                frame_extent_um=extent_um,
            )
            return [], mvs_io.METADATA_TRANSFORM_KEY

        finally:
            reg_logger.setLevel(saved_level)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _apply_scope_profile(self, acquisition_dir) -> None:
        """Overlay this microscope + objective's saved options onto the config.

        Skipped when `scope_profile_source` is already set: the GUI resolves the
        profile per queue item so the user can see and override the values
        before starting, and re-applying here would undo that override.

        This is the layer that makes the CLI honour a rig's tuning at all —
        without it a headless run gets the dataclass defaults regardless of
        which instrument produced the data.
        """
        if getattr(self.config, "scope_profile_source", ""):
            return
        try:
            from flamingo_stitcher import scope_profiles

            scope, objective, values, source = (
                scope_profiles.resolve_for_acquisition(acquisition_dir)
            )
            applied, message = scope_profiles.describe_resolution(
                scope, objective, values, source
            )
            self.logger.info(message)
            if not applied:
                return
            self.config.scope_profile_source = source
            for line in scope_profiles.apply_profile(self.config, values):
                self.logger.info(f"  {line}")
        except Exception as e:  # noqa: BLE001 - a preferences file must never
            # stop a stitch; the defaults are always a working configuration.
            self.logger.warning(f"Could not apply per-microscope options: {e}")

    def _frame_extent_um(self, tile_data, voxel_size_um) -> Dict[str, float]:
        """What one tile covers along each axis, in processing-frame µm."""
        try:
            shp = tile_data[0][0].shape
            return {
                "z": int(shp[-3]) * float(voxel_size_um.get("z", 1.0)),
                "y": int(shp[-2]) * float(voxel_size_um.get("y", 1.0)),
                "x": int(shp[-1]) * float(voxel_size_um.get("x", 1.0)),
            }
        except Exception:
            return {}

    def _registration_overlap_gate(self, tiles, extent_um) -> Optional[str]:
        """Reason to skip registration entirely, or None to proceed.

        Phase correlation needs shared content. Below a few percent overlap it
        does not return "no answer" — it returns a confident wrong one, and the
        clamp then reverts tile after tile while the log reads like a successful
        registration. Refusing up front is both cheaper and more honest.
        """
        if not extent_um:
            return None  # can't measure; let registration try rather than veto it
        # Same falsy-zero trap as min_registered_seam_frac: 0.0 means "do not
        # gate on overlap", and `or` would silently restore the default.
        configured = getattr(self.config, "min_registration_overlap_frac", None)
        threshold = float(
            MIN_REGISTRATION_OVERLAP if configured is None else configured
        )
        layout = tile_geometry.grid_overlap(
            tiles, extent_x_um=extent_um["x"], extent_y_um=extent_um["y"]
        )
        fractions = {
            axis: layout[axis].fraction
            for axis in ("x", "y")
            if layout[axis].fraction is not None
        }
        if not fractions:
            return None  # a single row or column: no pitch to judge, so proceed
        worst_axis = min(fractions, key=lambda a: fractions[a])
        worst = fractions[worst_axis]
        if worst >= threshold:
            return None
        return (
            f"measured tile overlap is only {worst * 100:.1f}% on {worst_axis.upper()} "
            f"(need >= {threshold * 100:.0f}%). Phase correlation on this little "
            f"shared content returns a confident wrong shift rather than failing, "
            f"so tiles are placed by stage position instead. Re-acquire with ~10% "
            f"tile overlap."
        )

    @staticmethod
    def _pairwise_reg_kwargs(upsample_factor) -> Optional[Dict[str, Any]]:
        """kwargs for the pairwise registration function, or None to leave MVS's.

        None rather than {} so an unconfigured run reaches multiview-stitcher
        exactly as it did before this option existed.
        """
        try:
            factor = int(upsample_factor or 0)
        except (TypeError, ValueError):
            return None
        return {"upsample_factor": factor} if factor > 0 else None

    def _refine_z_shifts(
        self, msims, params, tile_data, voxel_size_um, registration, clamp
    ):
        """Second registration pass whose only contribution is Z.

        The main pass registers at ``registration_binning`` (z=2 by default) and
        multiview-stitcher upsamples 3-D phase correlation by only 2, so Z comes
        out quantized to about one raw plane — on the axis with the coarsest
        voxel and, per the field reports, the largest error. This pass re-runs
        the same machinery at full Z resolution starting from the first pass's
        result, so it estimates the RESIDUAL, and contributes **only** its Z
        component: X/Y stay exactly as pass 1 left them, because two passes
        arguing about the same axis is how a mosaic oscillates.

        Returns ``(params, ZRefineSummary)``. Any failure here returns pass 1's
        params untouched — a refinement that cannot run must never cost the
        registration that already worked.
        """
        summary = registration_report.ZRefineSummary(
            binning=dict(self.config.registration_z_refine_binning),
            upsample_factor=int(self.config.registration_z_refine_upsample),
            search_range_um=float(self.config.registration_z_refine_range_um),
        )
        if not self.config.registration_z_refine:
            summary.reason = "disabled (registration_z_refine)"
            return params, summary

        # Keep "Registering" in the message: the phase attribution in the
        # estimator and in the memory probes keys off that substring, so the
        # second pass lands in the register phase rather than in Other/setup.
        self._progress_fn(48, "Registering tiles (Z refinement)...")
        self.logger.info(
            "  Z refinement pass: binning z="
            f"{summary.binning.get('z')}, upsample={summary.upsample_factor}, "
            f"search ±{summary.search_range_um:.0f} µm"
        )
        try:
            import dask.diagnostics

            with dask.diagnostics.ProgressBar():
                refined = registration.register(
                    msims,
                    reg_channel_index=0,
                    # Start from pass 1's (clamped) placement, so what comes
                    # back is the residual rather than a competing absolute.
                    transform_key="registered",
                    registration_binning=dict(
                        self.config.registration_z_refine_binning
                    ),
                    post_registration_do_quality_filter=True,
                    post_registration_quality_threshold=self.config.quality_threshold,
                    pairwise_reg_func_kwargs=self._pairwise_reg_kwargs(
                        self.config.registration_z_refine_upsample
                    ),
                    # No pruning, and it has to be none.
                    #
                    # Both of multiview-stitcher's grid-aware methods assume the
                    # views still sit on a clean grid. "keep_axis_aligned" drops
                    # any edge whose vector is more than 0.05 rad off an axis,
                    # and pass 2 runs in the frame pass 1 just finished nudging
                    # tiles out of: on a 2-tile phantom a 1.6 µm lateral
                    # correction across a 10 µm separation is 9 degrees, and it
                    # deleted the only edge in the graph. The checkerboard
                    # default would halve the evidence for no reason too.
                    #
                    # So every overlapping pair contributes. That costs roughly
                    # 1.7x the edges of pass 1 on a regular grid (the diagonals
                    # come back), which is the bulk of this pass's ~3x cost —
                    # and it buys more independent Z estimates, which is the
                    # entire point of running it.
                    pre_registration_pruning_method=None,
                    groupwise_resolution_kwargs={
                        "abs_tol": self.config.global_opt_abs_tol,
                        "rel_tol": self.config.global_opt_rel_tol,
                    },
                )
        except Exception as exc:
            self.logger.warning(
                f"  Z refinement failed ({exc}); keeping the first pass's result"
            )
            summary.reason = f"failed: {exc}"
            return params, summary

        merged, summary = self._merge_z_refinement(
            params, list(refined), summary, clamp
        )
        self.logger.info(
            f"  Z refinement: adjusted {summary.n_tiles_moved}/{len(params)} tiles "
            f"(|dz| median {summary.median_abs_dz_um:.2f} µm, "
            f"max {summary.max_abs_dz_um:.2f} µm)"
        )
        if summary.n_hit_search_limit:
            self.logger.warning(
                f"  Z refinement: {summary.n_hit_search_limit} corrections came back "
                f"at the ±{summary.search_range_um:.0f} µm search limit and were "
                f"REJECTED — those are floors, not measurements."
            )
        summary.ran = True
        return merged, summary

    def _merge_z_refinement(self, params, residuals, summary, clamp):
        """Fold pass 2's Z residual into pass 1's params. Z only, by design."""
        import numpy as _np

        search = float(summary.search_range_um)
        bound_z = clamp.bound_z_um if clamp.bound_z_um is not None else search
        applied: List[float] = []
        merged = []
        for index, param in enumerate(params):
            if index >= len(residuals):
                merged.append(param)
                continue
            try:
                base = _np.asarray(param, dtype=float)
                extra = _np.asarray(residuals[index], dtype=float)
                base_mat = base[0] if base.ndim == 3 else base
                extra_mat = extra[0] if extra.ndim == 3 else extra
                # Additive composition (t2 + t1) is only valid for pure
                # translations; for anything else the axes are coupled and a
                # Z-only graft is meaningless. Keep pass 1 in that case.
                eye = _np.eye(3)
                if not (
                    _np.allclose(base_mat[:3, :3], eye, atol=1e-9)
                    and _np.allclose(extra_mat[:3, :3], eye, atol=1e-9)
                ):
                    merged.append(param)
                    continue
                dz = float(extra_mat[0, 3])
            except Exception:
                merged.append(param)
                continue

            if abs(dz) >= search:
                # At the search limit the number is a lower bound on the error,
                # not the error. Applying it would move the tile by an amount we
                # know to be wrong; reporting it as a result would be worse.
                summary.n_hit_search_limit += 1
                merged.append(param)
                continue
            total_z = float(base_mat[0, 3]) + dz
            if abs(total_z) > bound_z:
                merged.append(param)
                continue
            if abs(dz) < 1e-9 or not hasattr(param, "copy"):
                merged.append(param)
                continue

            updated = param.copy()
            buf = updated.values if hasattr(updated, "values") else updated
            out = buf[0] if buf.ndim == 3 else buf
            out[0, 3] = total_z
            merged.append(updated)
            applied.append(abs(dz))

        summary.n_tiles_moved = len(applied)
        summary.max_abs_dz_um = max(applied) if applied else 0.0
        if applied:
            ordered = sorted(applied)
            summary.median_abs_dz_um = ordered[len(ordered) // 2]
        return merged, summary

    def _pairwise_shift_reject(self, tile_data, voxel_size_um):
        """A predicate that throws out a pairwise shift the geometry forbids.

        Applied inside `capture_prefilter_graph`, which multiview-stitcher calls
        immediately before the global solve — so a bad edge never gets to move
        the tiles it touches.

        This is the same physical ceiling `_clamp_registration_shifts` applies,
        moved earlier because earlier is where it works. The post-solve clamp can
        only revert whole TILES, and a solved component is only meaningful whole:
        reverting one member tears it from the rest. On the run that motivated
        this, the clamp reverted 4 tiles out of an 8-tile component and broke
        three seams that had genuinely been measured.

        The bound catches what the quality threshold cannot. Quality here is a
        Spearman rank correlation over the overlap, which stays high when an
        elongated structure slides along its own axis — so a confident wrong peak
        scores WELL. That run accepted a pairwise dx of -345.7 µm across a
        156.7 µm overlap: at that shift the two tiles do not overlap at all, so
        it cannot be a measurement of where they sit relative to each other.

        Returns None when the bound cannot be sized, rather than guessing one.
        """
        try:
            shp = tile_data[0][0].shape
            nz, ny, nx = int(shp[-3]), int(shp[-2]), int(shp[-1])
        except Exception:
            return None
        vx = float(voxel_size_um.get("x", 1.0))
        vy = float(voxel_size_um.get("y", 1.0))
        vz = float(voxel_size_um.get("z", 1.0))
        bound_xy, source_xy = self._lateral_shift_bound(
            [t for _v, t in tile_data], nx * vx, ny * vy, vx, vy
        )
        bound_z, source_z = self._axial_shift_bound(nz * vz, vz)
        self.logger.info(
            f"  Pairwise shift bounds: xy {bound_xy:.1f} µm ({source_xy}), "
            f"z {bound_z:.1f} µm ({source_z}) — a seam proposing more than this "
            f"is dropped before the global solve, not clamped after it."
        )

        def reject(_a, _b, data):
            translation = registration_report.translation_from_param(
                data.get("transform") if hasattr(data, "get") else None
            )
            if translation is None:
                return None
            dz, dy, dx = translation
            lateral = max(abs(dy), abs(dx))
            if lateral > bound_xy:
                return (
                    f"proposed a lateral shift of {lateral:.1f} µm, beyond the "
                    f"{bound_xy:.1f} µm the geometry allows ({source_xy}); at "
                    f"that shift these tiles no longer overlap"
                )
            if abs(dz) > bound_z:
                return (
                    f"proposed a Z shift of {abs(dz):.1f} µm, beyond the "
                    f"{bound_z:.1f} µm bound ({source_z})"
                )
            return None

        return reject

    def _bind_unconstrained_tiles(self, params, coverage):
        """Move tiles nothing measured WITH the mosaic, instead of past it.

        multiview-stitcher resolves each connected component of the registration
        graph independently and hands an edgeless component the identity. So a
        tile with no registered seam keeps its stage position while the tiles
        beside it move — and the seam between them opens by the whole of that
        correction. On a real 4x7 run where the sample sat in the middle two
        columns, that put 90-194 px steps into 14 of 45 seams that had no step
        before registration.

        The fix follows the same observation the clamp's consensus rests on: a
        correction every tile shares slides the mosaic and opens no seam at all.
        So an unconstrained tile is given the dominant component's consensus
        shift rather than zero. Its own placement is still unmeasured — it just
        stops being unmeasured in a way that tears its neighbours.

        Returns ``(params, bound_indices)``.
        """
        import numpy as _np

        if not coverage.components or coverage.is_safe_to_apply:
            return params, []
        dominant = set(coverage.components[0])
        loose = [i for i in range(len(params)) if i not in dominant]
        if not loose or len(dominant) < _MIN_TILES_FOR_CONSENSUS:
            return params, []

        cz, cy, cx = self._median_shift(params, sorted(dominant))
        out = list(params)
        bound = []
        for index in loose:
            param = out[index]
            if not hasattr(param, "copy"):
                continue
            try:
                moved = param.copy()
                buf = moved.values if hasattr(moved, "values") else moved
                mat = buf[0] if buf.ndim == 3 else buf
                if not _np.allclose(mat[:3, :3], _np.eye(3), atol=1e-9):
                    continue  # not a pure translation; leave it alone
                mat[0, 3], mat[1, 3], mat[2, 3] = cz, cy, cx
                out[index] = moved
                bound.append(index)
            except Exception:
                continue
        if bound:
            self.logger.info(
                f"  Registration: {len(bound)} tiles had no registered seam and "
                f"were moved with the mosaic (consensus dz={cz:.1f} dy={cy:.1f} "
                f"dx={cx:.1f} µm) rather than left behind. Their own placement "
                f"is unmeasured — see the seam table."
            )
        return out, bound

    def _untrustworthy_registration_reason(self, coverage) -> Optional[str]:
        """Why this registration should not be believed at all, or None.

        Distinct from the geometry problem `_bind_unconstrained_tiles` solves.
        Binding makes a partly-registered mosaic *safe*; it cannot make it
        *right*. When only a small fraction of seams registered, the few that
        did are as likely to be a confident wrong peak as a measurement — and on
        the run that motivated this, the surviving seams proposed lateral shifts
        of up to 345 µm across a 157 µm overlap, which cannot be a measurement
        of anything.

        The fraction is a judgement about a sample and a scope, not a constant,
        so it is configuration: `min_registered_seam_frac`.
        """
        # `or` would swallow a deliberate 0.0 — which is exactly the value that
        # means "do not check this" — so fall back only on a genuinely absent
        # setting.
        configured = getattr(self.config, "min_registered_seam_frac", None)
        threshold = float(
            DEFAULT_MIN_REGISTERED_SEAM_FRAC if configured is None else configured
        )
        if threshold <= 0.0 or coverage.n_expected_seams <= 0:
            return None
        fraction = coverage.n_registered_seams / coverage.n_expected_seams
        if fraction >= threshold:
            return None
        return (
            f"{coverage.describe()} — {fraction * 100:.0f}% of seams, below the "
            f"{threshold * 100:.0f}% needed to trust the result. With this "
            f"little agreement the seams that did pass are as likely to be a "
            f"confident wrong correlation peak as a measurement. The seam table "
            f"gives a quality score for every pair: if the sample genuinely "
            f"covers only part of the grid, either restrict the run to the "
            f"tiles that contain it, or lower 'Minimum share of seams that must "
            f"register' for this microscope in the Options tab."
        )

    def _unconstrained_coverage_reason(self, coverage, params) -> str:
        """Why this registration was refused, with the damage it would have done.

        A rule the user cannot check is a rule the user will switch off, so this
        measures rather than asserts: for every adjacent pair whose two halves
        landed in different components, the difference between their corrections
        is exactly the step that pair's seam would have shown in the output.
        """
        worst = 0.0
        for index_a, index_b in coverage.unconstrained:
            a = registration_report.translation_from_param(
                params[index_a] if index_a < len(params) else None
            )
            b = registration_report.translation_from_param(
                params[index_b] if index_b < len(params) else None
            )
            if a is None or b is None:
                continue
            worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))

        reason = (
            f"the registered seams do not connect the mosaic — "
            f"{coverage.describe()}. multiview-stitcher solves each connected "
            f"group independently, so nothing measured where those groups sit "
            f"relative to each other"
        )
        if worst > 0.0:
            reason += (
                f"; applying this would have opened a step of up to "
                f"{worst:.1f} µm at seams that are currently only off by stage "
                f"placement error"
            )
        return (
            reason + ". Tiles were placed by stage position instead. The seam "
            "table shows which pairs had too little shared content to register."
        )

    def _registration_settings(
        self, clamp, voxel_size_um, coverage=None
    ) -> Dict[str, Any]:
        settings = {
            "quality_threshold": self.config.quality_threshold,
            "registration_binning": dict(self.config.registration_binning),
            "upsample_factor": self.config.registration_upsample_factor,
            "voxel_z_um": float(voxel_size_um.get("z", 0.0)),
            "voxel_y_um": float(voxel_size_um.get("y", 0.0)),
            "voxel_x_um": float(voxel_size_um.get("x", 0.0)),
        }
        if clamp is not None:
            settings.update(
                {
                    "bound_xy_um": clamp.bound_xy_um,
                    "bound_xy_source": clamp.source_xy,
                    "bound_z_um": clamp.bound_z_um,
                    "bound_z_source": clamp.source_z,
                }
            )
        if coverage is not None:
            settings.update(
                {
                    "seams_registered": coverage.n_registered_seams,
                    "seams_expected": coverage.n_expected_seams,
                    "tile_groups": len(coverage.components),
                    "tiles_unconnected": coverage.n_isolated,
                    "unconstrained_pairs": len(coverage.unconstrained),
                    "tiles_bound_to_mosaic": len(coverage.bound_tiles),
                }
            )
        return settings

    def _clamp_registration_shifts(self, params, tile_data, voxel_size_um):
        """Revert corrections registration cannot plausibly have measured.

        multiview-stitcher's phase correlation bounds a pairwise shift to the
        tile SIZE, not the overlap (see registration.phase_correlation_
        registration: ``max_shift_per_dim = max(im.shape)``), so a low-content
        tile — background, or a featureless bright blur — can be flung most of a
        tile away on a garbage correlation peak and open a gap between tiles.

        **Lateral and axial are bounded separately**, because they are not the
        same measurement and no single number describes both. X/Y are bounded by
        the tile overlap width: a tile that moves more than one overlap opens a
        gap, so that is a real physical ceiling. Z has no such ceiling on an XY
        mosaic — every tile spans the same depth, so there is no Z overlap to
        derive one from — and the old code applied the lateral number to Z
        anyway. That was not conservative, it was arbitrary: being large, it
        left Z effectively unguarded, while any single over-budget axis reverted
        the whole matrix and threw away that tile's other two good corrections.

        X and Y are still reverted **together**: they come out of one joint
        lateral correlation peak, so if one is garbage the peak is garbage. Z is
        independent, and is what the optional refinement pass measures on its
        own.

        A clamped axis is reverted to the mosaic's **consensus** shift, not to
        zero: zero means "this tile's stage position", which is the wrong answer
        whenever the mosaic as a whole moved, and pulls the tile away from
        neighbours that did move. See the comment at the revert itself.

        Returns a `ClampResult`: the (possibly clamped) params plus a per-tile
        record of what was seen and what was reverted, which the registration
        report turns into rows. A clamped axis means **not measured** — never
        report it as a shift of the bound.
        """
        import numpy as _np

        n = len(params)
        if n == 0:
            return ClampResult(params=params, records=[])

        # Frame extent (µm) from the reference volume + processing voxel size.
        try:
            shp = tile_data[0][0].shape
            nz, ny, nx = int(shp[-3]), int(shp[-2]), int(shp[-1])
        except Exception:
            # Can't size the bound → leave params untouched rather than guess.
            return ClampResult(params=params, records=[])
        vx = float(voxel_size_um.get("x", 1.0))
        vy = float(voxel_size_um.get("y", 1.0))
        vz = float(voxel_size_um.get("z", 1.0))
        frame_x_um, frame_y_um, depth_um = nx * vx, ny * vy, nz * vz

        bound_xy, source_xy = self._lateral_shift_bound(
            [t for _v, t in tile_data], frame_x_um, frame_y_um, vx, vy
        )
        bound_z, source_z = self._axial_shift_bound(depth_um, vz)

        # Judge each tile against where the mosaic AS PLACED put it, not
        # against its original stage position: a correction every tile shares
        # slides the whole mosaic and opens no seam. See
        # _consensus_reference_shift.
        reference = self._consensus_reference_shift(params)

        clamped = []
        records: List[ClampRecord] = []
        for index, param in enumerate(params):
            try:
                arr = _np.asarray(param)
                mat = arr[0] if arr.ndim == 3 else arr
                abs_z, abs_y, abs_x = (float(v) for v in mat[:3, 3])  # µm, (z, y, x)
                # Judge the correction against the mosaic's consensus.
                ref_z, ref_y, ref_x = reference
                dz, dy, dx = abs_z - ref_z, abs_y - ref_y, abs_x - ref_x
                # Per-axis surgery is only meaningful for a pure translation.
                # For a general affine the translation column is the motion of
                # the world ORIGIN, not of the tile, and a rotation puts real Z
                # displacement into the 3x3 block — so zeroing one component
                # would neither remove that axis's error nor leave the others
                # alone. Today the global solve is translation-only, but ANTsPy
                # or transform="rigid" would not be.
                translation_only = _np.allclose(mat[:3, :3], _np.eye(3), atol=1e-9)
            except Exception:
                clamped.append(param)
                continue

            if translation_only:
                clamp_xy = max(abs(dy), abs(dx)) > bound_xy
                clamp_z = abs(dz) > bound_z
                whole = False
            else:
                clamp_xy = clamp_z = whole = max(abs(dz), abs(dy), abs(dx)) > max(
                    bound_xy, bound_z
                )

            records.append(
                ClampRecord(
                    index=index,
                    # The correction registration proposed, absolute...
                    dz_um=abs_z,
                    dy_um=abs_y,
                    dx_um=abs_x,
                    # ...and the part of it that disagrees with the neighbours,
                    # which is what the bound was actually applied to.
                    rel_z_um=dz,
                    rel_y_um=dy,
                    rel_x_um=dx,
                    clamped_xy=clamp_xy,
                    clamped_z=clamp_z,
                    whole_matrix=whole,
                )
            )

            if not (clamp_xy or clamp_z) or not hasattr(param, "copy"):
                clamped.append(param)
                continue

            # Revert to the mosaic's CONSENSUS, not to zero.
            #
            # Zero means "this tile's stage position", and that is only the
            # right answer if the rest of the mosaic also stayed there. When the
            # mosaic as a whole moved — which is exactly what the consensus
            # measures — dropping one tile back to zero pulls it away from every
            # neighbour and opens a seam by the size of the shared correction.
            # That is the same mistake multiview-stitcher makes with an edgeless
            # tile, and the one _bind_unconstrained_tiles exists to undo.
            #
            # What the clamp is entitled to reject is this tile's DISAGREEMENT
            # with its neighbours, which is the part nothing measured. The
            # shared part it keeps.
            #
            # Done on a COPY so the param keeps its exact type and coords
            # (xr.DataArray for real MVS params, ndarray in tests) — a bare
            # identity would drop the x_in/x_out coords rebase_affine needs.
            reverted = param.copy()
            buf = reverted.values if hasattr(reverted, "values") else reverted
            out = buf[0] if buf.ndim == 3 else buf
            if whole:
                # A non-translational param cannot be split per axis, so there
                # is no consensus to fall back to — only identity.
                out[:3, :3] = _np.eye(3)
                out[:3, 3] = 0.0
            else:
                if clamp_z:
                    out[0, 3] = ref_z
                if clamp_xy:
                    out[1, 3] = ref_y
                    out[2, 3] = ref_x
            clamped.append(reverted)

        result = ClampResult(
            params=clamped,
            records=records,
            bound_xy_um=bound_xy,
            bound_z_um=bound_z,
            source_xy=source_xy,
            source_z=source_z,
        )
        self.logger.info(result.summary_line())
        return result

    def _consensus_reference_shift(self, params):
        """The correction the mosaic as a whole agreed on, as (dz, dy, dx) µm.

        The clamp asks "did this tile move somewhere implausible?", and the only
        implausible move is one that separates a tile from the tiles it has to
        line up with. A correction shared by every tile slides the whole mosaic
        and opens no seam at all — so measuring each tile against **where the
        stage said it was** rejects exactly the corrections that were working:
        on a large grid a systematic offset accumulates past one overlap while
        every adjacent pair stays perfectly matched.

        The reference is the per-axis MEDIAN over all tiles, not the mean: a
        handful of tiles flung away by a garbage correlation peak is the case
        this guard exists for, and they must not be allowed to move the
        baseline they are judged against.

        **Limitation, stated rather than hidden:** this removes a uniform
        offset, not an arbitrary gradient. If a mosaic drifts steadily from one
        end to the other by much more than an overlap, the extremes still read
        as outliers. Per-tile neighbour medians would handle that, but they
        collapse where a tile has one or two neighbours — an edge or a sparse
        montage — which is where a drift is largest. If the seam report ever
        shows gradient-driven clamping on real data, that is the evidence to
        revisit this with.
        """
        import numpy as _np

        # A consensus needs a majority, and two tiles do not have one: when two
        # tiles disagree there is no way to tell which one moved. Below three,
        # fall back to judging the absolute displacement from the stage
        # position — the only check available, and the conservative one.
        if len(params) < _MIN_TILES_FOR_CONSENSUS:
            return (0.0, 0.0, 0.0)

        return self._median_shift(params, range(len(params)))

    def _median_shift(self, params, indices):
        """Per-axis median translation over `indices`, as (dz, dy, dx) µm.

        Median, not mean: the case this exists for is a handful of tiles flung
        away by a garbage correlation peak, and they must not be able to move
        the baseline they are judged against.
        """
        import numpy as _np

        shifts = []
        for index in indices:
            try:
                arr = _np.asarray(params[index])
                mat = arr[0] if arr.ndim == 3 else arr
                shifts.append((float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])))
            except Exception:
                shifts.append((0.0, 0.0, 0.0))
        if not shifts:
            return (0.0, 0.0, 0.0)

        def _median(axis):
            ordered = sorted(v[axis] for v in shifts)
            return float(ordered[len(ordered) // 2])

        return (_median(0), _median(1), _median(2))

    def _lateral_shift_bound(self, tiles, frame_x_um, frame_y_um, vx, vy):
        """(bound µm, source) for X/Y: one tile overlap width.

        A tile that moves more than the overlap has, by definition, been pushed
        past its neighbour's content — which opens a gap rather than closing
        one. That makes the overlap a real ceiling rather than a taste setting.
        """
        configured = float(getattr(self.config, "max_registration_shift_um", 0.0) or 0.0)
        if configured > 0.0:
            return max(configured, 2.0 * max(vx, vy)), "config"

        layout = tile_geometry.grid_overlap(
            tiles, extent_x_um=frame_x_um, extent_y_um=frame_y_um
        )
        widths = [
            ov.overlap_um
            for ov in (layout["x"], layout["y"])
            if ov.overlap_um is not None and ov.overlap_um > 0
        ]
        if widths:
            bound, source = min(widths), "auto: min overlap width"
        else:
            # No measurable overlap on either axis (single row/column, or a
            # gapped grid). 10% of a frame is a fallback, not a measurement.
            bound, source = 0.1 * min(frame_x_um, frame_y_um), "auto: 10% of frame"
        # Never revert on a sub-pixel rounding quirk.
        return max(bound, 2.0 * max(vx, vy)), source

    def _axial_shift_bound(self, depth_um, vz):
        """(bound µm, source) for Z, which has no overlap width to lean on.

        Sized to admit the error we actually see — a few frames of stage/focus
        drift between tiles — while still rejecting a correlation peak found
        halfway down the stack.
        """
        configured = float(
            getattr(self.config, "max_registration_shift_z_um", 0.0) or 0.0
        )
        binning_z = 1
        try:
            binning_z = max(1, int(self.config.registration_binning.get("z", 1) or 1))
        except Exception:
            pass
        # Pass 1 registers at `binning_z` Z voxels, so it can only EXPRESS
        # multiples of that. A bound below one binned step would clamp pure
        # quantization and report it as a rejected measurement.
        floor = 2.0 * binning_z * vz

        if configured > 0.0:
            return max(configured, floor), "config"

        bound = max(_Z_CLAMP_MIN_STEPS * vz, _Z_CLAMP_FLOOR_UM)
        if depth_um > 0:
            bound = min(bound, _Z_CLAMP_STACK_FRACTION * depth_um)
        return max(bound, floor), "auto"

    def _fuse_chunksize(self, tile_data) -> Dict[str, int]:
        """Output chunks for this fuse, sized against the final grid.

        Logs when auto-sizing changed the configured value, so a run's chunking
        is never a silent surprise.
        """
        tile_infos = [ti for _, ti in tile_data] if tile_data else []
        chunks = resolve_output_chunksize(self.config, tile_infos)
        configured = self.config.output_chunksize or {}
        if chunks != {k: int(configured.get(k, v)) for k, v in chunks.items()}:
            self.logger.info(
                f"  Output chunks auto-sized to the final grid: "
                f"z={chunks['z']} y={chunks['y']} x={chunks['x']} "
                f"(configured {configured}) — keeps the fused block near one "
                f"tile pitch so heavy downsample keeps saving memory"
            )
        return chunks

    def _fuse_with_fallback(self, fuse_fn, sims, fuse_kwargs):
        """Call multiview_stitcher.fusion.fuse, retrying without
        ``weights_func`` if the content-based path raises.

        multiview-stitcher's ``content_based`` weighting has known edge
        cases with large tiles / NaN-heavy overlaps that surface as
        ``'NoneType' object is not subscriptable`` deep inside the fuse
        graph. Rather than fail the whole run, log the traceback, drop
        the weights_func, and retry with default cosine blending.
        """
        import traceback

        try:
            return fuse_fn(sims, **fuse_kwargs)
        except Exception as e:
            if "weights_func" not in fuse_kwargs:
                raise
            self.logger.error(
                f"  Content-based fusion failed: {e}\n{traceback.format_exc()}"
            )
            self.logger.warning(
                "  Falling back to default cosine blending "
                "(turn off 'Content-based fusion' in Processing Options "
                "to silence this warning)."
            )
            retry_kwargs = dict(fuse_kwargs)
            retry_kwargs.pop("weights_func", None)
            retry_kwargs.pop("weights_func_kwargs", None)
            return fuse_fn(sims, **retry_kwargs)

    def _build_fusion_inputs(
        self,
        tile_data: List[Tuple[Any, RawTileInfo]],
        voxel_size_um: Dict[str, float],
        reg_params: list,
        transform_key: str,
    ) -> Tuple[list, Dict[str, Any]]:
        """Build the per-tile SpatialImages + the fuse() kwargs for a channel.

        Split out of :meth:`_fuse_channel` so the streaming path can build the
        inputs ONCE and then fuse many chunk-aligned sub-regions from them
        (super-block batching, item E), instead of one graph over the whole
        output. Returns ``(sims, fuse_kwargs)``; ``transform_key`` is embedded
        in ``fuse_kwargs``.
        """
        from multiview_stitcher import io as mvs_io
        from multiview_stitcher import msi_utils
        from multiview_stitcher import spatial_image_utils as si_utils

        import dask.array as da

        if self.config.tile_overlap_fusion == "brightest":
            # Rank tiles by mean intensity and place them brightest-first, so the
            # winner-take-all fusion_func (which fills each pixel from the first
            # covering view) draws every overlap pixel from the brightest tile.
            # reg_params is positionally aligned with tile_data, so reorder both
            # together to keep each tile's registration transform attached.
            order = sorted(
                range(len(tile_data)),
                key=lambda i: _tile_brightness(tile_data[i][0]),
                reverse=True,
            )
            tile_data = [tile_data[i] for i in order]
            if reg_params and len(reg_params) == len(order):
                reg_params = [reg_params[i] for i in order]

        # Multi-view: rotation axis for placing angled views (None-op when off).
        rot_center_um = self._resolve_rotation_center_um(
            [ti for _, ti in tile_data]
        )
        msims = []
        for volume, tile_info in tile_data:
            translation_um = {
                "z": tile_info.z_min_mm * 1000.0,
                "y": (
                    -tile_info.y_mm if self.config.reverse_y_tiles else tile_info.y_mm
                )
                * 1000.0,
                "x": (
                    -tile_info.x_mm if self.config.reverse_x_tiles else tile_info.x_mm
                )
                * 1000.0,
            }
            if not isinstance(volume, da.Array):
                volume = da.from_array(volume, chunks=_DASK_PROCESSING_CHUNKS)

            sim = si_utils.get_sim_from_array(
                volume,
                dims=["z", "y", "x"],
                scale=voxel_size_um,
                translation=translation_um,
                affine=self._tile_metadata_affine(tile_info, rot_center_um),
                transform_key=mvs_io.METADATA_TRANSFORM_KEY,
            )
            msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
            msims.append(msim)

        if reg_params and transform_key != mvs_io.METADATA_TRANSFORM_KEY:
            for msim, param in zip(msims, reg_params):
                msi_utils.set_affine_transform(
                    msim,
                    param,
                    transform_key=transform_key,
                    base_transform_key=mvs_io.METADATA_TRANSFORM_KEY,
                )

        sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

        fuse_kwargs: Dict[str, Any] = dict(
            transform_key=transform_key,
            output_chunksize=self._fuse_chunksize(tile_data),
            blending_widths=self.config.blending_widths,
        )

        if self.config.tile_overlap_fusion == "max":
            # Pixel-wise maximum across tiles: keeps the brighter tile in the
            # overlap so a sparse sample can't be diluted against a neighbour's
            # background. max_fusion ignores blending/content weights (it has no
            # blending_weights kwarg), so we skip them entirely.
            from multiview_stitcher.fusion import max_fusion

            fuse_kwargs["fusion_func"] = max_fusion
            self.logger.info("  Tile-overlap fusion: maximum intensity (np.nanmax)")
        elif self.config.tile_overlap_fusion == "brightest":
            # Winner-take-all: each overlap pixel is taken whole from the
            # brightest tile covering it (tiles pre-sorted brightest-first
            # above). No per-pixel mixing and no blending/content weights —
            # like max_fusion but selecting a whole tile rather than a pixel.
            fuse_kwargs["fusion_func"] = _priority_coalesce_fusion
            self.logger.info(
                "  Tile-overlap fusion: brightest tile wins whole overlap "
                "(winner-take-all, global priority)"
            )
        else:
            self.logger.info("  Tile-overlap fusion: weighted blend (cosine)")
            if self.config.content_based_fusion:
                try:
                    from multiview_stitcher.weights import content_based

                    fuse_kwargs["weights_func"] = content_based
                    # content_based defaults (sigma_1=5, sigma_2=11). Must be
                    # provided explicitly: fusion.fuse calls
                    # calculate_required_overlap(weights_func, weights_func_kwargs)
                    # which unconditionally dereferences kwargs["sigma_2"], so
                    # passing None here crashes with a NoneType subscript error.
                    fuse_kwargs["weights_func_kwargs"] = {
                        "sigma_1": 5,
                        "sigma_2": 11,
                    }
                    self.logger.info("  Using content-based tile-overlap weighting")
                except ImportError:
                    self.logger.warning(
                        "  content_based weights not available — using default blending"
                    )

        return sims, fuse_kwargs

    def _fuse_channel(
        self,
        tile_data: List[Tuple[Any, RawTileInfo]],
        voxel_size_um: Dict[str, float],
        reg_params: list,
        transform_key: str,
        output_stack_properties: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """Fuse tiles for a single channel using pre-computed registration.

        Args:
            tile_data: [(volume, tile_info), ...] for this channel
            voxel_size_um: Voxel sizes dict
            reg_params: Affine params from _register_tiles (one per tile)
            transform_key: Transform key to use for fusion
            output_stack_properties: If given, fuse ONLY this sub-region
                (origin/shape/spacing) instead of the whole output — used by
                super-block batching. Must be chunk-aligned for the region to
                match whole-output fusion bit-for-bit.

        Returns:
            (fused_sim, origin_um) — fused SpatialImage and origin dict
        """
        try:
            from multiview_stitcher import fusion
        except ImportError as exc:
            # Surface the REAL missing module (e.g. a dependency stripped from a
            # frozen build) instead of hiding it behind a generic message — the
            # underlying ImportError names exactly what's absent.
            raise ImportError(
                "multiview-stitcher (or one of its dependencies) failed to import. "
                f"Underlying error: {exc}. "
                "If running from source, install with: pip install multiview-stitcher"
            ) from exc

        sims, fuse_kwargs = self._build_fusion_inputs(
            tile_data, voxel_size_um, reg_params, transform_key
        )
        if output_stack_properties is not None:
            fuse_kwargs["output_stack_properties"] = output_stack_properties

        fused = self._fuse_with_fallback(fusion.fuse, sims, fuse_kwargs)

        origin_um = {
            "z": float(fused.coords["z"].values[0]),
            "y": float(fused.coords["y"].values[0]),
            "x": float(fused.coords["x"].values[0]),
        }
        self.logger.info(
            f"  Fused origin (µm): Z={origin_um['z']:.1f} "
            f"Y={origin_um['y']:.1f} X={origin_um['x']:.1f}"
        )

        return fused, origin_um

    def _full_stack_properties(self, sims, fuse_kwargs) -> Dict[str, Any]:
        """Compute the whole-output stack properties (origin/shape/spacing)
        without fusing — used to lay out super-block regions."""
        from multiview_stitcher import fusion
        from multiview_stitcher import spatial_image_utils as si_utils

        tk = fuse_kwargs["transform_key"]
        params = [si_utils.get_affine_from_sim(s, transform_key=tk) for s in sims]
        spacing = si_utils.get_spacing_from_sim(sims[0])
        return fusion.calc_fusion_stack_properties(
            sims, params=params, spacing=spacing, mode="union"
        )

    def _iter_superblock_regions(self, full_props, region_chunks: int):
        """Yield ``(region_props, (z0,z1,y0,y1,x0,x1))`` covering the full
        output in chunk-aligned regions of ``region_chunks`` output-chunks per
        axis. Chunk alignment is what makes each region fuse bit-identically to
        the whole output (see tests/test_superblock_fusion.py)."""
        sp = full_props["spacing"]
        org = full_props["origin"]
        shp = full_props["shape"]
        cs = self.config.output_chunksize
        step = {
            d: max(1, int(cs.get(d, 256))) * max(1, int(region_chunks))
            for d in ("z", "y", "x")
        }
        for z0 in range(0, int(shp["z"]), step["z"]):
            z1 = min(z0 + step["z"], int(shp["z"]))
            for y0 in range(0, int(shp["y"]), step["y"]):
                y1 = min(y0 + step["y"], int(shp["y"]))
                for x0 in range(0, int(shp["x"]), step["x"]):
                    x1 = min(x0 + step["x"], int(shp["x"]))
                    region_props = {
                        "origin": {
                            "z": org["z"] + z0 * sp["z"],
                            "y": org["y"] + y0 * sp["y"],
                            "x": org["x"] + x0 * sp["x"],
                        },
                        "shape": {"z": z1 - z0, "y": y1 - y0, "x": x1 - x0},
                        "spacing": dict(sp),
                    }
                    yield region_props, (z0, z1, y0, y1, x0, x1)

    def _write_multichannel_output(
        self,
        stacked: np.ndarray,
        channel_names: List[str],
        voxel_size_um: Dict[str, float],
        output_dir: Path,
        basename: str = "stitched",
    ) -> None:
        """Write multi-channel stacked volume to the configured output format.

        For multi-channel data, produces a single store (e.g. {basename}.ome.zarr)
        instead of per-channel files.
        """
        fmt = self.config.output_format

        if fmt == "ome-zarr-sharded" or fmt == "both":
            out_path = output_dir / f"{basename}.ome.zarr"
            self.logger.info(f"  Writing sharded OME-Zarr v0.5: {out_path}")
            try:
                from flamingo_stitcher.writers.ome_zarr_writer import (
                    package_as_ozx,
                    write_ome_zarr_sharded,
                )

                write_ome_zarr_sharded(
                    data=stacked,
                    output_path=out_path,
                    voxel_size_um=voxel_size_um,
                    chunks=self.config.zarr_chunks,
                    shard_chunks=self.config.zarr_shard_chunks,
                    compression=self.config.zarr_compression,
                    compression_level=self.config.zarr_compression_level,
                    pyramid_levels=self.config.pyramid_levels,
                    pyramid_method=self.config.pyramid_method,
                    channel_names=channel_names,
                    use_tensorstore=self.config.zarr_use_tensorstore,
                )

                if self.config.package_ozx:
                    ozx_path = output_dir / f"{basename}.ozx"
                    self.logger.info(f"  Packaging as .ozx: {ozx_path}")
                    package_as_ozx(out_path, ozx_path)

            except ImportError as e:
                self.logger.error(f"  OME-Zarr sharded write failed: {e}")
                self.logger.info("  Falling back to OME-TIFF")
                fmt = "ome-tiff"

        if fmt == "ome-zarr-v2":
            out_path = output_dir / f"{basename}.ome.zarr"
            self.logger.info(f"  Writing OME-Zarr v2 (Fiji compatible): {out_path}")
            try:
                from flamingo_stitcher.writers.ome_zarr_writer import (
                    write_ome_zarr_v2,
                )

                write_ome_zarr_v2(
                    data=stacked,
                    output_path=out_path,
                    voxel_size_um=voxel_size_um,
                    chunks=self.config.zarr_chunks,
                    compression=self.config.zarr_compression,
                    compression_level=self.config.zarr_compression_level,
                    pyramid_levels=self.config.pyramid_levels,
                )
            except Exception as e:
                self.logger.error(f"  OME-Zarr v2 write failed: {e}")
                self.logger.info("  Falling back to OME-TIFF")
                fmt = "ome-tiff"

        if fmt in ("ome-tiff", "both"):
            tiff_path = output_dir / f"{basename}.ome.tif"
            self.logger.info(f"  Writing pyramidal OME-TIFF: {tiff_path}")
            try:
                from flamingo_stitcher.writers.ome_tiff_writer import (
                    write_pyramidal_ome_tiff,
                )

                write_pyramidal_ome_tiff(
                    data=stacked,
                    output_path=tiff_path,
                    voxel_size_um=voxel_size_um,
                    tile_size=self.config.tiff_tile_size,
                    compression=self.config.tiff_compression,
                    pyramid_levels=(
                        self.config.pyramid_levels if self.config.tiff_pyramids else 0
                    ),
                    channel_names=channel_names,
                )
            except ImportError as e:
                self.logger.error(f"  OME-TIFF write failed: {e}")

        if fmt == "imaris":
            ims_path = output_dir / f"{basename}.ims"
            self.logger.info(f"  Writing Imaris .ims (direct): {ims_path}")
            try:
                from flamingo_stitcher.writers import imaris_writer

                if not imaris_writer.is_available():
                    self.logger.error(
                        f"  Imaris writer unavailable: {imaris_writer.unavailable_reason()}"
                    )
                else:
                    imaris_writer.write_imaris_from_array(
                        stacked=stacked,
                        output_path=ims_path,
                        voxel_size_um=voxel_size_um,
                        channel_names=channel_names,
                        progress_callback=self._progress_fn,
                    )
            except Exception as e:
                # 'imaris' is the sole requested format, so a failure here means
                # no usable output — surface it so the item is marked failed
                # instead of being reported as a successful run.
                self.logger.error(f"  Imaris .ims write failed: {e}", exc_info=True)
                raise RuntimeError(f"Imaris .ims write failed: {e}") from e

    def _write_multichannel_streaming(
        self,
        dask_data,
        channel_names: List[str],
        voxel_size_um: Dict[str, float],
        output_dir: Path,
        basename: str = "stitched",
    ) -> None:
        """Write dask array to output format in streaming mode (low memory).

        Dispatches to streaming writers that compute and write chunk-by-chunk.
        """
        fmt = self.config.output_format

        if fmt in ("ome-zarr-sharded", "ome-zarr-v2", "both"):
            out_path = output_dir / f"{basename}.ome.zarr"
            self.logger.info(f"  Writing OME-Zarr (streaming): {out_path}")
            try:
                from flamingo_stitcher.writers.ome_zarr_writer import (
                    write_ome_zarr_streaming,
                )

                write_ome_zarr_streaming(
                    dask_data=dask_data,
                    output_path=out_path,
                    voxel_size_um=voxel_size_um,
                    chunks=self.config.zarr_chunks,
                    compression=self.config.zarr_compression,
                    compression_level=self.config.zarr_compression_level,
                    pyramid_levels=self.config.pyramid_levels,
                    channel_names=channel_names,
                )
            except Exception as e:
                self.logger.error(
                    f"  Streaming OME-Zarr write failed: {e}", exc_info=True
                )
                if fmt != "both":
                    raise

        if fmt in ("ome-tiff", "both"):
            tiff_path = output_dir / f"{basename}.ome.tif"
            self.logger.info(f"  Writing OME-TIFF (streaming): {tiff_path}")
            try:
                from flamingo_stitcher.writers.ome_tiff_writer import (
                    write_pyramidal_ome_tiff_streaming,
                )

                write_pyramidal_ome_tiff_streaming(
                    dask_data=dask_data,
                    output_path=tiff_path,
                    voxel_size_um=voxel_size_um,
                    tile_size=self.config.tiff_tile_size,
                    compression=self.config.tiff_compression,
                    pyramid_levels=self.config.pyramid_levels,
                    channel_names=channel_names,
                )
            except Exception as e:
                self.logger.error(
                    f"  Streaming OME-TIFF write failed: {e}", exc_info=True
                )

    def _write_stitch_metadata_v2(
        self,
        output_dir: Path,
        channel_ids: List[int],
        origin_um: Dict[str, float],
        tiles: List[RawTileInfo],
        voxel_size_um: Dict[str, float],
        acquisition_dir: Path,
        basename: str = "stitched",
    ) -> None:
        """Write stitch_metadata.json v2 for single multi-channel store."""
        origin_list = [origin_um["z"], origin_um["y"], origin_um["x"]]

        # Determine the store filename
        fmt = self.config.output_format
        if fmt in ("ome-zarr-sharded", "ome-zarr-v2", "both"):
            store_path = f"{basename}.ome.zarr"
        elif fmt == "ome-tiff":
            store_path = f"{basename}.ome.tif"
        elif fmt == "imaris":
            store_path = f"{basename}.ims"
        else:
            store_path = f"{basename}.ome.zarr"

        # Build per-channel dict (all point to same store, for backward compat)
        channels_meta = {}
        for ch_id in channel_ids:
            channels_meta[str(ch_id)] = {
                "path": store_path,
                "origin_um": origin_list,
            }

        # Per-tile coverage descriptor (additive; older readers ignore unknown
        # keys). Records which illumination sides and rotation angle each tile
        # carried, so an asymmetric / multi-view acquisition is self-describing.
        # `partial_coverage` flags when the cuboid was NOT collected uniformly:
        # any tile missing a side, or more than one distinct angle present.
        tiles_meta = [
            {
                "x_mm": t.x_mm,
                "y_mm": t.y_mm,
                "z_min_mm": t.z_min_mm,
                "z_max_mm": t.z_max_mm,
                "illumination_sides": list(t.illumination_sides),
                "angle_deg": t.angle_deg,
                "view": t.view,
            }
            for t in tiles
        ]
        all_sides = sorted({s for t in tiles for s in t.illumination_sides})
        all_angles = sorted({t.angle_deg for t in tiles})
        partial_coverage = (
            any(list(t.illumination_sides) != all_sides for t in tiles)
            or len(all_angles) > 1
        )

        metadata = {
            "version": 2,
            **stitcher_provenance(),
            "source_acquisition": str(acquisition_dir),
            "voxel_size_um": voxel_size_um,
            "store_path": store_path,
            "origin_um": origin_list,
            "channel_ids": channel_ids,
            "downsample_xy": self.config.downsample_xy,
            "downsample_z": self.config.downsample_z,
            "output_format": self.config.output_format,
            "channels": channels_meta,
            "tile_count": len(tiles),
            "illumination_sides": all_sides,
            "angles_deg": all_angles,
            "partial_coverage": partial_coverage,
            # --- World frame (v2.1) ---
            # WITHOUT these the output is not self-describing: `origin_um` and
            # the voxel grid live in a frame the tile placement chose, and a
            # consumer cannot tell which. `reverse_x_tiles` NEGATES world X
            # (see the translation_um construction in the fuse paths), so a
            # mosaic spanning stage X 2.34–8.35 mm is written with
            # origin_um.x = -8350 — a number a reader will happily mistake for
            # a stage coordinate and place the volume mirrored and displaced.
            # `tile_orientation` is the per-tile camera→stage transform, which
            # fixes how the pixel axes relate to the stage axes.
            "world_frame": {
                "tile_orientation": str(
                    getattr(self.config, "tile_orientation", "") or ""
                ),
                "reverse_x_tiles": bool(
                    getattr(self.config, "reverse_x_tiles", False)
                ),
                "reverse_y_tiles": bool(
                    getattr(self.config, "reverse_y_tiles", False)
                ),
                # True when world X/Y are the NEGATION of stage X/Y, i.e. a
                # consumer must negate them back to recover stage coordinates.
                "x_axis_negated": bool(
                    getattr(self.config, "reverse_x_tiles", False)
                ),
                "y_axis_negated": bool(
                    getattr(self.config, "reverse_y_tiles", False)
                ),
                # The rotation-stage angle the sample was physically at. Already
                # present as `angles_deg`, repeated here so everything needed to
                # place the volume is in one object.
                "acquisition_angle_deg": (all_angles[0] if all_angles else 0.0),
            },
            "tiles": tiles_meta,
            # Full processing settings used for this run, so the GUI's "Load
            # Configuration" can reproduce a setup that worked on another
            # acquisition (skipping the file-specific fields on load).
            "stitching_config": serialize_stitching_config(self.config),
        }

        meta_path = output_dir / "stitch_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        self.logger.info(f"  Wrote {meta_path}")

    def _register_and_fuse(
        self,
        tile_data: List[Tuple[Any, RawTileInfo]],
        channel_id: int,
        voxel_size_um: Dict[str, float],
        output_dir: Path,
    ) -> Tuple[Path, Dict[str, float]]:
        """Register tiles and fuse into a single stitched volume.

        Uses multiview-stitcher for phase-correlation registration
        and blended fusion.

        Returns:
            Tuple of (output_path, origin_um) where origin_um is
            {"z": ..., "y": ..., "x": ...} in micrometers.
        """
        try:
            from multiview_stitcher import (
                fusion,
            )
            from multiview_stitcher import io as mvs_io
            from multiview_stitcher import (
                msi_utils,
                registration,
            )
            from multiview_stitcher import spatial_image_utils as si_utils
        except ImportError as exc:
            # Surface the REAL missing module (e.g. a dependency stripped from a
            # frozen build) instead of hiding it behind a generic message — the
            # underlying ImportError names exactly what's absent.
            raise ImportError(
                "multiview-stitcher (or one of its dependencies) failed to import. "
                f"Underlying error: {exc}. "
                "If running from source, install with: pip install multiview-stitcher"
            ) from exc

        import dask.array as da

        # --- Build SpatialImages with stage positions ---
        self.logger.info("  Building tile spatial images...")
        msims = []
        for volume, tile_info in tile_data:
            # Convert stage positions from mm to µm
            translation_um = {
                "z": tile_info.z_min_mm * 1000.0,
                "y": (
                    -tile_info.y_mm if self.config.reverse_y_tiles else tile_info.y_mm
                )
                * 1000.0,
                "x": (
                    -tile_info.x_mm if self.config.reverse_x_tiles else tile_info.x_mm
                )
                * 1000.0,
            }

            # Wrap as dask array for lazy computation
            if not isinstance(volume, da.Array):
                volume = da.from_array(volume, chunks=_DASK_PROCESSING_CHUNKS)

            sim = si_utils.get_sim_from_array(
                volume,
                dims=["z", "y", "x"],
                scale=voxel_size_um,
                translation=translation_um,
                transform_key=mvs_io.METADATA_TRANSFORM_KEY,
            )

            msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
            msims.append(msim)

        self.logger.info(f"  Built {len(msims)} multiscale spatial images")

        # --- Registration ---
        if len(msims) > 1:
            self.logger.info(
                f"  Running phase correlation registration "
                f"(quality threshold={self.config.quality_threshold})..."
            )
            try:
                import dask.diagnostics

                with dask.diagnostics.ProgressBar():
                    params = registration.register(
                        msims,
                        reg_channel_index=0,
                        transform_key=mvs_io.METADATA_TRANSFORM_KEY,
                        new_transform_key="registered",
                        registration_binning=self.config.registration_binning,
                        post_registration_do_quality_filter=True,
                        post_registration_quality_threshold=self.config.quality_threshold,
                        # Global optimization with iterative edge pruning —
                        # inspired by BigStitcher (Hörl et al., Nature Methods
                        # 2019).  Edges with residuals above abs_tol are
                        # removed (preserving graph connectivity) and the
                        # optimization re-runs.
                        groupwise_resolution_kwargs={
                            "abs_tol": self.config.global_opt_abs_tol,
                            "rel_tol": self.config.global_opt_rel_tol,
                        },
                    )

                params = self._clamp_registration_shifts(
                    list(params), tile_data, voxel_size_um
                ).params

                # Apply transforms
                for msim, param in zip(msims, params):
                    msi_utils.set_affine_transform(
                        msim,
                        param,
                        transform_key="registered",
                        base_transform_key=mvs_io.METADATA_TRANSFORM_KEY,
                    )

                fuse_transform_key = "registered"
                self.logger.info("  Registration complete")

            except Exception as e:
                self.logger.error(f"  Registration failed: {e}")
                self.logger.info("  Falling back to metadata positions only")
                fuse_transform_key = mvs_io.METADATA_TRANSFORM_KEY
        else:
            self.logger.info("  Single tile — skipping registration")
            fuse_transform_key = mvs_io.METADATA_TRANSFORM_KEY

        # --- Fusion ---
        # Cosine blending widths + optional content-based weighting are
        # inspired by BigStitcher's fusion algorithm (Hörl et al., Nature
        # Methods 2019).  multiview-stitcher implements both natively.
        self.logger.info(f"  Fusing tiles (transform_key={fuse_transform_key})...")
        sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

        fuse_kwargs: Dict[str, Any] = dict(
            transform_key=fuse_transform_key,
            output_chunksize=self._fuse_chunksize(tile_data),
            blending_widths=self.config.blending_widths,
        )

        if self.config.content_based_fusion:
            try:
                from multiview_stitcher.weights import content_based

                fuse_kwargs["weights_func"] = content_based
                # content_based defaults (sigma_1=5, sigma_2=11). Must be
                # provided explicitly: fusion.fuse calls
                # calculate_required_overlap(weights_func, weights_func_kwargs)
                # which unconditionally dereferences kwargs["sigma_2"], so
                # passing None here crashes with a NoneType subscript error.
                fuse_kwargs["weights_func_kwargs"] = {"sigma_1": 5, "sigma_2": 11}
                self.logger.info(
                    "  Using content-based tile-overlap weighting "
                    "(Preibisch local-variance algorithm)"
                )
            except ImportError:
                self.logger.warning(
                    "  content_based weights not available in this "
                    "multiview-stitcher version — using default blending"
                )

        fused = self._fuse_with_fallback(fusion.fuse, sims, fuse_kwargs)

        # --- Extract world-space origin from fused SpatialImage coords ---
        origin_um = {
            "z": float(fused.coords["z"].values[0]),
            "y": float(fused.coords["y"].values[0]),
            "x": float(fused.coords["x"].values[0]),
        }
        self.logger.info(
            f"  Fused origin (µm): Z={origin_um['z']:.1f} "
            f"Y={origin_um['y']:.1f} X={origin_um['x']:.1f}"
        )

        # --- Save output ---
        fmt = self.config.output_format
        out_path = self._write_output(fused, channel_id, voxel_size_um, output_dir, fmt)

        self.logger.info(f"  Channel {channel_id} done → {out_path}")
        return out_path, origin_um

    def _write_output(
        self,
        fused,
        channel_id: int,
        voxel_size_um: Dict[str, float],
        output_dir: Path,
        fmt: str,
    ) -> Path:
        """Write fused result in the configured output format."""
        out_path = output_dir / f"channel_{channel_id:02d}_stitched.tif"  # fallback

        if fmt == "ome-zarr-sharded" or fmt == "both":
            out_path = output_dir / f"channel_{channel_id:02d}.ome.zarr"
            self.logger.info(f"  Writing sharded OME-Zarr v0.5: {out_path}")
            try:
                from flamingo_stitcher.writers.ome_zarr_writer import (
                    package_as_ozx,
                    write_ome_zarr_sharded,
                )

                write_ome_zarr_sharded(
                    data=fused,
                    output_path=out_path,
                    voxel_size_um=voxel_size_um,
                    chunks=self.config.zarr_chunks,
                    shard_chunks=self.config.zarr_shard_chunks,
                    compression=self.config.zarr_compression,
                    compression_level=self.config.zarr_compression_level,
                    pyramid_levels=self.config.pyramid_levels,
                    pyramid_method=self.config.pyramid_method,
                    use_tensorstore=self.config.zarr_use_tensorstore,
                )

                # Package as .ozx if requested
                if self.config.package_ozx:
                    ozx_path = output_dir / f"channel_{channel_id:02d}.ozx"
                    self.logger.info(f"  Packaging as .ozx: {ozx_path}")
                    package_as_ozx(out_path, ozx_path)

            except ImportError as e:
                self.logger.error(f"  OME-Zarr sharded write failed: {e}")
                self.logger.info("  Falling back to OME-TIFF")
                fmt = "ome-tiff"

        if fmt == "ome-zarr-v2":
            out_path = output_dir / f"channel_{channel_id:02d}.ome.zarr"
            self.logger.info(f"  Writing OME-Zarr v2 (Fiji compatible): {out_path}")
            try:
                from flamingo_stitcher.writers.ome_zarr_writer import (
                    write_ome_zarr_v2,
                )

                write_ome_zarr_v2(
                    data=fused,
                    output_path=out_path,
                    voxel_size_um=voxel_size_um,
                    chunks=self.config.zarr_chunks,
                    compression=self.config.zarr_compression,
                    compression_level=self.config.zarr_compression_level,
                    pyramid_levels=self.config.pyramid_levels,
                )
            except Exception as e:
                self.logger.error(f"  OME-Zarr v2 write failed: {e}")
                self.logger.info("  Falling back to OME-TIFF")
                fmt = "ome-tiff"

        if fmt in ("ome-tiff", "both"):
            tiff_path = output_dir / f"channel_{channel_id:02d}_stitched.ome.tif"
            self.logger.info(f"  Writing pyramidal OME-TIFF: {tiff_path}")
            try:
                from flamingo_stitcher.writers.ome_tiff_writer import (
                    write_pyramidal_ome_tiff,
                )

                write_pyramidal_ome_tiff(
                    data=fused,
                    output_path=tiff_path,
                    voxel_size_um=voxel_size_um,
                    tile_size=self.config.tiff_tile_size,
                    compression=self.config.tiff_compression,
                    pyramid_levels=(
                        self.config.pyramid_levels if self.config.tiff_pyramids else 0
                    ),
                )
                if fmt == "ome-tiff":
                    out_path = tiff_path
            except ImportError as e:
                self.logger.error(f"  OME-TIFF write failed: {e}")
                self.logger.info("  Falling back to flat TIFF")
                tiff_path = output_dir / f"channel_{channel_id:02d}_stitched.tif"
                self._save_as_tiff(fused, tiff_path)
                out_path = tiff_path

        elif fmt == "ome-zarr":
            out_path = output_dir / f"channel_{channel_id:02d}.zarr"
            self.logger.info(f"  Writing OME-Zarr: {out_path}")
            try:
                from multiview_stitcher import ngff_utils

                ngff_utils.write_sim_to_ome_zarr(
                    fused,
                    str(out_path),
                    overwrite=True,
                )
            except Exception as e:
                self.logger.error(f"  OME-Zarr write failed: {e}, falling back to TIFF")
                out_path = output_dir / f"channel_{channel_id:02d}_stitched.tif"
                self._save_as_tiff(fused, out_path)

        elif fmt == "tiff":
            out_path = output_dir / f"channel_{channel_id:02d}_stitched.tif"
            self.logger.info(f"  Writing TIFF: {out_path}")
            self._save_as_tiff(fused, out_path)

        return out_path

    def _write_stitch_metadata(
        self,
        output_dir: Path,
        results: Dict[int, Tuple[Path, Dict[str, float]]],
        tiles: List[RawTileInfo],
        voxel_size_um: Dict[str, float],
        acquisition_dir: Path,
    ) -> None:
        """Write stitch_metadata.json sidecar with world origin per channel."""
        channels_meta = {}
        for ch_id, (ch_path, origin_um) in results.items():
            channels_meta[str(ch_id)] = {
                "path": ch_path.name,
                "origin_um": [origin_um["z"], origin_um["y"], origin_um["x"]],
            }

        metadata = {
            **stitcher_provenance(),
            "source_acquisition": str(acquisition_dir),
            "voxel_size_um": voxel_size_um,
            "downsample_xy": self.config.downsample_xy,
            "downsample_z": self.config.downsample_z,
            "output_format": self.config.output_format,
            "channels": channels_meta,
            "tile_count": len(tiles),
        }

        meta_path = output_dir / "stitch_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        self.logger.info(f"  Wrote {meta_path}")

    def _save_as_tiff(self, sim, path: Path) -> None:
        """Save a SpatialImage to TIFF via tifffile."""
        import dask.diagnostics

        try:
            from multiview_stitcher import io as mvs_io

            with dask.diagnostics.ProgressBar():
                mvs_io.save_sim_as_tif(str(path), sim)
        except Exception:
            # Fallback: compute dask array and save directly
            import dask.diagnostics
            import tifffile

            self.logger.info("  Computing fused volume into memory...")
            with dask.diagnostics.ProgressBar():
                data = sim.data.compute()
            tifffile.imwrite(str(path), data)

    def _deconvolve_tile(self, volume: np.ndarray, tile: RawTileInfo) -> np.ndarray:
        """Apply GPU deconvolution to a single tile."""
        try:
            from flamingo_stitcher.deconvolution import (
                DeconvolutionConfig,
                deconvolve_tile,
            )

            decon_config = DeconvolutionConfig(
                enabled=True,
                engine=self.config.deconvolution_engine,
                num_iterations=self.config.deconvolution_iterations,
                na=self.config.deconvolution_na,
                wavelength_nm=self.config.deconvolution_wavelength_nm,
                n_immersion=self.config.deconvolution_n_immersion,
                psf_path=self.config.deconvolution_psf_path,
            )

            z_step_um = self.config.z_step_um
            if z_step_um is None:
                z_step_um = tile.z_step_mm * 1000.0

            self.logger.info(
                f"    Deconvolving ({self.config.deconvolution_engine}, "
                f"{self.config.deconvolution_iterations} iterations)..."
            )
            return deconvolve_tile(
                volume, decon_config, self.config.pixel_size_um, z_step_um
            )
        except ImportError as e:
            self.logger.warning(f"    Deconvolution skipped: {e}")
            return volume
        except Exception as e:
            self.logger.error(f"    Deconvolution failed: {e}")
            return volume

    def _log_tile_summary(self, tiles: List[RawTileInfo]) -> None:
        """Log a summary of discovered tiles."""
        xs = sorted(set(t.x_mm for t in tiles))
        ys = sorted(set(t.y_mm for t in tiles))
        all_ch = sorted(set(ch for t in tiles for ch in t.channels))
        all_illum = sorted(set(il for t in tiles for il in t.illumination_sides))

        self.logger.info(f"  {len(tiles)} tiles in ~{len(xs)}x{len(ys)} grid")
        self.logger.info(
            f"  X range: {min(xs):.2f} – {max(xs):.2f} mm  "
            f"Y range: {min(ys):.2f} – {max(ys):.2f} mm"
        )
        self.logger.info(f"  Channels: {describe_channel_set(sorted(all_ch))}")
        self.logger.info(f"  Illumination side list: {all_illum}")
        # Depth is per-tile: an acquisition with per-tile Z ranges images only
        # the span where the sample is, so plane counts differ. Report the
        # spread, not tile 0 — logging one tile's depth as "planes per tile"
        # is what let a 97-tile run die at tile 7 with nothing in the header
        # hinting the tiles were ever different sizes.
        plane_counts = [int(t.n_planes) for t in tiles]
        z_lo = min(t.z_min_mm for t in tiles)
        z_hi = max(t.z_max_mm for t in tiles)
        if min(plane_counts) == max(plane_counts):
            self.logger.info(
                f"  Planes per tile: {plane_counts[0]} "
                f"(Z range: {z_lo:.3f} – {z_hi:.3f} mm)"
            )
        else:
            self.logger.info(
                f"  Planes per tile: {min(plane_counts)}–{max(plane_counts)} "
                f"(varies — per-tile Z ranges; overall Z {z_lo:.3f} – "
                f"{z_hi:.3f} mm)"
            )

        # A single voxel size is applied to every tile, so the Z STEP has to
        # match even though the depth need not. Differing steps would place
        # each tile's planes at the wrong physical spacing — silently, as a
        # Z-direction stretch — so say so rather than fusing nonsense.
        # Judge the spread RELATIVE to the step, not in absolute mm. Each tile's
        # step is derived as (z_max - z_min) / (n_planes - 1) from bounds the
        # acquisition rounds, so a spread of a few nm is arithmetic noise, not a
        # different step: a real 97-tile run showed 5.0022–5.0047 µm, which is
        # 0.05% — under one plane of drift across even the deepest tile. An
        # actual step change (5 vs 8 µm) is tens of percent.
        steps = [t.z_step_mm for t in tiles if int(t.n_planes) > 1]
        if steps:
            lo, hi = min(steps), max(steps)
            median_step = sorted(steps)[len(steps) // 2]
            spread_frac = (hi - lo) / median_step if median_step > 0 else 0.0
            if spread_frac > Z_STEP_SPREAD_WARN_FRAC:
                drift_planes = (hi - lo) * max(plane_counts) / median_step
                self.logger.warning(
                    f"  ⚠ Z step differs between tiles "
                    f"({lo * 1000:.4f}–{hi * 1000:.4f} µm, "
                    f"{spread_frac * 100:.1f}%). Fusion applies ONE Z voxel "
                    f"size to every tile, so tiles acquired at a different "
                    f"step are stretched or squashed along Z — up to "
                    f"~{drift_planes:.0f} planes of drift across the deepest "
                    f"tile. Re-acquire with a single Z step."
                )

        # On-disk input size: sum of every raw file across tiles, channels,
        # and illumination sides. Log both a per-tile average and the total
        # so the user sees what the pipeline has to read end-to-end.
        total_bytes = 0
        counted = 0
        missing = 0
        for t in tiles:
            for ch_map in t.raw_files.values():
                for raw_path in ch_map.values():
                    try:
                        total_bytes += raw_path.stat().st_size
                        counted += 1
                    except OSError:
                        missing += 1
        if counted:
            total_gb = total_bytes / (1024**3)
            avg_tile_mb = (total_bytes / counted) / (1024**2)
            n_ch = max(len(all_ch), 1)
            n_illum = max(len(all_illum), 1)
            msg = (
                f"  Input data on disk: ~{total_gb:.1f} GB across {counted} raw files "
                f"(avg {avg_tile_mb:.0f} MB/file, "
                f"{len(tiles)} tiles × {n_ch} ch × {n_illum} illum)"
            )
            if missing:
                msg += f"  [warning: {missing} files could not be stat'd]"
            self.logger.info(msg)

    def _apply_and_log_geometry(
        self, tiles: List[RawTileInfo], acquisition_dir: Path
    ) -> None:
        """Resolve frame size, verify pixel size against the objective, and warn
        about partial-capture / multi-angle acquisitions.

        Runs once at the start of run(), after tiles are known. Mutates tile
        frame dims only when the user forced an override in the config.
        """
        # 1) Frame size (camera AOI). A manual config override wins over the
        #    auto-detected per-tile dims (which already prefer the file size).
        if self.config.frame_width and self.config.frame_height:
            for t in tiles:
                t.frame_width = int(self.config.frame_width)
                t.frame_height = int(self.config.frame_height)
            self.logger.info(
                f"Frame size (AOI): {self.config.frame_width}×"
                f"{self.config.frame_height} px (manual override)"
            )
        else:
            fw, fh = _resolve_frame_size(tiles)
            distinct = sorted({(t.frame_width, t.frame_height) for t in tiles})
            if len(distinct) > 1:
                self.logger.warning(
                    f"Frame size (AOI): tiles disagree {distinct}; using {fw}×{fh}. "
                    f"Check for mixed acquisitions."
                )
            else:
                self.logger.info(f"Frame size (AOI): {fw}×{fh} px (from data)")
            if (fw, fh) != (FRAME_WIDTH, FRAME_HEIGHT):
                self.logger.info(
                    f"  (differs from hardware-config default "
                    f"{FRAME_WIDTH}×{FRAME_HEIGHT} — cropped/binned acquisition)"
                )

        # 2) Pixel size — derive it per-acquisition from this acquisition's own
        #    objective when in auto mode; otherwise sanity-check the configured
        #    value against it. Runs before the voxel-size / fusion math below, so
        #    a batch mixing objectives renders each item at the right scale.
        try:
            mag = read_objective_magnification(acquisition_dir)
            suggested = suggested_pixel_size_um(acquisition_dir)
            if mag and suggested:
                self.logger.info(
                    f"Objective (ScopeSettings.txt): {mag:.3f}× → effective pixel "
                    f"~{suggested:.3f} µm (sensor {_sensor_pixel_size_um():.2f} µm)"
                )
            # Cross-check: ScopeSettings.txt vs FlamingoMetaData*.txt objective.
            # These are two independent records of the same objective; when they
            # disagree, one is stale and every tile is rendered at the wrong
            # scale — producing mis-registered "ghost" duplicates of the same
            # feature that grow with distance from each tile centre. Warn loudly
            # so it is caught at discovery, not after a full stitch.
            meta_mag = read_objective_magnification_metadata(acquisition_dir)
            if mag and meta_mag and abs(mag - meta_mag) / max(mag, meta_mag) > 0.02:
                sensor = _sensor_pixel_size_um()
                self.logger.warning(
                    f"⚠ Objective magnification DISAGREES between metadata files: "
                    f"ScopeSettings.txt = {mag:.3g}× (→ {sensor / mag:.4f} µm/px, "
                    f"used for stitching) vs FlamingoMetaData = {meta_mag:.3g}× "
                    f"(→ {sensor / meta_mag:.4f} µm/px). One is stale. If the wrong "
                    f"one is used, tiles are placed by stage position but rendered "
                    f"at the wrong scale → duplicated/ghosted objects. Set the "
                    f"correct objective (or override the XY pixel size) before "
                    f"trusting this stitch."
                )
            if getattr(self.config, "auto_pixel_size", False):
                # Per-entry: this acquisition uses the pixel size implied by ITS
                # OWN objective, independent of any other queued item.
                if suggested and suggested > 0:
                    prev = self.config.pixel_size_um
                    self.config.pixel_size_um = round(suggested, 4)
                    self.logger.info(
                        f"Auto XY pixel size (per-acquisition): "
                        f"{self.config.pixel_size_um:.4f} µm from objective "
                        f"{mag:.3f}× (fallback was {prev:.4f} µm)."
                    )
                else:
                    self.logger.warning(
                        f"Auto XY pixel size requested but no objective was found "
                        f"in this acquisition's ScopeSettings.txt; falling back to "
                        f"{self.config.pixel_size_um:.4f} µm. Set the pixel size "
                        f"manually if that is wrong."
                    )
            elif mag and suggested:
                cur = self.config.pixel_size_um
                if cur > 0 and abs(cur - suggested) / suggested > 0.15:
                    self.logger.warning(
                        f"Configured XY pixel size {cur:.3f} µm differs from the "
                        f"objective-derived ~{suggested:.3f} µm by "
                        f"{abs(cur - suggested) / suggested * 100:.0f}%. If the "
                        f"objective changed, tiles will be placed by stage spacing "
                        f"but rendered at the wrong scale (gaps/overlap). Verify the "
                        f"XY pixel size. [Note: a stale value may reflect the known "
                        f"nominal-vs-system magnification ambiguity.]"
                    )
        except Exception as e:  # never let diagnostics break a run
            self.logger.debug(f"Objective/pixel-size check skipped: {e}")

        # 2b) Tile-spacing sanity: warn if tiles are stepped farther apart than
        #     one frame covers (blank gaps = missing acquired data). Runs after
        #     the pixel size is finalized above so the coverage math is correct.
        #     Catches the non-square-AOI-stepped-as-square failure the server's
        #     Tile expansion can produce (full-width tiles, black bands between
        #     rows). Faithfully rendering that is correct — the gaps are real —
        #     so the value here is naming the cause instead of a head-scratch.
        try:
            gap_fw, gap_fh = _resolve_frame_size(tiles, self.config)
            for msg in _detect_tile_spacing_gaps(
                tiles, gap_fw, gap_fh, self.config.pixel_size_um
            ):
                self.logger.warning(f"⚠ {msg}")
        except Exception as e:  # never let diagnostics break a run
            self.logger.debug(f"Tile-spacing gap check skipped: {e}")

        # 3) Tile orientation. In auto mode, resolve THIS acquisition's
        #    orientation from its own microscope name (user preset > bundled
        #    YAML), so a batch mixing systems — or a stale choice from another
        #    scope — can't mis-orient this item. A microscope with NO chosen
        #    orientation is refused (no guessed default): OrientationUnknownError
        #    propagates so the run stops with clear guidance instead of silently
        #    producing a broken (non-connecting) stitch.
        if getattr(self.config, "auto_tile_orientation", False):
            from flamingo_stitcher.orientation import (
                OrientationUnknownError,
                has_orientation_preview_data,
                read_microscope_name,
                resolve_tile_orientation,
            )

            ori = resolve_tile_orientation(acquisition_dir)
            if ori is not None and ori.name:
                self.config.tile_orientation = ori.name
                self.config.reverse_x_tiles = bool(ori.reverse_x)
                self.config.reverse_y_tiles = bool(ori.reverse_y)
            else:
                scope = read_microscope_name(acquisition_dir) or "(unnamed)"
                if has_orientation_preview_data(acquisition_dir):
                    raise OrientationUnknownError(
                        f"No tile orientation is set for microscope '{scope}'. "
                        f"Each microscope needs its orientation chosen once: open "
                        f"the Orientation Preview, pick the panel where tiles "
                        f"connect, and click 'Use for stitching' (GUI) or pass "
                        f"--tile-orientation (CLI). Refusing to stitch with a "
                        f"guessed orientation."
                    )
                raise OrientationUnknownError(
                    f"Tile orientation is unknown for microscope '{scope}', and "
                    f"this dataset has no MIPs to determine it from. Use a dataset "
                    f"that includes per-tile MIPs (*_MP.tif) or readable raw "
                    f"stacks for '{scope}' to choose the orientation, then re-run."
                )

        # Log the effective orientation (always — otherwise it's invisible).
        try:
            effective = self.config.tile_orientation or (
                "flip_h" if self.config.camera_x_inverted else "identity"
            )
            revs = []
            if self.config.reverse_x_tiles:
                revs.append("reverse-X order")
            if self.config.reverse_y_tiles:
                revs.append("reverse-Y order")
            self.logger.info(
                f"Tile orientation: {effective}"
                + (f" + {', '.join(revs)}" if revs else "")
                + (" (auto, per-acquisition)" if self.config.auto_tile_orientation else "")
            )
        except Exception as e:  # never let the log line break a run
            self.logger.debug(f"Tile-orientation log skipped: {e}")

        # 3) Partial-capture & multi-angle flags.
        try:
            wf = acquisition_dir / "Workflow.txt"
            if not wf.exists():
                wf = tiles[0].folder / "Workflow.txt"
            flags = _read_capture_and_angles(wf if wf.exists() else None)
            modes = flags.get("capture_modes") or []
            pcts = flags.get("capture_percents") or []
            if any(m not in (0, 3) for m in modes) or any(p < 100 for p in pcts):
                self.logger.warning(
                    f"Partial Z-capture detected (capture modes={modes}, "
                    f"percentages={pcts}). Saved planes are a sub-range of the full "
                    f"stack; 'from back' (mode 2) captures shift the Z origin. Tiles "
                    f"are placed from the Start-Position Z, so verify Z alignment."
                )
            if int(flags.get("n_angles", 1) or 1) > 1:
                if self.config.multiview_fusion:
                    self.logger.info(
                        f"Multi-angle acquisition (Number of angles="
                        f"{flags.get('n_angles')}). Multi-view fusion is ON — "
                        f"views are placed by rotation about the Y axis."
                    )
                else:
                    self.logger.warning(
                        f"Multi-angle acquisition (Number of angles="
                        f"{flags.get('n_angles')}). Multi-view fusion is OFF — "
                        f"rotation between angles is not applied. Enable "
                        f"multi-view fusion (--multiview) to fuse the angles into "
                        f"one frame."
                    )
        except Exception as e:
            self.logger.debug(f"Capture/angle check skipped: {e}")

    def _log_preflight(
        self,
        tiles: List[RawTileInfo],
        channels: List[int],
        output_path: Path,
        mem_est: Dict[str, float],
        use_streaming: bool,
    ) -> None:
        """Log RAM/disk headroom and format-specific warnings before the run.

        ``mem_est`` comes from :func:`estimate_memory_usage` and only covers
        our own graph. Writers (PyImarisWriter especially) add their own
        overhead, and temp-spill memmaps can eat the output drive — both
        are surfaced here so the user sees them in the log instead of
        getting a mid-run OOM or ENOSPC.
        """
        try:
            import shutil as _shutil

            import psutil as _psutil
        except ImportError:
            _psutil = None
            _shutil = None

        # --- System RAM ---
        if _psutil is not None:
            sys_ram_gb = _psutil.virtual_memory().total / (1024**3)
            avail_ram_gb = _psutil.virtual_memory().available / (1024**3)
            mode = "streaming" if use_streaming else "in-memory"
            peak = mem_est["streaming_gb" if use_streaming else "in_memory_gb"]
            self.logger.info(
                f"System RAM: {sys_ram_gb:.0f} GB total, "
                f"{avail_ram_gb:.0f} GB available; "
                f"projected peak {peak:.1f} GB ({mode})"
            )
            if peak > avail_ram_gb * 0.9:
                self.logger.warning(
                    f"  [warning] projected peak ({peak:.1f} GB) is close to or "
                    f"exceeds available RAM ({avail_ram_gb:.1f} GB). "
                    f"Consider Streaming mode or increasing downsample."
                )

        # --- Content-based fusion cost warning ---
        # Two NaN Gaussian filters per output chunk on float32 scale
        # super-linearly with tile overlap + tile count. Real-world
        # datasets (66-tile 750 GB acq) hit ~5-8 hours of fuse time
        # with threads×4; the default cosine blending is 5-10× faster.
        # Only actually used when blending overlaps; max-mode fusion
        # (max_fusion) has no weights kwarg and skips content weights entirely,
        # so announcing it there is misleading (and it costs no extra memory).
        if self.config.content_based_fusion and self.config.tile_overlap_fusion != "max":
            self.logger.warning(
                "  [info] Content-based fusion is enabled. Expect "
                "the fuse-store step to be significantly slower (often "
                "5-10× the default cosine blending). Disable under "
                "Processing Options → 'Content-based blending' if "
                "wall-clock matters more than per-overlap blending polish."
            )

        # --- Format-specific writer overhead ---
        fmt = self.config.output_format
        output_gb = mem_est["output_gb"]
        if fmt == "imaris":
            # PyImarisWriter keeps its own per-block scratch + HDF5 cache +
            # pyramid working set. Empirically ~25% of the uncompressed
            # output size on top of our own peak. Block reads are file I/O
            # from the fused memmap, not dask graph recomputes, so write
            # time is roughly output_gb ÷ writer_throughput.
            ims_overhead_gb = output_gb * 0.25
            self.logger.info(
                f"Imaris writer overhead: ~{ims_overhead_gb:.0f} GB "
                f"(PyImarisWriter block cache + pyramid buffers, "
                f"added on top of pipeline peak)"
            )
            if _psutil is not None:
                combined = peak + ims_overhead_gb
                if combined > avail_ram_gb * 0.9:
                    self.logger.warning(
                        f"  [warning] Imaris write may OOM: pipeline peak "
                        f"({peak:.0f} GB) + writer overhead "
                        f"(~{ims_overhead_gb:.0f} GB) vs "
                        f"{avail_ram_gb:.0f} GB available. "
                        f"Consider exporting OME-TIFF first, then using "
                        f"ImarisFileConverter."
                    )
        elif fmt == "ome-zarr-v2":
            self.logger.info(
                "OME-Zarr v2 writer reads pyramid levels from the fused "
                "memmap (not fully lazy). Level 0 roughly matches "
                f"output_gb (~{output_gb:.0f} GB) during write."
            )

        # --- Disk free space ---
        if _shutil is None:
            return
        output_path.mkdir(parents=True, exist_ok=True)

        # Estimated temp spill (per-tile memmaps for streaming mode) and
        # final output footprint.
        bpv = 2
        ds_xy = max(self.config.downsample_xy, 1)
        ds_z = max(self.config.downsample_z, 1)
        n_planes = max(t.n_planes for t in tiles)
        frame_w, frame_h = _resolve_frame_size(tiles, self.config)
        tile_bytes = (
            (n_planes // ds_z if ds_z > 1 else n_planes)
            * (frame_w // ds_xy if ds_xy > 1 else frame_w)
            * (frame_h // ds_xy if ds_xy > 1 else frame_h)
            * bpv
        )
        # Streaming spills one channel of tiles at a time, then fuses
        # the full (C,Z,Y,X) stack to a second memmap on disk before the
        # writer runs. Both live simultaneously for most of the run.
        spill_gb = (len(tiles) * tile_bytes / (1024**3)) if use_streaming else 0.0
        fused_memmap_gb = output_gb if use_streaming else 0.0
        scratch_gb = spill_gb + fused_memmap_gb  # lives under .stitch_tmp

        # Scratch (temp) lives at config.scratch_dir when set, else next to the
        # output. Report/guard the correct drive for each portion.
        scratch_base = _scratch_base_dir(self.config, output_path)
        scratch_set = getattr(self.config, "scratch_dir", None) is not None
        try:
            scratch_base.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        # `is not True`: when the volume check is undeterminable (None, e.g. a
        # stat error) but a scratch dir was explicitly set, assume it IS separate
        # so the scratch drive still gets a free-space check rather than silently
        # falling back to charging everything to the output drive.
        separate_scratch = (
            scratch_set and _same_volume(scratch_base, output_path) is not True
        )

        try:
            out_free_gb = _shutil.disk_usage(output_path).free / (1024**3)
            if separate_scratch:
                # Only the final output lands on the output drive; the tile spill
                # + fused memmap go to the (separate) scratch drive.
                self.logger.info(
                    f"Output drive ({output_path}): {out_free_gb:.0f} GB free, "
                    f"need ~{output_gb:.1f} GB output"
                )
                if out_free_gb < output_gb * 1.1:
                    self.logger.warning(
                        f"  [warning] output drive may run out of space: need "
                        f"~{output_gb:.0f} GB, {out_free_gb:.0f} GB free."
                    )
                try:
                    scr_free_gb = _shutil.disk_usage(scratch_base).free / (1024**3)
                    self.logger.info(
                        f"Scratch drive ({scratch_base}): {scr_free_gb:.0f} GB free, "
                        f"need ~{scratch_gb:.1f} GB (tile spill + fused memmap; "
                        f"deleted at the end)"
                    )
                    if scratch_gb > 0 and scr_free_gb < scratch_gb * 1.1:
                        self.logger.warning(
                            f"  [warning] scratch drive may run out of space: need "
                            f"~{scratch_gb:.0f} GB, {scr_free_gb:.0f} GB free. Point "
                            f"the scratch dir at a larger fast drive."
                        )
                except OSError as e:
                    self.logger.debug(f"Could not probe scratch drive: {e}")
            else:
                # Single drive: output + spill + fused memmap all together.
                needed_gb = output_gb + scratch_gb
                parts = [f"~{output_gb:.1f} GB output"]
                if spill_gb > 0:
                    parts.append(f"~{spill_gb:.1f} GB tile spill")
                if fused_memmap_gb > 0:
                    parts.append(f"~{fused_memmap_gb:.1f} GB fused memmap")
                self.logger.info(
                    f"Output drive ({output_path}): {out_free_gb:.0f} GB free, "
                    f"need {' + '.join(parts)}"
                )
                if scratch_set and scratch_gb > 0:
                    # Scratch requested but on the SAME drive as the output —
                    # it won't relieve the I/O contention it's meant to fix.
                    self.logger.warning(
                        "  [warning] scratch dir is on the same drive as the "
                        "output — no I/O benefit. Put it on a separate fast "
                        "local disk (NVMe/SSD)."
                    )
                if out_free_gb < needed_gb * 1.1:
                    self.logger.warning(
                        f"  [warning] output drive may run out of space: "
                        f"need ~{needed_gb:.0f} GB, {out_free_gb:.0f} GB free. "
                        f"Free up space or point output to a larger drive."
                    )
        except OSError as e:
            self.logger.debug(f"Could not probe output drive free space: {e}")

    def _enforce_resource_limits(
        self,
        tiles: List[RawTileInfo],
        output_path: Path,
        mem_est: Dict[str, float],
        use_streaming: bool,
    ) -> None:
        """Abort this item before any allocation if it would exhaust RAM/disk.

        Raises ``RuntimeError`` (caught per-item by the batch worker, so the
        rest of the queue survives) when the projected peak RAM for the chosen
        mode — plus writer overhead — exceeds ``resource_guard_ram_fraction`` of
        AVAILABLE RAM, or the output + tile-spill + fused-memmap footprint
        exceeds ``resource_guard_disk_fraction`` of free space on the output
        drive. The measurement is skipped (not failed) when psutil/shutil are
        unavailable.
        """
        if not getattr(self.config, "resource_guard_enabled", True):
            return
        try:
            import shutil as _shutil

            import psutil as _psutil
        except ImportError:
            return  # cannot measure — let the run proceed as before

        ram_frac = float(getattr(self.config, "resource_guard_ram_fraction", 0.95))
        disk_frac = float(getattr(self.config, "resource_guard_disk_fraction", 0.95))
        output_gb = float(mem_est.get("output_gb", 0.0))

        # --- RAM ---
        peak_gb = float(
            mem_est.get("streaming_gb" if use_streaming else "in_memory_gb", 0.0)
        )
        if self.config.output_format == "imaris":
            peak_gb += output_gb * 0.25  # PyImarisWriter scratch (see _log_preflight)
        try:
            avail_ram_gb = _psutil.virtual_memory().available / (1024**3)
        except Exception:
            avail_ram_gb = 0.0
        if avail_ram_gb > 0 and peak_gb > avail_ram_gb * ram_frac:
            mode = "streaming" if use_streaming else "in-memory"
            raise RuntimeError(
                f"Aborting before run: projected peak RAM ~{peak_gb:.0f} GB "
                f"({mode} mode) exceeds the safety limit "
                f"{ram_frac * 100:.0f}% of {avail_ram_gb:.0f} GB available. "
                f"Remedies: increase downsample (XY/Z), force Streaming mode, "
                f"split the acquisition, close other apps, or — if the XY pixel "
                f"size looks wrong (too small → giant output), fix it. Override "
                f"with resource_guard_enabled=False to force the run."
            )

        # --- Disk (output + spill + fused memmap, mirrors _log_preflight) ---
        try:
            bpv = 2
            ds_xy = max(self.config.downsample_xy, 1)
            ds_z = max(self.config.downsample_z, 1)
            n_planes = max(t.n_planes for t in tiles)
            frame_w, frame_h = _resolve_frame_size(tiles, self.config)
            tile_bytes = (
                (n_planes // ds_z if ds_z > 1 else n_planes)
                * (frame_w // ds_xy if ds_xy > 1 else frame_w)
                * (frame_h // ds_xy if ds_xy > 1 else frame_h)
                * bpv
            )
            spill_gb = (len(tiles) * tile_bytes / (1024**3)) if use_streaming else 0.0
            fused_memmap_gb = output_gb if use_streaming else 0.0
            scratch_gb = spill_gb + fused_memmap_gb  # under .stitch_tmp

            scratch_base = _scratch_base_dir(self.config, output_path)
            scratch_set = getattr(self.config, "scratch_dir", None) is not None
            try:
                scratch_base.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            # is not True: undeterminable + explicit scratch dir => assume separate
            # so the scratch drive is guarded (see _log_preflight for rationale).
            separate = (
                scratch_set and _same_volume(scratch_base, output_path) is not True
            )

            output_path.mkdir(parents=True, exist_ok=True)
            # Output drive: final output always; spill+fused too UNLESS scratch is
            # on a separate drive.
            out_need = output_gb + (0.0 if separate else scratch_gb)
            out_free_gb = _shutil.disk_usage(output_path).free / (1024**3)
            if out_free_gb > 0 and out_need > out_free_gb * disk_frac:
                raise RuntimeError(
                    f"Aborting before run: needs ~{out_need:.0f} GB on the output "
                    f"drive ({output_gb:.0f} GB output"
                    + (
                        f" + {spill_gb:.0f} GB spill + {fused_memmap_gb:.0f} GB "
                        f"fused memmap"
                        if use_streaming and not separate
                        else ""
                    )
                    + f") but only {out_free_gb:.0f} GB free (limit "
                    f"{disk_frac * 100:.0f}%). Remedies: point output to a larger "
                    f"drive, increase downsample, or fix the XY pixel size if the "
                    f"output looks far too large. Override with "
                    f"resource_guard_enabled=False."
                )
            # Separate scratch drive: guard it for the spill + fused memmap.
            if separate and scratch_gb > 0:
                scr_free_gb = _shutil.disk_usage(scratch_base).free / (1024**3)
                if scr_free_gb > 0 and scratch_gb > scr_free_gb * disk_frac:
                    raise RuntimeError(
                        f"Aborting before run: scratch dir needs ~{scratch_gb:.0f} GB "
                        f"({spill_gb:.0f} GB spill + {fused_memmap_gb:.0f} GB fused "
                        f"memmap) on {scratch_base} but only {scr_free_gb:.0f} GB free "
                        f"(limit {disk_frac * 100:.0f}%). Point the scratch dir at a "
                        f"larger fast drive, increase downsample, or unset it. "
                        f"Override with resource_guard_enabled=False."
                    )
        except OSError as e:
            self.logger.debug(f"Disk guard probe failed (allowing run): {e}")
