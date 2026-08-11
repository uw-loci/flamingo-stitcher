"""What registration actually did, per tile and per seam.

Registration is the one step in this pipeline whose effect is invisible. It
either silently improved the mosaic or silently made it worse, and the only
evidence a whole run left behind was a single clamp summary line. So when tiles
did not line up there was no way to tell whether phase correlation had found the
error and been overruled, found nothing, or never run at all — which is how
registration ended up switched off for months without anyone being able to say
what turning it back on would cost.

This module turns the affine params and the pairwise registration graph into
three files written next to the stitched output:

  ``registration_report.csv``  one row per tile: the correction applied, in µm
                               and in pixels/frames, plus per-axis clamp flags.
  ``registration_seams.csv``   one row per EXPECTED neighbour pair, taken from
                               the stage grid — not per surviving graph edge —
                               so a pair registration rejected appears as a row
                               saying so rather than as a silent absence.
  ``registration_report.txt``  the human summary, echoed into the run log.

**Honesty rules**, learned from border QC reporting a clamped search limit as if
it were a measurement (57% of flagged seams, in the end):

  * A clamped axis was **not measured**. It is never printed as "moved by the
    bound", it is excluded from the summary statistics, and the clamped count is
    in the header rather than buried in a column.
  * An unknown number is an **empty cell**. Never ``0``, never ``nan`` — a
    quality of 0.0 and a quality nobody could recover are different facts.
  * A refinement value at its search limit is a **floor**, and says so.
  * A run where registration did not happen still writes all three files, naming
    the reason. The absence of a file is ambiguous; a file saying "skipped,
    overlap was 3%" is not.

Pure: numpy, stdlib, and two leaf modules (`border_qc.find_neighbor_pairs`,
`tile_geometry`). No pipeline import and no multiview-stitcher import — the
registration graph is read structurally through ``.edges`` and
``.get_edge_data``, so a unit test can hand it a ``SimpleNamespace``.
"""

from __future__ import annotations

import contextlib
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from flamingo_stitcher import tile_geometry
from flamingo_stitcher.border_qc import find_neighbor_pairs

__all__ = [
    "RegistrationReport",
    "SeamResult",
    "TileShift",
    "ZRefineSummary",
    "SEAM_CSV_HEADER",
    "TILE_CSV_HEADER",
    "STATUS_BELOW_QUALITY",
    "STATUS_DROPPED",
    "STATUS_NOT_RUN",
    "STATUS_PRUNED",
    "STATUS_REGISTERED",
    "build_report",
    "capture_prefilter_graph",
    "extract_seams",
    "format_report_text",
    "report_to_json",
    "seam_rows_csv",
    "skipped_report",
    "tile_rows_csv",
    "translation_from_param",
    "write_report",
]

REPORT_VERSION = 1

TILE_CSV_NAME = "registration_report.csv"
SEAM_CSV_NAME = "registration_seams.csv"
TEXT_NAME = "registration_report.txt"
JSON_NAME = "registration_report.json"

# Seam outcomes. Ordered from "we used this" to "we never got a number".
STATUS_REGISTERED = "registered"  # edge survived quality AND fed the global solve
STATUS_PRUNED = "pruned"  # survived quality, dropped by edge pruning
STATUS_BELOW_QUALITY = "below_quality"  # correlation below the threshold
STATUS_DROPPED = "dropped"  # expected pair, no edge, reason unrecoverable
STATUS_NOT_RUN = "not_run"  # registration skipped / gated off / failed

TILE_CSV_HEADER: Tuple[str, ...] = (
    "tile_index",
    "tile_name",
    "x_mm",
    "y_mm",
    "z_min_mm",
    "dz_um",
    "dy_um",
    "dx_um",
    "dz_frames",
    "dy_px",
    "dx_px",
    "clamped_z",
    "clamped_y",
    "clamped_x",
    "dz_um_before_clamp",
    "dy_um_before_clamp",
    "dx_um_before_clamp",
    "note",
)

SEAM_CSV_HEADER: Tuple[str, ...] = (
    "tile_a",
    "tile_b",
    "index_a",
    "index_b",
    "axis",
    "status",
    "quality",
    "dz_um",
    "dy_um",
    "dx_um",
    "dz_frames",
    "dy_px",
    "dx_px",
    "residual_px",
    "overlap_frac",
    "note",
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class TileShift:
    """One tile's net correction from its stage position to where it was placed."""

    index: int
    name: str
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_min_mm: float = 0.0
    dz_um: float = 0.0
    dy_um: float = 0.0
    dx_um: float = 0.0
    # The same shifts in the units a person can act on: Z in processing frames
    # (post-downsample planes), Y/X in processing pixels.
    dz_frames: float = 0.0
    dy_px: float = 0.0
    dx_px: float = 0.0
    # True when the clamp reverted this axis. NOT "moved by the bound" — the
    # tile kept its stage position and the real offset is unknown.
    clamped_z: bool = False
    clamped_y: bool = False
    clamped_x: bool = False
    # What registration proposed before the clamp overruled it, when it did.
    shift_before_clamp_um: Optional[Tuple[float, float, float]] = None  # (z, y, x)
    note: str = ""

    @property
    def any_clamped(self) -> bool:
        return self.clamped_z or self.clamped_y or self.clamped_x


@dataclass
class SeamResult:
    """One expected neighbour pair, and what registration made of it."""

    tile_a: str
    tile_b: str
    index_a: int
    index_b: int
    axis: str
    status: str
    quality: Optional[float] = None
    # The PAIRWISE shift for this seam, not either tile's global correction.
    dz_um: Optional[float] = None
    dy_um: Optional[float] = None
    dx_um: Optional[float] = None
    dz_frames: Optional[float] = None
    dy_px: Optional[float] = None
    dx_px: Optional[float] = None
    residual_px: Optional[float] = None
    overlap_frac: Optional[float] = None
    note: str = ""


@dataclass
class ZRefineSummary:
    """Outcome of the optional dedicated Z-refinement pass."""

    ran: bool = False
    reason: str = ""
    binning: Dict[str, int] = field(default_factory=dict)
    upsample_factor: int = 0
    search_range_um: float = 0.0
    n_tiles_moved: int = 0
    max_abs_dz_um: float = 0.0
    median_abs_dz_um: float = 0.0
    # Corrections that came back at the search limit. These are floors, not
    # measurements, and are rejected rather than applied.
    n_hit_search_limit: int = 0


@dataclass
class RegistrationReport:
    ran: bool = False
    reason: str = ""
    transform_key: str = ""
    tiles: List[TileShift] = field(default_factory=list)
    seams: List[SeamResult] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    z_refine: Optional[ZRefineSummary] = None
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def n_expected_pairs(self) -> int:
        return len(self.seams)

    def count(self, status: str) -> int:
        return sum(1 for s in self.seams if s.status == status)

    @property
    def n_tiles_clamped(self) -> int:
        return sum(1 for t in self.tiles if t.any_clamped)


# ---------------------------------------------------------------------------
# Reading multiview-stitcher's output, defensively
# ---------------------------------------------------------------------------


def _as_float(value) -> Optional[float]:
    """A scalar out of an ndarray / xr.DataArray / number, or None."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size == 0:
            return None
        out = float(arr[0])
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(out) else out


def translation_from_param(param) -> Optional[Tuple[float, float, float]]:
    """(dz, dy, dx) in µm from an affine, or None if it cannot be read.

    Tolerates the leading singleton time axis real multiview-stitcher params
    carry (dims ``t, x_in, x_out``) as well as a bare 4x4 from a test.
    """
    try:
        arr = np.asarray(param, dtype=float)
        mat = arr[0] if arr.ndim == 3 else arr
        return (float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]))
    except Exception:
        return None


def _first_time_slice(metric) -> Any:
    """multiview-stitcher keys groupwise metrics by time index; take t=0."""
    if isinstance(metric, Mapping):
        if not metric:
            return None
        if 0 in metric:
            return metric[0]
        return next(iter(metric.values()))
    return metric


def _edge_key(a, b) -> Tuple[int, int]:
    return (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))


def _graph_edges(graph) -> Dict[Tuple[int, int], Any]:
    """{(lo, hi): edge_data} read structurally, so a fake graph works."""
    if graph is None:
        return {}
    out: Dict[Tuple[int, int], Any] = {}
    try:
        for edge in graph.edges:
            a, b = edge[0], edge[1]
            key = _edge_key(a, b)
            try:
                out[key] = graph.get_edge_data(a, b) or {}
            except Exception:
                out[key] = {}
    except Exception:
        return {}
    return out


@contextlib.contextmanager
def capture_prefilter_graph(sink: dict):
    """Capture the registration graph BEFORE the quality filter removes edges.

    ``register()`` applies ``post_registration_quality_threshold`` inside itself
    (``mv_graph.filter_edges``), so the graph it hands back contains only the
    edges that passed. The rows most worth reporting — this seam scored 0.31
    against a threshold of 0.40 — are therefore unrecoverable from the return
    value, and would show up as an unexplained gap.

    ``register()`` reaches ``filter_edges`` through the module object, so
    wrapping that attribute for the duration of our own call intercepts it
    without patching multiview-stitcher. Strictly best-effort: if the import
    fails, the attribute is missing, or a future version stops calling it, the
    sink stays empty and the caller falls back to set arithmetic — which still
    yields the seam's status, just not its score. Never guess the number.
    """
    try:
        from multiview_stitcher import mv_graph

        original = getattr(mv_graph, "filter_edges", None)
    except Exception:
        yield
        return
    if original is None:
        yield
        return

    def _shim(graph, *args, **kwargs):
        try:
            sink["prefilter"] = graph.copy()
        except Exception:
            pass
        return original(graph, *args, **kwargs)

    mv_graph.filter_edges = _shim
    try:
        yield
    finally:
        mv_graph.filter_edges = original


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _tile_name(tile, index: int) -> str:
    """The folder name — the only key present for every layout.

    ``RawTileInfo.tile_index`` is None for folder-layout acquisitions, and
    ``orientation._tile_label`` returns "" in that case, so neither is safe here.
    """
    try:
        name = tile.folder.name
        if name:
            return str(name)
    except Exception:
        pass
    return f"tile_{index:03d}"


def _voxel(voxel_size_um: Optional[Mapping[str, float]], axis: str) -> float:
    try:
        value = float((voxel_size_um or {}).get(axis, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if value > 0 else 0.0


def _in_voxels(shift_um: Optional[float], voxel_um: float) -> Optional[float]:
    if shift_um is None or voxel_um <= 0:
        return None
    return shift_um / voxel_um


def build_tile_shifts(
    *,
    tiles: Sequence,
    params: Sequence,
    voxel_size_um: Mapping[str, float],
    clamp_records: Optional[Sequence] = None,
) -> List[TileShift]:
    """Per-tile rows from the final params, index-aligned with `tiles`."""
    vz, vy, vx = (_voxel(voxel_size_um, a) for a in ("z", "y", "x"))
    by_index = {int(r.index): r for r in (clamp_records or []) if hasattr(r, "index")}

    rows: List[TileShift] = []
    for index, tile in enumerate(tiles):
        row = TileShift(
            index=index,
            name=_tile_name(tile, index),
            x_mm=float(getattr(tile, "x_mm", 0.0) or 0.0),
            y_mm=float(getattr(tile, "y_mm", 0.0) or 0.0),
            z_min_mm=float(getattr(tile, "z_min_mm", 0.0) or 0.0),
        )
        translation = (
            translation_from_param(params[index]) if index < len(params) else None
        )
        if translation is None:
            row.note = "no registration param for this tile"
        else:
            row.dz_um, row.dy_um, row.dx_um = translation
            row.dz_frames = _in_voxels(row.dz_um, vz) or 0.0
            row.dy_px = _in_voxels(row.dy_um, vy) or 0.0
            row.dx_px = _in_voxels(row.dx_um, vx) or 0.0

        record = by_index.get(index)
        if record is not None:
            clamped_xy = bool(getattr(record, "clamped_xy", False))
            row.clamped_z = bool(getattr(record, "clamped_z", False))
            row.clamped_y = clamped_xy
            row.clamped_x = clamped_xy
            if row.any_clamped:
                row.shift_before_clamp_um = (
                    float(getattr(record, "dz_um", 0.0)),
                    float(getattr(record, "dy_um", 0.0)),
                    float(getattr(record, "dx_um", 0.0)),
                )
                if getattr(record, "whole_matrix", False):
                    row.note = (
                        "non-translational param reverted whole; axes are coupled"
                    )
        rows.append(row)
    return rows


def extract_seams(
    *,
    tiles: Sequence,
    voxel_size_um: Mapping[str, float],
    reg_dict: Optional[Mapping[str, Any]] = None,
    prefilter_graph: Any = None,
    quality_threshold: Optional[float] = None,
    frame_extent_um: Optional[Mapping[str, float]] = None,
    ran: bool = True,
) -> List[SeamResult]:
    """One row per EXPECTED neighbour pair, with what became of it.

    Driven by the stage grid rather than by the graph, because a pair that
    registration threw away is exactly the row worth having and it is not in the
    graph. Never raises: a malformed ``reg_dict`` degrades to statuses without
    numbers.
    """
    vz, vy, vx = (_voxel(voxel_size_um, a) for a in ("z", "y", "x"))
    try:
        expected = find_neighbor_pairs(tiles, include_z=False)
    except Exception:
        expected = []

    surviving: Dict[Tuple[int, int], Any] = {}
    prefilter: Dict[Tuple[int, int], Any] = {}
    used: set = set()
    # An EMPTY used_edges means the global solve used nothing — every edge was
    # pruned, which is a real and alarming finding. An ABSENT one means this MVS
    # version did not tell us. Collapsing the two would make an older version
    # report a perfectly good run as entirely pruned.
    have_used_edges = False
    residuals: Dict[Tuple[int, int], float] = {}
    if ran and isinstance(reg_dict, Mapping):
        pairwise = reg_dict.get("pairwise_registration")
        if isinstance(pairwise, Mapping):
            surviving = _graph_edges(pairwise.get("graph"))
        groupwise = reg_dict.get("groupwise_resolution")
        if isinstance(groupwise, Mapping):
            metrics = groupwise.get("metrics")
            if isinstance(metrics, Mapping):
                used_edges = _first_time_slice(metrics.get("used_edges"))
                if used_edges is not None:
                    try:
                        used = {_edge_key(e[0], e[1]) for e in used_edges}
                        have_used_edges = True
                    except Exception:
                        used, have_used_edges = set(), False
                raw_residuals = _first_time_slice(metrics.get("edge_residuals"))
                if isinstance(raw_residuals, Mapping):
                    for edge, value in raw_residuals.items():
                        try:
                            residuals[_edge_key(edge[0], edge[1])] = _as_float(value)
                        except Exception:
                            continue
    prefilter = _graph_edges(prefilter_graph)

    rows: List[SeamResult] = []
    for index_a, index_b, axis in expected:
        key = _edge_key(index_a, index_b)
        row = SeamResult(
            tile_a=_tile_name(tiles[index_a], index_a),
            tile_b=_tile_name(tiles[index_b], index_b),
            index_a=index_a,
            index_b=index_b,
            axis=axis,
            status=STATUS_NOT_RUN,
        )
        extent = (frame_extent_um or {}).get(axis)
        if extent:
            try:
                overlap_um = tile_geometry.pair_overlap_um(
                    tiles[index_a], tiles[index_b], axis, extent
                )
                row.overlap_frac = overlap_um / extent
            except Exception:
                pass

        if not ran:
            rows.append(row)
            continue

        edge = surviving.get(key)
        if edge is not None:
            if have_used_edges and key not in used:
                row.status = STATUS_PRUNED
                row.note = "dropped by global-optimization edge pruning"
            else:
                row.status = STATUS_REGISTERED
                if not have_used_edges:
                    row.note = "edge used status not reported by multiview-stitcher"
            row.quality = _as_float(edge.get("quality") if isinstance(edge, Mapping) else None)
            translation = translation_from_param(
                edge.get("transform") if isinstance(edge, Mapping) else None
            )
            if translation is not None:
                row.dz_um, row.dy_um, row.dx_um = translation
                row.dz_frames = _in_voxels(row.dz_um, vz)
                row.dy_px = _in_voxels(row.dy_um, vy)
                row.dx_px = _in_voxels(row.dx_um, vx)
            row.residual_px = residuals.get(key)
        elif key in prefilter:
            row.status = STATUS_BELOW_QUALITY
            data = prefilter[key]
            row.quality = _as_float(data.get("quality") if isinstance(data, Mapping) else None)
            if quality_threshold is not None:
                row.note = f"below quality threshold {quality_threshold:g}"
            else:
                row.note = "below the quality threshold"
        else:
            row.status = STATUS_DROPPED
            # Say WHY the number is missing rather than leaving a bare gap: this
            # is the one status where the reason is genuinely unrecoverable.
            row.note = (
                "no edge returned; either pruned before registration or filtered "
                "out — multiview-stitcher does not report which"
            )
        rows.append(row)
    return rows


def build_report(
    *,
    tiles: Sequence,
    params: Sequence,
    voxel_size_um: Mapping[str, float],
    transform_key: str = "",
    clamp_records: Optional[Sequence] = None,
    reg_dict: Optional[Mapping[str, Any]] = None,
    prefilter_graph: Any = None,
    quality_threshold: Optional[float] = None,
    frame_extent_um: Optional[Mapping[str, float]] = None,
    settings: Optional[Dict[str, Any]] = None,
    z_refine: Optional[ZRefineSummary] = None,
    elapsed_s: float = 0.0,
) -> RegistrationReport:
    """Assemble the full report. Never raises on malformed registration output."""
    report = RegistrationReport(
        ran=True,
        transform_key=transform_key,
        settings=dict(settings or {}),
        z_refine=z_refine,
        elapsed_s=float(elapsed_s),
    )
    try:
        report.tiles = build_tile_shifts(
            tiles=tiles,
            params=params,
            voxel_size_um=voxel_size_um,
            clamp_records=clamp_records,
        )
    except Exception as exc:  # pragma: no cover - defensive
        report.warnings.append(f"could not read per-tile shifts: {exc}")
    try:
        report.seams = extract_seams(
            tiles=tiles,
            voxel_size_um=voxel_size_um,
            reg_dict=reg_dict,
            prefilter_graph=prefilter_graph,
            quality_threshold=quality_threshold,
            frame_extent_um=frame_extent_um,
        )
    except Exception as exc:  # pragma: no cover - defensive
        report.warnings.append(f"could not read seam results: {exc}")
    if prefilter_graph is None and report.count(STATUS_DROPPED):
        report.warnings.append(
            "rejected seams have no quality score: the pre-filter graph could "
            "not be captured, so their status is inferred rather than read"
        )
    return report


def skipped_report(
    reason: str,
    *,
    tiles: Optional[Sequence] = None,
    voxel_size_um: Optional[Mapping[str, float]] = None,
    frame_extent_um: Optional[Mapping[str, float]] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> RegistrationReport:
    """A report for a run where registration did not happen.

    Still lists every tile and every expected seam, all at zero and `not_run`.
    Written out like any other, because a missing file reads as "nobody
    implemented this" while a file naming the reason answers the question.
    """
    report = RegistrationReport(
        ran=False, reason=reason, settings=dict(settings or {})
    )
    tiles = tiles or []
    report.tiles = [
        TileShift(
            index=index,
            name=_tile_name(tile, index),
            x_mm=float(getattr(tile, "x_mm", 0.0) or 0.0),
            y_mm=float(getattr(tile, "y_mm", 0.0) or 0.0),
            z_min_mm=float(getattr(tile, "z_min_mm", 0.0) or 0.0),
            note="registration did not run",
        )
        for index, tile in enumerate(tiles)
    ]
    try:
        report.seams = extract_seams(
            tiles=tiles,
            voxel_size_um=voxel_size_um or {},
            frame_extent_um=frame_extent_um,
            ran=False,
        )
    except Exception:  # pragma: no cover - defensive
        report.seams = []
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(value, digits: int = 3) -> str:
    """A number, or an EMPTY cell when it is unknown.

    Not 0, not "nan". A quality of 0.0 and a quality nobody could recover are
    different facts, and a spreadsheet cannot tell them apart once both are 0.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _csv(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(header))
    for row in rows:
        writer.writerow(list(row))
    return buffer.getvalue()


def tile_rows_csv(report: RegistrationReport) -> str:
    rows = []
    for t in report.tiles:
        before = t.shift_before_clamp_um
        rows.append(
            [
                t.index,
                t.name,
                _cell(t.x_mm, 4),
                _cell(t.y_mm, 4),
                _cell(t.z_min_mm, 4),
                _cell(t.dz_um),
                _cell(t.dy_um),
                _cell(t.dx_um),
                _cell(t.dz_frames, 2),
                _cell(t.dy_px, 2),
                _cell(t.dx_px, 2),
                _cell(t.clamped_z),
                _cell(t.clamped_y),
                _cell(t.clamped_x),
                _cell(before[0] if before else None),
                _cell(before[1] if before else None),
                _cell(before[2] if before else None),
                t.note,
            ]
        )
    return _csv(TILE_CSV_HEADER, rows)


def seam_rows_csv(report: RegistrationReport) -> str:
    rows = []
    for s in report.seams:
        rows.append(
            [
                s.tile_a,
                s.tile_b,
                s.index_a,
                s.index_b,
                s.axis,
                s.status,
                _cell(s.quality),
                _cell(s.dz_um),
                _cell(s.dy_um),
                _cell(s.dx_um),
                _cell(s.dz_frames, 2),
                _cell(s.dy_px, 2),
                _cell(s.dx_px, 2),
                _cell(s.residual_px, 2),
                _cell(s.overlap_frac, 4),
                s.note,
            ]
        )
    return _csv(SEAM_CSV_HEADER, rows)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def format_report_text(report: RegistrationReport, *, acquisition: str = "") -> str:
    rule = "=" * 71
    lines = [rule]
    lines.append(
        " REGISTRATION SHIFT REPORT" + (f" — {acquisition}" if acquisition else "")
    )

    if not report.ran:
        lines.append("")
        lines.append(" Registration DID NOT RUN for this stitch.")
        lines.append(f"   Reason: {report.reason or 'not recorded'}")
        lines.append(
            f"   {report.n_tiles} tiles were placed by stage position alone, and"
        )
        lines.append(
            "   every seam below is unmeasured — not zero, unmeasured."
        )
        lines.append(rule)
        return "\n".join(lines)

    lines.append(
        f" Transform: {report.transform_key or 'n/a'}   "
        f"Tiles: {report.n_tiles}   Expected seams: {report.n_expected_pairs}"
    )
    lines.append(
        f" Seams: {report.count(STATUS_REGISTERED)} registered · "
        f"{report.count(STATUS_PRUNED)} pruned · "
        f"{report.count(STATUS_BELOW_QUALITY)} below quality · "
        f"{report.count(STATUS_DROPPED)} no edge"
    )
    s = report.settings
    if s:
        lines.append(
            " Effective voxel: "
            f"{s.get('voxel_y_um', 0):.3f} µm XY · {s.get('voxel_z_um', 0):.3f} µm Z"
            f"   Quality threshold: {s.get('quality_threshold', 0):g}"
        )
        lines.append(
            f" Clamp bounds: XY {s.get('bound_xy_um', 0):.1f} µm "
            f"({s.get('bound_xy_source', '?')}) · "
            f"Z {s.get('bound_z_um', 0):.1f} µm ({s.get('bound_z_source', '?')})"
        )
    lines.append(f" Elapsed: {report.elapsed_s:.1f} s")
    lines.append("-" * 71)

    # Headline the clamp, for the same reason border QC headlines its search
    # limit: a run where many tiles were clamped is not "aligned by these small
    # numbers", it is unaligned by an amount nobody measured.
    n_clamped = report.n_tiles_clamped
    if n_clamped:
        lines.append(
            f" ** {n_clamped} of {report.n_tiles} tiles hit a clamp bound and kept"
        )
        lines.append(
            "    their stage position. Those tiles are NOT aligned by the numbers"
        )
        lines.append(
            "    below — the true shift was larger and was NOT MEASURED. The"
        )
        lines.append(
            "    proposed value is in the *_before_clamp columns of the CSV."
        )

    clean = [t for t in report.tiles if not t.any_clamped]
    if clean:
        lines.append(
            f" Net shift ({len(clean)} unclamped tiles):"
        )
        lines.append(
            f"   |dz| median {_median([abs(t.dz_um) for t in clean]):.1f} µm "
            f"({_median([abs(t.dz_frames) for t in clean]):.1f} frames), "
            f"max {max(abs(t.dz_um) for t in clean):.1f} µm"
        )
        lines.append(
            f"   |dy| median {_median([abs(t.dy_um) for t in clean]):.1f} µm "
            f"({_median([abs(t.dy_px) for t in clean]):.1f} px), "
            f"max {max(abs(t.dy_um) for t in clean):.1f} µm"
        )
        lines.append(
            f"   |dx| median {_median([abs(t.dx_um) for t in clean]):.1f} µm "
            f"({_median([abs(t.dx_px) for t in clean]):.1f} px), "
            f"max {max(abs(t.dx_um) for t in clean):.1f} µm"
        )

    zr = report.z_refine
    if zr is not None:
        lines.append("")
        if zr.ran:
            lines.append(
                f" Z refinement: {zr.n_tiles_moved} tiles adjusted, "
                f"|dz| median {zr.median_abs_dz_um:.2f} µm, max {zr.max_abs_dz_um:.2f} µm"
            )
            lines.append(
                f"   binning z={zr.binning.get('z', '?')} "
                f"upsample={zr.upsample_factor} search=±{zr.search_range_um:.0f} µm"
            )
            if zr.n_hit_search_limit:
                lines.append(
                    f"   ** {zr.n_hit_search_limit} corrections came back AT the "
                    f"search limit and were rejected —"
                )
                lines.append(
                    "      those are floors, not measurements. Widen the search "
                    "or check tile order."
                )
        else:
            lines.append(f" Z refinement: not run — {zr.reason or 'disabled'}")

    worst = sorted(
        (t for t in report.tiles if not t.any_clamped),
        key=lambda t: max(abs(t.dz_um), abs(t.dy_um), abs(t.dx_um)),
        reverse=True,
    )[:5]
    if worst and max(
        max(abs(t.dz_um), abs(t.dy_um), abs(t.dx_um)) for t in worst
    ) > 0:
        lines.append("")
        lines.append(" LARGEST CORRECTIONS (worst first)")
        for rank, t in enumerate(worst, 1):
            lines.append(
                f"   {rank}. {t.name}   dz={t.dz_um:+.1f} µm ({t.dz_frames:+.1f} fr)"
                f"  dy={t.dy_um:+.1f} µm ({t.dy_px:+.1f} px)"
                f"  dx={t.dx_um:+.1f} µm ({t.dx_px:+.1f} px)"
            )

    rejected = [
        s
        for s in report.seams
        if s.status in (STATUS_BELOW_QUALITY, STATUS_DROPPED, STATUS_PRUNED)
    ]
    if rejected:
        lines.append("")
        lines.append(" SEAMS NOT USED")
        for rank, s in enumerate(rejected[:10], 1):
            quality = "quality " + (
                f"{s.quality:.2f}" if s.quality is not None else "unrecorded"
            )
            overlap = (
                f"overlap {s.overlap_frac * 100:.0f}%"
                if s.overlap_frac is not None
                else "overlap unknown"
            )
            lines.append(
                f"   {rank}. {s.tile_a} <-> {s.tile_b}  [{s.axis.upper()}-seam]  "
                f"{s.status}  {quality}  {overlap}"
            )
        if len(rejected) > 10:
            lines.append(f"   ... and {len(rejected) - 10} more (see {SEAM_CSV_NAME})")

    for warning in report.warnings:
        lines.append(f" NOTE: {warning}")
    lines.append(rule)
    return "\n".join(lines)


def report_to_json(
    report: RegistrationReport, *, acquisition: str = ""
) -> Dict[str, Any]:
    def _tile(t: TileShift) -> Dict[str, Any]:
        return {
            "index": t.index,
            "name": t.name,
            "x_mm": t.x_mm,
            "y_mm": t.y_mm,
            "z_min_mm": t.z_min_mm,
            "dz_um": t.dz_um,
            "dy_um": t.dy_um,
            "dx_um": t.dx_um,
            "dz_frames": t.dz_frames,
            "dy_px": t.dy_px,
            "dx_px": t.dx_px,
            "clamped_z": t.clamped_z,
            "clamped_y": t.clamped_y,
            "clamped_x": t.clamped_x,
            "shift_before_clamp_um": list(t.shift_before_clamp_um)
            if t.shift_before_clamp_um
            else None,
            "note": t.note,
        }

    def _seam(s: SeamResult) -> Dict[str, Any]:
        return {
            "tile_a": s.tile_a,
            "tile_b": s.tile_b,
            "index_a": s.index_a,
            "index_b": s.index_b,
            "axis": s.axis,
            "status": s.status,
            "quality": s.quality,
            "dz_um": s.dz_um,
            "dy_um": s.dy_um,
            "dx_um": s.dx_um,
            "residual_px": s.residual_px,
            "overlap_frac": s.overlap_frac,
            "note": s.note,
        }

    payload: Dict[str, Any] = {
        "version": REPORT_VERSION,
        "acquisition": acquisition,
        "ran": report.ran,
        "reason": report.reason,
        "transform_key": report.transform_key,
        "elapsed_s": round(report.elapsed_s, 3),
        "n_tiles": report.n_tiles,
        "n_tiles_clamped": report.n_tiles_clamped,
        "n_expected_pairs": report.n_expected_pairs,
        "seam_status_counts": {
            status: report.count(status)
            for status in (
                STATUS_REGISTERED,
                STATUS_PRUNED,
                STATUS_BELOW_QUALITY,
                STATUS_DROPPED,
                STATUS_NOT_RUN,
            )
        },
        "settings": report.settings,
        "warnings": report.warnings,
        "tiles": [_tile(t) for t in report.tiles],
        "seams": [_seam(s) for s in report.seams],
    }
    if report.z_refine is not None:
        zr = report.z_refine
        payload["z_refine"] = {
            "ran": zr.ran,
            "reason": zr.reason,
            "binning": zr.binning,
            "upsample_factor": zr.upsample_factor,
            "search_range_um": zr.search_range_um,
            "n_tiles_moved": zr.n_tiles_moved,
            "max_abs_dz_um": zr.max_abs_dz_um,
            "median_abs_dz_um": zr.median_abs_dz_um,
            "n_hit_search_limit": zr.n_hit_search_limit,
        }
    return payload


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_report(
    output_dir,
    report: RegistrationReport,
    *,
    acquisition: str = "",
    write_json: bool = False,
    logger=None,
) -> Dict[str, Path]:
    """Write the CSVs and the text summary into `output_dir`.

    Into the OUTPUT directory, deliberately — not beside the log where border QC
    puts its report. These numbers describe the store sitting next to them and
    have to travel with it when someone copies the result off the rig.

    Best-effort per file: an unwritable path warns and is omitted from the
    returned mapping. Evidence about a run must never be the thing that fails
    the run.
    """
    written: Dict[str, Path] = {}
    directory = Path(output_dir)
    payloads = [
        ("tiles_csv", TILE_CSV_NAME, lambda: tile_rows_csv(report)),
        ("seams_csv", SEAM_CSV_NAME, lambda: seam_rows_csv(report)),
        (
            "text",
            TEXT_NAME,
            lambda: format_report_text(report, acquisition=acquisition) + "\n",
        ),
    ]
    if write_json:
        import json

        payloads.append(
            (
                "json",
                JSON_NAME,
                lambda: json.dumps(
                    report_to_json(report, acquisition=acquisition), indent=2
                ),
            )
        )

    for key, name, render in payloads:
        path = directory / name
        try:
            path.write_text(render(), encoding="utf-8")
            written[key] = path
        except Exception as exc:
            if logger is not None:
                logger.warning(f"Could not write {name}: {exc}")
    return written
