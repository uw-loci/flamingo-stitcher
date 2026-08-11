"""Persistent timing cache for stitching pipeline ETA.

Stores per-key total wall time and per-phase share-of-total as EMAs.
The key is built from the variables that most strongly affect cost
(tile count, channels, pyramid levels, timepoints, output format,
fusion method, registration on/off, planes per tile). Different
acquisitions naturally land in different keys, so the EMA for any
given key only blends "similar" runs.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("microscope_settings") / "stitching_timing_cache.json"
EMA_ALPHA = 0.3

# Phase identifiers used by the estimator. Order matters: it's the
# expected execution order, and the estimator uses it to determine
# which phases are "remaining" once the current phase is known.
PHASE_ORDER: List[str] = [
    "discover",
    "register",
    "preprocess",
    "fuse",
    "write",
    "metadata",
]


# ---------------------------------------------------------------------------
# Key bucketing
# ---------------------------------------------------------------------------


def _bucket_tiles(n: int) -> str:
    """Bucket tile count so adjacent counts share a cache key."""
    if n <= 4:
        return "1-4"
    if n <= 9:
        return "5-9"
    if n <= 24:
        return "10-24"
    if n <= 49:
        return "25-49"
    if n <= 99:
        return "50-99"
    if n <= 249:
        return "100-249"
    return "250+"


def _bucket_planes(n: int) -> str:
    """Bucket planes-per-tile (Z-stack depth)."""
    if n <= 50:
        return "1-50"
    if n <= 150:
        return "51-150"
    if n <= 400:
        return "151-400"
    if n <= 1000:
        return "401-1000"
    return "1000+"


@dataclass(frozen=True)
class StitchingTimingKey:
    """All key axes for the cache. Use ``.serialize()`` to get a flat
    string suitable for JSON dict lookup."""

    n_tiles: int
    n_channels: int
    n_pyramid_levels: int  # 0 if none
    n_timepoints: int  # 1 if not a time series
    output_format: str  # e.g. "ome-zarr-sharded", "imaris", "ome-tiff"
    fusion_method: str  # "content_based" | "cosine"
    skip_registration: bool
    planes_per_tile: int
    # Downsample factors strongly change fuse/write time, so they're part of
    # the key — otherwise a full-res and a heavily-downsampled run would share
    # (and average) one cached total. -1 = iso (auto). Default 1 keeps older
    # call sites valid.
    downsample_xy: int = 1
    downsample_z: int = 1
    # Source (read) and destination (write) drive roots, e.g. "H:" / "C:" on
    # Windows. I/O speed dominates load + write, so bucketing by drive lets the
    # cache learn per-drive throughput empirically. "" = unknown (POSIX / no
    # drive letter) and just collapses to one bucket.
    source_drive: str = ""
    dest_drive: str = ""
    # Per-tile PREPROCESSING steps. These dominate the load+preprocess phase
    # (61% of a measured 135-tile run), so a destripe-on and a destripe-off run
    # sharing one cache entry averages two very different costs into a number
    # that fits neither — the same reasoning that already puts downsample in the
    # key. `destripe_fast` is separate because it filters the DOWNSAMPLED tile,
    # which is a different cost class again (4x fewer pixels at xy2).
    destripe: bool = False
    destripe_fast: bool = False
    deconvolution: bool = False
    flat_field: bool = False
    # The Z-refinement pass roughly doubles-to-triples registration time (twice
    # the overlap voxels at Z binning 1, plus ~1.7x the edges with pruning off).
    # `skip_registration` is already a key axis precisely because
    # registration-on and registration-off are different cost classes; this is
    # the same effect one level down, and without it a refined run and a plain
    # one average into a number that fits neither.
    z_refine: bool = False

    def serialize(self) -> str:
        return (
            f"t={_bucket_tiles(self.n_tiles)}|"
            f"c={self.n_channels}|"
            f"p={self.n_pyramid_levels}|"
            f"tp={self.n_timepoints}|"
            f"fmt={self.output_format}|"
            f"fus={self.fusion_method}|"
            f"skipreg={int(self.skip_registration)}|"
            f"pl={_bucket_planes(self.planes_per_tile)}|"
            f"dxy={self.downsample_xy}|"
            f"dz={self.downsample_z}|"
            f"ds={int(self.destripe)}{int(self.destripe_fast)}|"
            f"dec={int(self.deconvolution)}|"
            f"ff={int(self.flat_field)}|"
            f"src={self.source_drive}|"
            f"dst={self.dest_drive}"
            # Appended only when set, so every key string already in a user's
            # cache stays valid and their learned timings survive this change.
            + ("|zr=1" if self.z_refine else "")
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class StitchingTimingCache:
    """JSON-backed EMA cache of stitching phase timings.

    Each key maps to::

        {
          "total_s": {"mean": float, "samples": int},
          "phases":  {phase_name: {"mean_share": float, "samples": int}, ...}
        }

    Shares are fractions of total wall time (sum-to-1 across phases
    actually run; some phases — like ``register`` when
    ``skip_registration=True`` — may be absent).
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else DEFAULT_CACHE_PATH
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    self._data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load stitching timing cache: {e}")
            self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save stitching timing cache: {e}")

    # A cached total is scaled by at most this factor before we stop trusting
    # the linearity assumption. Well outside a single tile bucket's span.
    _MAX_TILE_SCALE = 4.0

    def get_total_s(
        self, key: StitchingTimingKey, n_tiles: Optional[int] = None
    ) -> Optional[float]:
        """Cached mean total wall time for this key, scaled to ``n_tiles``.

        Tile counts are BUCKETED in the key, and the widest buckets span 2.5x
        (100-249). Returning the bucket's mean unscaled meant a 249-tile run's
        recorded time seeded a 135-tile run: one measured case predicted 204 s
        for a run that took 115 s — almost exactly the 249/135 ratio. Wall time
        is close to linear in tile count for a fixed geometry (preprocess is
        per-tile, and fused area grows with the mosaic), so scale by the ratio
        of actual to recorded tile count.

        Falls back to the raw mean when the entry predates tile-count tracking
        or the ratio is implausible.
        """
        with self._lock:
            entry = self._data.get(key.serialize())
            if not entry:
                return None
            total = entry.get("total_s", {}).get("mean")
            if not total or total <= 0:
                return None
            total = float(total)

            recorded_tiles = entry.get("n_tiles", {}).get("mean")
            if not n_tiles or not recorded_tiles or recorded_tiles <= 0:
                return total
            scale = float(n_tiles) / float(recorded_tiles)
            if not (1.0 / self._MAX_TILE_SCALE) <= scale <= self._MAX_TILE_SCALE:
                return total
            return total * scale

    def get_phase_shares(self, key: StitchingTimingKey) -> Dict[str, float]:
        """Cached mean share-of-total per phase. Empty dict if no data."""
        with self._lock:
            entry = self._data.get(key.serialize())
            if not entry:
                return {}
            phases = entry.get("phases", {})
            return {
                name: float(p["mean_share"])
                for name, p in phases.items()
                if p.get("mean_share")
            }

    def record_run(
        self,
        key: StitchingTimingKey,
        total_s: float,
        phase_durations_s: Dict[str, float],
        *,
        alpha: float = EMA_ALPHA,
        n_tiles: Optional[int] = None,
    ) -> None:
        """Update the EMA with one completed run.

        ``phase_durations_s`` is the absolute wall time spent in each
        phase; shares are computed here from ``total_s``. ``n_tiles`` is
        recorded alongside so :meth:`get_total_s` can scale the cached total to
        a different tile count within the same (bucketed) key.
        """
        if total_s <= 0:
            return
        with self._lock:
            k = key.serialize()
            entry = self._data.setdefault(k, {"total_s": {}, "phases": {}})

            # Track the tile count this timing was measured at, so a later run
            # elsewhere in the same bucket can scale rather than inherit.
            if n_tiles and n_tiles > 0:
                tile_block = entry.setdefault("n_tiles", {})
                prev_tiles = float(tile_block.get("mean", 0.0))
                if not prev_tiles:
                    tile_block["mean"] = float(n_tiles)
                else:
                    tile_block["mean"] = alpha * n_tiles + (1.0 - alpha) * prev_tiles

            # Update total_s EMA
            total_block = entry["total_s"]
            prev_total = float(total_block.get("mean", 0.0))
            prev_samples = int(total_block.get("samples", 0))
            if prev_samples == 0:
                total_block["mean"] = total_s
            else:
                total_block["mean"] = alpha * total_s + (1.0 - alpha) * prev_total
            total_block["samples"] = prev_samples + 1

            # Update per-phase share EMA
            phases_block = entry["phases"]
            for phase, dur in phase_durations_s.items():
                share = max(0.0, dur / total_s)
                pb = phases_block.setdefault(phase, {})
                prev_share = float(pb.get("mean_share", 0.0))
                prev_n = int(pb.get("samples", 0))
                if prev_n == 0:
                    pb["mean_share"] = share
                else:
                    pb["mean_share"] = alpha * share + (1.0 - alpha) * prev_share
                pb["samples"] = prev_n + 1

            self._save()

    def clear(self, key: Optional[StitchingTimingKey] = None) -> None:
        with self._lock:
            if key is None:
                self._data.clear()
            else:
                self._data.pop(key.serialize(), None)
            self._save()
