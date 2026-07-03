"""Run ONE headless stitch under a memory monitor and print JSON results.

Invoked as a subprocess by ``test_memory_scaling.py`` so each data point gets a
fresh process — otherwise Python's monotonic RSS (freed memory isn't returned to
the OS) would make sequential in-process runs pollute each other's peak.

Usage:  python _mem_probe.py '<json-config>'
Prints one JSON line: {peak_delta_mb, phase_peaks_mb, estimate, tiles, planes, ...}
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # _synth_acq
sys.path.insert(0, str(_HERE.parent / "src"))  # flamingo_stitcher

from _synth_acq import write_synth_acquisition  # noqa: E402

from flamingo_stitcher.memory_monitor import MemoryMonitor  # noqa: E402
from flamingo_stitcher.pipeline import (  # noqa: E402
    StitchingConfig,
    StitchingPipeline,
    discover_tiles,
    estimate_memory_usage,
)

# Map a progress message to a coarse phase name for RSS attribution.
_PHASE_KEYWORDS = [
    ("register", "register"),
    ("registrat", "register"),
    ("preprocess", "preprocess"),
    ("materializ", "preprocess"),
    ("loading", "preprocess"),
    ("fusing", "fuse"),
    ("fuse", "fuse"),
    ("computing", "fuse"),
    ("writing", "write"),
    ("pyramid", "write"),
    ("metadata", "write"),
]


def _phase_for(msg: str) -> str:
    low = (msg or "").lower()
    for kw, phase in _PHASE_KEYWORDS:
        if kw in low:
            return phase
    return "other"


def main() -> None:
    cfg_in = json.loads(sys.argv[1])

    with tempfile.TemporaryDirectory() as d:
        acq = write_synth_acquisition(
            Path(d) / "acq",
            grid=tuple(cfg_in["grid"]),
            n_planes=cfg_in["n_planes"],
            channels=cfg_in.get("channels", [1]),
            illum_sides=cfg_in.get("illum_sides", [0]),
            frame_size=tuple(cfg_in.get("frame_size", [32, 32])),
            overlap=cfg_in.get("overlap", 0.15),
        )
        tiles = discover_tiles(acq)

        cfg = StitchingConfig.with_yaml_defaults()
        cfg.skip_registration = cfg_in.get("skip_registration", True)
        cfg.streaming_mode = cfg_in.get("streaming", True)
        cfg.output_format = cfg_in.get("output_format", "ome-tiff")
        cfg.tile_overlap_fusion = cfg_in.get("tile_overlap_fusion", "max")
        cfg.illumination_fusion = cfg_in.get("illumination_fusion", "max")
        cfg.content_based_fusion = cfg_in.get("content_based_fusion", False)
        cfg.destripe = cfg_in.get("destripe", False)
        cfg.fusion_superblock_chunks = cfg_in.get("superblock", 0)
        cfg.resource_guard_enabled = False  # never abort tiny probe runs
        # Chunk finer than the mosaic so the fusion geometry mirrors real
        # scale: each output block overlaps only its LOCAL tile neighbourhood,
        # not the whole grid. Without this, a tiny mosaic fits in one chunk and
        # views_per_block degenerates to n_tiles (a small-data artifact).
        if "output_chunksize" in cfg_in:
            cfg.output_chunksize = dict(cfg_in["output_chunksize"])

        channels = sorted({c for t in tiles for c in t.channels})
        est = estimate_memory_usage(tiles, channels, cfg)

        # Warm the heavy lazy imports (multiview_stitcher / dask / scipy) BEFORE
        # the monitor baseline so their fixed ~120 MB high-water is excluded and
        # peak_delta reflects only the pipeline's dataset-dependent working set.
        # Otherwise import noise swamps the signal at tiny test sizes and an
        # O(dataset) allocation could hide beneath it.
        import dask.array  # noqa: F401
        import multiview_stitcher.fusion  # noqa: F401
        import multiview_stitcher.registration  # noqa: F401

        mon = MemoryMonitor(interval_s=0.02, metric=cfg_in.get("metric", "uss"))

        def progress_fn(pct, msg):
            mon.set_phase(_phase_for(msg))

        mon.start()
        try:
            StitchingPipeline(cfg, progress_fn=progress_fn).run(acq, Path(d) / "out")
        finally:
            mon.stop()

        result = {
            "tiles": len(tiles),
            "planes": tiles[0].n_planes if tiles else 0,
            "channels": len(channels),
            "illum_sides": len(cfg_in.get("illum_sides", [0])),
            "streaming": cfg.streaming_mode,
            "peak_delta_mb": round(mon.peak_delta_bytes / 1e6, 2),
            "phase_peaks_mb": {
                p: round(v / 1e6, 2) for p, v in mon.phase_peaks_delta().items()
            },
            "estimate": est,
            "psutil": mon.available,
        }
        print(json.dumps(result))


if __name__ == "__main__":
    main()
