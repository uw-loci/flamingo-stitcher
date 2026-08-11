"""Independent memory-boundedness check for the streaming stitching pipeline.

This is the verifier described in the memory-boundedness plan. It runs the REAL
headless pipeline on tiny synthetic acquisitions at several dataset sizes,
measures actual peak RSS (each run in its own subprocess for an honest peak),
and asserts the peak does NOT scale with dataset volume in streaming mode.

The guarantee under test:

    In streaming mode, peak RAM depends on workers x (tile/block working set),
    NOT on n_tiles or n_planes.

So a large dataset increase (e.g. 16x the tiles or planes) must produce only a
small peak increase. If a new feature introduces an O(dataset) allocation (like
the pre-v0.4.4 registration bug that held all tiles in RAM), the corresponding
ratio blows up and this test fails -- regardless of whether anyone updated the
memory estimator. That independence is the point.

Marked slow: spawns several subprocess stitch runs. Requires multiview_stitcher
and psutil; skipped otherwise.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROBE = _HERE / "_mem_probe.py"

try:
    import multiview_stitcher  # noqa: F401
    import psutil  # noqa: F401

    _HAVE_DEPS = True
except Exception:
    _HAVE_DEPS = False

# A finer-than-mosaic chunk so fusion overlaps only the local neighbourhood,
# mirroring real scale (see _mem_probe / plan). Shared by every probe here.
_CHUNK = {"z": 8, "y": 32, "x": 32}

# Peak is measured as USS (private/committed memory) — the proxy for "will I
# OOM". USS excludes the memmap page-cache that made RSS overcount spilled
# tiles, and excludes the fixed ~120 MB library-import high-water, so the
# dataset-dependent working set is exposed cleanly.

# Per-tile private-memory slope ceiling (MB per added tile). The CURRENT
# streaming pipeline sits at ~0.4 MB/tile — that's the dask fusion graph
# (O(n_output_blocks)), a known term we accept for now and drive toward zero
# with spatial super-block batching (see the boundedness plan). A data-hold
# regression (the pre-v0.4.4 class: keep a full tile per tile) adds
# tile_bytes/tile — MB at test scale, GB at production scale — blowing far past
# this ceiling. So this catches the catastrophic class while tolerating the
# graph floor.
_MAX_SLOPE_MB_PER_TILE = 1.5

# For the plane axis, peak SHOULD grow ~linearly (a deeper stack = a bigger
# per-worker tile working set — the healthy workers x tile_bytes bound). We only
# guard against SUPER-linear growth (e.g. holding all-Z of every tile), so a
# generous ratio ceiling over a 16x plane increase.
_MAX_PLANE_RATIO = 3.0


def _run_probe(**cfg) -> dict:
    cfg.setdefault("output_chunksize", _CHUNK)
    cfg.setdefault("streaming", True)
    out = subprocess.run(
        [sys.executable, str(_PROBE), json.dumps(cfg)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"probe failed (rc={out.returncode})\ncfg={cfg}\nstderr:\n{out.stderr[-2000:]}"
        )
    # Take the last JSON-parseable stdout line (libraries may print noise).
    last = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                pass
    if last is None:
        raise RuntimeError(f"no JSON from probe\ncfg={cfg}\nstdout:\n{out.stdout[-2000:]}")
    return last


@unittest.skipUnless(_HAVE_DEPS, "needs multiview_stitcher + psutil")
class TestStreamingMemoryBounded(unittest.TestCase):
    def test_slope_flat_vs_tile_count(self):
        """Private memory must not scale with tile DATA as tiles are added.

        Peak per added tile must stay below a small ceiling: the healthy floor
        is the dask graph (~0.4 MB/tile); a data-hold regression adds
        tile_bytes/tile and blows past it.
        """
        small = _run_probe(grid=[2, 2], n_planes=16)  # 4 tiles
        large = _run_probe(grid=[8, 8], n_planes=16)  # 64 tiles
        self.assertEqual(small["tiles"], 4)
        self.assertEqual(large["tiles"], 64)
        slope = (large["peak_delta_mb"] - small["peak_delta_mb"]) / (
            large["tiles"] - small["tiles"]
        )
        self.assertLess(
            slope,
            _MAX_SLOPE_MB_PER_TILE,
            f"streaming private memory scales with tile COUNT at "
            f"{slope:.2f} MB/tile (ceiling {_MAX_SLOPE_MB_PER_TILE}). "
            f"{small['peak_delta_mb']:.0f} MB @ {small['tiles']} tiles -> "
            f"{large['peak_delta_mb']:.0f} MB @ {large['tiles']} tiles. "
            f"An O(n_tiles) data allocation was likely introduced. "
            f"Phase peaks: small={small['phase_peaks_mb']} "
            f"large={large['phase_peaks_mb']}",
        )

    def test_slope_flat_vs_tile_count_with_registration(self):
        """The same bound, with registration and its Z-refinement pass ON.

        Every other probe in this file skips registration, so until this one
        existed the whole registration stage — the largest single allocation
        this pipeline ever made, and the source of the 570 GB OOM — was outside
        CI's reach. Pairwise registration is bounded per PAIR (multiview-stitcher
        crops to the overlap bbox before materializing, and forces 3-D pairs to
        run sequentially), so the slope must stay just as flat as without it.
        """
        small = _run_probe(
            grid=[2, 2], n_planes=16, skip_registration=False, z_refine=True
        )
        large = _run_probe(
            grid=[8, 8], n_planes=16, skip_registration=False, z_refine=True
        )
        slope = (large["peak_delta_mb"] - small["peak_delta_mb"]) / (
            large["tiles"] - small["tiles"]
        )
        self.assertLess(
            slope,
            _MAX_SLOPE_MB_PER_TILE,
            f"registration memory scales with tile COUNT at {slope:.2f} MB/tile "
            f"(ceiling {_MAX_SLOPE_MB_PER_TILE}). "
            f"{small['peak_delta_mb']:.0f} MB @ {small['tiles']} tiles -> "
            f"{large['peak_delta_mb']:.0f} MB @ {large['tiles']} tiles. "
            f"Registration must hold one PAIR's overlap, never all tiles. "
            f"Phase peaks: small={small['phase_peaks_mb']} "
            f"large={large['phase_peaks_mb']}",
        )

    def test_registration_peaks_are_attributed_to_the_register_phase(self):
        """A registration regression must be traceable to registration.

        The Z-refinement pass reports its status as "Registering tiles (Z
        refinement)...", which the phase keyword table maps to `register`. If
        that wording changes, its memory lands in Other/setup and the next
        person hunts it in the wrong stage.
        """
        r = _run_probe(
            grid=[4, 4], n_planes=16, skip_registration=False, z_refine=True
        )
        self.assertIn("register", r["phase_peaks_mb"])

    def test_bounded_vs_plane_count(self):
        """16x the Z planes must not SUPER-linearly grow the peak (tiles are
        spilled to disk / paged, not all held whole in RAM)."""
        small = _run_probe(grid=[2, 2], n_planes=16)
        large = _run_probe(grid=[2, 2], n_planes=256)
        self.assertEqual(small["planes"], 16)
        self.assertEqual(large["planes"], 256)
        ratio = large["peak_delta_mb"] / max(small["peak_delta_mb"], 1.0)
        self.assertLess(
            ratio,
            _MAX_PLANE_RATIO,
            f"streaming peak grew {ratio:.2f}x for 16x the planes "
            f"({small['peak_delta_mb']:.0f} -> {large['peak_delta_mb']:.0f} MB) "
            f"— super-linear in Z suggests a whole-stack hold. "
            f"Phase peaks: small={small['phase_peaks_mb']} "
            f"large={large['phase_peaks_mb']}",
        )

    def test_phase_attribution_present(self):
        """Peaks are attributed to phases so a regression is traceable to a
        stage, not a mystery."""
        r = _run_probe(grid=[4, 4], n_planes=16)
        phases = r["phase_peaks_mb"]
        self.assertIn("fuse", phases)
        # register + preprocess should also have been observed.
        self.assertTrue(
            {"preprocess", "register"} & set(phases),
            f"expected preprocess/register phases, got {set(phases)}",
        )


if __name__ == "__main__":
    unittest.main()
