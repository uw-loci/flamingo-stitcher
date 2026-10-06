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

import pytest
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

# Plane axis. Replaced a 16x-planes RATIO ceiling on 2026-10-06, which failed
# on a RAM-starved dev box AND on a clean CI runner, and could not have worked
# either way: at the 32x32 probe frame a plane is 2 KB, so a FULL-STACK hold of
# every tile would move the peak less than the run-to-run noise. The ratio was
# reporting fixed overhead, and no choice of ceiling fixes a quantity that
# cannot see the regression it exists to catch.
#
# What the measurement showed instead (4 plane counts x 2 frame sizes):
#   * preprocess and register are FLAT -- 1.1-1.2 MB across 8x the planes, at
#     both frame sizes. Tiles are streamed, not held. That is the guarantee.
#   * fuse/write carry ALL the growth, R2=0.997, and it is ANONYMOUS, not the
#     output memmap (anon 88->435 MB while mapped went 109->239). It is the
#     dask fusion graph, O(n_output_blocks) -- the same known, accepted term
#     the tile-axis slope above documents, showing up along Z.
#
# So the guard is a SLOPE on the phases where a data-hold would appear, which
# removes the fixed overhead that made the ratio meaningless, and is expressed
# as a share of what holding a plane of every tile would cost -- so it scales
# with the probe geometry instead of being a magic number.
#
# Margin at the current frame: ceiling 0.033 MB/plane against a measured
# ~0.0004. A whole-stack hold lands at 0.131. Roughly 300x of headroom, where
# the ratio had none.
_MAX_HOLD_SLOPE_FRACTION = 0.25

# Plane counts to fit across. Four points, not two: a slope needs a line, and
# R2 is what says the fit means anything.
_PLANE_POINTS = (32, 64, 128, 256)

# Frame big enough that a plane of a tile is a real quantity (0.033 MB), so a
# hold is separable from overhead. The 32x32 default could not do that.
_SLOPE_FRAME = (128, 128)

# Phases where an O(dataset) hold would show. fuse/write legitimately carry the
# graph term, so they are REPORTED but not gated -- gating a known accepted
# cost is how a suite trains people to ignore it.
_HOLD_PHASES = ("preprocess", "register")


def _fit_line(xs, ys):
    """Least-squares (intercept, slope, R2). Hand-rolled so the test needs no
    extra dependency, and so the intercept — the fixed overhead that made a
    ratio meaningless here — is visible rather than folded in."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs) or 1e-9
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys) or 1e-9
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return intercept, slope, 1 - ss_res / ss_tot


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

    def test_no_phase_holds_the_dataset_as_planes_grow(self):
        """Peak must not grow with Z in the phases that touch whole tiles.

        Fits anonymous peak against plane count and checks the SLOPE, so the
        fixed overhead that dominates at these sizes drops out. The ceiling is
        a share of what holding one plane of every tile would cost, so it means
        the same thing at any probe geometry.

        Prints the full fit every run, pass or fail: a number with no visible
        working is how the previous version of this guard went unexamined for
        two months.
        """
        plane_mb = _SLOPE_FRAME[0] * _SLOPE_FRAME[1] * 2 / 1e6
        grid = [2, 2]
        n_tiles = grid[0] * grid[1]
        # Slope if every tile's whole stack were held: one more plane costs one
        # plane of every tile.
        hold_slope = n_tiles * plane_mb
        ceiling = _MAX_HOLD_SLOPE_FRACTION * hold_slope

        runs = [
            _run_probe(grid=grid, n_planes=p, frame_size=list(_SLOPE_FRAME))
            for p in _PLANE_POINTS
        ]
        xs = [r["planes"] for r in runs]

        report = [
            f"frame {_SLOPE_FRAME[0]}x{_SLOPE_FRAME[1]}, {n_tiles} tiles, "
            f"plane={plane_mb:.4f} MB/tile",
            f"a full-stack hold would slope at {hold_slope:.4f} MB/plane; "
            f"ceiling {ceiling:.4f}",
            f"{'planes':>7}{'uss':>9}{'anon':>9}{'mapped':>9}  anon per phase",
        ]
        for r in runs:
            ph = r.get("phase_anon_peaks_mb") or r.get("phase_peaks_mb", {})
            per = " ".join(f"{k}={v:.1f}" for k, v in sorted(ph.items()))
            report.append(
                f"{r['planes']:>7}{r['peak_delta_mb']:>9.1f}"
                f"{r.get('peak_anon_mb', float('nan')):>9.1f}"
                f"{r.get('peak_mapped_mb', 0):>9.1f}  {per}"
            )

        def phase_series(name):
            return [
                (r.get("phase_anon_peaks_mb") or r.get("phase_peaks_mb", {})).get(name, 0.0)
                for r in runs
            ]

        failures = []
        for phase in _HOLD_PHASES:
            ys = phase_series(phase)
            if not any(ys):
                continue  # phase never reported; test_phase_attribution_present covers that
            a, b, r2 = _fit_line(xs, ys)
            report.append(
                f"  {phase:>10}: {a:7.3f} MB + {b:+.5f} MB/plane  R2={r2:5.3f}  "
                f"= {100 * b / hold_slope:6.2f}% of a hold"
                f"{'   (R2 is meaningless on a flat series — read the slope)' if abs(b) < ceiling / 100 else ''}"
            )
            if b > ceiling:
                failures.append(
                    f"{phase} grows {b:.4f} MB/plane ({100 * b / hold_slope:.1f}% of a "
                    f"whole-stack hold), over the {ceiling:.4f} ceiling"
                )
        # Reported, never gated: the dask fusion graph is O(n_output_blocks),
        # a known cost that super-block batching drives down.
        for phase in ("fuse", "write"):
            ys = phase_series(phase)
            if any(ys):
                a, b, r2 = _fit_line(xs, ys)
                report.append(
                    f"  {phase:>10}: {a:7.3f} MB + {b:+.5f} MB/plane  R2={r2:5.3f}  "
                    f"(reported, not gated — dask graph term)"
                )

        print("\n".join(report))
        self.assertFalse(failures, "\n".join(failures + [""] + report))

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
