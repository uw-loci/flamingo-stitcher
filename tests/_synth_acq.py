"""Minimal, self-contained synthetic raw-acquisition generator for tests.

flamingo-stitcher's own CI must be able to build tiny acquisitions that exercise
the real discovery -> preprocess -> fuse -> write code paths without depending on
the Flamingo_Control ``py2flamingo.testing.phantom_dataset`` module. This is a
stripped-down port of that generator: just enough to satisfy ``discover_tiles``.

Produces the native Flamingo layout:
  <acq>/ScopeSettings.txt                      (Objective lens magnification)
  <acq>/X{x:.2f}_Y{y:.2f}/Workflow.txt         (Start/End Z, AOI width/height)
  <acq>/X{x:.2f}_Y{y:.2f}/S000_t000000_V000_R0000_X000_Y000_C{ch:02d}_I{s}_D1_P{P:05d}.raw

Each .raw is contiguous C-order uint16 (P, H, W), byte size exactly P*H*W*2.
Content is a deterministic bright-blob field cropped per tile so overlaps share
structure (registration works when enabled). Supports 1 or 2 illumination sides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

_UINT16_MAX = 65535


def _phantom_field(shape: Tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """Deterministic (Z, Y, X) uint16 volume: smooth background + bright blobs."""
    rng = np.random.default_rng(seed)
    z, y, x = shape
    vol = np.full(shape, 200, dtype=np.float32)
    n_blobs = max(4, (y * x) // 512)
    yy, xx = np.mgrid[0:y, 0:x]
    for _ in range(n_blobs):
        cy, cx = rng.integers(0, y), rng.integers(0, x)
        r2 = ((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
        blob = np.exp(-r2 / (2.0 * (max(y, x) / 12.0) ** 2)) * 3000.0
        for zi in range(z):
            vol[zi] += blob * (0.6 + 0.4 * rng.random())
    return np.clip(vol, 0, _UINT16_MAX).astype(np.uint16)


def _workflow_text(z_start_mm, z_end_mm, z_step_um, w, h) -> str:
    return "\n".join(
        [
            "<Workflow Settings>",
            "  <Start Position>",
            f"    Z (mm) = {z_start_mm:.4f}",
            "  </Start Position>",
            "  <End Position>",
            f"    Z (mm) = {z_end_mm:.4f}",
            "  </End Position>",
            f"  AOI width = {w}",
            f"  AOI height = {h}",
            f"  Plane spacing (um) = {z_step_um:.3f}",
            "</Workflow Settings>",
            "",
        ]
    )


def write_synth_acquisition(
    out_dir,
    *,
    grid: Tuple[int, int] = (2, 2),
    overlap: float = 0.15,
    n_planes: int = 16,
    channels: Sequence[int] = (1,),
    illum_sides: Sequence[int] = (0,),
    frame_size: Tuple[int, int] = (32, 32),
    pixel_size_um: float = 0.406,
    sensor_pixel_size_um: float = 6.5,
    z_start_mm: float = 10.0,
    z_step_um: float = 5.0,
    seed: int = 0,
    inject_border_step: Optional[dict] = None,
    tile_planes: Optional[dict] = None,
) -> Path:
    """Write a tiny synthetic raw acquisition. Returns the acquisition dir.

    ``inject_border_step`` (for border-QC tests) adds a constant to one tile's
    edge strip so that tile disagrees with its neighbor along that seam. Keys:
    ``tile=(ix,iy)``, ``edge`` in {left,right,top,bottom}, ``magnitude``,
    optional ``width_px`` (default = the overlap width) and ``z_slice=(z0,z1)``.

    ``tile_planes`` maps ``(ix, iy)`` to that tile's plane count, overriding
    ``n_planes``. This is what an acquisition with per-tile Z ranges looks like
    on disk: the scope images only the Z span where the sample actually is, so
    every tile can be a different depth. Tiles share ``z_start_mm`` and the Z
    step; only the end (and so the plane count) differs.
    """
    tile_planes = {tuple(k): int(v) for k, v in (tile_planes or {}).items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ny, nx = grid
    h, w = frame_size
    channels = [int(c) for c in channels]
    illum_sides = [int(s) for s in illum_sides]

    objective_mag = sensor_pixel_size_um / pixel_size_um
    (out_dir / "ScopeSettings.txt").write_text(
        "<Scope Settings>\n"
        f"  Objective lens magnification = {objective_mag:.4f}\n"
        "</Scope Settings>\n"
    )

    overlap = float(np.clip(overlap, 0.0, 0.9))
    step_px_x = max(1, int(round(w * (1.0 - overlap))))
    step_px_y = max(1, int(round(h * (1.0 - overlap))))
    field_w = step_px_x * (nx - 1) + w
    field_h = step_px_y * (ny - 1) + h
    # The field must be deep enough for the deepest tile; shallower tiles just
    # take a shorter slice of it, so overlapping tiles still agree plane-wise.
    max_planes = max([n_planes, *tile_planes.values()])
    field = _phantom_field((max_planes, field_h, field_w), seed=seed)

    fov_mm_x = pixel_size_um / 1000.0 * w
    fov_mm_y = pixel_size_um / 1000.0 * h
    stage_step_mm_x = fov_mm_x * (1.0 - overlap)
    stage_step_mm_y = fov_mm_y * (1.0 - overlap)
    z_end_mm = z_start_mm + (n_planes - 1) * (z_step_um / 1000.0)

    for iy in range(ny):
        for ix in range(nx):
            x_mm = ix * stage_step_mm_x
            y_mm = iy * stage_step_mm_y
            folder = out_dir / f"X{x_mm:.2f}_Y{y_mm:.2f}"
            folder.mkdir(parents=True, exist_ok=True)
            tile_n_planes = tile_planes.get((ix, iy), n_planes)
            tile_z_end_mm = (
                z_start_mm + (tile_n_planes - 1) * (z_step_um / 1000.0)
                if tile_planes
                else z_end_mm
            )
            (folder / "Workflow.txt").write_text(
                _workflow_text(z_start_mm, tile_z_end_mm, z_step_um, w, h)
            )
            y0, x0 = iy * step_px_y, ix * step_px_x
            crop = field[:tile_n_planes, y0 : y0 + h, x0 : x0 + w]
            if inject_border_step and tuple(inject_border_step["tile"]) == (ix, iy):
                spec = inject_border_step
                mag = float(spec["magnitude"])
                edge = spec.get("edge", "right")
                zc0, zc1 = spec.get("z_slice", (0, tile_n_planes))
                ow = max(1, int(round(w * overlap)))
                oh = max(1, int(round(h * overlap)))
                wpx = int(spec.get("width_px", ow if edge in ("left", "right") else oh))
                # along-seam extent (Y for left/right edges, X for top/bottom)
                a0, a1 = spec.get("along_slice", (0, h if edge in ("left", "right") else w))
                crop = crop.astype(np.float32).copy()
                if edge == "right":
                    crop[zc0:zc1, a0:a1, -wpx:] += mag
                elif edge == "left":
                    crop[zc0:zc1, a0:a1, :wpx] += mag
                elif edge == "bottom":
                    crop[zc0:zc1, -wpx:, a0:a1] += mag
                elif edge == "top":
                    crop[zc0:zc1, :wpx, a0:a1] += mag
                crop = np.clip(crop, 0, _UINT16_MAX).astype(field.dtype)
            for ci, ch in enumerate(channels):
                scale = 1.0 - 0.25 * ci
                for s in illum_sides:
                    # Give each illumination side a slightly different tilt so
                    # max/mean fusion has something non-trivial to combine.
                    side_bias = 1.0 + 0.1 * s
                    tile = np.clip(
                        crop.astype(np.float32) * scale * side_bias, 0, _UINT16_MAX
                    ).astype(np.uint16)
                    tile = np.ascontiguousarray(tile)
                    fname = (
                        f"S000_t000000_V000_R0000_X000_Y000_"
                        f"C{ch:02d}_I{s}_D1_P{tile_n_planes:05d}.raw"
                    )
                    tile.tofile(str(folder / fname))
    return out_dir
