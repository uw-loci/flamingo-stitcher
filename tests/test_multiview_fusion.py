"""Rotation-affine multi-view fusion, validated on a phantom.

The physical convention (rotation sign / center) is NOT yet confirmed on the
instrument — that needs a real two-angle acquisition (flip ``rotation_sign`` if
views come out mirrored). Here we validate the *machinery*: a view acquired at
angle θ, placed with the rotation affine, lands back in the common frame. The
decisive integration check: fusing an angle-0 view with an angle-180 view
carrying the same content (pre-rotated into the camera frame) reconstructs the
original volume when multi-view fusion is ON, and does NOT when it is OFF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from flamingo_stitcher.pipeline import (  # noqa: E402
    RawTileInfo,
    StitchingConfig,
    StitchingPipeline,
    _rotation_affine_zyx,
)


# --------------------------------------------------------------------------- #
# Pure affine math
# --------------------------------------------------------------------------- #
def test_affine_identity_at_zero():
    A = _rotation_affine_zyx(0.0, 8.0, 4.0)
    assert np.allclose(A, np.eye(4))


def test_affine_180_is_point_reflection_about_center():
    cx, cz = 8.0, 4.0
    A = _rotation_affine_zyx(180.0, cx, cz)
    # Homogeneous world point (z, y, x, 1); 180° about (cx,cz) → (2cz-z, y, 2cx-x)
    for z, y, x in [(0, 3, 0), (2, 1, 5), (4, 0, 8)]:
        out = A @ np.array([z, y, x, 1.0])
        assert np.allclose(out[:3], [2 * cz - z, y, 2 * cx - x], atol=1e-9)


def test_affine_roundtrip_inverse_sign():
    cx, cz = 8.0, 4.0
    fwd = _rotation_affine_zyx(37.0, cx, cz, rotation_sign=1.0)
    back = _rotation_affine_zyx(37.0, cx, cz, rotation_sign=-1.0)
    assert np.allclose(fwd @ back, np.eye(4), atol=1e-9)


# --------------------------------------------------------------------------- #
# Integration: two views fused through the real _fuse_channel seam
# --------------------------------------------------------------------------- #
def _rot180_xz(vol):
    """180° rotation in the X–Z plane (exact, grid-preserving): flip z and x."""
    return vol[::-1, :, ::-1].copy()


def _smooth_phantom(nz, ny, nx):
    """Spatially-structured volume (smooth blobs) — a 1-voxel misalignment stays
    highly correlated, unlike per-voxel white noise."""
    from _synth_acq import _phantom_field

    return _phantom_field((nz, ny, nx), seed=3).astype(np.uint16)


def _extract_footprint(fused, nz, ny, nx):
    """Pull the world [0..N-1] block out of a fused sim using its coords, so a
    fractional/shifted fused origin doesn't misalign the comparison."""
    arr = np.nan_to_num(np.asarray(fused)).squeeze()  # -> (z, y, x)

    def start(coord_vals):
        return int(np.argmin(np.abs(np.asarray(coord_vals) - 0.0)))

    z0 = start(fused.coords["z"].values)
    y0 = start(fused.coords["y"].values)
    x0 = start(fused.coords["x"].values)
    return arr[z0 : z0 + nz, y0 : y0 + ny, x0 : x0 + nx].astype(float)


def _fuse_two_views(multiview: bool):
    from multiview_stitcher import io as mvs_io

    nz, ny, nx = 9, 5, 17           # odd z/x => integer rotation center, exact 180°
    vol = _smooth_phantom(nz, ny, nx)

    def tile(angle):
        return RawTileInfo(
            folder=Path("."), x_mm=0.0, y_mm=0.0, z_min_mm=0.0, z_max_mm=0.0,
            n_planes=nz, illumination_sides=[0], angle_deg=angle,
        )

    voxel = {"z": 1.0, "y": 1.0, "x": 1.0}   # world µm == voxel index
    # Angle-0 view holds vol; angle-180 view holds vol pre-rotated into the
    # camera frame, at the SAME stage position. Rotation center = tile center.
    tile_data = [(vol, tile(0.0)), (_rot180_xz(vol), tile(180.0))]

    cfg = StitchingConfig(
        multiview_fusion=multiview,
        rotation_center_um=((nx - 1) / 2.0, (nz - 1) / 2.0),  # (x, z)
        skip_registration=True,
        tile_overlap_fusion="max",
    )
    pipe = StitchingPipeline(cfg)
    fused, _ = pipe._fuse_channel(
        tile_data, voxel, reg_params=[], transform_key=mvs_io.METADATA_TRANSFORM_KEY
    )
    return vol.astype(float), _extract_footprint(fused, nz, ny, nx)


def _corr(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1])


def test_multiview_on_reconstructs_original():
    vol, fused = _fuse_two_views(multiview=True)
    assert fused.shape == vol.shape
    assert _corr(vol, fused) > 0.98


def test_multiview_off_does_not_reconstruct():
    # With multi-view OFF the angle-180 view is placed by translation only, so the
    # fusion mixes vol with its 180° rotation and correlates far worse.
    vol, fused_off = _fuse_two_views(multiview=False)
    _, fused_on = _fuse_two_views(multiview=True)
    assert _corr(vol, fused_on) > _corr(vol, fused_off) + 0.2
