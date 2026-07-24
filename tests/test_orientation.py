"""Whole-mosaic orientation model + MIP-mosaic preview.

Covers:
  * the eight dihedral orientations (distinct, correct, volume matches image),
  * microscope-name reading + output_orientation preset lookup,
  * building a MIP-mosaic preview from per-tile *_MP.tif files and generating
    all eight orientation previews.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import tifffile  # noqa: E402

from flamingo_stitcher.orientation import (  # noqa: E402
    MosaicOrientation,
    build_mip_mosaic,
    orientation_previews,
    preset_for_microscope,
    read_microscope_name,
)


# --------------------------------------------------------------------------- #
# Dihedral orientation transforms
# --------------------------------------------------------------------------- #
def test_all_eight_orientations_are_distinct():
    # An asymmetric image so every dihedral element gives a different result.
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    results = [MosaicOrientation(n).apply2d(img) for n in MosaicOrientation.NAMES]
    seen = []
    for r in results:
        assert not any(
            r.shape == s.shape and np.array_equal(r, s) for s in seen
        ), "orientations must be distinct"
        seen.append(r)
    assert len(results) == 8


def test_orientation_specific_transforms():
    img = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert np.array_equal(MosaicOrientation("identity").apply2d(img), img)
    assert np.array_equal(
        MosaicOrientation("flip_h").apply2d(img), np.array([[2, 1], [4, 3]])
    )
    assert np.array_equal(
        MosaicOrientation("flip_v").apply2d(img), np.array([[3, 4], [1, 2]])
    )
    assert np.array_equal(
        MosaicOrientation("transpose").apply2d(img), np.array([[1, 3], [2, 4]])
    )
    assert np.array_equal(
        MosaicOrientation("rot180").apply2d(img), np.array([[4, 3], [2, 1]])
    )


def test_volume_xy_matches_per_plane_image_transform():
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 100, size=(3, 5, 4)).astype(np.uint16)
    for name in MosaicOrientation.NAMES:
        ori = MosaicOrientation(name)
        out = ori.apply_volume_xy(vol)
        for z in range(vol.shape[0]):
            assert np.array_equal(out[z], ori.apply2d(vol[z])), name


def test_unknown_orientation_raises_but_from_name_is_lenient():
    with pytest.raises(ValueError):
        MosaicOrientation("sideways")
    assert MosaicOrientation.from_name("sideways").name == "identity"
    assert MosaicOrientation.from_name(None).name == "identity"


# --------------------------------------------------------------------------- #
# Microscope name + presets
# --------------------------------------------------------------------------- #
def test_read_microscope_name_from_scope_settings(tmp_path):
    acq = tmp_path / "acq"
    acq.mkdir()
    (acq / "ScopeSettings.txt").write_text(
        "<Type>\n  Microscope name = n7\n  Objective lens magnification = 25.69\n"
    )
    assert read_microscope_name(acq) == "n7"


def test_preset_for_bundled_microscope():
    # The bundled microscope_hardware.yaml ships an n7 tile_orientation preset
    # (flip_h — the legacy per-tile X-flip).
    assert preset_for_microscope("n7") == "flip_h"
    assert preset_for_microscope("N7") == "flip_h"  # case-insensitive
    assert preset_for_microscope("no-such-scope") is None


# --------------------------------------------------------------------------- #
# MIP-mosaic preview
# --------------------------------------------------------------------------- #
_AOI = 8
_N_PLANES = 2


def _write_flat_acq_with_mips(tmp_path: Path) -> Path:
    """2×2 flat tiles, each with a raw + a distinctive *_MP.tif companion."""
    acq = tmp_path / "acq_prev"
    acq.mkdir()
    (acq / "Workflow.txt").write_text(
        "<Camera Settings>\n  AOI width = 8\n  AOI height = 8\n"
        "</Camera Settings>\n"
        "<Start Position>\n  X (mm) = 0.0\n  Y (mm) = 0.0\n  Z (mm) = 0.0\n"
        "</Start Position>\n"
        "<End Position>\n  X (mm) = 0.7\n  Y (mm) = 0.7\n  Z (mm) = 1.0\n"
        "</End Position>\n"
    )
    raw_bytes = b"\x01" * (_N_PLANES * _AOI * _AOI * 2)
    # Stage grid ~0.7 mm apart so tiles tile at pixel_size 100 µm (fov 0.8 mm).
    coords = {(0, 0): (0.0, 0.0), (1, 0): (0.7, 0.0),
              (0, 1): (0.0, 0.7), (1, 1): (0.7, 0.7)}
    for (xi, yi), (xmm, ymm) in coords.items():
        base = (
            f"S000_t000000_V000_R0000_X{xi:03d}_Y{yi:03d}"
            f"_C02_I0_D1_P{_N_PLANES:05d}"
        )
        (acq / f"{base}.raw").write_bytes(raw_bytes)
        # Asymmetric per-tile content (a corner ramp) so a per-tile transform
        # (flip/rot/transpose) actually changes the mosaic — a uniform tile
        # would look identical under every orientation.
        ramp = np.add.outer(
            np.arange(_AOI, dtype=np.uint16) * 100,
            np.arange(_AOI, dtype=np.uint16) * 10,
        )
        mip = (ramp + (xi * 2 + yi + 1) * 1000).astype(np.uint16)
        tifffile.imwrite(str(acq / f"{base}_MP.tif"), mip)
        # settings companion so discovery reads real positions
        (acq / f"{base}_Settings.txt").write_text(
            "<Start Position>\n"
            f"  X (mm) = {xmm}\n  Y (mm) = {ymm}\n  Z (mm) = 0.0\n"
            "</Start Position>\n<End Position>\n  Z (mm) = 1.0\n</End Position>\n"
        )
    return acq


def test_build_mip_mosaic_and_previews(tmp_path):
    acq = _write_flat_acq_with_mips(tmp_path)
    mosaic = build_mip_mosaic(acq, pixel_size_um=100.0, target_long_px=200)
    assert mosaic is not None
    assert mosaic.ndim == 2
    # Normalised to [0, 1] and actually contains signal.
    assert 0.0 <= float(mosaic.min())
    assert float(mosaic.max()) == pytest.approx(1.0, abs=1e-6)
    assert mosaic.any()

    previews = orientation_previews(acq, pixel_size_um=100.0, target_long_px=200)
    assert set(previews) == set(MosaicOrientation.NAMES)
    # Per-tile transforms keep the SAME placement grid, so every preview shares
    # one canvas shape — the transform reorients tile CONTENT, not the mosaic.
    shapes = {p.shape for p in previews.values()}
    assert len(shapes) == 1
    # Different orientations produce different mosaics (tiles reoriented in place).
    assert not np.array_equal(previews["identity"], previews["rot180"])
    assert not np.array_equal(previews["identity"], previews["transpose"])


def test_build_mip_mosaic_no_tiles_returns_none(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build_mip_mosaic(empty) is None


def test_build_mip_mosaic_z_range_reads_raw(tmp_path):
    """A Z sub-range projects from the raw stack (not the full *_MP.tif)."""
    acq = _write_flat_acq_with_mips(tmp_path)
    m = build_mip_mosaic(
        acq, pixel_size_um=100.0, target_long_px=200, z_range=(0.0, 0.5)
    )
    assert m is not None and m.ndim == 2 and m.any()


def test_tile_label_from_index():
    from flamingo_stitcher.orientation import _tile_label

    class _T:
        tile_index = (3, 5)

    class _T2:
        tile_index = None

    assert _tile_label(_T()) == "X3Y5"
    assert _tile_label(_T2()) == ""


def test_plane_bounds():
    from flamingo_stitcher.orientation import _plane_bounds

    assert _plane_bounds(100, None)[:2] == (0, 100)
    z0, z1, _ = _plane_bounds(100, (0.0, 0.25))
    assert (z0, z1) == (0, 25)
    z0, z1, _ = _plane_bounds(100, (0.75, 1.0))
    assert (z0, z1) == (75, 100)
    # A sub-range always yields at least one plane.
    z0, z1, _ = _plane_bounds(4, (0.9, 0.95))
    assert z1 > z0
