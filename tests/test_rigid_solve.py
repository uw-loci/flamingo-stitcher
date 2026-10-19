"""Optional rigid global solve (shift + rotation).

2026-10-18 Ket_Trial1: the Z offset between the bottom two rows grew 105 ->
140 µm across four columns -- a tilt no per-tile shift can represent -- and
the translation-only solve kept 1 of 4 row-to-row seams. multiview-stitcher's
solve can also rotate tiles, inferring the rotation from where the different
neighbours' overlaps sit.

Everything after the solve (carry, clamp, Z snap, report, coverage) judges a
tile by its shift. For a rotated tile the translation column is the motion of
the world ORIGIN, so these work on translation-only copies holding each tile's
CENTRE displacement, and the rotation goes back on at the end.
"""

from __future__ import annotations

import csv
import logging

import numpy as np
import pytest

from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline


def _rot_zyx(deg_about_z):
    """4x4 in (z, y, x) order: rotation in the YX plane plus a tilt-free frame."""
    a = np.radians(deg_about_z)
    m = np.eye(4)
    m[1, 1], m[1, 2] = np.cos(a), -np.sin(a)
    m[2, 1], m[2, 2] = np.sin(a), np.cos(a)
    return m


class TestCentreShift:
    def setup_method(self):
        self.pipe = StitchingPipeline(StitchingConfig())

    def test_a_translation_is_its_own_centre_shift(self):
        m = np.eye(4)
        m[:3, 3] = [5.0, -3.0, 7.0]
        c = np.array([1000.0, 20000.0, 15000.0])
        view = self.pipe._centre_shift_view([m], [c])[0]
        np.testing.assert_allclose(view, m)

    def test_a_rotation_far_from_the_origin_is_mostly_not_a_shift(self):
        # 0.2 degrees about a tile 10 mm out: the translation column moves
        # ~35 µm, the tile centre does not move at all.
        c = np.array([500.0, 10000.0, 0.0])
        R = _rot_zyx(0.2)
        m = R.copy()
        m[:3, 3] = c - R[:3, :3] @ c          # rotation about the tile centre
        assert np.abs(m[:3, 3]).max() > 30
        view = self.pipe._centre_shift_view([m], [c])[0]
        np.testing.assert_allclose(view[:3, :3], np.eye(3))
        np.testing.assert_allclose(view[:3, 3], 0.0, atol=1e-9)

    def test_recombine_puts_the_rotation_back_around_the_new_shift(self):
        c = np.array([500.0, 10000.0, 3000.0])
        R = _rot_zyx(0.3)
        full = R.copy()
        full[:3, 3] = [4.0, 12.0, -9.0]
        view = self.pipe._centre_shift_view([full], [c])[0]
        # unchanged shift -> exactly the original transform
        back = self.pipe._recombine_rotation([full], [view], [c], drop=set())[0]
        np.testing.assert_allclose(back, full, atol=1e-9)
        # a later step moved the centre by +2.5 µm in Z (a snap, a refinement)
        moved = view.copy()
        moved[0, 3] += 2.5
        out = self.pipe._recombine_rotation([full], [moved], [c], drop=set())[0]
        np.testing.assert_allclose(out[:3, :3], R[:3, :3])
        np.testing.assert_allclose(out[:3, :3] @ c + out[:3, 3] - c, moved[:3, 3])

    def test_clamped_or_carried_tiles_stay_shift_only(self):
        c = np.array([0.0, 5000.0, 5000.0])
        full = _rot_zyx(0.5)
        view = self.pipe._centre_shift_view([full], [c])[0]
        out = self.pipe._recombine_rotation([full], [view], [c], drop={0})[0]
        np.testing.assert_allclose(out[:3, :3], np.eye(3))


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def _run(tmp_path, transform, caplog):
    pytest.importorskip("multiview_stitcher")
    from _synth_acq import write_synth_acquisition

    acq = write_synth_acquisition(
        tmp_path / f"acq_{transform}", grid=(2, 2), overlap=0.3, n_planes=16,
        frame_size=(64, 64), z_step_um=2.0, z_texture=True,
        tile_xy_offset_px={(1, 1): (3, -2)},
    )
    cfg = StitchingConfig.with_yaml_defaults()
    for k, v in dict(
        skip_registration=False, streaming_mode=True, output_format="ome-tiff",
        resource_guard_enabled=False, flat_field_correction=False, reg_channel=1,
        output_chunksize={"z": 4, "y": 16, "x": 16}, downsample_xy=1,
        downsample_z=1, tile_orientation="identity", camera_x_inverted=False,
        auto_tile_orientation=False, registration_solve_transform=transform,
        registration_z_spacing_um=0.0,
    ).items():
        setattr(cfg, k, v)
    out = tmp_path / f"out_{transform}"
    with caplog.at_level(logging.INFO):
        StitchingPipeline(cfg).run(acq, out)
    with (out / "registration_report.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows, caplog.text


def test_rigid_runs_end_to_end_and_agrees_with_shift_only_when_nothing_rotates(
    tmp_path, caplog
):
    rows_t, _ = _run(tmp_path, "translation", caplog)
    caplog.clear()
    rows_r, log = _run(tmp_path, "rigid", caplog)
    assert "Solve model: shift + rotation (rigid, experimental)" in log
    assert "Rigid solve: tile rotations" in log
    # Untouched synthetic tiles have nothing to rotate: the centre shifts the
    # report gives must match the shift-only solve to within a pixel.
    for a, b in zip(rows_t, rows_r):
        for k in ("dz_um", "dy_um", "dx_um"):
            assert abs(float(a[k]) - float(b[k])) <= 1.0, (k, a, b)
