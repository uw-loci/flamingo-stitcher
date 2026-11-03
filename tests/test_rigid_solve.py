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

    def test_dropped_tiles_stay_shift_only(self):
        c = np.array([0.0, 5000.0, 5000.0])
        full = _rot_zyx(0.5)
        view = self.pipe._centre_shift_view([full], [c])[0]
        out = self.pipe._recombine_rotation([full], [view], [c], drop={0})[0]
        np.testing.assert_allclose(out[:3, :3], np.eye(3))

    def test_clamped_and_orphan_tiles_drop_rotation_carried_tiles_keep_it(self):
        """Carried tiles that followed their neighbours' rotation keep it
        (without it an edge tile lands 20-40 µm out on a sample that turned
        between rows, Ket_Trial1 run 29); orphans and clamped tiles do not."""
        from types import SimpleNamespace as NS

        coverage = NS(bound_tiles=[3, 5, 7])          # 3, 5 carried; 7 orphan
        carry = NS(rotations={3: np.eye(3), 5: np.eye(3)})
        clamp = NS(records=[NS(index=2, clamped_xy=True, clamped_z=False, whole_matrix=False)])
        assert StitchingPipeline._rotation_drop_set(coverage, clamp, carry) == {2, 7}
        # The `default` approach binds with no carry: everything bound drops.
        assert StitchingPipeline._rotation_drop_set(coverage, clamp, None) == {2, 3, 5, 7}

    def test_carried_rotation_is_written_into_the_full_params(self):
        c = np.array([0.0, 5000.0, 5000.0])
        full = [np.eye(4), np.eye(4)]
        R = _rot_zyx(0.7)[:3, :3]
        out = self.pipe._with_rotations(full, {1: R})
        np.testing.assert_allclose(out[1][:3, :3], R)
        np.testing.assert_allclose(full[1], np.eye(4))  # input untouched
        view = self.pipe._centre_shift_view([out[1]], [c])[0]
        view[:3, 3] = [1.0, 2.0, 3.0]
        back = self.pipe._recombine_rotation([out[1]], [view], [c], drop=set())[0]
        np.testing.assert_allclose(back[:3, :3] @ c + back[:3, 3] - c, [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------
# Judging a seam where the tiles meet, not at their centres
# --------------------------------------------------------------------------


def _tilt_zx(deg):
    """3x3 in (z, y, x) order: Z changes along X (a tilt out of the XY plane)."""
    a = np.radians(deg)
    m = np.eye(3)
    m[0, 0], m[0, 2] = np.cos(a), -np.sin(a)
    m[2, 0], m[2, 2] = np.sin(a), np.cos(a)
    return m


def _disagree(measured, a, b, Ra, Rb, ca, cb):
    """What the registration map computes: measured minus placed, at the seam."""
    pipe = StitchingPipeline(StitchingConfig())
    da = Ra @ ca - ca   # centre shifts of a rigid mosaic motion (t = 0)
    db = Rb @ cb - cb
    term = pipe._rotation_at_seam_um((Ra, ca), (Rb, cb))
    return measured - (da - db) - term, measured - (da - db)


@pytest.mark.parametrize("R", [_rot_zyx(1.25)[:3, :3], _tilt_zx(0.64)])
def test_a_mosaic_rotated_as_one_body_leaves_no_seam_off(R):
    """Every seam honoured: the measured pairwise shift is 0 and both tiles
    share one rotation about the world origin. The centre shifts still differ
    by (R - I)(c_a - c_b), about angle x pitch: run 35 listed seams 39-49 µm
    off from exactly this (1.17-1.32° tiles, 1931 µm pitch)."""
    ca = np.array([18000.0, 19207.0, 5596.0])
    cb = np.array([18000.0, 19207.0, 7527.0])      # x neighbour, 1931 µm on
    fixed, old = _disagree(np.zeros(3), 0, 1, R, R, ca, cb)
    np.testing.assert_allclose(fixed, 0.0, atol=1e-9)
    assert np.linalg.norm(old) > 20.0               # what the old check reported


def test_no_rotation_adds_nothing():
    pipe = StitchingPipeline(StitchingConfig())
    c = np.array([0.0, 1000.0, 2000.0])
    np.testing.assert_allclose(
        pipe._rotation_at_seam_um((np.eye(3), c), (np.eye(3), c + 1931.0)), 0.0
    )
    np.testing.assert_allclose(pipe._rotation_at_seam_um(None, None), 0.0)


def test_a_real_misplacement_is_still_reported():
    """The term removes the rotation's share, not a genuine error: shift tile
    b by 15 µm in Z beyond the rigid motion and the check sees -15 µm."""
    R = _rot_zyx(1.0)[:3, :3]
    ca = np.array([18000.0, 19207.0, 5596.0])
    cb = np.array([18000.0, 19207.0, 7527.0])
    pipe = StitchingPipeline(StitchingConfig())
    da = R @ ca - ca
    db = R @ cb - cb + np.array([15.0, 0.0, 0.0])
    off = np.zeros(3) - (da - db) - pipe._rotation_at_seam_um((R, ca), (R, cb))
    np.testing.assert_allclose(off, [15.0, 0.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("extra_um", [0.0, 15.0])
@pytest.mark.parametrize("R", [_rot_zyx(1.25)[:3, :3], _tilt_zx(0.64)])
def test_the_check_agrees_with_multiview_stitchers_own_residual(R, extra_um):
    """Sign and size checked against the library, not against this file.

    multiview-stitcher scores a seam by placing the overlap's corners with
    both tiles' final affines (``compute_edge_residuals``). The map's number,
    with the rotation term, must match it; without the term it did not.
    """
    pytest.importorskip("multiview_stitcher")
    import networkx as nx
    import xarray as xr
    from multiview_stitcher.param_resolution.utils import compute_edge_residuals

    ca = np.array([3750.0, 19207.0, 5596.0])
    cb = np.array([3750.0, 19207.0, 7527.0])
    mid = 0.5 * (ca + cb)
    half = np.array([3750.0, 1070.0, 107.0])          # the shared strip
    s = np.array([11.25, 0.0, -16.0])                 # measured pairwise shift
    T = np.eye(4)
    T[:3, 3] = s

    def affine(R_, c, d):                              # rotate about c, centre moves d
        m = np.eye(4)
        m[:3, :3] = R_
        m[:3, 3] = c + d - R_ @ c
        return m

    da = np.array([5.0, -3.0, 2.0])
    db = da - s + np.array([extra_um, 0.0, 0.0])      # honour s, then add an error
    # Rigid body: b's centre also follows the rotation about a's centre.
    db = db + (R - np.eye(3)) @ (cb - ca)
    Aa, Ab = affine(R, ca, da), affine(R, cb, db)

    g = nx.Graph()
    g.add_edge(0, 1, bbox=xr.DataArray(np.stack([mid - half, mid + half])),
               transform=xr.DataArray(T))
    mvs = compute_edge_residuals(g, {0: Aa, 1: Ab}, ndim=3)[(0, 1)]

    pipe = StitchingPipeline(StitchingConfig())
    ours = s - (da - db) - pipe._rotation_at_seam_um((R, ca), (R, cb))
    old = s - (da - db)
    assert abs(np.linalg.norm(ours) - mvs) < 1.0, (ours, mvs)
    if extra_um == 0.0:
        assert np.linalg.norm(old) > 20.0             # the false alarm it replaces


def test_rotations_by_tile_keeps_every_tile_with_a_centre():
    pipe = StitchingPipeline(StitchingConfig())
    rot = np.eye(4)
    rot[:3, :3] = _rot_zyx(0.5)[:3, :3]
    out = pipe._rotations_by_tile(
        [np.eye(4), rot, np.eye(4)],
        [np.zeros(3), np.ones(3), None],
    )
    assert sorted(out) == [0, 1]
    np.testing.assert_allclose(out[0][0], np.eye(3))
    np.testing.assert_allclose(out[1][0], rot[:3, :3])


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
