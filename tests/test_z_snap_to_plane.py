"""A tile must never be resampled BETWEEN two acquired Z planes.

A Z step is 10 um where a pixel is ~1 um: a plane is the unit the data arrives
in, not a sample of a smooth axial signal. multiview-stitcher fuses with
``interpolation_order=1``, so a tile whose Z origin misses the output grid is
linear-interpolated from two acquired planes — and because the whole tile shifts
together, that mixes two planes across the ENTIRE XY field, which is the plane
people then analyse.

Measured here on a single-bright-plane phantom at a 10 um step (see
``test_measures_the_blur_it_prevents``): 0.96 um of shift keeps 91% of the peak,
2.72 um keeps 73%, half a plane keeps 51%. A whole-plane shift is exact.

Those are not hypothetical numbers: the 2026-09-03 3x3 run's Z refinement pass
(``registration_z_refine_upsample=10``, sub-plane by design) adjusted 8 of 9
tiles by a median 0.96 um and a max 2.72 um, so 8 tiles were resampled.

Run: python -m pytest tests/test_z_snap_to_plane.py -q
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("multiview_stitcher")
from multiview_stitcher import (  # noqa: E402
    fusion,
    msi_utils,
    param_utils,
    spatial_image_utils as si_utils,
)
from multiview_stitcher.fusion import max_fusion  # noqa: E402

from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline  # noqa: E402

VZ = 10.0
_VOXEL = {"z": VZ, "y": 2.095, "x": 2.095}
_NZ, _NY, _NX = 24, 48, 48
_DELTA, _BG, _SIG = 10, 100, 4000


def _params(dzs):
    return [
        param_utils.affine_from_translation(np.asarray((dz, 0.0, 0.0), float))
        for dz in dzs
    ]


def _dz(param):
    arr = np.asarray(param)
    return float((arr[0] if arr.ndim == 3 else arr)[0, 3])


def _tile_data(n, z_min_mm=None):
    vol = np.zeros((160, 8, 8), np.uint16)  # only .shape is read
    zs = z_min_mm if z_min_mm is not None else [16.931] * n
    return [(vol, SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=z)) for z in zs]


def _fuse_two_tiles(dz_um):
    """Fuse two real tiles, each a single bright Z plane, tile B shifted."""
    step_x = int(_NX * 0.85)
    sims = []
    for i in range(2):
        vol = np.full((_NZ, _NY, _NX), _BG, np.uint16)
        vol[_DELTA] = _SIG
        sim = si_utils.get_sim_from_array(
            vol,
            dims=["z", "y", "x"],
            scale=_VOXEL,
            translation={"z": 0.0, "y": 0.0, "x": i * step_x * _VOXEL["x"]},
            transform_key="metadata",
        )
        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        m = np.eye(4)
        m[0, 3] = dz_um if i == 1 else 0.0
        msi_utils.set_affine_transform(
            msim,
            param_utils.affine_to_xaffine(m, t_coords=[0]),
            transform_key="reg",
            base_transform_key="metadata",
        )
        sims.append(msi_utils.get_sim_from_msim(msim))
    arr = np.squeeze(
        np.asarray(
            fusion.fuse(sims, transform_key="reg", fusion_func=max_fusion)
            .compute()
            .data
        )
    )
    column = arr[:, _NY // 2, -4]  # deep inside tile B, outside the overlap
    lit = int((column > _BG + 50).sum())
    return lit, int(column.max())


class TestTheBlurIsReal:
    """Establish the damage first — the fix is only worth its cost if so."""

    @pytest.mark.parametrize(
        "dz_um, planes, min_peak_frac, max_peak_frac",
        [
            (0.00, 1, 0.99, 1.01),   # unshifted: exact
            (0.96, 2, 0.88, 0.94),   # the run's MEDIAN refinement
            (2.72, 2, 0.70, 0.78),   # the run's MAX refinement
            (5.00, 2, 0.48, 0.54),   # half a plane: split down the middle
            (10.00, 1, 0.99, 1.01),  # a WHOLE plane: exact again
        ],
    )
    def test_subplane_shift_smears_one_plane_across_two(
        self, dz_um, planes, min_peak_frac, max_peak_frac
    ):
        lit, peak = _fuse_two_tiles(dz_um)
        assert lit == planes, f"dz={dz_um}: {lit} lit planes, expected {planes}"
        frac = peak / _SIG
        assert min_peak_frac <= frac <= max_peak_frac, f"dz={dz_um}: peak {frac:.2f}"


class TestSnapRemovesIt:
    def test_snapped_shift_is_a_whole_number_of_planes(self):
        pipe = StitchingPipeline(StitchingConfig())
        dzs = [0.0, 0.96, 2.72, -4.5, 12.3, -26.0]
        out, n_moved, worst = pipe._snap_z_shifts_to_planes(
            _params(dzs), _tile_data(len(dzs)), _VOXEL
        )
        z_ref = min(dzs)
        for param, original in zip(out, dzs):
            offset = (_dz(param) - z_ref) / VZ
            assert abs(offset - round(offset)) < 1e-9, f"{original} -> {_dz(param)}"
        # The reference is the LOWEST tile (-26.0), so "already on the grid"
        # means a whole number of planes above it — which dz=0.0 is not (2.6
        # planes). Every tile but the reference moves here.
        assert n_moved == 5
        assert worst <= VZ / 2 + 1e-9

    def test_a_snapped_shift_fuses_without_blurring(self):
        """The end-to-end point: snap, then fuse, and the plane is intact."""
        pipe = StitchingPipeline(StitchingConfig())
        out, _n, _w = pipe._snap_z_shifts_to_planes(
            _params([0.0, 2.72]), _tile_data(2), _VOXEL
        )
        lit, peak = _fuse_two_tiles(_dz(out[1]))
        assert lit == 1, "snapped tile still spread across two planes"
        assert peak == _SIG, f"snapped tile lost intensity: {peak}/{_SIG}"

    def test_never_moves_a_tile_more_than_half_a_plane(self):
        pipe = StitchingPipeline(StitchingConfig())
        rng = np.random.default_rng(0)
        dzs = list(rng.uniform(-40, 40, 24))
        out, _n, worst = pipe._snap_z_shifts_to_planes(
            _params(dzs), _tile_data(len(dzs)), _VOXEL
        )
        assert worst <= VZ / 2 + 1e-9
        for param, original in zip(out, dzs):
            assert abs(_dz(param) - original) <= VZ / 2 + 1e-9

    def test_lowest_tile_is_the_reference_and_never_moves(self):
        pipe = StitchingPipeline(StitchingConfig())
        dzs = [3.3, -7.7, 1.1]
        out, _n, _w = pipe._snap_z_shifts_to_planes(
            _params(dzs), _tile_data(3), _VOXEL
        )
        assert _dz(out[1]) == pytest.approx(-7.7)


class TestOffGridTileOrigins:
    def test_per_tile_z_starts_are_snapped_too(self):
        """Collect Tiles gives tiles their own z_min; those can miss the grid
        on their own, and a zero registration shift would still interpolate."""
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data(2, z_min_mm=[16.931, 16.9343])  # 3.3 µm apart
        out, n_moved, worst = pipe._snap_z_shifts_to_planes(_params([0.0, 0.0]), td, _VOXEL)
        assert n_moved == 1
        assert _dz(out[1]) == pytest.approx(-3.3, abs=1e-6)
        assert worst == pytest.approx(3.3, abs=1e-6)


class TestTheOptOut:
    def test_disabled_leaves_the_measured_shifts_alone(self, caplog):
        pipe = StitchingPipeline(
            StitchingConfig(registration_z_snap_to_plane=False)
        )
        params = _params([0.0, 2.72])
        with caplog.at_level(logging.INFO, logger=pipe.logger.name):
            out = pipe._apply_z_snap(params, _tile_data(2), _VOXEL)
        assert _dz(out[1]) == pytest.approx(2.72)
        assert "Z snap OFF" in caplog.text

    def test_enabled_by_default_and_says_what_it_did(self, caplog):
        pipe = StitchingPipeline(StitchingConfig())
        assert pipe.config.registration_z_snap_to_plane is True
        with caplog.at_level(logging.INFO, logger=pipe.logger.name):
            out = pipe._apply_z_snap(_params([0.0, 2.72]), _tile_data(2), _VOXEL)
        assert _dz(out[1]) == pytest.approx(0.0)
        assert "Z snap: moved 1/2 tiles" in caplog.text


class TestDoesNotKillTheRun:
    def test_no_params(self):
        pipe = StitchingPipeline(StitchingConfig())
        assert pipe._snap_z_shifts_to_planes([], [], _VOXEL) == ([], 0, 0.0)

    def test_zero_z_voxel(self):
        pipe = StitchingPipeline(StitchingConfig())
        p = _params([2.72])
        out, n, _w = pipe._snap_z_shifts_to_planes(p, _tile_data(1), {"z": 0.0})
        assert out is p and n == 0

    def test_tile_info_without_z_min(self):
        pipe = StitchingPipeline(StitchingConfig())
        p = _params([2.72])
        td = [(np.zeros((4, 8, 8)), SimpleNamespace())]
        out, n, _w = pipe._snap_z_shifts_to_planes(p, td, _VOXEL)
        assert out is p and n == 0

class TestTheRealParamType:
    """`registration.register` returns xarray affines with a `t` dim, not bare
    arrays. That is the branch the rig takes, so it has to be the branch under
    test — a snap that only works on ndarrays would pass every other test here
    and do nothing on hardware."""

    @staticmethod
    def _xparams(dzs):
        out = []
        for dz in dzs:
            m = np.eye(4)
            m[0, 3] = dz
            out.append(param_utils.affine_to_xaffine(m, t_coords=[0]))
        return out

    def test_the_test_fixture_matches_what_mvs_returns(self):
        xp = self._xparams([2.72])[0]
        assert xp.dims == ("t", "x_in", "x_out")
        assert np.asarray(xp).ndim == 3

    def test_snapping_an_xarray_param_lands_on_the_grid(self):
        pipe = StitchingPipeline(StitchingConfig())
        out, n_moved, worst = pipe._snap_z_shifts_to_planes(
            self._xparams([0.0, 2.72, -4.5]), _tile_data(3), _VOXEL
        )
        assert n_moved == 2
        assert worst <= VZ / 2 + 1e-9
        z_ref = -4.5
        for param in out:
            offset = (_dz(param) - z_ref) / VZ
            assert abs(offset - round(offset)) < 1e-9, _dz(param)

    def test_it_stays_an_xarray_affine_with_its_dims(self):
        pipe = StitchingPipeline(StitchingConfig())
        out, _n, _w = pipe._snap_z_shifts_to_planes(
            self._xparams([0.0, 2.72]), _tile_data(2), _VOXEL
        )
        for param in out:
            assert hasattr(param, "dims"), "snap degraded the param to an ndarray"
            assert param.dims == ("t", "x_in", "x_out")

    def test_it_does_not_mutate_the_caller_s_params(self):
        pipe = StitchingPipeline(StitchingConfig())
        given = self._xparams([0.0, 2.72])
        pipe._snap_z_shifts_to_planes(given, _tile_data(2), _VOXEL)
        assert _dz(given[1]) == pytest.approx(2.72), "snap wrote through to the input"

    def test_only_z_moves(self):
        """X/Y come from a joint lateral peak; Z is the only axis this touches."""
        pipe = StitchingPipeline(StitchingConfig())
        m = np.eye(4)
        m[0, 3], m[1, 3], m[2, 3] = 2.72, 13.5, -7.25
        given = [
            param_utils.affine_to_xaffine(np.eye(4), t_coords=[0]),
            param_utils.affine_to_xaffine(m, t_coords=[0]),
        ]
        out, _n, _w = pipe._snap_z_shifts_to_planes(given, _tile_data(2), _VOXEL)
        moved = np.asarray(out[1])[0]
        assert moved[0, 3] == pytest.approx(0.0)      # Z snapped
        assert moved[1, 3] == pytest.approx(13.5)     # Y untouched
        assert moved[2, 3] == pytest.approx(-7.25)    # X untouched
        assert np.allclose(moved[:3, :3], np.eye(3))  # no rotation introduced
