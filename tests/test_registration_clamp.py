"""Unit tests for the post-registration shift clamp.

multiview-stitcher's phase correlation bounds a pairwise shift to the tile size,
not the overlap, so a low-content tile can be flung ~a full tile away and open a
gap. `_clamp_registration_shifts` reverts corrections it cannot plausibly have
measured. These tests drive the pure clamp with hand-built translation params.

Lateral and axial are bounded separately and the tests say why: X/Y share one
correlation peak so they revert together, while Z has no overlap width to lean
on and gets its own budget.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("multiview_stitcher")
from multiview_stitcher import param_utils  # noqa: E402

from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline  # noqa: E402


def _params(translations):
    # translation is (z, y, x) in µm; MVS puts it in the affine's last column.
    return [
        param_utils.affine_from_translation(np.asarray(t, float)) for t in translations
    ]


def _trans(param):
    arr = np.asarray(param)
    if arr.ndim == 3:
        arr = arr[0]
    return arr[:3, 3]


def _tile_data(xs, ys, shape=(8, 100, 100)):
    vol = np.zeros(shape, np.uint16)  # only .shape is read
    return [
        (vol, SimpleNamespace(x_mm=x, y_mm=y, z_min_mm=0.0)) for x, y in zip(xs, ys)
    ]


# 3 tiles in a row: pitch 80 µm, frame 100 µm (voxel 1 µm) -> overlap 20 µm.
_VOXEL = {"z": 1.0, "y": 1.0, "x": 1.0}

# A stack deep enough that the quarter-stack cap does not dominate the Z bound,
# so tests that mean to exercise the Z budget actually do.
_DEEP = (400, 100, 100)


class TestLateralBound:
    def test_auto_bound_reverts_overbudget_tile(self):
        pipe = StitchingPipeline(StitchingConfig())  # max_registration_shift_um=0
        td = _tile_data([0.0, 0.08, 0.16], [0.0, 0.0, 0.0])
        params = _params([(0, 2, 3), (0, 5, 60), (0, 1, -4)])  # middle exceeds 20 µm
        out = pipe._clamp_registration_shifts(params, td, _VOXEL).params
        assert np.max(np.abs(_trans(out[0]))) == pytest.approx(3, abs=1e-6)  # kept
        assert np.allclose(_trans(out[1]), 0.0)  # reverted to stage
        assert np.max(np.abs(_trans(out[2]))) == pytest.approx(4, abs=1e-6)  # kept

    def test_explicit_bound_overrides_auto(self):
        cfg = StitchingConfig()
        cfg.max_registration_shift_um = 10.0
        pipe = StitchingPipeline(cfg)
        td = _tile_data([0.0, 0.08], [0.0, 0.0])  # overlap 20 µm, but bound is 10
        params = _params([(0, 0, 8), (0, 0, 12)])
        out = pipe._clamp_registration_shifts(params, td, _VOXEL).params
        assert np.max(np.abs(_trans(out[0]))) == pytest.approx(8, abs=1e-6)
        assert np.allclose(_trans(out[1]), 0.0)  # 12 > 10 reverted

    def test_noop_when_all_within_bound(self):
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08, 0.16], [0.0, 0.0, 0.0])
        params = _params([(0, 1, 1), (0, 2, 2), (0, -3, 1)])  # all << 20 µm
        out = pipe._clamp_registration_shifts(params, td, _VOXEL).params
        for p_in, p_out in zip(params, out):
            assert np.allclose(_trans(p_in), _trans(p_out))

    def test_empty_params_returns_empty(self):
        pipe = StitchingPipeline(StitchingConfig())
        result = pipe._clamp_registration_shifts([], _tile_data([0.0], [0.0]), _VOXEL)
        assert result.params == []
        assert result.records == []

    def test_x_and_y_revert_together_because_they_share_one_peak(self):
        # A lateral phase-correlation peak is one measurement in two axes. If X
        # is garbage there is no reason to trust the Y that came out of the same
        # peak, so keeping it would be false precision.
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08], [0.0, 0.0])
        params = _params([(0, 0, 0), (0, 5, 60)])  # dy fine, dx way over
        out = pipe._clamp_registration_shifts(params, td, _VOXEL).params
        assert np.allclose(_trans(out[1]), 0.0)


class TestAxialBound:
    def test_the_z_bound_is_independent_of_the_lateral_one(self):
        # The whole point: a Z correction survives a run whose lateral budget is
        # far tighter than it. Before, one shared bound meant a 30 µm Z shift on
        # a 20 µm-overlap mosaic was discarded along with its good X/Y.
        cfg = StitchingConfig()
        cfg.max_registration_shift_z_um = 50.0
        pipe = StitchingPipeline(cfg)
        td = _tile_data([0.0, 0.08], [0.0, 0.0], shape=_DEEP)  # lateral bound 20 µm
        params = _params([(0, 0, 0), (30, 2, 3)])
        out = pipe._clamp_registration_shifts(params, td, _VOXEL).params
        assert _trans(out[1])[0] == pytest.approx(30, abs=1e-6)  # Z kept
        assert _trans(out[1])[1] == pytest.approx(2, abs=1e-6)  # Y kept
        assert _trans(out[1])[2] == pytest.approx(3, abs=1e-6)  # X kept

    def test_an_overbudget_z_leaves_the_lateral_correction_alone(self):
        cfg = StitchingConfig()
        cfg.max_registration_shift_z_um = 10.0
        pipe = StitchingPipeline(cfg)
        td = _tile_data([0.0, 0.08], [0.0, 0.0], shape=_DEEP)
        params = _params([(0, 0, 0), (99, 2, 3)])
        out = pipe._clamp_registration_shifts(params, td, _VOXEL).params
        assert _trans(out[1])[0] == pytest.approx(0.0, abs=1e-9)  # Z reverted
        assert _trans(out[1])[1] == pytest.approx(2, abs=1e-6)  # Y survives
        assert _trans(out[1])[2] == pytest.approx(3, abs=1e-6)  # X survives

    def test_an_overbudget_lateral_leaves_the_z_correction_alone(self):
        cfg = StitchingConfig()
        cfg.max_registration_shift_z_um = 50.0
        pipe = StitchingPipeline(cfg)
        td = _tile_data([0.0, 0.08], [0.0, 0.0], shape=_DEEP)
        params = _params([(0, 0, 0), (30, 5, 60)])
        out = pipe._clamp_registration_shifts(params, td, _VOXEL).params
        assert _trans(out[1])[0] == pytest.approx(30, abs=1e-6)  # Z survives
        assert np.allclose(_trans(out[1])[1:], 0.0)  # lateral reverted

    def test_the_auto_z_bound_admits_the_error_we_actually_see(self):
        # 3-6 frames of stage/focus drift is the reported symptom. At a 5 µm Z
        # step that is 15-30 µm, and a bound that rejected it would defeat the
        # feature. Deep stack so the quarter-stack cap is not the binding term.
        pipe = StitchingPipeline(StitchingConfig())
        voxel = {"z": 5.0, "y": 1.0, "x": 1.0}
        td = _tile_data([0.0, 0.08], [0.0, 0.0], shape=(500, 100, 100))
        params = _params([(0, 0, 0), (30, 0, 0)])
        out = pipe._clamp_registration_shifts(params, td, voxel).params
        assert _trans(out[1])[0] == pytest.approx(30, abs=1e-6)

    def test_a_shallow_stack_caps_the_bound_at_a_quarter_of_its_depth(self):
        # 40 planes x 1 µm = 40 µm deep. A "correction" of 30 µm is most of the
        # volume and is a garbage peak, not a measurement, whatever the step size.
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08], [0.0, 0.0], shape=(40, 100, 100))
        params = _params([(0, 0, 0), (30, 0, 0)])
        out = pipe._clamp_registration_shifts(params, td, _VOXEL).params
        assert _trans(out[1])[0] == pytest.approx(0.0, abs=1e-9)

    def test_the_bound_never_falls_below_one_binned_z_step(self):
        # Pass 1 registers at registration_binning["z"], so it can only express
        # multiples of that. A tighter bound would clamp pure quantization and
        # record it as a rejected measurement.
        cfg = StitchingConfig()
        cfg.registration_binning = {"z": 4, "y": 4, "x": 4}
        cfg.max_registration_shift_z_um = 0.5  # absurdly tight on purpose
        pipe = StitchingPipeline(cfg)
        bound, source = pipe._axial_shift_bound(depth_um=5000.0, vz=10.0)
        assert bound == pytest.approx(2.0 * 4 * 10.0)
        assert source == "config"


class TestClampRecords:
    def test_records_carry_the_pre_clamp_shift_not_the_zero(self):
        # "We measured +60 µm and did not believe it" and "we measured nothing"
        # are different findings; the report must be able to tell them apart.
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08], [0.0, 0.0])
        result = pipe._clamp_registration_shifts(
            _params([(0, 0, 0), (0, 5, 60)]), td, _VOXEL
        )
        assert result.records[1].dx_um == pytest.approx(60.0)
        assert result.records[1].clamped_xy is True
        assert result.records[1].clamped_z is False

    def test_records_are_index_aligned_with_the_params(self):
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08, 0.16], [0.0, 0.0, 0.0])
        result = pipe._clamp_registration_shifts(
            _params([(0, 1, 1), (0, 5, 60), (0, 2, 2)]), td, _VOXEL
        )
        assert [r.index for r in result.records] == [0, 1, 2]
        assert [r.clamped_xy for r in result.records] == [False, True, False]

    def test_the_bounds_and_their_provenance_are_reported(self):
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08], [0.0, 0.0])
        result = pipe._clamp_registration_shifts(_params([(0, 0, 0)] * 2), td, _VOXEL)
        assert result.bound_xy_um == pytest.approx(20.0)
        assert result.source_xy == "auto: min overlap width"
        assert result.source_z == "auto"

    def test_the_summary_line_says_a_clamped_axis_was_not_measured(self):
        # The border-QC lesson: a clamped value read as a small tidy offset once
        # before, and 57% of seams were actually unmeasured.
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08], [0.0, 0.0])
        result = pipe._clamp_registration_shifts(
            _params([(0, 0, 0), (0, 5, 60)]), td, _VOXEL
        )
        assert "NOT measured" in result.summary_line()


class TestNonTranslationalParams:
    def test_a_rotated_param_falls_back_to_a_whole_matrix_revert(self):
        # For a general affine the translation column is the motion of the world
        # ORIGIN, not of the tile, and the axes are coupled — so zeroing one
        # component would be meaningless surgery.
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08], [0.0, 0.0], shape=_DEEP)
        params = _params([(0, 0, 0), (0, 0, 0)])
        rotated = np.asarray(params[1]).copy()
        mat = rotated[0] if rotated.ndim == 3 else rotated
        mat[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        mat[:3, 3] = [0.0, 0.0, 900.0]
        result = pipe._clamp_registration_shifts(
            [params[0], rotated], td, _VOXEL
        )
        assert result.records[1].whole_matrix is True
        out = np.asarray(result.params[1])
        out = out[0] if out.ndim == 3 else out
        assert np.allclose(out[:3, :3], np.eye(3))
        assert np.allclose(out[:3, 3], 0.0)

    def test_a_within_budget_rotated_param_is_left_alone(self):
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08], [0.0, 0.0], shape=_DEEP)
        rotated = np.asarray(_params([(0, 0, 0)])[0]).copy()
        mat = rotated[0] if rotated.ndim == 3 else rotated
        mat[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        result = pipe._clamp_registration_shifts([rotated], td, _VOXEL)
        assert result.records[0].whole_matrix is False
        out = np.asarray(result.params[0])
        out = out[0] if out.ndim == 3 else out
        assert np.allclose(out[:3, :3], mat[:3, :3])


class TestTheBoundIsRelativeToWhereTilesWerePlaced:
    """A correction every tile shares opens no seam, so it must not be clamped.

    The clamp used to measure each tile against its ORIGINAL stage position.
    On a large mosaic a systematic offset accumulates past one overlap while
    every adjacent pair stays perfectly matched — so the far end of the grid
    got reverted for doing exactly what registration is for.
    """

    def test_a_mosaic_wide_offset_is_not_clamped(self):
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08, 0.16, 0.24, 0.32], [0.0] * 5)
        # Every tile moved 60 µm in X — three times the 20 µm overlap, but the
        # mosaic simply slid and not one seam changed.
        out = pipe._clamp_registration_shifts(_params([(0, 0, 60)] * 5), td, _VOXEL)
        for param in out.params:
            assert _trans(param)[2] == pytest.approx(60, abs=1e-6)

    def test_a_single_tile_flung_away_is_still_clamped(self):
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08, 0.16, 0.24, 0.32], [0.0] * 5)
        shifts = [(0, 0, 60), (0, 0, 60), (0, 0, 400), (0, 0, 60), (0, 0, 60)]
        out = pipe._clamp_registration_shifts(_params(shifts), td, _VOXEL)
        assert out.records[2].clamped_xy is True
        assert np.allclose(_trans(out.params[2]), 0.0)
        # ...and its well-behaved neighbours are untouched.
        assert _trans(out.params[0])[2] == pytest.approx(60, abs=1e-6)

    def test_outliers_cannot_drag_the_baseline_they_are_judged_against(self):
        # Two bad tiles out of five: the median holds, so both are caught.
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08, 0.16, 0.24, 0.32], [0.0] * 5)
        shifts = [(0, 0, 0), (0, 0, 300), (0, 0, 0), (0, 0, -300), (0, 0, 0)]
        out = pipe._clamp_registration_shifts(_params(shifts), td, _VOXEL)
        assert [r.clamped_xy for r in out.records] == [
            False, True, False, True, False,
        ]

    def test_the_record_keeps_both_the_absolute_and_the_relative_shift(self):
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08, 0.16], [0.0] * 3)
        out = pipe._clamp_registration_shifts(
            _params([(0, 0, 50), (0, 0, 50), (0, 0, 130)]), td, _VOXEL
        )
        assert out.records[2].dx_um == pytest.approx(130.0)  # proposed
        assert out.records[2].rel_x_um == pytest.approx(80.0)  # judged

    def test_two_tiles_fall_back_to_the_absolute_bound(self):
        # No majority, so no consensus: with two disagreeing tiles there is no
        # way to say which one moved, and the stage position is all there is.
        pipe = StitchingPipeline(StitchingConfig())
        td = _tile_data([0.0, 0.08], [0.0, 0.0])
        out = pipe._clamp_registration_shifts(_params([(0, 0, 60)] * 2), td, _VOXEL)
        assert all(np.allclose(_trans(p), 0.0) for p in out.params)
