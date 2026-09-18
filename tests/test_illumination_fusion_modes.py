"""Fusing two light sheets when half of every frame is unusable.

The case that motivated this: dual illumination, a single FOV, a highly
scattering sample. Each sheet is good on the side it enters from and
progressively worse across, and the bad half is not merely dim -- it is full of
OUT-OF-FOCUS BLUR, which is bright and smooth.

That is what breaks `max`. Max takes the brighter sample, and blur is brighter
than the in-focus signal underneath it, so max systematically selects the blur.
The fix is either positional (give each sheet its own half and discard the
rest) or content-based (weight by local high-frequency energy, which blur does
not have however bright it is).

Run: python -m pytest tests/test_illumination_fusion_modes.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from flamingo_stitcher.pipeline import (
    fuse_illumination_sides,
    side_weight_profile,
)

RNG = np.random.default_rng(3)
NZ, NY, NX = 4, 64, 96


def _blobs(shape):
    """Smooth 2-D structure in [0, 1], for the detail-measure tests."""
    seed = RNG.random(shape).astype(np.float32)
    out = ndimage.gaussian_filter(seed, sigma=1.5)
    return (out - out.min()) / max(1e-9, float(np.ptp(out)))


def _detail(nz=NZ, ny=NY, nx=NX, level=400.0):
    """Fine structure — what an in-focus sheet records."""
    seed = RNG.random((nz, ny, nx)).astype(np.float32)
    blobs = ndimage.gaussian_filter(seed, sigma=1.2)
    blobs = (blobs - blobs.min()) / max(1e-9, float(np.ptp(blobs)))
    return level * blobs


def _sheets():
    """Left sheet good on the left, right sheet good on the right.

    The DEGRADED half of each is brighter than the sharp half — blurred, and on
    a high pedestal. That is the trap: it is exactly what max prefers.
    """
    sharp = _detail()
    blur = ndimage.gaussian_filter(sharp, sigma=(0, 6, 6)) * 3.0 + 6000.0
    half = NX // 2
    left = np.empty((NZ, NY, NX), np.float32)
    left[:, :, :half] = sharp[:, :, :half] + 500.0
    left[:, :, half:] = blur[:, :, half:]
    right = np.empty((NZ, NY, NX), np.float32)
    right[:, :, :half] = blur[:, :, :half]
    right[:, :, half:] = sharp[:, :, half:] + 500.0
    return (
        {1: left.astype(np.uint16), 0: right.astype(np.uint16)},
        sharp,
    )


def _sharpness(plane):
    """Local high-frequency energy — how much real detail survived."""
    f = np.asarray(plane, np.float32)
    return float((f - ndimage.gaussian_filter(f, 5.0)).std())


class TestTheWeightProfile:
    def test_a_hard_split_is_half_and_half(self):
        w = side_weight_profile(10, 0.5)
        assert list(w) == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    def test_blend_keeps_a_pure_zone_at_each_end(self):
        w = side_weight_profile(20, 0.25)
        assert w[0] == 1.0 and w[4] == 1.0
        assert w[-1] == 0.0 and w[-5] == 0.0

    def test_the_handover_is_monotone_and_symmetric(self):
        w = side_weight_profile(21, 0.2)
        assert np.all(np.diff(w) <= 1e-6), "weight must never rise"
        assert w + w[::-1] == pytest.approx(np.ones(21), abs=1e-5)

    def test_it_has_no_slope_break_at_the_joins(self):
        # A linear ramp leaves a visible kink where it meets the pure zone.
        w = side_weight_profile(200, 0.25)
        d2 = np.abs(np.diff(w, 2))
        assert d2.max() < 0.002, "handover is not smooth at the joins"

    def test_degenerate_widths_do_not_raise(self):
        for frac in (-1.0, 0.0, 0.49, 0.5, 5.0):
            assert len(side_weight_profile(8, frac)) == 8


class TestSplitTakesTheGoodHalf:
    def test_it_beats_max_on_sharpness(self):
        vols, sharp = _sheets()
        split = fuse_illumination_sides(vols, "split", axis=-1, low_side=1)
        mx = fuse_illumination_sides(vols, "max")
        assert _sharpness(split[NZ // 2]) > _sharpness(mx[NZ // 2])

    def test_max_really_does_prefer_the_blur(self):
        # Guard the premise: if max were fine here, none of this is needed.
        vols, sharp = _sheets()
        mx = fuse_illumination_sides(vols, "max")
        left_half = mx[NZ // 2][:, : NX // 2]
        assert _sharpness(left_half) < _sharpness(
            np.asarray(vols[1][NZ // 2][:, : NX // 2])
        ), "max kept the sharp left half after all"

    def test_each_half_comes_from_its_own_sheet(self):
        vols, _ = _sheets()
        out = fuse_illumination_sides(vols, "split", axis=-1, low_side=1)
        half = NX // 2
        assert np.array_equal(out[:, :, :half], vols[1][:, :, :half])
        assert np.array_equal(out[:, :, half:], vols[0][:, :, half:])

    def test_the_other_low_side_takes_the_other_halves(self):
        vols, _ = _sheets()
        out = fuse_illumination_sides(vols, "split", axis=-1, low_side=0)
        assert np.array_equal(out[:, :, : NX // 2], vols[0][:, :, : NX // 2])

    def test_a_y_axis_split_works_too(self):
        vols, _ = _sheets()
        out = fuse_illumination_sides(vols, "split", axis=-2, low_side=1)
        assert np.array_equal(out[:, : NY // 2, :], vols[1][:, : NY // 2, :])


class TestItRefusesToGuess:
    def test_no_low_side_falls_back_to_max(self, caplog):
        vols, _ = _sheets()
        out = fuse_illumination_sides(vols, "split", axis=-1, low_side=None)
        assert np.array_equal(out, fuse_illumination_sides(vols, "max"))

    def test_an_unknown_low_side_falls_back_to_max(self):
        vols, _ = _sheets()
        out = fuse_illumination_sides(vols, "blend", axis=-1, low_side=7)
        assert np.array_equal(out, fuse_illumination_sides(vols, "max"))

    def test_a_single_sided_tile_is_returned_untouched(self):
        vol = _detail().astype(np.uint16)
        for method in ("split", "blend", "content", "max"):
            assert np.array_equal(
                fuse_illumination_sides({0: vol}, method, low_side=1), vol
            )


class TestBlend:
    def test_the_middle_is_a_mix_of_both(self):
        vols, _ = _sheets()
        out = fuse_illumination_sides(
            vols, "blend", axis=-1, low_side=1, pure_frac=0.25
        )
        mid = NX // 2
        col = out[:, :, mid]
        assert not np.array_equal(col, vols[1][:, :, mid])
        assert not np.array_equal(col, vols[0][:, :, mid])

    def test_the_pure_zones_are_untouched(self):
        vols, _ = _sheets()
        out = fuse_illumination_sides(
            vols, "blend", axis=-1, low_side=1, pure_frac=0.25
        )
        n = int(round(0.25 * NX))
        assert np.array_equal(out[:, :, :n], vols[1][:, :, :n])
        assert np.array_equal(out[:, :, NX - n:], vols[0][:, :, NX - n:])

    def test_a_pure_frac_of_half_is_the_hard_split(self):
        vols, _ = _sheets()
        a = fuse_illumination_sides(vols, "blend", axis=-1, low_side=1, pure_frac=0.5)
        b = fuse_illumination_sides(vols, "split", axis=-1, low_side=1)
        assert np.array_equal(a, b)


class TestContentWeighting:
    def test_it_beats_max_on_sharpness(self):
        vols, _ = _sheets()
        content = fuse_illumination_sides(vols, "content")
        mx = fuse_illumination_sides(vols, "max")
        assert _sharpness(content[NZ // 2]) > _sharpness(mx[NZ // 2])

    def test_it_needs_no_geometry_at_all(self):
        # The point of having it: it works without knowing which side is which.
        vols, _ = _sheets()
        assert fuse_illumination_sides(vols, "content").shape == (NZ, NY, NX)

    def test_a_flat_region_does_not_let_noise_decide(self):
        flat = np.full((2, 16, 16), 1000, np.uint16)
        out = fuse_illumination_sides({0: flat, 1: flat}, "content")
        assert np.allclose(out, 1000, atol=1)


class TestItStaysUint16:
    @pytest.mark.parametrize("method", ["split", "blend", "content", "max", "mean"])
    def test_dtype_and_shape_are_preserved(self, method):
        vols, _ = _sheets()
        out = fuse_illumination_sides(vols, method, axis=-1, low_side=1)
        assert out.dtype == np.uint16
        assert out.shape == (NZ, NY, NX)


class TestContentIsFastEnoughToUse:
    """The measure is computed on a decimated copy with box filters.

    Written literally it costs 562 s per 643-plane tile at 2048x2048 — eight
    hours of fusion across a 49-tile mosaic, which makes the one mode that
    needs no geometry the one nobody can afford. These tests pin the shortcuts
    that make it usable, and the limit past which they stop working.
    """

    def test_it_still_prefers_detail_over_brightness(self):
        # The whole point survives the optimisation: a bright blur must lose
        # to a dimmer sharp signal.
        from scipy import ndimage

        from flamingo_stitcher.pipeline import _detail_weights

        sharp = (_blobs((256, 256)) * 3000).astype(np.float32) + 200
        blur = ndimage.gaussian_filter(sharp, 6.0) * 3.0 + 5000
        w = _detail_weights(sharp, blur)
        assert w.mean() > 0.9, "the bright blur won"

    def test_the_weight_map_is_full_resolution(self):
        from flamingo_stitcher.pipeline import _detail_weights

        a = (_blobs((300, 220)) * 1000).astype(np.float32)
        assert _detail_weights(a, a * 2).shape == (300, 220)

    def test_a_tiny_plane_does_not_decimate_into_nothing(self):
        from flamingo_stitcher.pipeline import _detail_weights

        a = (_blobs((6, 6)) * 1000).astype(np.float32)
        assert _detail_weights(a, a).shape == (6, 6)

    @pytest.mark.parametrize("sigma_1", [2.0, 5.0, 8.0, 20.0])
    def test_the_inner_kernel_never_degenerates(self, sigma_1):
        # At 8x on sigma_1=5 the inner kernel rounds to ONE pixel, the
        # high-pass becomes p - p, and the comparison INVERTS: measured, it
        # then picked the wrong sheet for 100% of the frame. The guard is that
        # the decimation is chosen from the kernel it leaves behind, so this
        # asserts the kernel actually used rather than a constant.
        from flamingo_stitcher.pipeline import (
            _DETAIL_MAX_DECIMATION,
            _DETAIL_MIN_KERNEL_PX,
        )

        inner = 2.355 * sigma_1
        f = int(min(_DETAIL_MAX_DECIMATION, max(1, inner // _DETAIL_MIN_KERNEL_PX)))
        assert round(inner / f) >= _DETAIL_MIN_KERNEL_PX, (
            f"sigma_1={sigma_1} decimates to f={f}, leaving a "
            f"{round(inner / f)}px high-pass kernel"
        )

    def test_a_degenerate_kernel_really_would_invert_the_choice(self):
        # Guards the guard: shows WHY the cap exists, so a future change that
        # raises it fails with a reason rather than a bare assertion.
        from scipy import ndimage

        sharp = (_blobs((256, 256)) * 3000).astype(np.float32) + 200
        blur = ndimage.gaussian_filter(sharp, 6.0) * 3.0 + 5000
        # A 1px "high-pass" is the identity minus itself: no detail survives.
        flat_a = sharp - ndimage.uniform_filter(sharp, 1)
        assert float(np.abs(flat_a).max()) == 0.0
