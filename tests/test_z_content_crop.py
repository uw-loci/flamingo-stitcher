"""Registering on the Z planes that actually contain something.

Measured on a two-tile phantom with INDEPENDENT per-tile noise (sharing the
noise makes empty planes correlate perfectly with each other and hides the
whole effect), with content occupying 25% / 10% / 5% of the stack:

    recovered shift   exactly right in all three cases
    seam quality      0.72 / 0.52 / 0.43   (full stack)
                      0.99 / 0.91 / 0.76   (cropped to content)

So empty planes do not push tiles around -- they dilute the rank-correlation
quality score until a seam is rejected for a reason unrelated to its alignment.
At 5% content that lands on a 0.4 threshold.

The dangerous failure mode here is NOT a bad crop, it is a crop whose
translation offset is wrong: that shifts every tile by its own crop and looks
exactly like a registration result. Hence the world-position tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from flamingo_stitcher import tile_content as tc

RNG = np.random.default_rng(7)


def slab_stack(n_planes=160, c0=60, c1=100, amp=800.0, noise=20.0, size=64):
    """A stack with structure only in planes [c0, c1)."""
    from scipy import ndimage

    vol = RNG.normal(300.0, noise, (n_planes, size, size)).astype(np.float32)
    body = ndimage.gaussian_filter(
        RNG.random((c1 - c0, size, size)).astype(np.float32), sigma=2.5
    )
    body = (body - body.min()) / max(1e-9, float(np.ptp(body)))
    vol[c0:c1] += (amp * body).astype(np.float32)
    return vol


class TestTheProfile:
    def test_it_finds_the_slab(self):
        profile = tc.plane_structure_profile(slab_stack())
        assert profile is not None and len(profile) == 160
        assert profile[60:100].mean() > profile[:50].mean() * 2

    def test_empty_planes_form_a_flat_floor_well_below_the_content(self):
        # NOT compared against DEFAULT_MIN_STRUCTURE: that constant is
        # calibrated for the 3-D score, which smooths across Z as well and so
        # sits far lower (~0.088 vs ~0.195 here). Mixing the two is exactly the
        # error that silently disables the crop, so the assertion is about
        # separation, which is what content_z_range actually uses.
        profile = tc.plane_structure_profile(slab_stack())
        floor, content = profile[:40], profile[65:95]
        assert float(np.ptp(floor)) < 0.1          # flat
        assert content.min() > floor.max() * 3     # and far below the sample

    def test_a_stack_too_thin_to_profile_returns_none(self):
        assert tc.plane_structure_profile(np.zeros((2, 8, 8), np.float32)) is None


class TestTheRange:
    def test_it_brackets_the_content(self):
        z0, z1 = tc.content_z_range(slab_stack())
        # Margin either side, but it must contain the slab and not the whole
        # stack.
        assert z0 <= 60 and z1 >= 100
        assert z0 >= 60 - 2 * tc._Z_MARGIN_PLANES
        assert z1 <= 100 + 2 * tc._Z_MARGIN_PLANES

    def test_content_filling_the_stack_is_not_cropped(self):
        # Nothing to save, and the case where the profile is least trustworthy.
        full = slab_stack(n_planes=100, c0=0, c1=100)
        assert tc.content_z_range(full) is None

    def test_an_empty_stack_is_not_cropped(self):
        # None means "do not crop" -- the content gate decides what to do with
        # a tile that has nothing, and it is not this function's call.
        flat = RNG.normal(3000.0, 40.0, (120, 64, 64)).astype(np.float32)
        assert tc.content_z_range(flat) is None

    def test_a_narrower_slab_crops_harder(self):
        wide = tc.content_z_range(slab_stack(c0=40, c1=120))
        narrow = tc.content_z_range(slab_stack(c0=70, c1=90))
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


class TestTheCropKeepsWorldPosition:
    """The failure that would be invisible: cropping without moving the
    translation shifts every tile by its own crop, which looks like a
    registration result rather than a bug."""

    def _sim_z_origin(self, pipe, tile_data, voxel):
        import dask.array as da
        from multiview_stitcher import io as mvs_io
        from multiview_stitcher import spatial_image_utils as si_utils

        z_ranges = pipe._content_z_ranges(tile_data)
        volume, tile_info = tile_data[0]
        translation_z = tile_info.z_min_mm * 1000.0
        if z_ranges and z_ranges[0] is not None:
            z0, _z1 = z_ranges[0]
            volume = volume[z0:]
            translation_z += z0 * voxel["z"]
        sim = si_utils.get_sim_from_array(
            da.from_array(volume, chunks=(16, 32, 32)),
            dims=["z", "y", "x"], scale=voxel,
            translation={"z": translation_z, "y": 0.0, "x": 0.0},
            transform_key=mvs_io.METADATA_TRANSFORM_KEY,
        )
        return float(sim.coords["z"].values[0])

    def test_the_first_retained_plane_keeps_its_world_z(self):
        pytest.importorskip("multiview_stitcher")
        from types import SimpleNamespace

        from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

        voxel = {"z": 2.0, "y": 1.0, "x": 1.0}
        vol = slab_stack()
        tile_data = [(vol, SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=10.0))]

        cropped = StitchingPipeline(StitchingConfig())
        whole = StitchingPipeline(
            StitchingConfig(registration_z_content_crop=False)
        )
        z_cropped = self._sim_z_origin(cropped, tile_data, voxel)
        z_whole = self._sim_z_origin(whole, tile_data, voxel)

        z0, _z1 = tc.content_z_range(vol)
        # The cropped sim starts exactly z0 planes further down the stack, in
        # world units -- not at the same place, and not somewhere arbitrary.
        assert z_cropped - z_whole == pytest.approx(z0 * voxel["z"])

    def test_disabling_it_uses_the_whole_stack(self):
        from types import SimpleNamespace

        from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

        pipe = StitchingPipeline(
            StitchingConfig(registration_z_content_crop=False)
        )
        tile_data = [
            (slab_stack(), SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=10.0))
        ]
        assert pipe._content_z_ranges(tile_data) is None


class TestItNeverBreaksARun:
    def test_a_volume_it_cannot_profile_is_left_whole(self):
        from types import SimpleNamespace

        from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

        pipe = StitchingPipeline(StitchingConfig())
        bad = np.zeros((2, 2, 2), np.float32)
        tile_data = [(bad, SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=0.0))]
        assert pipe._content_z_ranges(tile_data) is None

    def test_no_tiles_is_not_an_error(self):
        from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

        assert StitchingPipeline(StitchingConfig())._content_z_ranges([]) is None


class TestItRescuesSeamsTheFilterWouldHaveThrownAway:
    """The end-to-end claim, through the real registration path.

    This is the whole justification for the feature: not a better shift, but a
    seam that stops being rejected for a reason unrelated to its alignment.
    """

    @staticmethod
    def _tile_pair(n_planes=320, content=40, size=96, step=72, seed=3):
        from types import SimpleNamespace

        from scipy import ndimage

        rng = np.random.default_rng(seed)
        c0 = n_planes // 2 - content // 2
        body = ndimage.gaussian_filter(
            rng.random((content, size, size + step)).astype(np.float32), sigma=2.5
        )
        body = (body - body.min()) / max(1e-9, float(np.ptp(body)))
        field = np.zeros((n_planes, size, size + step), np.float32)
        field[c0 : c0 + content] = 400.0 * body
        out = []
        for i, x0 in enumerate((0, step)):
            vol = 300.0 + field[:, :, x0 : x0 + size]
            # Independent noise per tile. Sharing it makes empty planes
            # correlate perfectly and hides the entire effect under test.
            vol = vol + rng.normal(0, 40.0, vol.shape).astype(np.float32)
            out.append(
                (
                    vol.astype(np.float32),
                    SimpleNamespace(
                        x_mm=x0 * 0.001, y_mm=0.0, z_min_mm=0.0,
                        z_max_mm=n_planes * 0.001, folder=None, tile_index=i,
                        illumination_sides=[0], angle_deg=0.0, view=0,
                    ),
                )
            )
        return out

    def _quality(self, crop):
        from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

        pipe = StitchingPipeline(
            StitchingConfig(
                skip_registration=False,
                registration_z_content_crop=crop,
                quality_threshold=0.0,     # measure it, do not filter on it
                min_tile_structure=0.0,    # isolate the crop from the gate
            )
        )
        pipe._register_tiles(self._tile_pair(), {"z": 1.0, "y": 1.0, "x": 1.0})
        scores = [
            s.quality for s in pipe._registration_report.seams if s.quality is not None
        ]
        assert scores, "no seam quality was recorded"
        return scores[0]

    def test_content_in_12_percent_of_the_stack_passes_only_when_cropped(self):
        pytest.importorskip("multiview_stitcher")
        whole = self._quality(False)
        cropped = self._quality(True)
        # Measured 0.32 vs 0.95. The default quality threshold is 0.4, so the
        # uncropped seam is thrown away and the cropped one is kept -- same
        # tiles, same alignment.
        assert whole < 0.4 < cropped
        assert cropped > whole * 2
