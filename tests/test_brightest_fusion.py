"""The "brightest tile" tile-overlap fusion mode (winner-take-all).

Unlike ``max`` (per-pixel nanmax) or ``blend`` (cosine average), this mode ranks
tiles by overall mean intensity and, in every overlap, takes ALL pixels from the
brighter tile — a clean whole-tile patchwork with no per-pixel mixing. Locks in:

  * ``_priority_coalesce_fusion`` — per pixel, first non-NaN view along axis 0;
  * ``_tile_brightness`` — ranks tiles by mean;
  * end-to-end semantics — the overlap comes wholesale from the brighter tile,
    NOT a per-pixel max (a locally-bright spike in the dimmer tile is discarded);
  * super-block bit-identity — regioned fusion == whole-output fusion when
    chunk-aligned, exactly like ``max`` (the mode is region-independent).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from flamingo_stitcher.pipeline import (  # noqa: E402
    RawTileInfo,
    StitchingConfig,
    StitchingPipeline,
    _priority_coalesce_fusion,
    _tile_brightness,
)

try:
    from multiview_stitcher import fusion, io as mvs_io  # noqa: E402
    from multiview_stitcher import spatial_image_utils as si_utils  # noqa: E402

    _HAVE_MVS = True
except Exception:
    _HAVE_MVS = False


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
class TestPriorityCoalesce(unittest.TestCase):
    def test_first_non_nan_wins(self):
        nan = np.nan
        # view 0 is highest priority; it covers pixel [0,1] but not [0,0].
        tv = np.array([[[nan, 1.0]], [[2.0, 3.0]]])  # (2 views, 1, 2)
        out = _priority_coalesce_fusion(tv)
        # [0,0] falls through to view 1 (=2); [0,1] taken from view 0 (=1).
        np.testing.assert_array_equal(out, [[2.0, 1.0]])

    def test_all_nan_stays_nan(self):
        tv = np.array([[[np.nan]], [[np.nan]]])
        out = _priority_coalesce_fusion(tv)
        self.assertTrue(np.isnan(out).all())

    def test_single_view_passthrough(self):
        tv = np.array([[[5.0, np.nan]]])
        out = _priority_coalesce_fusion(tv)
        np.testing.assert_array_equal(out[np.isfinite(out)], [5.0])


class TestTileBrightness(unittest.TestCase):
    def test_ranks_by_mean(self):
        dim = np.full((4, 8, 8), 100, dtype=np.uint16)
        bright = np.full((4, 8, 8), 900, dtype=np.uint16)
        self.assertGreater(_tile_brightness(bright), _tile_brightness(dim))

    def test_bad_volume_sorts_last(self):
        class _Boom:
            ndim = 3

            def __getitem__(self, k):
                raise RuntimeError("unreadable")

        self.assertEqual(_tile_brightness(_Boom()), float("-inf"))


# --------------------------------------------------------------------------- #
# End-to-end: overlap is taken whole from the brighter tile, not per-pixel max
# --------------------------------------------------------------------------- #
def _tile(x_mm, nz, ny, nx):
    return RawTileInfo(
        folder=Path("."), x_mm=x_mm, y_mm=0.0, z_min_mm=0.0, z_max_mm=0.0,
        n_planes=nz, illumination_sides=[0], angle_deg=0.0,
    )


def _fuse_pair(mode):
    """Two tiles overlapping by 4 columns in X. The LEFT (dimmer) tile carries
    bright spikes inside the overlap; the RIGHT tile is uniformly brighter on
    average. Returns (fused_array, x_coords, overlap_mask)."""
    nz, ny, nx = 2, 6, 12
    left = np.full((nz, ny, nx), 100, dtype=np.uint16)
    left[:, :, 8:] = 100
    left[0, 0, 9] = 5000  # local spike in the overlap — max() would grab this
    left[1, 3, 10] = 5000
    right = np.full((nz, ny, nx), 1000, dtype=np.uint16)  # brighter on average

    # Right tile placed at world x = 8 µm (voxel = 1 µm) → overlaps left cols 8..11.
    tile_data = [(left, _tile(0.0, nz, ny, nx)), (right, _tile(0.008, nz, ny, nx))]
    cfg = StitchingConfig(tile_overlap_fusion=mode, skip_registration=True)
    pipe = StitchingPipeline(cfg)
    fused, _ = pipe._fuse_channel(
        tile_data, {"z": 1.0, "y": 1.0, "x": 1.0},
        reg_params=[], transform_key=mvs_io.METADATA_TRANSFORM_KEY,
    )
    arr = np.nan_to_num(np.asarray(fused)).squeeze()
    xs = np.asarray(fused.coords["x"].values)
    overlap = (xs >= 8) & (xs <= 11)
    return arr, xs, overlap


@unittest.skipUnless(_HAVE_MVS, "needs multiview_stitcher")
class TestWinnerTakeAll(unittest.TestCase):
    def test_brightest_takes_whole_brighter_tile_in_overlap(self):
        arr, xs, overlap = _fuse_pair("brightest")
        ov = arr[:, :, overlap]
        # The brighter (right) tile is uniform 1000; winner-take-all means the
        # WHOLE overlap is 1000 — the left tile's 5000 spikes are discarded.
        np.testing.assert_array_equal(ov, np.full_like(ov, 1000))

    def test_max_keeps_the_spikes(self):
        # Contrast: per-pixel max DOES surface the dimmer tile's local spikes,
        # which is exactly what "brightest" avoids.
        arr, xs, overlap = _fuse_pair("max")
        ov = arr[:, :, overlap]
        self.assertEqual(int(ov.max()), 5000)


# --------------------------------------------------------------------------- #
# Super-block bit-identity (region-independent, like max)
# --------------------------------------------------------------------------- #
_CHUNK = {"z": 8, "y": 32, "x": 32}


def _make_sims(seed=0):
    rng = np.random.default_rng(seed)
    sims = []
    for tx in (0.0, 40.0, 80.0):
        vol = rng.integers(100, 4000, size=(4, 64, 64)).astype(np.uint16)
        sims.append(
            si_utils.get_sim_from_array(
                vol, dims=["z", "y", "x"], scale={"z": 5.0, "y": 1.0, "x": 1.0},
                translation={"z": 0.0, "y": 0.0, "x": tx},
                transform_key=mvs_io.METADATA_TRANSFORM_KEY,
            )
        )
    return sims


@unittest.skipUnless(_HAVE_MVS, "needs multiview_stitcher")
class TestSuperblockIdentical(unittest.TestCase):
    def test_brightest_regioned_matches_whole(self):
        sims = _make_sims()
        kw = {"fusion_func": _priority_coalesce_fusion}
        whole = np.asarray(
            fusion.fuse(
                sims, transform_key=mvs_io.METADATA_TRANSFORM_KEY,
                output_chunksize=_CHUNK, **kw,
            ).data
        ).squeeze()

        params = [
            si_utils.get_affine_from_sim(s, transform_key=mvs_io.METADATA_TRANSFORM_KEY)
            for s in sims
        ]
        fp = fusion.calc_fusion_stack_properties(
            sims, params=params, spacing=si_utils.get_spacing_from_sim(sims[0]),
            mode="union",
        )
        sp, org, shp = fp["spacing"], fp["origin"], fp["shape"]
        out = np.zeros((shp["z"], shp["y"], shp["x"]), dtype=np.uint16)
        step = _CHUNK["x"] * 2  # chunk-aligned region
        edges = list(range(0, shp["x"], step)) + [shp["x"]]
        for x0, x1 in zip(edges, edges[1:]):
            rprops = {
                "origin": {"z": org["z"], "y": org["y"], "x": org["x"] + x0 * sp["x"]},
                "shape": {"z": shp["z"], "y": shp["y"], "x": int(x1 - x0)},
                "spacing": dict(sp),
            }
            r = fusion.fuse(
                sims, transform_key=mvs_io.METADATA_TRANSFORM_KEY,
                output_chunksize=_CHUNK, output_stack_properties=rprops, **kw,
            ).data
            out[:, :, x0:x1] = np.asarray(r).squeeze()

        np.testing.assert_array_equal(whole, out)


if __name__ == "__main__":
    unittest.main()
