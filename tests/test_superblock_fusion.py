"""Correctness of super-block (regioned) fusion vs whole-output fusion.

Item E bounds the fusion dask-graph by fusing the output in spatial regions
instead of one giant graph. This is only safe if regioned fusion is
BIT-IDENTICAL to whole-output fusion. The proven condition: region boundaries
must be integer multiples of output_chunksize (so MVS's internal per-block grid
is identical either way). This test locks that in for all three tile-overlap
resolution methods — the interaction the design must protect.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from multiview_stitcher import fusion, io as mvs_io
    from multiview_stitcher import spatial_image_utils as si_utils
    from multiview_stitcher.fusion import max_fusion
    from multiview_stitcher.weights import content_based

    _HAVE_MVS = True
except Exception:
    _HAVE_MVS = False

_CHUNK = {"z": 8, "y": 32, "x": 32}


def _make_sims(seed=0):
    rng = np.random.default_rng(seed)
    sims = []
    for tx in (0.0, 40.0, 80.0):  # overlapping tiles along X
        vol = rng.integers(100, 4000, size=(4, 64, 64)).astype(np.uint16)
        sims.append(
            si_utils.get_sim_from_array(
                vol,
                dims=["z", "y", "x"],
                scale={"z": 5.0, "y": 1.0, "x": 1.0},
                translation={"z": 0.0, "y": 0.0, "x": tx},
                transform_key=mvs_io.METADATA_TRANSFORM_KEY,
            )
        )
    return sims


def _full_props(sims):
    params = [
        si_utils.get_affine_from_sim(s, transform_key=mvs_io.METADATA_TRANSFORM_KEY)
        for s in sims
    ]
    return fusion.calc_fusion_stack_properties(
        sims, params=params, spacing=si_utils.get_spacing_from_sim(sims[0]), mode="union"
    )


def _fuse_whole(sims, **kw):
    return np.asarray(
        fusion.fuse(
            sims,
            transform_key=mvs_io.METADATA_TRANSFORM_KEY,
            output_chunksize=_CHUNK,
            **kw,
        ).data
    ).squeeze()


def _fuse_regioned(sims, region_chunks_x=2, **kw):
    fp = _full_props(sims)
    sp, org, shp = fp["spacing"], fp["origin"], fp["shape"]
    out = np.zeros((shp["z"], shp["y"], shp["x"]), dtype=np.uint16)
    step = _CHUNK["x"] * region_chunks_x  # chunk-aligned region size
    edges = list(range(0, shp["x"], step)) + [shp["x"]]
    for x0, x1 in zip(edges, edges[1:]):
        rprops = {
            "origin": {"z": org["z"], "y": org["y"], "x": org["x"] + x0 * sp["x"]},
            "shape": {"z": shp["z"], "y": shp["y"], "x": int(x1 - x0)},
            "spacing": dict(sp),
        }
        r = fusion.fuse(
            sims,
            transform_key=mvs_io.METADATA_TRANSFORM_KEY,
            output_chunksize=_CHUNK,
            output_stack_properties=rprops,
            **kw,
        ).data
        out[:, :, x0:x1] = np.asarray(r).squeeze()
    return out


@unittest.skipUnless(_HAVE_MVS, "needs multiview_stitcher")
class TestSuperblockIdentical(unittest.TestCase):
    def _check(self, label, **kw):
        sims = _make_sims()
        whole = _fuse_whole(sims, **kw)
        reg = _fuse_regioned(sims, **kw)
        self.assertEqual(whole.shape, reg.shape, label)
        np.testing.assert_array_equal(
            whole, reg, err_msg=f"{label}: chunk-aligned regioned fusion must match whole"
        )

    def test_max(self):
        self._check("max", fusion_func=max_fusion)

    def test_blend_cosine(self):
        self._check("blend")

    def test_content_based(self):
        self._check(
            "content",
            weights_func=content_based,
            weights_func_kwargs={"sigma_1": 5, "sigma_2": 11},
        )


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_HAVE_MVS, "needs multiview_stitcher")
class TestRegionsUseTheFusionChunkSize(unittest.TestCase):
    """The region step must come from the chunk size FUSION uses.

    `resolve_output_chunksize` caps XY near one tile pitch, so a heavily
    downsampled run fuses at a smaller chunk than `config.output_chunksize`
    names. Stepping the regions by the configured value then puts every region
    boundary off the fusion grid, and the bit-identical guarantee above no
    longer applies to anything.
    """

    def _pipeline(self, configured, resolved):
        from types import SimpleNamespace
        from flamingo_stitcher.pipeline import StitchingPipeline

        p = StitchingPipeline.__new__(StitchingPipeline)
        p.config = SimpleNamespace(output_chunksize=configured)
        self.resolved = resolved
        return p

    def _props(self):
        return {
            "origin": {"z": 0.0, "y": 0.0, "x": 0.0},
            "shape": {"z": 16, "y": 1024, "x": 1024},
            "spacing": {"z": 5.0, "y": 1.0, "x": 1.0},
        }

    def test_the_passed_chunksize_wins_over_the_configured_one(self):
        p = self._pipeline({"z": 8, "y": 256, "x": 256}, {"z": 8, "y": 64, "x": 64})
        regions = list(
            p._iter_superblock_regions(self._props(), 2, chunksize=self.resolved)
        )
        starts_x = sorted({r[1][4] for r in regions})
        # 64 * 2 = 128 px steps, every one a multiple of the fusion chunk (64).
        self.assertEqual(starts_x, list(range(0, 1024, 128)))
        for s in starts_x:
            self.assertEqual(s % self.resolved["x"], 0)

    def test_it_falls_back_to_the_configured_size(self):
        p = self._pipeline({"z": 8, "y": 256, "x": 256}, None)
        regions = list(p._iter_superblock_regions(self._props(), 2))
        starts_x = sorted({r[1][4] for r in regions})
        self.assertEqual(starts_x, [0, 512])

    def test_regions_tile_the_output_exactly(self):
        p = self._pipeline({"z": 8, "y": 256, "x": 256}, {"z": 8, "y": 96, "x": 96})
        regions = list(
            p._iter_superblock_regions(self._props(), 3, chunksize=self.resolved)
        )
        covered = np.zeros((16, 1024, 1024), dtype=np.uint8)
        for _rprops, (z0, z1, y0, y1, x0, x1) in regions:
            covered[z0:z1, y0:y1, x0:x1] += 1
        self.assertTrue((covered == 1).all(), "regions must partition the output")
