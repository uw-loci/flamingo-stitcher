"""The intensity adjustment matches BigStitcher's, stage by stage.

Ground truth was produced by running BigStitcher's OWN code
(multiview-reconstruction 9.1.1 + mpicbg 1.6.6) through
``tools/bigstitcher_intensity/IntensityHarness.java`` on the synthetic cases
``make_case.py`` builds, three runs each (BigStitcher's solver shuffles its
tile order every sweep, so runs differ slightly). The Java pieces the port
re-implements -- ``java.util.Random``, ``HashMap`` iteration order, the
``ArrayList`` thinning -- were dumped by ``JavaBits.java``. Both outputs are in
``tests/data/bigstitcher/``; regenerating them needs a JDK and the jars listed
in ``tools/bigstitcher_intensity/README.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from flamingo_stitcher import intensity_adjust as ia

DATA = Path(__file__).parent / "data" / "bigstitcher"
TOOLS = Path(__file__).parent.parent / "tools" / "bigstitcher_intensity"
sys.path.insert(0, str(TOOLS))
import make_case  # noqa: E402

JAVA = json.loads((DATA / "java_bits.json").read_text())
GOLDEN = json.loads((DATA / "intensity_cases.json").read_text())["cases"]


# --------------------------------------------------------------------------
# The Java pieces, against Java
# --------------------------------------------------------------------------


def test_java_random_next_int_matches_java():
    r = ia.JavaRandom(344)
    bounds = JAVA["random_bounds"]
    got = [r.next_int(bounds[k % len(bounds)]) for k in range(len(JAVA["random_344"]))]
    assert got == JAVA["random_344"]


@pytest.mark.parametrize("m", [9, 20, 42, 70])
def test_hashmap_iteration_order_matches_java(m):
    assert [list(k) for k in ia.java_hashmap_order(m)] == JAVA[f"order_{m}"]


def test_thinning_keeps_the_same_samples_as_arraylist_remove():
    rnd = ia.JavaRandom(344)  # ONE stream across lists, as upstream
    for case in JAVA["thin_cases"]:
        kept = ia.thin_like_arraylist(case["n"], case["max"], rnd)
        assert kept.tolist() == case["survivors"], (case["n"], case["max"])


# --------------------------------------------------------------------------
# Whole cases, against BigStitcher
# --------------------------------------------------------------------------


def _case(name):
    g = GOLDEN[name]
    a = g["make_case"]
    ny, nx = (int(v) for v in a.get("grid", "3x3").split("x"))
    views, truth = make_case.make_views(grid=(ny, nx), seed=a.get("seed", 0))
    params = ia.IntensityAdjustParams(
        downsampling=a["downsampling"],
        max_matches=a["max_matches"],
        affine=bool(a.get("affine", 1)),
        offset_regularization=a.get("reg_trans", 0.1),
        identity_regularization=a.get("reg_identity", 0.1),
    )
    return [ia.View(d, aff) for d, aff in views], params, g, truth


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_grid_and_samples_match_bigstitcher(name):
    views, params, g, _ = _case(name)
    raw, bbox, grid = ia.collect_matches(views, params)
    assert [bbox[0].tolist(), bbox[1].tolist()] == g["bb"]
    if params.downsampling != 1.0:
        assert [grid[0].tolist(), grid[1].tolist()] == g["scaledbb"]
    assert [list(k) for k in ia.java_hashmap_order(len(views))] == [
        list(k) for k in g["maporder"]
    ]
    thinned = ia.thin_matches(raw, len(views), params.max_matches)
    for key, jp in g["pairs"].items():
        k = tuple(int(v) for v in key.split())
        assert (len(raw[k]), len(thinned[k])) == (jp["before"], jp["after"]), key
        if jp["after"]:
            t = thinned[k]
            ours = [t[:, 0].sum(), t[:, 1].sum(), (t[:, 0] ** 2).sum(), (t[:, 0] * t[:, 1]).sum()]
            # float32 interpolation upstream vs float64 here: same samples,
            # last-bit differences only.
            np.testing.assert_allclose(ours, jp["sums"], rtol=1e-6)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_fitted_models_match_bigstitcher(name):
    views, params, g, _ = _case(name)
    r = ia.compute(views, params)
    a_runs = np.array(g["scale_runs"])
    b_runs = np.array(g["offset_runs"])
    # BigStitcher stops after 300 shuffled sweeps; this solves the configuration
    # those sweeps converge to. Agreement is to well within a count.
    np.testing.assert_allclose(r.scale, a_runs.mean(0), atol=1e-5)
    np.testing.assert_allclose(r.offset, b_runs.mean(0), atol=0.01)


def test_it_does_what_it_is_for():
    """Known per-tile gains and offsets: the brightness step across overlaps
    drops by an order of magnitude."""
    views, params, _, _ = _case("case1")
    raw, _, _ = ia.collect_matches(views, params)
    r = ia.compute(views, params)
    before, after = [], []
    for (i, j), arr in raw.items():
        if len(arr):
            before.append(np.abs(arr[:, 0] - arr[:, 1]).mean())
            after.append(
                np.abs(
                    r.scale[i] * arr[:, 0] + r.offset[i]
                    - (r.scale[j] * arr[:, 1] + r.offset[j])
                ).mean()
            )
    assert np.mean(after) < np.mean(before) / 10


def test_the_smallest_offset_is_zero_as_upstream():
    views, params, _, _ = _case("case1")
    r = ia.compute(views, params)
    assert r.offset.min() == 0.0


def test_a_tile_with_no_overlap_is_left_unchanged():
    views, params, _, _ = _case("case1")
    lonely = ia.View(views[0].data, views[0].affine.copy())
    lonely.affine[0, 3] += 1e5  # far from everything
    r = ia.compute(views + [lonely], params)
    assert not r.connected[-1]
    assert r.scale[-1] == 1.0
    assert r.offset[-1] == pytest.approx(-r.min_offset)


def test_apply_rounds_and_clips_to_uint16():
    v = np.array([[[0, 100, 60000]]], dtype=np.uint16)
    out = ia.apply_uint16(v, 1.5, -20.4)
    assert out.dtype == np.uint16
    assert out.ravel().tolist() == [0, 130, 65535]


# --------------------------------------------------------------------------
# In the pipeline
# --------------------------------------------------------------------------


def test_index_to_world_affine_matches_the_tile_placement():
    """A voxel of a placed tile lands where multiview-stitcher puts it."""
    pytest.importorskip("multiview_stitcher")
    from multiview_stitcher import io as mvs_io
    from multiview_stitcher import spatial_image_utils as si_utils

    from flamingo_stitcher.pipeline import _index_to_world_px_affine

    vol = np.zeros((4, 5, 6), dtype=np.uint16)
    sim = si_utils.get_sim_from_array(
        vol,
        dims=["z", "y", "x"],
        scale={"z": 10.0, "y": 8.0, "x": 8.0},
        translation={"z": 100.0, "y": -40.0, "x": 16.0},
        transform_key=mvs_io.METADATA_TRANSFORM_KEY,
    )
    a = _index_to_world_px_affine(sim, mvs_io.METADATA_TRANSFORM_KEY, native_px_um=1.0)
    # voxel (x=2, y=3, z=1) -> world (16 + 2*8, -40 + 3*8, 100 + 1*10) um
    xyz = a[:, :3] @ np.array([2, 3, 1]) + a[:, 3]
    np.testing.assert_allclose(xyz, [32.0, -16.0, 110.0])


def test_the_pipeline_measures_and_applies(tmp_path, caplog):
    """Real discover -> preprocess -> fusion inputs on a synthetic mosaic with
    one tile 40% too bright: Measure logs it and changes nothing, Apply fixes
    the step."""
    pytest.importorskip("multiview_stitcher")
    import logging

    from _synth_acq import write_synth_acquisition
    from multiview_stitcher import io as mvs_io

    from flamingo_stitcher.pipeline import (
        StitchingConfig,
        StitchingPipeline,
        discover_tiles,
    )

    acq = write_synth_acquisition(
        tmp_path / "acq", grid=(2, 2), overlap=0.2, n_planes=8, channels=(1,),
        illum_sides=(0,), frame_size=(48, 48), pixel_size_um=1.0,
    )
    cfg = StitchingConfig.with_yaml_defaults()
    for k, v in dict(
        downsample_xy=1, downsample_z=1, flat_field_correction=False,
        skip_registration=True, tile_orientation="identity",
        camera_x_inverted=False, auto_tile_orientation=False,
        intensity_adjust_downsampling=2.0, intensity_adjust_max_matches=2000,
    ).items():
        setattr(cfg, k, v)
    tiles = discover_tiles(acq)
    voxel = {"z": 5.0, "y": 1.0, "x": 1.0}

    def inputs(mode):
        cfg.intensity_adjustment = mode
        pipe = StitchingPipeline(cfg)
        ctd = pipe._load_and_preprocess(tiles, [1])
        data = [(v.astype(np.float32), t) for v, t in ctd[1]]
        data[0] = ((data[0][0] * 1.4).astype(np.uint16), data[0][1])
        data = [(v.astype(np.uint16), t) for v, t in data]
        sims, _ = pipe._build_fusion_inputs(
            data, voxel, [], mvs_io.METADATA_TRANSFORM_KEY
        )
        return sims

    with caplog.at_level(logging.INFO):
        measured = inputs("measure")
    assert any("Intensity adjustment (measure" in r.getMessage() for r in caplog.records)
    assert any("fused UNCHANGED" in r.getMessage() for r in caplog.records)
    applied = inputs("apply")
    m0 = float(np.asarray(measured[0].data).mean())
    a0 = float(np.asarray(applied[0].data).mean())
    m1 = float(np.asarray(measured[1].data).mean())
    a1 = float(np.asarray(applied[1].data).mean())
    # the bright tile is brought toward its neighbour
    assert abs(a0 / a1 - 1) < abs(m0 / m1 - 1) / 2


def test_streaming_tiles_backed_by_dask_give_the_same_fit():
    """Streaming fuses from dask arrays over the tile spill, which do not
    support scattered indexing; the first version skipped the adjustment there
    with only a warning."""
    import dask.array as da

    views, params, _, _ = _case("case1")
    lazy = [ia.View(da.from_array(v.data, chunks=(4, 16, 16)), v.affine) for v in views]
    a = ia.compute(views, params)
    b = ia.compute(lazy, params)
    np.testing.assert_array_equal(a.scale, b.scale)
    np.testing.assert_array_equal(a.offset, b.offset)
