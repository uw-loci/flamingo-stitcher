"""The registration map: one PNG of the tiles as fused, tinted by Z correction,
overlaps coloured by what registration made of each seam, tiles labelled.

Requested 2026-10-19 after a run whose bottom row hung on one of four seams:
the log said so in a table, and nobody should have to read a 71-row table to
see which row is held by what.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from flamingo_stitcher import registration_map as rm  # noqa: E402


def _grid(dz=(0.0, 0.0, 0.0, 40.0)):
    frame, pitch = 1000.0, 900.0
    tiles = []
    for i, (iy, ix) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        y0, x0 = iy * pitch, ix * pitch
        img = np.random.default_rng(i).random((128, 128)).astype(np.float32)
        tiles.append(rm.MapTile(f"X{ix:03d} Y{iy:03d}", (y0, x0, y0 + frame, x0 + frame),
                                dz[i], "carried" if i == 3 else "", img))
    seams = [
        rm.MapSeam(0, 1, "registered", 0.95, 1.0),
        rm.MapSeam(2, 3, "pruned", 0.80, 60.0),
        rm.MapSeam(0, 2, "below_quality", 0.10, None),
        rm.MapSeam(1, 3, "empty_overlap", None, None),
    ]
    return tiles, seams


def test_categories():
    assert rm.seam_category("registered", 2.0, 10.0) == "agrees"
    assert rm.seam_category("registered", 20.0, 10.0) == "strained"
    assert rm.seam_category("pruned", 50.0, 10.0) == "pruned"
    assert rm.seam_category("below_quality", None, 10.0) == "below_quality"
    assert rm.seam_category("implausible_shift", None, 10.0) == "implausible"
    assert rm.seam_category("empty_overlap", None, 10.0) == "empty"
    assert rm.seam_category("not_run", None, 10.0) == "not_measured"


def test_it_draws_and_tiles_are_big_enough_for_their_labels(tmp_path):
    tiles, seams = _grid()
    out = rm.render(tiles, seams, tmp_path / "m.png", tol_um=10.0, title="t")
    assert out is not None and out.is_file()
    img = Image.open(out)
    # four 1000 µm tiles at 900 µm pitch: the mosaic alone is 1.9 tiles wide,
    # and a tile is at least 240 px so its three lines of text fit.
    assert img.width >= int(1.9 * rm.MIN_TILE_PX)


def test_seam_strips_carry_their_category_colour(tmp_path):
    tiles, seams = _grid()
    out = rm.render(tiles, seams, tmp_path / "m.png", tol_um=10.0)
    img = np.asarray(Image.open(out).convert("RGB")).astype(int)
    green = np.array(rm._SEAM_RGB["agrees"])
    orange = np.array(rm._SEAM_RGB["pruned"])
    # The seam outlines are drawn in the category colour at full strength.
    assert (np.abs(img - green).sum(axis=2) < 6).sum() > 50
    assert (np.abs(img - orange).sum(axis=2) < 6).sum() > 50


def test_z_scale_runs_from_the_run_min_to_its_max():
    scale = rm.z_scale([-229, -64, 0, 33, 98])
    assert scale == (-229, 98)
    # each side reaches full colour at its own extreme
    np.testing.assert_allclose(rm.z_colour(-229, scale), rm._Z_NEG)
    np.testing.assert_allclose(rm.z_colour(98, scale), rm._Z_POS)
    np.testing.assert_allclose(rm.z_colour(0, scale), rm._Z_MID)
    neg, pos = rm.z_colour(-100, scale), rm.z_colour(40, scale)
    assert neg[2] > neg[0] and pos[0] > pos[2]      # blue / orange
    assert rm.tint_alpha(98, scale) == pytest.approx(rm.TILE_TINT_ALPHA_MAX)
    assert rm.tint_alpha(0, scale) == pytest.approx(rm.TILE_TINT_ALPHA_MIN)
    # all one sign: white at zero, the scale still includes it
    assert rm.z_scale([5, 20]) == (0.0, 20)


def test_projections_are_shrunk_on_the_way_in():
    big = np.zeros((2048, 2048), np.float32)
    big[1000, 1000] = 7.0
    small = rm.shrink_for_map(big, 512)
    assert max(small.shape) <= 512 and small.max() == 7.0  # block MAX keeps it


def test_the_pipeline_writes_it_from_the_microscope_mips(tmp_path, caplog):
    pytest.importorskip("multiview_stitcher")
    import tifffile
    from _synth_acq import write_synth_acquisition

    from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

    frame = 64
    acq = write_synth_acquisition(
        tmp_path / "acq", grid=(2, 2), overlap=0.25, n_planes=8,
        frame_size=(frame, frame), z_texture=True,
        tile_z_offset_planes={(1, 1): 2},
    )
    for raw in acq.rglob("*.raw"):
        vol = np.fromfile(raw, dtype="<u2").reshape(-1, frame, frame)
        tifffile.imwrite(raw.with_name(raw.stem + "_MP.tif"), vol.max(axis=0))
    cfg = StitchingConfig.with_yaml_defaults()
    for k, v in dict(
        skip_registration=False, streaming_mode=True, output_format="ome-tiff",
        resource_guard_enabled=False, flat_field_correction=False, reg_channel=1,
        output_chunksize={"z": 4, "y": 16, "x": 16}, downsample_xy=1,
        downsample_z=1, tile_orientation="identity", camera_x_inverted=False,
        auto_tile_orientation=False, registration_z_spacing_um=0.0,
    ).items():
        setattr(cfg, k, v)
    with caplog.at_level(logging.INFO):
        StitchingPipeline(cfg).run(acq, tmp_path / "out")
    maps = list((tmp_path / "out").glob("*_registration_map.png"))
    assert len(maps) == 1
    assert "registration map:" in caplog.text
    assert "have no projection" not in caplog.text


def test_without_microscope_mips_the_map_still_has_an_image(tmp_path, caplog):
    """The 2026-10-20 rig run saved no *_MP.tif and its map came out as tint
    and labels over black. The projection is now taken from each tile as it
    is preprocessed, so the map never depends on the microscope saving MIPs."""
    pytest.importorskip("multiview_stitcher")
    from _synth_acq import write_synth_acquisition

    from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

    acq = write_synth_acquisition(
        tmp_path / "acq", grid=(2, 2), overlap=0.25, n_planes=8,
        frame_size=(64, 64), z_texture=True,
    )
    assert not list(acq.rglob("*_MP.tif"))
    cfg = StitchingConfig.with_yaml_defaults()
    for k, v in dict(
        skip_registration=False, streaming_mode=True, output_format="ome-tiff",
        resource_guard_enabled=False, flat_field_correction=False, reg_channel=1,
        output_chunksize={"z": 4, "y": 16, "x": 16}, downsample_xy=1,
        downsample_z=1, tile_orientation="identity", camera_x_inverted=False,
        auto_tile_orientation=False, registration_z_spacing_um=0.0,
    ).items():
        setattr(cfg, k, v)
    pipe = StitchingPipeline(cfg)
    with caplog.at_level(logging.INFO):
        pipe.run(acq, tmp_path / "out")
    assert "have no projection" not in caplog.text
    assert "max projection of each tile as fused" in caplog.text
    projs = list(pipe._map_proj.values())
    assert len(projs) == 4 and all(p.max() > 0 for p in projs)
    # The PNG's mosaic area carries image, not just flat tint: many grey levels.
    (png,) = (tmp_path / "out").glob("*_registration_map.png")
    img = np.asarray(Image.open(png).convert("L"))
    assert len(np.unique(img[: img.shape[0] // 2, : img.shape[1] // 2])) > 50


def test_the_projection_is_kept_small():
    from flamingo_stitcher.pipeline import RawTileInfo, StitchingConfig, StitchingPipeline

    pipe = StitchingPipeline(StitchingConfig())
    tile = RawTileInfo(folder=Path("/t"), x_mm=0, y_mm=0, z_min_mm=0, z_max_mm=1, n_planes=3)
    vol = np.zeros((3, 1024, 1024), np.uint16)
    vol[1, 500, 500] = 900
    pipe._record_map_projection(tile, 1, vol)
    vol2 = np.zeros_like(vol)
    vol2[0, 10, 10] = 50                      # the other light path, same tile
    pipe._record_map_projection(tile, 1, vol2)
    (proj,) = pipe._map_proj.values()
    assert max(proj.shape) <= 512 and proj.max() == 900 and (proj == 50).any()
    pipe.config.registration_map_enabled = False
    pipe._map_proj.clear()
    pipe._record_map_projection(tile, 1, vol)
    assert not pipe._map_proj


def test_it_can_be_turned_off(tmp_path):
    pytest.importorskip("multiview_stitcher")
    from _synth_acq import write_synth_acquisition

    from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

    acq = write_synth_acquisition(
        tmp_path / "acq", grid=(2, 1), overlap=0.25, n_planes=8, frame_size=(48, 48)
    )
    cfg = StitchingConfig.with_yaml_defaults()
    for k, v in dict(
        skip_registration=True, streaming_mode=True, output_format="ome-tiff",
        resource_guard_enabled=False, flat_field_correction=False,
        registration_map_enabled=False, tile_orientation="identity",
        camera_x_inverted=False, auto_tile_orientation=False,
    ).items():
        setattr(cfg, k, v)
    StitchingPipeline(cfg).run(acq, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*_registration_map.png"))


# --------------------------------------------------------------------------
# Blend width from the measured overlap (shipped with the map in 0.13.12)
# --------------------------------------------------------------------------


def _blend_widths(frac, overlap=0.25, frame=64, fusion="blend", caplog=None):
    from pathlib import Path as _P

    from flamingo_stitcher.pipeline import RawTileInfo, StitchingConfig, StitchingPipeline

    px = 0.406
    pitch_mm = frame * px * (1 - overlap) / 1000.0
    tiles = [
        RawTileInfo(folder=_P(f"/t{i}"), x_mm=i % 2 * pitch_mm, y_mm=i // 2 * pitch_mm,
                    z_min_mm=0.0, z_max_mm=0.01, n_planes=4)
        for i in range(4)
    ]
    vol = np.zeros((4, frame, frame), np.uint16)
    cfg = StitchingConfig(blending_width_overlap_fraction=frac, tile_overlap_fusion=fusion)
    pipe = StitchingPipeline(cfg)
    return pipe._blending_widths(
        [(vol, t) for t in tiles], {"z": 2.5, "y": px, "x": px}
    ), frame * px * overlap


def test_blend_width_follows_the_measured_overlap(caplog):
    with caplog.at_level(logging.INFO):
        w, overlap_um = _blend_widths(1.0)
    assert w["x"] == pytest.approx(overlap_um, rel=0.02)
    assert w["y"] == pytest.approx(overlap_um, rel=0.02)
    assert w["z"] == 50.0                     # Z keeps the fixed setting
    assert "measured" in caplog.text and "Blend width" in caplog.text


def test_half_overlap_and_fixed():
    w_half, overlap_um = _blend_widths(0.5)
    assert w_half["x"] == pytest.approx(overlap_um / 2, rel=0.02)
    w_fixed, _ = _blend_widths(0.0)
    assert (w_fixed["x"], w_fixed["y"]) == (100.0, 100.0)


def test_only_blend_logs_it(caplog):
    with caplog.at_level(logging.INFO):
        _blend_widths(1.0, fusion="max")
    assert "Blend width" not in caplog.text


# --------------------------------------------------------------------------
# Joins held by too few links, and broken links
# --------------------------------------------------------------------------


def _row_join(statuses):
    """A 2-row strip of len(statuses) columns; seam i joins (i,0)-(i,1)."""
    tiles = []
    for iy in range(2):
        for ix in range(len(statuses)):
            y0, x0 = iy * 900.0, ix * 900.0
            tiles.append(rm.MapTile(f"X{ix:03d} Y{iy:03d}", (y0, x0, y0 + 1000, x0 + 1000),
                                    grid=(ix, iy)))
    n = len(statuses)
    seams = [
        rm.MapSeam(i, n + i, st, 0.8, 30.0, axis="y",
                   disagree_um=(0.0, -40.0 - i, 2.0) if st != "below_quality" else None)
        for i, st in enumerate(statuses)
    ]
    return tiles, seams


def test_a_row_held_by_one_link_is_called_out():
    tiles, seams = _row_join(["registered", "pruned", "pruned", "pruned"])
    (b,) = rm.boundaries(tiles, seams)
    assert (b.axis, b.a, b.b, b.kept, b.measured) == ("y", 0, 1, 1, 4)
    assert b.weak == "one"
    assert b.normal_um == [41.0, 42.0, 43.0]      # |dy| of the dropped links


def test_a_well_held_row_is_not():
    tiles, seams = _row_join(["registered", "registered", "registered", "pruned"])
    (b,) = rm.boundaries(tiles, seams)
    assert b.weak == ""


def test_the_largest_disagreement_names_its_axis():
    assert rm.dominant((5.0, -52.0, 8.0)) == ("y", -52.0)


def test_broken_links_get_a_red_x(tmp_path):
    tiles, seams = _row_join(["registered", "pruned", "below_quality", "registered"])
    out = rm.render(tiles, seams, tmp_path / "m.png", tol_um=10.0)
    img = np.asarray(Image.open(out).convert("RGB")).astype(int)
    red = np.array([235, 30, 30])
    assert (np.abs(img - red).sum(axis=2) < 10).sum() > 40


def test_a_kept_seam_off_in_z_is_flagged_even_inside_the_solve_tolerance():
    """2026-10-27, XY 4x / Z 1x: tolerance 25 lateral px = 105 µm let a seam
    sit tens of µm off in Z and still be drawn green."""
    limits = (5.0, 8.4)  # 2 planes of 2.5 µm; 2 output px of 4.19 µm
    assert rm.seam_category("registered", 30.0, 105.0, limits, (30.0, 0.0, 0.0)) == "strained"
    assert rm.seam_category("registered", 9.0, 105.0, limits, (0.0, 6.0, 6.0)) == "strained"
    assert rm.seam_category("registered", 5.0, 105.0, limits, (4.0, 3.0, 3.0)) == "agrees"
    # Without per-axis data the old rule still applies.
    assert rm.seam_category("registered", 30.0, 105.0) == "agrees"
    assert rm.seam_category("registered", 130.0, 105.0) == "strained"
