"""Where each tile's tissue is down Z, and how much of it each seam compared.

Asked for 2026-10-21: the bottom-row seams of Ket_Trial1 measured X shifts of
-148 to +144 µm, and the question was whether those tiles were nearly empty at
mid-stack. The Z crop already knew each tile's tissue band and nothing printed
it, and no line said how many planes a pair was actually compared over.

Run: python -m pytest tests/test_verbose_z_columns.py -q
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from flamingo_stitcher import registration_report as rr
from flamingo_stitcher import tile_content as tc
from test_z_content_crop import slab_stack


class TestTheBand:
    def test_a_slab_is_reported_and_cropped(self):
        b = tc.content_z_band(slab_stack())
        assert b.crop == b.band and b.why == "cropped"
        assert b.band[0] <= 60 and b.band[1] >= 100
        assert b.text() == f"{b.band[0]}-{b.band[1]}"

    def test_the_crop_is_unchanged(self):
        vol = slab_stack()
        assert tc.content_z_range(vol) == tc.content_z_band(vol).crop

    def test_a_band_too_wide_to_crop_is_still_reported(self):
        # 126 of 160 planes: the crop is not worth taking, but where the
        # tissue is still answers "is this tile empty at mid-stack". (Past
        # ~80% the empty planes no longer set the floor and it reads "no band".)
        b = tc.content_z_band(slab_stack(c0=10, c1=136))
        assert b.crop is None and b.band is not None
        assert b.why == "band covers most of the stack"

    def test_a_featureless_stack_says_so(self):
        flat = np.random.default_rng(1).normal(300, 20, (80, 64, 64)).astype(np.float32)
        b = tc.content_z_band(flat)
        assert b.band is None and b.crop is None and b.text() == "no band"


def _seam_report():
    report = rr.RegistrationReport(ran=True, applied=True, transform_key="registered")
    for i in range(2):
        report.tiles.append(rr.TileShift(index=i, name=f"t{i}"))
    report.seams.append(rr.SeamResult(
        tile_a="t0", tile_b="t1", index_a=0, index_b=1, axis="y",
        status=rr.STATUS_REGISTERED, quality=0.5,
    ))
    return report


class TestSharedPlanes:
    def _pipe(self):
        from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

        return StitchingPipeline(StitchingConfig())

    def test_two_bands_at_different_depths_share_only_their_overlap(self):
        pipe = self._pipe()
        # tile 0 registered on world Z 100-300 µm, tile 1 on 250-600 µm, 2.5 µm planes
        pipe._reg_z_bands = [tc.ZBand(400, band=(0, 80), crop=(0, 80), why="cropped"),
                             tc.ZBand(400, band=(20, 160), crop=(20, 160), why="cropped")]
        pipe._reg_z_spans_um = [(100.0, 300.0), (250.0, 600.0)]
        pipe._reg_z_voxel_um = 2.5
        tissue, shared = pipe._verbose_z_columns(_seam_report())
        assert tissue == {0: "0-80", 1: "20-160"}
        assert shared == {(0, 1): 20}            # 50 µm / 2.5 µm

    def test_disjoint_bands_share_nothing(self):
        pipe = self._pipe()
        pipe._reg_z_bands = None
        pipe._reg_z_spans_um = [(0.0, 100.0), (200.0, 300.0)]
        pipe._reg_z_voxel_um = 2.5
        tissue, shared = pipe._verbose_z_columns(_seam_report())
        assert tissue is None and shared == {(0, 1): 0}


class TestTheTables:
    def test_columns_only_when_given(self):
        text = "\n".join(rr.format_verbose_alignment(_seam_report()))
        assert "tissue planes" not in text and "shared pl" not in text

    def test_columns_carry_the_values(self):
        lines = rr.format_verbose_alignment(
            _seam_report(), tissue_z={0: "1277-2399", 1: "no band"},
            shared_z={(0, 1): 412},
        )
        text = "\n".join(lines)
        assert "tissue planes" in text and "shared pl" in text
        row0 = next(l for l in lines if l.lstrip().startswith("t0 ") and "registered" in l and "t1" not in l)
        assert "1277-2399" in row0
        seam = next(l for l in lines if l.lstrip().startswith("t0 ") and "t1" in l)
        assert " 412 " in seam


def test_a_real_run_logs_both_columns(tmp_path, caplog):
    pytest.importorskip("multiview_stitcher")
    from _synth_acq import write_synth_acquisition

    from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

    acq = write_synth_acquisition(
        tmp_path / "acq", grid=(2, 1), overlap=0.25, n_planes=8,
        frame_size=(48, 48), z_texture=True,
    )
    cfg = StitchingConfig.with_yaml_defaults()
    for k, v in dict(
        skip_registration=False, streaming_mode=True, output_format="ome-tiff",
        resource_guard_enabled=False, flat_field_correction=False, reg_channel=1,
        output_chunksize={"z": 4, "y": 16, "x": 16}, downsample_xy=1,
        downsample_z=1, tile_orientation="identity", camera_x_inverted=False,
        auto_tile_orientation=False, registration_z_spacing_um=0.0,
        registration_map_enabled=False,
    ).items():
        setattr(cfg, k, v)
    with caplog.at_level(logging.INFO):
        StitchingPipeline(cfg).run(acq, tmp_path / "out")
    assert "tissue planes" in caplog.text and "shared pl" in caplog.text
    # Two tiles at the same stage Z, uncropped: every plane is shared.
    seam_rows = [r.getMessage() for r in caplog.records if "registered" in r.getMessage() and "%" in r.getMessage()]
    assert any(" 8  " in m for m in seam_rows), seam_rows
