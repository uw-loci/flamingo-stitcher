"""The log says where a fade that repeats in every tile comes from.

2026-10-16: a stitched brain showed the same top-to-bottom brightness ramp in
every tile, and nothing in the log could say whether it was in the raw data,
came in with the illumination merge or the flat-field, or appeared later. The
pipeline now logs, per channel, the median brightness across a tile at each
stage. Here a 30% ramp along Y is written into ONE illumination side's raw
files and the log has to show it there, not in the other side, and carried
into the merged and stitched tile.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest

from _synth_acq import write_synth_acquisition

from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline


def _ramp_side(acq, side: int, frame: int, planes: int):
    ramp = np.linspace(0.7, 1.0, frame, dtype=np.float32)[None, :, None]
    for f in acq.rglob(f"*_I{side}_*.raw"):
        vol = np.fromfile(f, dtype="<u2").reshape(planes, frame, frame)
        (vol.astype(np.float32) * ramp).astype("<u2").tofile(f)


def _profiles(lines, axis):
    out = {}
    for line in lines:
        m = re.match(rf"\s+along {axis}\s+(.+?)\s{{2,}}([\d ]+?)\s+\(range (\d+)-(\d+)\)", line)
        if m:
            out[m.group(1).strip()] = (int(m.group(3)), int(m.group(4)))
    return out


@pytest.mark.parametrize("streaming", [False, True])
def test_a_ramp_in_one_side_is_reported_at_every_stage(tmp_path, caplog, streaming):
    pytest.importorskip("multiview_stitcher")
    frame, planes = 64, 8
    acq = write_synth_acquisition(
        tmp_path / "acq", grid=(2, 2), overlap=0.2, n_planes=planes,
        channels=(1,), illum_sides=(0, 1), frame_size=(frame, frame),
        pixel_size_um=1.0,
    )
    _ramp_side(acq, 1, frame, planes)
    cfg = StitchingConfig.with_yaml_defaults()
    for k, v in dict(
        skip_registration=True, streaming_mode=streaming, output_format="ome-tiff",
        resource_guard_enabled=False, flat_field_correction=False,
        downsample_xy=1, downsample_z=1, tile_orientation="identity",
        camera_x_inverted=False, auto_tile_orientation=False,
        illumination_fusion="mean",
    ).items():
        setattr(cfg, k, v)
    with caplog.at_level(logging.INFO):
        StitchingPipeline(cfg).run(acq, tmp_path / "out")
    lines = [r.getMessage() for r in caplog.records]
    assert any("Brightness across a tile, channel 1" in l for l in lines)
    y = _profiles(lines, "Y")
    lo1, hi1 = y["raw side 1"]
    lo0, hi0 = y["raw side 0"]
    assert hi1 - lo1 >= 20, y          # the planted ramp is seen
    assert hi0 - lo0 < hi1 - lo1 - 10, y  # and only in that side
    lo_m, hi_m = y["merged"]
    lo_s, hi_s = y["tile as stitched"]
    assert hi_m - lo_m >= 8 and hi_s - lo_s >= 8, y  # mean carries half of it


def test_streaming_logs_the_block_before_registration(tmp_path, caplog):
    """Streaming loads the reference channel inside Step 3's branch, so the
    block was logged with nothing in it and only appeared at the end."""
    pytest.importorskip("multiview_stitcher")
    frame, planes = 64, 8
    acq = write_synth_acquisition(
        tmp_path / "acq", grid=(2, 2), overlap=0.2, n_planes=planes,
        channels=(1,), illum_sides=(0, 1), frame_size=(frame, frame),
        pixel_size_um=1.0,
    )
    cfg = StitchingConfig.with_yaml_defaults()
    for k, v in dict(
        skip_registration=False, streaming_mode=True, output_format="ome-tiff",
        resource_guard_enabled=False, flat_field_correction=False,
        downsample_xy=1, downsample_z=1, tile_orientation="identity",
        camera_x_inverted=False, auto_tile_orientation=False,
    ).items():
        setattr(cfg, k, v)
    with caplog.at_level(logging.INFO):
        StitchingPipeline(cfg).run(acq, tmp_path / "out")
    lines = [r.getMessage() for r in caplog.records]
    block = next(i for i, l in enumerate(lines) if "Brightness across a tile" in l)
    step3 = next(i for i, l in enumerate(lines) if "Step 3: Registering" in l)
    assert block < step3
