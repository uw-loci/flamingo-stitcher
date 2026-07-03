"""Multi-view / asymmetric-coverage discovery + metadata (Increment 1).

These lock in the additive foundation for multi-view stitching:
  * the filename regexes capture V (view) and R (rotation) via named groups
    without breaking the existing channel/illumination/plane extraction,
  * ``_read_start_angle`` reads the rotation-stage angle from Workflow.txt,
  * ``discover_tiles`` populates view/rotation_index/angle_deg (0 for the
    ordinary single-angle case), and records asymmetric illumination coverage,
  * ``_write_stitch_metadata_v2`` emits the per-tile coverage descriptor and the
    ``partial_coverage`` flag while keeping ``version == 2``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _synth_acq import write_synth_acquisition  # noqa: E402

from flamingo_stitcher.pipeline import (  # noqa: E402
    FLAT_RAW_PATTERN,
    RAW_FILE_PATTERN,
    RawTileInfo,
    StitchingConfig,
    StitchingPipeline,
    _read_start_angle,
    discover_tiles,
)


# --------------------------------------------------------------------------- #
# Filename regex named groups
# --------------------------------------------------------------------------- #
def test_raw_pattern_named_groups():
    m = RAW_FILE_PATTERN.match(
        "S000_t000000_V002_R0180_X000_Y000_C03_I1_D1_P00360.raw"
    )
    assert m is not None
    assert int(m.group("view")) == 2
    assert int(m.group("rot")) == 180
    assert int(m.group("ch")) == 3
    assert int(m.group("illum")) == 1
    assert int(m.group("planes")) == 360


def test_flat_pattern_named_groups():
    m = FLAT_RAW_PATTERN.search(
        "S000_t000000_V001_R0090_X005_Y007_C02_I0_D1_P00120.btf"
    )
    assert m is not None
    assert int(m.group("view")) == 1
    assert int(m.group("rot")) == 90
    assert int(m.group("xidx")) == 5
    assert int(m.group("yidx")) == 7
    assert int(m.group("ch")) == 2
    assert int(m.group("illum")) == 0
    assert int(m.group("planes")) == 120


# --------------------------------------------------------------------------- #
# Angle reading
# --------------------------------------------------------------------------- #
def test_read_start_angle_absent_is_zero(tmp_path):
    wf = tmp_path / "Workflow.txt"
    wf.write_text("<Workflow Settings>\n  <Start Position>\n"
                  "    Z (mm) = 10.0\n  </Start Position>\n</Workflow Settings>\n")
    assert _read_start_angle(wf) == 0.0
    assert _read_start_angle(tmp_path / "missing.txt") == 0.0


def test_read_start_angle_present(tmp_path):
    wf = tmp_path / "Workflow.txt"
    wf.write_text(
        "<Workflow Settings>\n  <Start Position>\n"
        "    Z (mm) = 10.0\n    Angle (degrees) = 180.000\n"
        "  </Start Position>\n</Workflow Settings>\n"
    )
    assert _read_start_angle(wf) == 180.0


# --------------------------------------------------------------------------- #
# discover_tiles populates the new fields
# --------------------------------------------------------------------------- #
def test_discover_defaults_single_angle(tmp_path):
    acq = write_synth_acquisition(tmp_path / "acq", grid=(2, 2), channels=(1,))
    tiles = discover_tiles(acq)
    assert tiles
    for t in tiles:
        assert t.view == 0
        assert t.rotation_index == 0
        assert t.angle_deg == 0.0


def test_discover_reads_angle_from_workflow(tmp_path):
    acq = write_synth_acquisition(tmp_path / "acq", grid=(1, 1), channels=(1,))
    # Inject an Angle into the one tile's Workflow.txt.
    wf = next(acq.glob("X*_Y*/Workflow.txt"))
    text = wf.read_text().replace(
        "  <Start Position>\n", "  <Start Position>\n    Angle (degrees) = 90.0\n"
    )
    wf.write_text(text)
    tiles = discover_tiles(acq)
    assert tiles and tiles[0].angle_deg == 90.0


def test_discover_records_asymmetric_sides(tmp_path):
    acq = write_synth_acquisition(
        tmp_path / "acq", grid=(2, 1), channels=(1,), illum_sides=(0, 1)
    )
    # Make ONE tile single-sided by removing its right-side (I1) file.
    victim = sorted(acq.glob("X*_Y*"))[0]
    right = next(victim.glob("*_I1_*.raw"))
    right.unlink()

    tiles = discover_tiles(acq)
    sides = {tuple(t.illumination_sides) for t in tiles}
    assert (0,) in sides       # the tile we stripped
    assert (0, 1) in sides     # the untouched dual-side tile


# --------------------------------------------------------------------------- #
# Metadata coverage descriptor
# --------------------------------------------------------------------------- #
def _mk_tile(x, sides, angle=0.0):
    return RawTileInfo(
        folder=Path("."), x_mm=x, y_mm=0.0, z_min_mm=0.0, z_max_mm=1.0,
        n_planes=2, illumination_sides=list(sides), angle_deg=angle,
    )


def test_metadata_flags_partial_coverage(tmp_path):
    pipe = StitchingPipeline(StitchingConfig())
    tiles = [_mk_tile(0.0, [0, 1]), _mk_tile(1.0, [0])]  # asymmetric sides
    pipe._write_stitch_metadata_v2(
        tmp_path, [1], {"z": 0.0, "y": 0.0, "x": 0.0}, tiles,
        {"z": 5.0, "y": 0.4, "x": 0.4}, tmp_path,
    )
    meta = json.loads((tmp_path / "stitch_metadata.json").read_text())
    assert meta["version"] == 2                      # reader-compatible
    assert meta["partial_coverage"] is True
    assert meta["illumination_sides"] == [0, 1]
    assert len(meta["tiles"]) == 2


def test_metadata_full_coverage_not_flagged(tmp_path):
    pipe = StitchingPipeline(StitchingConfig())
    tiles = [_mk_tile(0.0, [0, 1]), _mk_tile(1.0, [0, 1])]
    pipe._write_stitch_metadata_v2(
        tmp_path, [1], {"z": 0.0, "y": 0.0, "x": 0.0}, tiles,
        {"z": 5.0, "y": 0.4, "x": 0.4}, tmp_path,
    )
    meta = json.loads((tmp_path / "stitch_metadata.json").read_text())
    assert meta["partial_coverage"] is False
    assert meta["angles_deg"] == [0.0]
