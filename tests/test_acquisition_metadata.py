"""Acquisitions from the 2026-10-06 server build on: SystemMetaData.txt and binning.

That build stopped writing ScopeSettings.txt; the experiment folder gets
SystemMetaData.txt (the system document plus the first stack's settings), and
Workflow.txt gains 'Bit binning width/height' and renames 'AOI' to 'ROI'. The
stitcher read the microscope name and objective only from ScopeSettings.txt,
so per-microscope settings and the pixel size were lost, and it never read
binning, so 2x2 data would have stitched at half its real scale.

Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/test_acquisition_metadata.py -q
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest

from flamingo_stitcher import acquisition_metadata as am
from flamingo_stitcher.orientation import microscope_names_found, read_microscope_name
from flamingo_stitcher.pipeline import (
    NonSquareBinningError,
    _read_aoi_from_workflow,
    _sensor_pixel_size_um,
    pixel_size_basis,
    read_camera_binning,
    read_objective_with_source,
    suggested_pixel_size_um,
)

# The shape of the server's document (MetaDataTemplate.cpp), cut down to the
# blocks the stitcher reads, with the template's example camera values kept.
DOCUMENT = """<MetaData>
  Version = 1.1
  <Experiment>
    <Microscope Settings>
      <Camera Settings>
        ROI width = 1024
        ROI height = 1024
        Bit binning width = {bw}
        Bit binning height = {bh}
      </Camera Settings>
    </Microscope Settings>
  </Experiment>
  <Instrument>
    <Microscope characteristics>
      Name = {name}
      Type = L-SPIM
      <Address>
        IP = 10.129.37.22
        Port = 53717
      </Address>
    </Microscope characteristics>
    <Microscope Hardware>
      <Detection>
        <Detection 1>
          Tube lens focal length (mm) = 200
          Image pixel size = 0.38
          <Detection lens>
            Magnification = {mag}
          </Detection lens>
          <Camera>
            Pixel pitch = 4.5
          </Camera>
        </Detection 1>
        <Detection 2>
          <Detection lens>
            Magnification =
          </Detection lens>
        </Detection 2>
      </Detection>
    </Microscope Hardware>
  </Instrument>
</MetaData>
"""


def _document(name="n7", mag="6.2", bw=1, bh=1):
    return DOCUMENT.format(name=name, mag=mag, bw=bw, bh=bh)


def _new_generation(folder: Path, binning=(1, 1), name="n7", mag="6.2"):
    """Turn an old-style experiment folder into what the 10-06+ server writes."""
    for scope in folder.rglob("ScopeSettings.txt"):
        scope.unlink()
    (folder / am.SYSTEM_METADATA_FILE).write_text(
        _document(name=name, mag=mag, bw=binning[0], bh=binning[1]))
    for wf in folder.rglob("Workflow.txt"):
        text = wf.read_text().replace("AOI width", "ROI width").replace("AOI height", "ROI height")
        text = re.sub(
            r"(\s*)</Camera Settings>",
            rf"\1  Bit binning width = {binning[0]}\1  Bit binning height = {binning[1]}\1</Camera Settings>",
            text, count=1)
        if "Bit binning" not in text:  # a Workflow.txt without a camera block
            text += f"\nBit binning width = {binning[0]}\nBit binning height = {binning[1]}\n"
        wf.write_text(text)
    return folder


# --- The document reader ----------------------------------------------------------


def test_document_paths_follow_the_server_template():
    doc = am.parse(_document(name="liara", mag="25.48", bw=2, bh=2))
    assert am.microscope_name(doc) == "liara"
    assert am.objective_magnification(doc) == pytest.approx(25.48)
    assert am.binning_from_document(doc) == (2, 2)
    assert am.camera_pixel_pitch_um(doc) == pytest.approx(4.5)  # reported, not used


def test_empty_values_count_as_missing():
    doc = am.parse(_document(name="", mag=""))
    assert am.microscope_name(doc) is None
    assert am.objective_magnification(doc) is None


def test_binning_from_workflow_text():
    text = "<Camera Settings>\n  Bit binning width = 4\n  Bit binning height = 4\n</Camera Settings>"
    assert am.binning_from_text(text) == (4, 4)
    assert am.binning_from_text("AOI width = 2048") is None
    assert am.binning_from_text("Bit binning width = 0\nBit binning height = 0") is None


def test_frame_size_reads_roi_and_aoi(tmp_path):
    wf = tmp_path / "Workflow.txt"
    wf.write_text("  ROI width = 1024\n  ROI height = 512\n")
    assert _read_aoi_from_workflow(wf) == (1024, 512)
    wf.write_text("  AOI width = 2048\n  AOI height = 2048\n")
    assert _read_aoi_from_workflow(wf) == (2048, 2048)


# --- Objective, name and pixel size on synthetic acquisitions -----------------------


sys.path.insert(0, str(Path(__file__).parent))


def _acq(tmp_path, name="acq"):
    from _synth_acq import write_synth_acquisition

    return write_synth_acquisition(tmp_path / name, grid=(2, 2), n_planes=12,
                                   frame_size=(48, 48),
                                   pixel_size_um=_sensor_pixel_size_um() / 6.2)


def test_old_generation_is_read_exactly_as_before(tmp_path):
    acq = _acq(tmp_path)
    mag, source = read_objective_with_source(acq)
    assert source == "ScopeSettings.txt" and mag == pytest.approx(6.2, rel=1e-3)
    assert read_camera_binning(acq) == ((1, 1), None)
    assert suggested_pixel_size_um(acq) == pytest.approx(_sensor_pixel_size_um() / 6.2, rel=1e-3)


def test_new_generation_objective_and_name_come_from_systemmetadata(tmp_path):
    acq = _new_generation(_acq(tmp_path), name="n7", mag="6.2")
    assert not list(acq.rglob("ScopeSettings.txt"))
    mag, source = read_objective_with_source(acq)
    assert (mag, source) == (pytest.approx(6.2), "SystemMetaData.txt")
    assert read_microscope_name(acq) == "n7"
    assert microscope_names_found(acq) == [(acq / "SystemMetaData.txt", "n7")]
    from flamingo_stitcher import scope_profiles

    scope, objective, _values, _src = scope_profiles.resolve_for_acquisition(acq)
    assert scope == "n7" and objective == pytest.approx(6.2)


@pytest.mark.parametrize("binning", [1, 2, 4])
def test_pixel_size_scales_with_binning(tmp_path, binning):
    acq = _new_generation(_acq(tmp_path), binning=(binning, binning))
    assert read_camera_binning(acq) == ((binning, binning), "Workflow.txt")
    expected = _sensor_pixel_size_um() * binning / 6.2
    assert suggested_pixel_size_um(acq) == pytest.approx(expected)
    basis = pixel_size_basis(acq)
    assert basis["pixel_um"] == pytest.approx(expected)


def test_binning_falls_back_to_the_document(tmp_path):
    acq = _new_generation(_acq(tmp_path), binning=(2, 2))
    for wf in acq.rglob("Workflow.txt"):  # a Workflow.txt without the keys
        wf.write_text(re.sub(r"\s*Bit binning (width|height) = \d+", "", wf.read_text()))
    assert read_camera_binning(acq) == ((2, 2), "SystemMetaData.txt")


def test_unequal_binning_is_refused_not_guessed(tmp_path):
    acq = _new_generation(_acq(tmp_path), binning=(2, 4))
    with pytest.raises(NonSquareBinningError, match="2x4"):
        suggested_pixel_size_um(acq)
    assert pixel_size_basis(acq)["pixel_um"] is None


def test_the_documents_camera_pitch_is_not_used(tmp_path):
    acq = _new_generation(_acq(tmp_path))
    # The document says 4.5 µm (the template's example); the config pitch wins.
    assert suggested_pixel_size_um(acq) == pytest.approx(_sensor_pixel_size_um() / 6.2)
    assert _sensor_pixel_size_um() != pytest.approx(4.5)


# --- A whole run on new-generation data ---------------------------------------------


def _run(acq, tmp_path, caplog, **cfg_overrides):
    pytest.importorskip("multiview_stitcher")
    from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

    cfg = StitchingConfig.with_yaml_defaults()
    cfg.streaming_mode = True
    cfg.skip_registration = True
    cfg.output_format = "ome-tiff"
    cfg.resource_guard_enabled = False
    cfg.auto_pixel_size = True
    cfg.scratch_dir = str(tmp_path / "scratch")
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    pipe = StitchingPipeline(cfg, progress_fn=lambda p, m: None)
    with caplog.at_level(logging.INFO):
        pipe.run(acq, tmp_path / "out")
    return pipe, caplog.text


def test_a_binned_new_generation_acquisition_stitches_at_its_real_scale(tmp_path, caplog):
    acq = _new_generation(_acq(tmp_path), binning=(2, 2))
    pipe, log = _run(acq, tmp_path, caplog)
    expected = round(_sensor_pixel_size_um() * 2 / 6.2, 4)
    assert pipe.config.pixel_size_um == pytest.approx(expected)
    assert "Microscope name: 'n7' (from SystemMetaData.txt)" in log
    assert "2x2 binning (Workflow.txt)" in log
    assert "objective 6.20x (SystemMetaData.txt)" in log
    assert "the document's value is not used" in log  # pitch 4.5 reported


def test_unequal_binning_stops_the_run_with_a_reason(tmp_path, caplog):
    acq = _new_generation(_acq(tmp_path), binning=(2, 4))
    with pytest.raises(NonSquareBinningError):
        _run(acq, tmp_path, caplog)
    assert "Cannot stitch" in caplog.text


def test_the_server_default_name_is_flagged(tmp_path, caplog):
    acq = _new_generation(_acq(tmp_path), name=am.SERVER_DEFAULT_NAME)
    _pipe, log = _run(acq, tmp_path, caplog)
    assert "every unconfigured microscope shares it" in log
