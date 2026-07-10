"""Tests for Load Stitching Configuration (share a setup that worked).

Covers the pipeline-side serializer that embeds the config in
stitch_metadata.json and the GUI-side loader that applies it back, skipping
file-specific / environment-specific fields.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flamingo_stitcher.pipeline import (  # noqa: E402
    SHAREABLE_CONFIG_FIELDS,
    StitchingConfig,
    serialize_stitching_config,
)

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def dialog(qapp):
    from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

    d = StitchingDialog()
    yield d
    d.deleteLater()


# ---------------------------------------------------------------------------
# Serializer (pipeline side)
# ---------------------------------------------------------------------------


def test_serialize_is_json_safe_and_stringifies_dict_keys():
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.background_zero_thresholds = {2: 100, 3: 150}
    blob = serialize_stitching_config(cfg)
    # Round-trips through JSON without error (int dict keys were stringified).
    round_tripped = json.loads(json.dumps(blob))
    assert round_tripped["background_zero_thresholds"] == {"2": 100, "3": 150}
    # Processing + file-specific provenance fields are present.
    assert "output_format" in blob
    assert "pixel_size_um" in blob


def test_serialize_omits_absent_fields():
    class Bare:
        output_format = "ome-tiff"

    blob = serialize_stitching_config(Bare())
    assert blob == {"output_format": "ome-tiff"}
    assert set(blob).issubset(set(SHAREABLE_CONFIG_FIELDS))


# ---------------------------------------------------------------------------
# Loader (GUI side)
# ---------------------------------------------------------------------------


def test_apply_sets_shareable_fields(dialog):
    cfg = serialize_stitching_config(dialog._build_config())
    cfg["output_format"] = "ome-tiff"
    cfg["downsample_xy"] = 2
    cfg["downsample_z"] = 2
    cfg["streaming_mode"] = True
    cfg["illumination_fusion"] = dialog._fusion_combo.itemData(
        dialog._fusion_combo.count() - 1
    )

    applied, skipped = dialog._apply_stitching_config(cfg)

    assert applied > 0
    assert dialog._format_combo.currentData() == "ome-tiff"
    assert dialog._downsample_xy_combo.currentData() == 2
    assert dialog._downsample_z_combo.currentData() == 2
    assert dialog._streaming_combo.currentData() is True


def test_apply_skips_file_specific_fields(dialog):
    before_px = dialog._pixel_size_spin.value()
    before_z = dialog._z_step_spin.value()
    before_frame = dialog._frame_size_combo.currentIndex()

    cfg = {
        "pixel_size_um": before_px + 123.0,
        "z_step_um": before_z + 99.0,
        "frame_width": 512,
        "frame_height": 512,
        "scratch_dir": "/some/other/machine/nvme",
        "downsample_xy": 4,  # one real shareable field so applied > 0
    }
    applied, skipped = dialog._apply_stitching_config(cfg)

    # Physical geometry untouched.
    assert dialog._pixel_size_spin.value() == before_px
    assert dialog._z_step_spin.value() == before_z
    assert dialog._frame_size_combo.currentIndex() == before_frame
    # ...and reported as skipped.
    assert {"pixel_size_um", "z_step_um", "frame_width", "frame_height"} <= skipped
    assert "scratch_dir" in skipped
    # The one shareable field still landed.
    assert dialog._downsample_xy_combo.currentData() == 4


def test_apply_stashes_bg_zero_thresholds_for_replay(dialog):
    cfg = {"background_zero_enabled": True, "background_zero_thresholds": {"2": 77}}
    dialog._apply_stitching_config(cfg)
    assert dialog._pending_bg_zero_thresholds == {2: 77}
    assert dialog._bg_zero_panel.is_enabled() is True


def test_roundtrip_through_metadata_file(dialog, tmp_path):
    """Serialize a config into a stitch_metadata.json, then load it back."""
    cfg = dialog._build_config()
    cfg.output_format = "ome-tiff"
    cfg.downsample_xy = 2
    meta = {
        "version": 2,
        "store_path": "stitched.ome.tif",
        "stitching_config": serialize_stitching_config(cfg),
    }
    meta_path = tmp_path / "stitch_metadata.json"
    meta_path.write_text(json.dumps(meta))

    loaded = json.loads(meta_path.read_text())
    applied, _ = dialog._apply_stitching_config(loaded["stitching_config"])
    assert applied > 0
    assert dialog._format_combo.currentData() == "ome-tiff"
    assert dialog._downsample_xy_combo.currentData() == 2


def test_both_dialog_classes_have_the_button(qapp):
    from flamingo_stitcher.gui.stitching_dialog import (
        NativeStitchingDialog,
        StitchingDialog,
    )

    for cls in (StitchingDialog, NativeStitchingDialog):
        d = cls()
        try:
            assert hasattr(d, "_load_config_btn")
            assert hasattr(d, "_apply_stitching_config")
        finally:
            d.deleteLater()
