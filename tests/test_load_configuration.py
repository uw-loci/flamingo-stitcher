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


# ---------------------------------------------------------------------------
# Settings with no control in this dialog
#
# The gap that made "Load Configuration" reproduce half a run: the loader only
# ever pushed values into WIDGETS, so every setting without one -- destripe
# tuning, the deconvolution parameters, depth attenuation, border QC, the
# registration thresholds that live in the Options tab -- was silently dropped
# even once it was in the file.
# ---------------------------------------------------------------------------


def _cfg_with_everything():
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.destripe = True
    cfg.destripe_params = {
        "sigma_foreground": 64.0,
        "sigma_background": 512.0,
        "level": 5,
        "wavelet": "db4",
    }
    cfg.destripe_direction = "horizontal"
    cfg.deconvolution_enabled = True
    cfg.deconvolution_iterations = 25
    cfg.deconvolution_na = 0.44
    cfg.depth_attenuation = True
    cfg.depth_attenuation_mu = 0.004
    cfg.quality_threshold = 0.55
    cfg.min_tile_structure = 0.22
    cfg.border_qc_alpha = 0.7
    return cfg


def _load_into(dialog, cfg):
    blob = json.loads(json.dumps(serialize_stitching_config(cfg)))
    dialog._apply_stitching_config(blob)
    return dialog._apply_loaded_overrides(StitchingConfig.with_yaml_defaults())


@pytest.mark.parametrize(
    "field,expected",
    [
        ("destripe_params", {
            "sigma_foreground": 64.0, "sigma_background": 512.0,
            "level": 5, "wavelet": "db4",
        }),
        ("destripe_direction", "horizontal"),
        ("deconvolution_iterations", 25),
        ("deconvolution_na", 0.44),
        ("depth_attenuation", True),
        ("depth_attenuation_mu", 0.004),
        ("quality_threshold", 0.55),
        ("min_tile_structure", 0.22),
        ("border_qc_alpha", 0.7),
    ],
)
def test_settings_without_a_widget_still_reach_the_run(dialog, field, expected):
    out = _load_into(dialog, _cfg_with_everything())
    assert getattr(out, field) == expected


def test_machine_limits_are_never_applied(dialog):
    # A 191 GB rig's ceiling must not follow the file onto a laptop.
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.max_memory_gb = 180.0
    cfg.preprocess_workers = 32
    out = _load_into(dialog, cfg)
    fresh = StitchingConfig.with_yaml_defaults()
    assert out.max_memory_gb == fresh.max_memory_gb
    assert out.preprocess_workers == fresh.preprocess_workers


def test_a_scratch_path_from_another_machine_is_never_applied(dialog):
    # Older files still carry one; taking it points this box at a disk it
    # does not have, and the run only discovers that when it needs the space.
    dialog._apply_stitching_config({"scratch_dir": "/other/machine/nvme"})
    out = dialog._apply_loaded_overrides(StitchingConfig.with_yaml_defaults())
    assert out.scratch_dir == StitchingConfig.with_yaml_defaults().scratch_dir


def test_a_psf_that_travelled_with_the_file_is_used(dialog, tmp_path):
    psf = tmp_path / "psf.tif"
    psf.write_bytes(b"stub")
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.deconvolution_psf_path = str(psf)
    assert _load_into(dialog, cfg).deconvolution_psf_path == str(psf)


def test_a_psf_path_that_does_not_exist_here_is_refused_and_reported(dialog):
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.deconvolution_psf_path = "/no/such/machine/psf.tif"
    out = _load_into(dialog, cfg)
    assert out.deconvolution_psf_path != "/no/such/machine/psf.tif"
    assert dialog._psf_path_unavailable == "/no/such/machine/psf.tif"


def test_an_unknown_field_from_a_newer_build_is_ignored(dialog):
    dialog._apply_stitching_config({"a_setting_from_the_future": 1})
    out = dialog._apply_loaded_overrides(StitchingConfig.with_yaml_defaults())
    assert not hasattr(out, "a_setting_from_the_future")


def test_int_keyed_dicts_come_back_as_ints(dialog):
    # JSON hands them back as strings; a string-keyed per-channel threshold
    # matches no channel and nothing raises.
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.registration_z_refine_binning = {"z": 1, "y": 2, "x": 2}
    out = _load_into(dialog, cfg)
    assert out.registration_z_refine_binning == {"z": 1, "y": 2, "x": 2}


def test_loading_nothing_leaves_the_config_alone(dialog):
    base = StitchingConfig.with_yaml_defaults()
    assert dialog._apply_loaded_overrides(base) is base


# ---------------------------------------------------------------------------
# Save Configuration
# ---------------------------------------------------------------------------


def test_saved_configuration_round_trips_through_the_loader(dialog, tmp_path):
    from flamingo_stitcher.pipeline import serialize_stitching_config as ser

    cfg = _cfg_with_everything()
    path = tmp_path / "shared.json"
    path.write_text(json.dumps({
        "kind": "flamingo-stitcher-configuration",
        "stitching_config": ser(cfg),
    }))

    blob = json.loads(path.read_text())["stitching_config"]
    dialog._apply_stitching_config(blob)
    out = dialog._apply_loaded_overrides(StitchingConfig.with_yaml_defaults())
    assert out.destripe_params == cfg.destripe_params
    assert out.deconvolution_iterations == cfg.deconvolution_iterations
    assert out.quality_threshold == cfg.quality_threshold


def test_both_dialog_classes_have_the_save_button(qapp):
    from flamingo_stitcher.gui.stitching_dialog import (
        NativeStitchingDialog,
        StitchingDialog,
    )

    for cls in (StitchingDialog, NativeStitchingDialog):
        d = cls()
        try:
            assert hasattr(d, "_save_config_btn")
        finally:
            d.deleteLater()
