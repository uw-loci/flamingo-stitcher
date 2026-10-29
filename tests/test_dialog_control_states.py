"""Controls that cannot apply are grayed out; shift limits live on the Options tab.

2026-10-28 tooltip review: 'Content-based blending' only shapes a Blend,
'.ozx' is only packaged from sharded OME-Zarr, and the main window's shift
fields were silently replaced at run time by the saved Options profile.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp):
    from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

    d = StitchingDialog()
    yield d
    d.deleteLater()


def _select(combo, data):
    i = combo.findData(data)
    assert i >= 0, data
    combo.setCurrentIndex(i)


def test_content_based_blending_needs_blend(dialog):
    _select(dialog._tile_fusion_combo, "max")
    assert not dialog._content_fusion_cb.isEnabled()
    _select(dialog._tile_fusion_combo, "blend")
    assert dialog._content_fusion_cb.isEnabled()


def test_ozx_needs_sharded_zarr(dialog):
    for fmt, ok in (
        ("ome-zarr-sharded", True),
        ("both", True),
        ("ome-zarr-v2", False),
        ("ome-tiff", False),
    ):
        if dialog._format_combo.findData(fmt) < 0:
            continue
        _select(dialog._format_combo, fmt)
        assert dialog._ozx_cb.isEnabled() is ok, fmt


def test_the_main_window_no_longer_sets_shift_limits(dialog):
    from flamingo_stitcher.pipeline import StitchingConfig

    assert not hasattr(dialog, "_max_reg_shift_spin")
    assert not hasattr(dialog, "_max_reg_shift_z_spin")
    blank = StitchingConfig()
    cfg = dialog._build_config()
    assert cfg.max_registration_shift_um == blank.max_registration_shift_um
    assert cfg.max_registration_shift_z_um == blank.max_registration_shift_z_um


def test_a_loaded_shift_limit_is_carried_and_named_in_the_log(dialog):
    dialog._apply_stitching_config({"max_registration_shift_z_um": 40.0})
    assert dialog._loaded_config_overrides.get("max_registration_shift_z_um") == 40.0


def test_fast_destripe_needs_a_downsample(dialog):
    """2026-10-29: 'Fast' does nothing useful at 1x XY and Z, so it is grayed
    out and not sent; the check mark returns with a downsample."""
    dialog._destripe_cb.setChecked(True)
    _select(dialog._downsample_xy_combo, 2)
    _select(dialog._downsample_z_combo, 1)
    assert dialog._destripe_fast_cb.isEnabled()
    dialog._destripe_fast_cb.setChecked(True)
    assert dialog._build_config().destripe_fast is True

    _select(dialog._downsample_xy_combo, 1)
    assert not dialog._destripe_fast_cb.isEnabled()
    assert dialog._destripe_fast_cb.isChecked()
    assert dialog._build_config().destripe_fast is False

    _select(dialog._downsample_z_combo, 2)  # a Z-only downsample still counts
    assert dialog._destripe_fast_cb.isEnabled()
    assert dialog._build_config().destripe_fast is True


def test_z_search_needs_refine_z(dialog):
    dialog._skip_reg_cb.setChecked(False)
    dialog._z_refine_cb.setChecked(False)
    assert not dialog._z_refine_range_spin.isEnabled()
    dialog._z_refine_cb.setChecked(True)
    assert dialog._z_refine_range_spin.isEnabled()
    dialog._skip_reg_cb.setChecked(True)
    assert not dialog._z_refine_range_spin.isEnabled()
    dialog._skip_reg_cb.setChecked(False)
    assert dialog._z_refine_range_spin.isEnabled()


def test_qc_detail_needs_border_qc(dialog):
    dialog._border_qc_cb.setChecked(False)
    assert not dialog._border_qc_mode_combo.isEnabled()
    dialog._border_qc_cb.setChecked(True)
    assert dialog._border_qc_mode_combo.isEnabled()


def test_align_light_paths_follows_the_merge(dialog):
    for fusion, ok in (("max", True), ("split", True), ("content", True),
                       ("separate", False), ("leonardo", False)):
        i = dialog._fusion_combo.findData(fusion)
        dialog._fusion_combo.model().item(i).setEnabled(True)
        dialog._fusion_combo.setCurrentIndex(i)
        assert dialog._illum_align_cb.isEnabled() is ok, fusion
    _select(dialog._fusion_combo, "split")
    dialog._illum_align_cb.setChecked(True)
    assert dialog._build_config().illumination_align is True
    _select(dialog._fusion_combo, "separate")
    assert dialog._build_config().illumination_align is False


def test_options_tab_grays_out_settings_switched_off(qapp):
    from flamingo_stitcher.gui.options_panel import OptionsPanel

    panel = OptionsPanel()
    auto = panel._spins["registration_channel_auto"]
    channel = panel._spins["reg_channel"]
    auto.setChecked(True)
    assert not channel.isEnabled()
    auto.setChecked(False)
    assert channel.isEnabled()
    affine = panel._spins["intensity_adjust_affine"]
    pull = panel._spins["intensity_adjust_offset_regularization"]
    affine.setChecked(False)
    assert not pull.isEnabled()
    affine.setChecked(True)
    assert pull.isEnabled()
    panel.deleteLater()


def test_zeroing_thresholds_need_the_box(qapp):
    from flamingo_stitcher.gui.stitching_dialog import BackgroundZeroPanel

    p = BackgroundZeroPanel()
    assert not p._channels_container.isEnabled()
    p._enable_cb.setChecked(True)
    assert p._channels_container.isEnabled()
    assert p._preview_btn.isEnabled()
    p.deleteLater()
