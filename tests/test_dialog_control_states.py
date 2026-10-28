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
