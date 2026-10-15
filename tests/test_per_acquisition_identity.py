"""Each acquisition keeps its OWN Z step and microscope name.

2026-10-14: a 2.5 um acquisition was stitched at 1.0 um. Nobody typed 1.0. The
dialog's Z spacing box is one value for the whole dialog -- restored from the
last session and refilled by auto-detect from the first queue item with tiles,
finished or not -- and _build_config passed it on to every queued run. The
likeliest source was an ASLM dataset stitched earlier.

Same release window: after a reinstall both rigs wrote the default microscope
name "princess" in some files and their own name in others, and per-microscope
settings key on whichever file is read first.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flamingo_stitcher.orientation import (  # noqa: E402
    microscope_names_found,
    read_microscope_name,
)
from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline  # noqa: E402

# --------------------------------------------------------------------------
# Z step: the dialog
# --------------------------------------------------------------------------

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


def _set_programmatically(d, value):
    """What session-restore and Discover's auto-detect both do."""
    d._setting_z_step_programmatically = True
    d._z_step_spin.setValue(value)
    d._setting_z_step_programmatically = False


def test_a_detected_or_restored_z_step_is_not_imposed_on_the_run(dialog):
    _set_programmatically(dialog, 1.0)        # e.g. left over from an ASLM run
    assert dialog._z_step_user_set is False
    assert dialog._build_config().z_step_um is None, (
        "a value nobody typed must not override each acquisition's own Z step"
    )


def test_a_typed_z_step_is_still_honoured(dialog):
    dialog._z_step_spin.setValue(2.5)         # a real edit fires the handler
    assert dialog._z_step_user_set is True
    assert dialog._build_config().z_step_um == pytest.approx(2.5)


def test_auto_still_means_from_the_data(dialog):
    _set_programmatically(dialog, 0.0)
    assert dialog._build_config().z_step_um is None


# --------------------------------------------------------------------------
# Z step: the pipeline, when one IS configured
# --------------------------------------------------------------------------


def test_the_pipeline_reports_where_the_z_step_came_from(caplog):
    from types import SimpleNamespace

    pipe = StitchingPipeline(StitchingConfig())
    with caplog.at_level(logging.INFO):
        pipe._check_configured_z_step(1.0, [SimpleNamespace(z_step_mm=0.0025)])
    assert any(
        r.levelno >= logging.WARNING and "2.500" in r.getMessage()
        for r in caplog.records
    )


# --------------------------------------------------------------------------
# Microscope name
# --------------------------------------------------------------------------


def _acq(tmp_path: Path, scope: str, meta: str = None) -> Path:
    acq = tmp_path / "acq"
    acq.mkdir()
    (acq / "ScopeSettings.txt").write_text(
        f"<Type>\n  Microscope name = {scope}\n</Type>\n"
    )
    if meta is not None:
        (acq / "FlamingoMetaData.txt").write_text(f"Microscope name = {meta}\n")
    return acq


def test_every_name_is_listed_with_its_file(tmp_path):
    acq = _acq(tmp_path, "princess", meta="n7")
    found = [(f.name, n) for f, n in microscope_names_found(acq)]
    assert found == [("ScopeSettings.txt", "princess"), ("FlamingoMetaData.txt", "n7")]
    # The lookup everything keys on is unchanged: first file wins.
    assert read_microscope_name(acq) == "princess"


def test_disagreeing_names_are_a_warning_naming_both(tmp_path, caplog):
    acq = _acq(tmp_path, "princess", meta="n7")
    pipe = StitchingPipeline(StitchingConfig())
    with caplog.at_level(logging.INFO):
        pipe._log_microscope_identity(acq)
    warn = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warn, "two instruments' names in one acquisition must not be silent"
    assert "princess" in warn[0] and "n7" in warn[0]
    assert "keyed on 'princess'" in warn[0]


def test_agreeing_names_are_one_quiet_line(tmp_path, caplog):
    acq = _acq(tmp_path, "n7", meta="n7")
    pipe = StitchingPipeline(StitchingConfig())
    with caplog.at_level(logging.INFO):
        pipe._log_microscope_identity(acq)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Microscope name: 'n7'" in r.getMessage() for r in caplog.records)
