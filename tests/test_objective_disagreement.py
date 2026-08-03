"""Objective magnification cross-check: ScopeSettings.txt vs FlamingoMetaData.

An acquisition records the objective in two independent places. When they
disagree, one is stale and every tile is rendered at the wrong scale, producing
mis-registered "ghost" duplicates of the same feature. Discovery must read the
FlamingoMetaData objective and warn loudly at the disagreement — not after a
full (wrong) stitch. Same value in both → silence.
"""

import logging
from pathlib import Path

from flamingo_stitcher.pipeline import (
    StitchingConfig,
    StitchingPipeline,
    discover_tiles,
    read_objective_magnification,
    read_objective_magnification_metadata,
)

from _synth_acq import write_synth_acquisition


def _write_scope(acq: Path, mag: float) -> None:
    (acq / "ScopeSettings.txt").write_text(
        "<Scope Settings>\n"
        f"  Objective lens magnification = {mag}\n"
        "</Scope Settings>\n"
    )


def _write_metadata(acq: Path, mag: float, name: str = "liara") -> None:
    (acq / "FlamingoMetaData.txt").write_text(
        "<Instrument>\n"
        "  <Type>\n"
        f"    Microscope name = {name}\n"
        f"    Objective lens magnification = {mag}\n"
        "  </Type>\n"
        "</Instrument>\n"
    )


def test_reads_objective_from_flamingo_metadata(tmp_path):
    acq = write_synth_acquisition(tmp_path / "a", grid=(2, 2))
    _write_metadata(acq, 23.8)
    assert read_objective_magnification_metadata(acq) == 23.8


def test_metadata_reader_none_when_absent(tmp_path):
    acq = write_synth_acquisition(tmp_path / "b", grid=(2, 2))
    assert read_objective_magnification_metadata(acq) is None


def test_metadata_reader_finds_nested(tmp_path):
    # Selected a name-level folder; metadata lives one level down.
    root = tmp_path / "Sample"
    nested = root / "20260728_HighRes"
    nested.mkdir(parents=True)
    _write_metadata(nested, 25.48)
    assert read_objective_magnification_metadata(root) == 25.48


def _run_geometry(acq: Path, caplog) -> None:
    pipe = StitchingPipeline(StitchingConfig(pixel_size_um=0.406, auto_pixel_size=False))
    tiles = discover_tiles(acq)
    assert tiles
    with caplog.at_level(logging.WARNING):
        pipe._apply_and_log_geometry(tiles, acq)


def test_disagreement_warns(tmp_path, caplog):
    # ScopeSettings holds a stale 17x; the acquisition metadata says 23.8x.
    acq = write_synth_acquisition(tmp_path / "c", grid=(2, 2))
    _write_scope(acq, 17.0)
    _write_metadata(acq, 23.8)
    assert read_objective_magnification(acq) == 17.0

    _run_geometry(acq, caplog)

    disagree = [r for r in caplog.records if "DISAGREE" in r.message]
    assert disagree, "expected an objective-disagreement warning"
    msg = disagree[0].message
    assert "17" in msg and "23.8" in msg


def test_agreement_is_silent(tmp_path, caplog):
    acq = write_synth_acquisition(tmp_path / "d", grid=(2, 2))
    _write_scope(acq, 23.8)
    _write_metadata(acq, 23.8)

    _run_geometry(acq, caplog)

    assert not [r for r in caplog.records if "DISAGREE" in r.message]
