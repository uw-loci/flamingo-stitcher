"""The Options tab must be able to measure the illumination side.

The microscope PC runs the frozen installer, whose entry point
(`installer/launcher.py`) goes straight to `gui.app.main` — there is no Python
on PATH and no CLI, so `python -m flamingo_stitcher.illumination_geometry`
cannot be run where the data is. A diagnostic that only exists on the command
line does not exist for the person who needs it.

PyQt5 is not installed in this environment and the release workflow only builds
the installer, so the dialog's *behaviour* cannot be exercised anywhere in
automation. These tests check the wiring statically (the pattern the other GUI
tests in this suite already use) and exercise the Qt-free measurement for real.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.ndimage import gaussian_filter

from flamingo_stitcher import illumination_geometry as ig

PANEL = Path(__file__).resolve().parents[1] / (
    "src/flamingo_stitcher/gui/options_panel.py"
)
LAUNCHER = Path(__file__).resolve().parents[1] / "installer/launcher.py"


def _tree():
    return ast.parse(PANEL.read_text())


def _func(name):
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


# --------------------------------------------------------------------------
# Why the button has to exist at all
# --------------------------------------------------------------------------


def test_the_frozen_entry_point_has_no_command_line():
    """Guards the premise. If the launcher ever grows argument handling, the
    CLI becomes reachable on the rig and this constraint changes."""
    src = LAUNCHER.read_text()
    assert "gui.app" in src
    assert "argv" not in src and "argparse" not in src


# --------------------------------------------------------------------------
# The wiring
# --------------------------------------------------------------------------


def test_the_handler_exists():
    assert _func("_on_measure_illumination_side") is not None


def test_the_button_is_built_only_for_the_illumination_side_control():
    ctrl = _func("_control_for")
    assert ctrl is not None
    guards = [
        n for n in ast.walk(ctrl)
        if isinstance(n, ast.If)
        and "illumination_low_side" in ast.dump(n.test)
    ]
    assert len(guards) == 1, "the button must be gated on that one field"
    body = ast.dump(guards[0])
    assert "QPushButton" in body
    assert "_on_measure_illumination_side" in body
    assert "addWidget" in body


def test_the_handler_asks_for_a_folder_and_measures_it():
    body = ast.dump(_func("_on_measure_illumination_side"))
    assert "getExistingDirectory" in body
    assert "measure_acquisition" in body


def test_the_handler_writes_the_answer_into_the_control():
    body = ast.dump(_func("_on_measure_illumination_side"))
    assert "setValue" in body, "measuring is useless if it cannot apply"
    assert "illumination_low_side" in body


def test_a_read_failure_is_caught():
    """A tab that dies on an unreadable folder loses the user's other edits."""
    fn = _func("_on_measure_illumination_side")
    assert any(isinstance(n, ast.Try) for n in ast.walk(fn))
    assert "warning" in ast.dump(fn)


def test_the_wait_cursor_is_always_restored():
    fn = _func("_on_measure_illumination_side")
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    assert any(t.finalbody for t in tries), "needs a finally, not just except"
    assert "restoreOverrideCursor" in ast.dump(fn)


def test_the_panel_imports_what_the_handler_uses():
    src = PANEL.read_text()
    for name in ("QFileDialog", "QApplication", "QMessageBox", "QPushButton"):
        assert f"    {name},\n" in src, name


# --------------------------------------------------------------------------
# The measurement the button and the CLI share (no Qt)
# --------------------------------------------------------------------------

NZ, NY, NX = 12, 128, 128


def _sheets(low_side=1):
    rng = np.random.default_rng(0)
    v = gaussian_filter(rng.normal(0, 1, (NZ, NY, NX)), (1.0, 3, 3))
    v -= v.min()
    v /= v.max()
    tile = ((v**2) * 3000 + 200).astype(np.float32)
    out = {}
    for side, near in ((low_side, 0), (1 - low_side, NX - 1)):
        d = (np.abs(np.arange(NX) - near) / (NX - 1)).reshape(1, 1, NX)
        out[side] = (
            ((1 - d) * tile + d * gaussian_filter(tile, (0, 5, 5)))
            * (0.25 + 0.75 * np.exp(-2.2 * d))
        ).astype(np.float32)
    return out


def _fake_tile():
    t = mock.Mock(
        raw_files={3: {0: "s0", 1: "s1"}}, n_planes=NZ, frame_width=NX, frame_height=NY
    )
    t.folder.name = "tileA"
    return t


def _run(tiles, sheets):
    with mock.patch(
        "flamingo_stitcher.pipeline.discover_tiles", return_value=tiles
    ), mock.patch(
        "flamingo_stitcher.pipeline.load_tile_volume",
        side_effect=lambda p, *a, **k: sheets[int(str(p)[1])],
    ):
        return ig.measure_acquisition("/acq")


def test_it_pools_votes_over_tiles():
    r = _run([_fake_tile() for _ in range(3)], _sheets(low_side=1))
    assert r.low_side == 1
    assert (r.agreed, r.total) == (3, 3)
    assert not r.split
    assert "Side 1" in r.summary()


def test_it_honours_max_tiles():
    tiles = [_fake_tile() for _ in range(10)]
    with mock.patch(
        "flamingo_stitcher.pipeline.discover_tiles", return_value=tiles
    ), mock.patch(
        "flamingo_stitcher.pipeline.load_tile_volume",
        side_effect=lambda p, *a, **k: _sheets(1)[int(str(p)[1])],
    ):
        r = ig.measure_acquisition("/acq", max_tiles=2)
    assert r.total == 2


def test_a_single_sided_acquisition_returns_no_rows():
    t = mock.Mock(raw_files={3: {0: "s0"}}, n_planes=NZ, frame_width=NX, frame_height=NY)
    t.folder.name = "solo"
    r = _run([t], _sheets())
    assert r.rows == [] and r.low_side is None


def test_disagreeing_tiles_are_called_out():
    r = ig.AcquisitionGeometry(low_side=1, votes={1: 2, 0: 1}, rows=[])
    assert r.split
    assert "disagreed" in r.summary()


def test_an_inconclusive_acquisition_says_what_to_do_next():
    r = ig.AcquisitionGeometry(low_side=None, votes={}, rows=[])
    assert not r.split
    assert "which half of the frame is in focus" in r.summary()
