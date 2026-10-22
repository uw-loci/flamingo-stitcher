"""tools/export_seam_strips.py cuts the edges that actually overlap: the two
strips of every seam hold the same content, whichever way round the pair is
named."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "export_seam_strips.py"
spec = importlib.util.spec_from_file_location("export_seam_strips", _TOOL)
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


def test_both_strips_of_each_seam_show_the_same_tissue(tmp_path):
    from _synth_acq import write_synth_acquisition

    acq = write_synth_acquisition(
        tmp_path / "acq", grid=(3, 2), overlap=0.25, n_planes=8,
        frame_size=(64, 64), illum_sides=(0, 1),
    )
    seams = ["X000 Y000:X001 Y000", "X001 Y000:X001 Y001",
             "X001 Y002:X001 Y001", "X001 Y002:X000 Y002"]
    argv = [str(acq), str(tmp_path / "out"), "--pixel-um", "0.406", "--margin-um", "0",
            "--tile-orientation", "identity", "--reverse-x", "no", "--reverse-y", "no"]
    for s in seams:
        argv += ["--seam", s]
    assert tool.main(argv) == 0
    dirs = sorted((tmp_path / "out").iterdir())
    assert len(dirs) == 4
    for d in dirs:
        meta = json.loads((d / "seam.json").read_text())
        a = np.load(d / "A.npy").astype(float)
        b = np.load(d / "B.npy").astype(float)
        assert a.shape == b.shape
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.9, d.name
        assert {meta["strips"]["A"]["edge"], meta["strips"]["B"]["edge"]} == {"high", "low"}


def test_an_unknown_tile_is_named(tmp_path):
    from _synth_acq import write_synth_acquisition

    acq = write_synth_acquisition(tmp_path / "acq", grid=(2, 2), frame_size=(32, 32))
    with pytest.raises(SystemExit, match="X009 Y009"):
        tool.main([str(acq), str(tmp_path / "o"), "--seam", "X009 Y009:X000 Y000",
                   "--pixel-um", "0.4", "--tile-orientation", "identity", "--reverse-x", "no", "--reverse-y", "no"])
