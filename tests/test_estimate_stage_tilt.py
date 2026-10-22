"""tools/estimate_stage_tilt.py recovers a known tilt from both inputs."""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "estimate_stage_tilt.py"
spec = importlib.util.spec_from_file_location("estimate_stage_tilt", _TOOL)
est = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = est      # dataclasses look their module up
spec.loader.exec_module(est)

A, B = -10.0, 20.0          # um of dz per mm of stage X, Y
PITCH = 1.931


def _write_csvs(tmp_path):
    tiles, seams = [], []
    for iy in range(4):
        for ix in range(3):
            i = iy * 3 + ix
            x, y = ix * PITCH, iy * PITCH
            tiles.append({"tile_index": i, "x_mm": x, "y_mm": y,
                          "dz_um": A * x + B * y + 5.0, "clamped_z": "False"})
    for t in tiles:
        i, ix, iy = t["tile_index"], t["tile_index"] % 3, t["tile_index"] // 3
        for j, axis in ((i + 1, "x") if ix < 2 else (None, None), (i + 3, "y") if iy < 3 else (None, None)):
            if j is None:
                continue
            seams.append({"index_a": i, "index_b": j, "axis": axis, "status": "registered",
                          "dz_um": tiles[i]["dz_um"] - tiles[j]["dz_um"]})
    tp, sp = tmp_path / "registration_report.csv", tmp_path / "registration_seams.csv"
    for path, rows in ((tp, tiles), (sp, seams)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return tp, sp


def test_plane_fit_and_seams_recover_the_tilt(tmp_path):
    data = est.read_csvs(*_write_csvs(tmp_path))
    p = est.plane_fit(list(data.tiles.values()))
    assert p["a"] == pytest.approx(A) and p["b"] == pytest.approx(B)
    slopes = est.seam_slopes(data, ("registered",))
    assert slopes["x"] == pytest.approx([A] * len(slopes["x"]))
    assert slopes["y"] == pytest.approx([B] * len(slopes["y"]))
    assert est._deg(B) == pytest.approx(math.degrees(math.atan(0.02)))


def test_it_reads_the_verbose_log(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "[t] p:  TILE PLACEMENT (2 tiles)\n"
        "[t] p:   X000 Y000         1.000   13.000    10.00    -1.00     7.00    4.00   -0.50    3.50  registered\n"
        "[t] p:   X001 Y000         3.000   13.000    30.00    -1.00     7.00   12.00   -0.50    3.50  carried r1\n"
        "[t] p:  SEAM MEASUREMENTS (1 adjacent pairs)\n"
        "[t] p:   X001 Y000      X000 Y000      x      registered           0.90    20.00     0.00     1.05      0.00   10%\n"
        "[t] p:   X000 Y000      X000 Y001      y      below_quality        0.04                                        10%  below quality threshold 0.4\n"
        "[t] p: =====\n",
        encoding="utf-8",
    )
    data = est.read_log(log)
    assert data.tiles["X001 Y000"].dz_um == 30.0
    assert [(s.status, s.dz_um) for s in data.seams] == [("registered", 20.0), ("below_quality", None)]
    assert est.seam_slopes(data, ("registered",))["x"] == pytest.approx([10.0])
    assert est.main([str(log)]) == 0
