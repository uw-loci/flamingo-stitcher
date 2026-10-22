"""tools/estimate_stage_tilt.py: recovers a known tilt, curve and drift from
seams, honours dropped joins, and reads the verbose log."""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "estimate_stage_tilt.py"
spec = importlib.util.spec_from_file_location("estimate_stage_tilt", _TOOL)
est = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = est      # dataclasses look their module up
spec.loader.exec_module(est)

PITCH = 1.931


def _grid(fz, nx=4, ny=5, bad_join=None):
    """Tiles on a grid with dz = fz(x, y, i), and every neighbour seam
    measuring exactly dz_A - dz_B (plus 300 um across ``bad_join``)."""
    data = est.Data()
    for iy in range(ny):
        for ix in range(nx):
            i = iy * nx + ix
            x, y = ix * PITCH, iy * PITCH
            lab = f"X{ix:03d} Y{iy:03d}"
            data.tiles[lab] = est.Tile(lab, x, y, fz(x, y, i), label=lab)
    for iy in range(ny):
        for ix in range(nx):
            a = f"X{ix:03d} Y{iy:03d}"
            for b, axis in ((f"X{ix + 1:03d} Y{iy:03d}", "x"), (f"X{ix:03d} Y{iy + 1:03d}", "y")):
                if b not in data.tiles:
                    continue
                dz = data.tiles[a].dz_um - data.tiles[b].dz_um
                if bad_join and axis == "y" and {a.split()[1], b.split()[1]} == set(bad_join.split("-")):
                    dz += 300.0
                data.seams.append(est.Seam(a, b, axis, "registered", dz))
    return data


def test_a_plane_is_recovered_as_a_tilt():
    data = _grid(lambda x, y, i: -10.0 * x + 20.0 * y + 5.0)
    seams = est.select_seams(data)
    f = est.fit_seams(seams, {"x": lambda t: t.x_mm, "y": lambda t: t.y_mm})
    assert f["coef"]["x"] == pytest.approx(-10.0) and f["coef"]["y"] == pytest.approx(20.0)
    assert f["rms"] == pytest.approx(0.0, abs=1e-9)
    assert est._deg(20.0) == pytest.approx(math.degrees(math.atan(0.02)))


def test_dropping_a_bad_join_removes_exactly_its_seams():
    data = _grid(lambda x, y, i: -10.0 * x, bad_join="Y003-Y004")
    terms = {"x": lambda t: t.x_mm, "y": lambda t: t.y_mm}
    polluted = est.fit_seams(est.select_seams(data), terms)
    clean = est.fit_seams(est.select_seams(data, drop_joins=["Y004-Y003"]), terms)
    assert abs(polluted["coef"]["y"]) > 10
    assert clean["coef"]["x"] == pytest.approx(-10.0) and clean["coef"]["y"] == pytest.approx(0.0, abs=1e-9)


def test_carried_tiles_are_not_evidence():
    data = _grid(lambda x, y, i: 0.0)
    data.tiles["X000 Y000"].how = "carried"
    used = est.select_seams(data)
    assert all("X000 Y000" not in (ta.label, tb.label) for _s, ta, tb in used)


def test_a_curve_is_told_from_a_tilt():
    data = _grid(lambda x, y, i: 3.0 * y * y)
    seams = est.select_seams(data)
    tilt = est.fit_seams(seams, {"x": lambda t: t.x_mm, "y": lambda t: t.y_mm})
    curve = est.fit_seams(seams, {"x": lambda t: t.x_mm, "y": lambda t: t.y_mm,
                                  "yy": lambda t: t.y_mm ** 2})
    assert tilt["rms"] > 1.0 and curve["rms"] < 1e-6
    assert curve["coef"]["yy"] == pytest.approx(3.0)


def test_drift_from_file_times(tmp_path):
    # Raster order, 10 minutes a tile, accelerating drift: not a plane.
    rate = lambda i: 2.0 * (i / 6.0) ** 2                     # noqa: E731  um, i in tiles
    data = _grid(lambda x, y, i: rate(i))
    for lab, t in data.tiles.items():
        ix, iy = int(lab[1:4]), int(lab[6:9])
        i = iy * 4 + ix
        f = tmp_path / f"S000_t000000_V000_R0000_X{ix:03d}_Y{iy:03d}_C00_I0_D0_P00010.raw"
        f.write_bytes(b"")
        os.utime(f, (1_000_000 + 600 * i, 1_000_000 + 600 * i))
    assert est.read_times(tmp_path, data) == len(data.tiles)
    seams = est.select_seams(data)
    T = lambda t: (t.t_s - 1_000_000) / 3600.0                # noqa: E731
    accel = est.fit_seams(seams, {"x": lambda t: t.x_mm, "y": lambda t: t.y_mm,
                                  "t": T, "tt": lambda t: T(t) ** 2})
    tilt = est.fit_seams(seams, {"x": lambda t: t.x_mm, "y": lambda t: t.y_mm})
    assert accel["rms"] < 1e-6 < tilt["rms"]


def test_it_reads_the_verbose_log_and_csvs(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "[t] p:  TILE PLACEMENT (3 tiles)\n"
        "[t] p:   X000 Y000         1.000   13.000    10.00    -1.00     7.00    4.00   -0.50    3.50  registered\n"
        "[t] p:   X001 Y000         3.000   13.000    30.00    -1.00     7.00   12.00   -0.50    3.50  registered\n"
        "[t] p:   X002 Y000         5.000   13.000    50.00    -1.00     7.00   12.00   -0.50    3.50  carried r1 \n"
        "[t] p:  SEAM MEASUREMENTS (2 adjacent pairs)\n"
        "[t] p:   X001 Y000      X000 Y000      x      registered           0.90    20.00     0.00     1.05      0.00   10%\n"
        "[t] p:   X000 Y000      X000 Y001      y      below_quality        0.04                                        10%  below quality threshold 0.4\n"
        "[t] p: =====\n",
        encoding="utf-8",
    )
    data = est.read_log(log)
    assert data.tiles["X002 Y000"].how == "carried r1"
    assert [(s.status, s.dz_um) for s in data.seams] == [("registered", 20.0), ("below_quality", None)]

    tp, sp = tmp_path / "registration_report.csv", tmp_path / "registration_seams.csv"
    with tp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_index", "x_mm", "y_mm", "dz_um", "clamped_z"])
        w.writerows([[0, 0.0, 0.0, 0.0, "False"], [1, PITCH, 0.0, -10.0, "False"]])
    with sp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index_a", "index_b", "axis", "status", "dz_um"])
        w.writerow([1, 0, "x", "registered", -10.0])
    d2 = est.read_csvs(tp, sp)
    assert d2.tiles["1"].label == "X001 Y000" and len(est.select_seams(d2)) == 1
