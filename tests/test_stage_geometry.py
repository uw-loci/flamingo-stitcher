"""Measure the stage-vs-image disagreement. Measure only — never apply.

On the 2026-09-06 7x7 the registration corrections were not scattered: they grew
monotonically across the grid, every Y-seam independently measuring about -32 um
between vertically adjacent tiles. That is a per-step disagreement between where
the stage went and where the image says it went, and if it repeats it is a
calibration fault worth ~400 px of drift across a 14x14 grid.

The reason this module only measures is written into these tests. A single
acquisition cannot separate a stage property from a sample-shaped effect, and a
hand-read of ONE ROW of that run suggested a 0.60% X scale error that the fit
over all 34 registered tiles then showed to be scattered (R^2 0.11). A model
fitted to one slice and applied to the next run is how registration made a
mosaic worse in v0.11.2.

Run: python -m pytest tests/test_stage_geometry.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flamingo_stitcher import stage_geometry as sg  # noqa: E402

PITCH_MM = 1.8235
PITCH_UM = 1823.5
EXTENT_UM = 2145.3


def _grid(n=7):
    tiles, index = [], {}
    for row in range(n):
        for col in range(n):
            index[(col, row)] = len(tiles)
            tiles.append(
                SimpleNamespace(
                    x_mm=1.333 + col * PITCH_MM,
                    y_mm=14.037 + row * PITCH_MM,
                    tile_index=(col, row),
                )
            )
    return tiles, index


def _shifts(index, fn):
    return [
        SimpleNamespace(index=i, **dict(zip(("dz_um", "dy_um", "dx_um"), fn(c, r))))
        for (c, r), i in sorted(index.items(), key=lambda kv: kv[1])
    ]


class TestItFindsARamp:
    def test_a_pure_y_ramp_is_systematic(self):
        tiles, index = _grid()
        shifts = _shifts(index, lambda c, r: (0.0, -32.0 * r, 0.0))
        g = sg.measure(tiles, shifts, pitch_um={"x": PITCH_UM, "y": PITCH_UM})
        fit = g.get("dy", "row")
        assert fit.slope_um_per_step == pytest.approx(-32.0)
        assert fit.r2 == pytest.approx(1.0)
        assert fit.is_systematic

    def test_the_slope_is_reported_as_a_share_of_the_pitch(self):
        tiles, index = _grid()
        shifts = _shifts(index, lambda c, r: (0.0, -32.0 * r, 0.0))
        g = sg.measure(tiles, shifts, pitch_um={"x": PITCH_UM, "y": PITCH_UM})
        assert g.get("dy", "row").percent_of_pitch == pytest.approx(
            -100 * 32.0 / PITCH_UM, rel=1e-3
        )

    def test_a_shear_shows_up_as_dx_against_row(self):
        """The 7x7 had one: columns drifting sideways as you go down."""
        tiles, index = _grid()
        shifts = _shifts(index, lambda c, r: (0.0, 0.0, 18.0 * r))
        g = sg.measure(tiles, shifts, pitch_um={"x": PITCH_UM, "y": PITCH_UM})
        assert g.get("dx", "row").is_systematic
        assert not g.get("dx", "col").is_systematic

    def test_a_z_tilt_shows_up_as_dz_against_col(self):
        tiles, index = _grid()
        shifts = _shifts(index, lambda c, r: (13.0 * c, 0.0, 0.0))
        g = sg.measure(tiles, shifts, pitch_um={"x": PITCH_UM, "y": PITCH_UM})
        assert g.get("dz", "col").is_systematic


class TestItRefusesToOverclaim:
    def test_noise_is_not_called_systematic(self):
        """The whole point: a slope through scatter must not read as a finding."""
        tiles, index = _grid()
        rng = np.random.default_rng(0)
        shifts = _shifts(
            index, lambda c, r: (0.0, float(rng.normal(0, 40)), float(rng.normal(0, 40)))
        )
        g = sg.measure(tiles, shifts, pitch_um={"x": PITCH_UM, "y": PITCH_UM})
        assert not g.get("dy", "row").is_systematic
        assert not g.get("dx", "col").is_systematic

    def test_a_flat_correction_is_not_a_perfect_fit(self):
        """Zero variance to explain is not the same as explaining everything."""
        tiles, index = _grid()
        shifts = _shifts(index, lambda c, r: (0.0, 0.0, 0.0))
        g = sg.measure(tiles, shifts, pitch_um={"x": PITCH_UM, "y": PITCH_UM})
        fit = g.get("dy", "row")
        assert fit.r2 == 0.0 and not fit.is_systematic
        assert fit.slope_um_per_step == pytest.approx(0.0)

    def test_too_few_tiles_refuses_to_fit(self):
        tiles, index = _grid(2)
        shifts = _shifts(index, lambda c, r: (0.0, -32.0 * r, 0.0))
        g = sg.measure(tiles, shifts)
        assert g.fits == [] and "curve-fitting" in g.reason

    def test_carried_tiles_are_excluded_from_the_fit(self):
        """A carried tile's correction IS its neighbours' average, so fitting it
        would be fitting the model to its own output."""
        tiles, index = _grid()
        shifts = _shifts(index, lambda c, r: (0.0, -32.0 * r, 0.0))
        core = [i for (c, r), i in index.items() if 0 < c < 6 and 0 < r < 6]
        g = sg.measure(tiles, shifts, placed=core, pitch_um={"x": PITCH_UM})
        assert g.n_tiles == len(core)


class TestTheRealRun:
    """The 2026-09-06 7x7, from its own TILE PLACEMENT table."""

    ROWS = {
        (4, 6): (45.19, -101.76, 81.07), (3, 6): (35.19, -101.76, 82.07),
        (2, 6): (45.19, -130.04, 86.11), (1, 6): (25.19, -124.80, 70.39),
        (5, 5): (55.19, -75.57, 66.46), (4, 5): (45.19, -76.62, 76.93),
        (3, 5): (45.19, -101.76, 81.99), (2, 5): (35.19, -99.66, 73.52),
        (1, 5): (5.19, -93.38, 57.81), (6, 4): (65.19, -58.87, 65.59),
        (5, 4): (65.19, -60.96, 58.25), (4, 4): (45.19, -64.84, 53.26),
        (3, 4): (35.19, -63.79, 39.64), (2, 4): (15.19, -63.00, 29.39),
        (1, 4): (-4.81, -57.69, 16.10), (6, 3): (55.19, -27.43, 56.21),
        (5, 3): (45.19, -31.62, 47.83), (4, 3): (35.19, -32.76, 37.80),
        (3, 3): (15.19, -31.06, 25.81), (2, 3): (5.19, -29.80, 14.50),
        (1, 3): (-24.81, -24.11, 1.77), (5, 2): (45.19, -1.14, 33.86),
        (4, 2): (35.19, -2.08, 23.04), (3, 2): (15.19, -1.02, 11.32),
        (2, 2): (5.19, 0.0, 0.0), (1, 2): (-14.81, 5.75, -13.78),
        (5, 1): (45.19, 32.88, 13.40), (4, 1): (25.19, 31.83, 2.93),
        (3, 1): (15.19, 32.22, -8.78), (2, 1): (-4.81, 33.39, -19.59),
        (1, 1): (-24.81, 38.63, -36.35), (4, 0): (25.19, 63.25, -11.74),
        (3, 0): (5.19, 63.65, -22.41), (2, 0): (-4.81, 64.70, -33.93),
    }

    def _measure(self):
        tiles, shifts = [], []
        for i, ((cx, cy), (dz, dy, dx)) in enumerate(self.ROWS.items()):
            tiles.append(
                SimpleNamespace(
                    x_mm=1.333 + cx * PITCH_MM, y_mm=14.037 + cy * PITCH_MM,
                    tile_index=(cx, cy),
                )
            )
            shifts.append(SimpleNamespace(index=i, dz_um=dz, dy_um=dy, dx_um=dx))
        return sg.measure(tiles, shifts, pitch_um={"x": PITCH_UM, "y": PITCH_UM})

    def test_the_y_pitch_error_is_the_strong_finding(self):
        fit = self._measure().get("dy", "row")
        assert fit.slope_um_per_step == pytest.approx(-30.2, abs=0.5)
        assert fit.r2 > 0.95 and fit.is_systematic

    def test_the_x_scale_error_is_NOT_supported(self):
        """Read off one row it looked like a clean 0.60%. Over all 34 tiles it
        is scatter, and the report has to say so."""
        fit = self._measure().get("dx", "col")
        assert fit.r2 < 0.2
        assert not fit.is_systematic

    def test_the_shear_and_the_tilt_survive(self):
        g = self._measure()
        assert g.get("dx", "row").is_systematic
        assert g.get("dz", "col").is_systematic


class TestTheReport:
    def test_it_says_the_measurement_is_not_applied(self):
        tiles, index = _grid()
        shifts = _shifts(index, lambda c, r: (0.0, -32.0 * r, 0.0))
        text = "\n".join(
            sg.format_report(sg.measure(tiles, shifts, pitch_um={"x": PITCH_UM}))
        )
        assert "measured, not applied" in text

    def test_it_warns_against_treating_one_run_as_calibration(self):
        tiles, index = _grid()
        shifts = _shifts(index, lambda c, r: (0.0, -32.0 * r, 0.0))
        text = "\n".join(sg.format_report(sg.measure(tiles, shifts)))
        assert "ONE acquisition is an observation" in text
        assert "SAME microscope" in text

    def test_it_reports_the_implied_overlap(self):
        tiles, index = _grid()
        shifts = _shifts(index, lambda c, r: (0.0, -32.0 * r, 0.0))
        g = sg.measure(tiles, shifts, pitch_um={"x": PITCH_UM, "y": PITCH_UM})
        text = "\n".join(
            sg.format_report(g, frame_extent_um={"x": EXTENT_UM, "y": EXTENT_UM})
        )
        assert "Implied Y overlap" in text and "the layout assumes 15.0%" in text

    def test_a_refusal_is_explained_rather_than_left_blank(self):
        tiles, index = _grid(2)
        shifts = _shifts(index, lambda c, r: (0.0, 0.0, 0.0))
        text = "\n".join(sg.format_report(sg.measure(tiles, shifts)))
        assert "Not measured:" in text
