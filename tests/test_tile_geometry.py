"""Tile spacing arithmetic, once, so the four old copies cannot drift again.

Run: python3 -m pytest tests/test_tile_geometry.py -q
"""

from types import SimpleNamespace

import pytest

from flamingo_stitcher import tile_geometry as tg


def _grid(xs, ys, z_mins=None):
    """Tiles at the cartesian product of xs and ys, in mm."""
    z_mins = z_mins or [0.0]
    return [
        SimpleNamespace(x_mm=x, y_mm=y, z_min_mm=z)
        for z in z_mins
        for y in ys
        for x in xs
    ]


class TestPitch:
    def test_a_regular_grid_has_one_pitch_by_either_statistic(self):
        tiles = _grid([0.0, 1.0, 2.0], [0.0, 1.0])
        assert tg.median_pitch_um(tiles, "x") == pytest.approx(1000.0)
        assert tg.min_pitch_um(tiles, "x") == pytest.approx(1000.0)

    def test_a_single_column_has_no_pitch_at_all(self):
        # None, not 0.0 — "unknown", not "the tiles coincide".
        tiles = _grid([5.0], [0.0, 1.0])
        assert tg.median_pitch_um(tiles, "x") is None
        assert tg.min_pitch_um(tiles, "x") is None

    def test_a_stray_position_drags_the_min_but_not_the_median(self):
        # The reason the layout math standardizes on the median: one tile
        # re-acquired 20 µm off would otherwise redefine the whole grid.
        tiles = _grid([0.0, 1.0, 1.02, 2.0], [0.0])
        assert tg.min_pitch_um(tiles, "x") == pytest.approx(20.0)
        assert tg.median_pitch_um(tiles, "x") == pytest.approx(980.0)

    def test_duplicate_positions_are_folded_before_differencing(self):
        tiles = _grid([0.0, 0.0, 1.0, 1.0, 2.0], [0.0])
        assert tg.median_pitch_um(tiles, "x") == pytest.approx(1000.0)

    def test_positions_are_rounded_so_float_noise_is_not_a_new_row(self):
        tiles = [
            SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=0.0),
            SimpleNamespace(x_mm=1.0 + 1e-9, y_mm=0.0, z_min_mm=0.0),
            SimpleNamespace(x_mm=1.0, y_mm=0.0, z_min_mm=0.0),
        ]
        assert len(tg.distinct_positions_um(tiles, "x")) == 2


class TestAxisOverlap:
    def test_overlap_is_extent_minus_pitch(self):
        tiles = _grid([0.0, 0.9], [0.0])
        ov = tg.axis_overlap(tiles, "x", extent_um=1000.0)
        assert ov.pitch_um == pytest.approx(900.0)
        assert ov.overlap_um == pytest.approx(100.0)
        assert ov.fraction == pytest.approx(0.1)
        assert not ov.is_gapped

    def test_a_gap_is_reported_as_a_negative_overlap_not_clamped_to_zero(self):
        # A gap is missing acquired data. Collapsing it to 0 would read as
        # "tiles just touch", which is a different and fixable situation.
        tiles = _grid([0.0, 1.5], [0.0])
        ov = tg.axis_overlap(tiles, "x", extent_um=1000.0)
        assert ov.overlap_um == pytest.approx(-500.0)
        assert ov.fraction == pytest.approx(-0.5)
        assert ov.is_gapped

    def test_a_single_position_yields_none_not_a_guess(self):
        ov = tg.axis_overlap(_grid([5.0], [0.0]), "x", extent_um=1000.0)
        assert (ov.pitch_um, ov.overlap_um, ov.fraction) == (None, None, None)
        assert ov.n_positions == 1
        assert "single position" in ov.describe()

    def test_a_nonpositive_extent_yields_none_rather_than_dividing_by_zero(self):
        ov = tg.axis_overlap(_grid([0.0, 1.0], [0.0]), "x", extent_um=0.0)
        assert ov.fraction is None

    def test_the_statistic_is_the_callers_choice(self):
        tiles = _grid([0.0, 1.0, 1.02, 2.0], [0.0])
        assert tg.axis_overlap(
            tiles, "x", 1000.0, statistic="min"
        ).pitch_um == pytest.approx(20.0)
        assert tg.axis_overlap(
            tiles, "x", 1000.0, statistic="median"
        ).pitch_um == pytest.approx(980.0)

    def test_an_unknown_axis_or_statistic_is_an_error_not_a_default(self):
        tiles = _grid([0.0, 1.0], [0.0])
        with pytest.raises(ValueError):
            tg.axis_overlap(tiles, "q", 1000.0)
        with pytest.raises(ValueError):
            tg.axis_overlap(tiles, "x", 1000.0, statistic="mean")


class TestGridOverlap:
    def test_x_and_y_are_measured_independently(self):
        tiles = _grid([0.0, 0.9], [0.0, 0.8])
        ov = tg.grid_overlap(tiles, extent_x_um=1000.0, extent_y_um=1000.0)
        assert ov["x"].fraction == pytest.approx(0.1)
        assert ov["y"].fraction == pytest.approx(0.2)
        assert "z" not in ov

    def test_z_appears_only_when_an_extent_is_supplied(self):
        tiles = _grid([0.0, 0.9], [0.0], z_mins=[0.0, 0.5])
        ov = tg.grid_overlap(
            tiles, extent_x_um=1000.0, extent_y_um=1000.0, extent_z_um=1000.0
        )
        assert ov["z"].pitch_um == pytest.approx(500.0)

    def test_the_z_key_reads_z_min_mm(self):
        tiles = [
            SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=1.0),
            SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=1.4),
        ]
        ov = tg.grid_overlap(
            tiles, extent_x_um=100.0, extent_y_um=100.0, extent_z_um=1000.0
        )
        assert ov["z"].overlap_um == pytest.approx(600.0)


class TestPairOverlap:
    def test_it_uses_these_two_tiles_not_the_grid_pitch(self):
        a = SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=0.0)
        b = SimpleNamespace(x_mm=0.75, y_mm=0.0, z_min_mm=0.0)
        assert tg.pair_overlap_um(a, b, "x", 1000.0) == pytest.approx(250.0)

    def test_order_does_not_matter(self):
        a = SimpleNamespace(x_mm=2.0, y_mm=0.0, z_min_mm=0.0)
        b = SimpleNamespace(x_mm=1.1, y_mm=0.0, z_min_mm=0.0)
        assert tg.pair_overlap_um(a, b, "x", 1000.0) == pytest.approx(
            tg.pair_overlap_um(b, a, "x", 1000.0)
        )


class TestRobustness:
    def test_no_tiles_is_not_an_exception(self):
        ov = tg.grid_overlap([], extent_x_um=1000.0, extent_y_um=1000.0)
        assert ov["x"].n_positions == 0 and ov["x"].pitch_um is None

    def test_a_tile_missing_the_attribute_is_skipped_not_fatal(self):
        tiles = [
            SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=0.0),
            SimpleNamespace(y_mm=0.0, z_min_mm=0.0),  # no x_mm
            SimpleNamespace(x_mm=1.0, y_mm=0.0, z_min_mm=0.0),
        ]
        assert tg.median_pitch_um(tiles, "x") == pytest.approx(1000.0)

    def test_an_unparseable_position_is_skipped_not_fatal(self):
        tiles = [
            SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=0.0),
            SimpleNamespace(x_mm="nope", y_mm=0.0, z_min_mm=0.0),
            SimpleNamespace(x_mm=1.0, y_mm=0.0, z_min_mm=0.0),
        ]
        assert tg.median_pitch_um(tiles, "x") == pytest.approx(1000.0)
