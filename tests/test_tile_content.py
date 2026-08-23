"""The content gate: has this tile anything phase correlation can lock onto?

The sample that motivated this is a fish, in agarose, in an FEP tube. That is
four distinct "backgrounds" at four different brightnesses -- air, tube wall,
gel, medium -- so an intensity threshold cannot separate content from
background: the agarose is well above any noise floor and completely
featureless. The measure has to be STRUCTURE relative to noise, which is
independent of absolute brightness.
"""

from __future__ import annotations

import numpy as np
import pytest

from flamingo_stitcher import tile_content as tc

RNG = np.random.default_rng(0)
SHAPE = (24, 96, 96)


def air():
    """Empty: near-zero constant plus shot noise."""
    return (RNG.normal(20, 4, SHAPE)).astype(np.float32)


def agarose(level=3000.0):
    """The case an intensity test gets wrong: BRIGHT and featureless."""
    return (RNG.normal(level, 40, SHAPE)).astype(np.float32)


def fish(level=800.0):
    """Structured: smooth blobs at cellular scale, on a moderate pedestal."""
    from scipy import ndimage

    seed = RNG.random(SHAPE).astype(np.float32)
    blobs = ndimage.gaussian_filter(seed, sigma=3.0)
    blobs = (blobs - blobs.min()) / max(1e-9, float(np.ptp(blobs)))
    return (level + 600.0 * blobs + RNG.normal(0, 15, SHAPE)).astype(np.float32)


def fish_at_cnr(cnr, noise=40.0):
    """Structured sample at a given contrast-to-noise ratio."""
    from scipy import ndimage

    seed = RNG.random(SHAPE).astype(np.float32)
    blobs = ndimage.gaussian_filter(seed, sigma=3.0)
    blobs = (blobs - blobs.min()) / max(1e-9, float(np.ptp(blobs)))
    return (500.0 + cnr * noise * blobs + RNG.normal(0, noise, SHAPE)).astype(np.float32)


def tube_wall():
    """A strong straight edge -- real structure, but a poor alignment target."""
    vol = np.full(SHAPE, 50.0, np.float32)
    vol[:, :, 40:46] = 4000.0
    return (vol + RNG.normal(0, 10, SHAPE)).astype(np.float32)


class TestBrightnessIndependence:
    """The whole point: the score must not track brightness."""

    def test_bright_featureless_agarose_scores_like_empty_air(self):
        assert tc.structure_score(agarose()) < tc.DEFAULT_MIN_STRUCTURE
        assert tc.structure_score(air()) < tc.DEFAULT_MIN_STRUCTURE

    def test_agarose_is_far_brighter_than_the_sample_and_still_scores_lower(self):
        gel, sample = agarose(), fish()
        assert gel.mean() > sample.mean()          # brighter...
        assert tc.structure_score(gel) < tc.structure_score(sample)  # ...less structure

    def test_scaling_a_tile_does_not_change_its_score(self):
        sample = fish()
        base = tc.structure_score(sample)
        assert tc.structure_score(sample * 8.0) == pytest.approx(base, abs=0.02)

    def test_adding_a_pedestal_does_not_change_its_score(self):
        sample = fish()
        base = tc.structure_score(sample)
        assert tc.structure_score(sample + 5000.0) == pytest.approx(base, abs=0.02)


class TestVerdicts:
    def test_a_structured_tile_is_kept(self):
        assert tc.structure_score(fish()) >= tc.DEFAULT_MIN_STRUCTURE

    def test_the_tube_wall_counts_as_structure(self):
        # It IS structure. That it is a BAD alignment target -- a straight edge
        # slides along its own length freely -- is the geometric shift bound's
        # problem, not this measure's.
        assert tc.structure_score(tube_wall()) >= tc.DEFAULT_MIN_STRUCTURE

    def test_a_perfectly_constant_tile_scores_zero(self):
        assert tc.structure_score(np.full(SHAPE, 700.0, np.float32)) == 0.0

    def test_scoring_a_mosaic_marks_only_the_empty_ones(self):
        volumes = [fish(), agarose(), fish(), air()]
        results = tc.score_tiles(volumes)
        assert [r.has_content for r in results] == [True, False, True, False]

    def test_an_excluded_tile_says_why_in_words(self):
        results = tc.score_tiles([agarose()])
        assert "nothing for phase correlation" in results[0].note


class TestItNeverGuesses:
    """"could not measure" and "measured, and it is low" are different findings."""

    def test_an_unmeasurable_tile_returns_none_not_zero(self):
        assert tc.structure_score(np.zeros((1, 1, 1), np.float32)) is None
        assert tc.structure_score(np.array([], np.float32)) is None

    def test_an_unmeasurable_tile_is_KEPT_in_the_registration(self):
        results = tc.score_tiles([np.zeros((1, 1, 1), np.float32)])
        assert results[0].has_content is True
        assert results[0].structure is None
        assert "could not be measured" in results[0].note

    def test_all_nan_is_unmeasurable_rather_than_empty(self):
        assert tc.structure_score(np.full(SHAPE, np.nan, np.float32)) is None

    def test_some_nan_is_tolerated(self):
        sample = fish()
        sample[0, 0, 0] = np.nan
        assert tc.structure_score(sample) is not None


class TestTheThresholdIsWhereTheMeasurementPutIt:
    """The default is calibrated, so a regression in the measure should move
    these numbers and fail here rather than quietly reclassify tiles."""

    def test_featureless_sits_at_the_floor_whatever_its_brightness(self):
        scores = [
            tc.structure_score(
                RNG.normal(level, level / 100 + 4, SHAPE).astype(np.float32)
            )
            for level in (20, 500, 3000, 20000, 60000)
        ]
        assert max(scores) - min(scores) < 0.02   # over a 3000x brightness range
        assert max(scores) < 0.12

    def test_dim_noisy_but_real_sample_is_kept(self):
        # CNR 5. An earlier 0.35 default excluded this -- the false negative
        # that throws away a genuine measurement.
        assert tc.structure_score(fish_at_cnr(5.0)) >= tc.DEFAULT_MIN_STRUCTURE

    def test_sample_below_CNR_one_is_correctly_indistinguishable_from_empty(self):
        # Not a limitation to apologise for: below CNR ~1 there is nothing for
        # phase correlation to find either.
        assert tc.structure_score(fish_at_cnr(0.33)) < tc.DEFAULT_MIN_STRUCTURE


class TestCost:
    def test_a_large_tile_is_subsampled_rather_than_measured_whole(self):
        big = np.zeros((4000, 512, 512), np.float32)
        assert tc._subsample(big).size <= tc._MAX_SAMPLE_VOXELS

    def test_subsampling_does_not_change_the_verdict(self):
        # Same content, tiled up past the sampling cap.
        small = fish()
        big = np.repeat(small, 40, axis=0)
        assert tc.structure_score(big) == pytest.approx(
            tc.structure_score(small), abs=0.1
        )


class TestDescribe:
    def test_it_reports_the_spread_that_produced_the_split(self):
        line = tc.describe(tc.score_tiles([fish(), agarose(), air()]))
        assert "2 of 3 tiles have nothing to register" in line
        assert "structure" in line

    def test_it_says_so_when_nothing_was_excluded(self):
        line = tc.describe(tc.score_tiles([fish(), fish()]))
        assert "all 2 tiles have structure" in line
