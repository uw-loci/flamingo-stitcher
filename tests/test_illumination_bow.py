"""The mid-frame sharpness bow: is there a light-sheet beam waist to correct?

`illumination_low_side` answers an ANTIsymmetric question (which end of the
frame does this sheet light better), and split/blend act on the answer. The bow
answers a SYMMETRIC one: is the frame sharpest in the MIDDLE, which is what a
Gaussian beam waist sitting mid-frame looks like. That matters because no choice
of illumination side can fix it -- if both sheets' waists are mid-frame then both
are spread at both ends -- so it is blur our fusion cannot touch, and the only
remedies are ASLM (optical) or a spatially varying axial deconvolution.

These tests build volumes whose blur profile is known by construction, so the
measure can be checked against ground truth rather than against a rig run.
"""

from __future__ import annotations

import numpy as np
import pytest

from flamingo_stitcher import illumination_geometry as ig

pytest.importorskip("scipy")


def _speckle(shape, seed):
    """Dense fine-grained detail, so a blur has something to destroy."""
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32) * 1000.0


def _blur_columns(vol, sigma_per_col):
    """Blur each column of `vol` laterally by its own sigma.

    Blurs along axis -2 only (within each column), so the varying blur is not
    itself smearing signal ACROSS the axis we then profile. That would be a
    circular test: it would create the band-to-band contrast the measure looks
    for regardless of the physics.
    """
    from scipy import ndimage

    out = np.empty_like(vol, dtype=np.float32)
    for x, sigma in enumerate(sigma_per_col):
        col = vol[..., x]
        out[..., x] = (
            ndimage.gaussian_filter1d(col, float(sigma), axis=-1)
            if sigma > 0
            else col
        )
    return out


def _waist_sigmas(n, *, centre, min_sigma=0.0, max_sigma=3.0):
    """Gaussian-beam-like spread: thinnest at `centre`, worse either side."""
    x = np.arange(n, dtype=np.float64)
    d = np.abs(x - centre) / max(1.0, max(centre, n - 1 - centre))
    return min_sigma + (max_sigma - min_sigma) * d


def _monotonic_sigmas(n, max_sigma=3.0):
    """Attenuation-like: progressively worse from one end to the other."""
    return np.linspace(0.0, max_sigma, n)


SHAPE = (6, 48, 48)


def test_a_centred_waist_reads_as_a_positive_bow():
    """The case the whole measure exists for."""
    base = _speckle(SHAPE, seed=0)
    vol = _blur_columns(base, _waist_sigmas(SHAPE[-1], centre=(SHAPE[-1] - 1) / 2))

    profile = ig._focus_profile(vol, axis=-1)
    assert profile is not None
    bow = ig._focus_bow(profile)

    assert bow > ig._MIN_BOW, (
        f"a sheet thinnest mid-frame should read sharpest mid-frame, got "
        f"bow={bow:+.3f} from profile {np.round(profile, 5)}"
    )
    # Sanity: the profile really is peaked in the middle, not just noisy.
    assert profile.argmax() in (2, 3), profile


def test_pure_attenuation_does_not_read_as_a_bow():
    """The thing we must NOT confuse it with.

    A sheet that simply decays end-to-end is what split and blend already
    handle. If that registered as a bow, the measure would recommend a
    deconvolution for blur that choosing a side removes for free.
    """
    base = _speckle(SHAPE, seed=1)
    vol = _blur_columns(base, _monotonic_sigmas(SHAPE[-1]))

    profile = ig._focus_profile(vol, axis=-1)
    assert profile is not None
    bow = ig._focus_bow(profile)

    assert bow < ig._MIN_BOW, (
        f"a monotonically decaying sheet has no mid-frame peak, got "
        f"bow={bow:+.3f} from profile {np.round(profile, 5)}"
    )


def test_a_flat_sheet_reads_as_no_bow():
    base = _speckle(SHAPE, seed=2)
    profile = ig._focus_profile(base, axis=-1)
    assert profile is not None
    assert abs(ig._focus_bow(profile)) < ig._MIN_BOW


def test_bowed_needs_both_sheets_to_agree():
    """`bowed` requires both sides, because attenuation cannot fake that.

    Attenuation pushes the two sides in OPPOSITE directions along the axis, so
    it can never make both of them peak mid-frame. Requiring agreement is what
    separates sheet geometry from a one-sided artifact.
    """
    one_sided = ig.IlluminationGeometry(
        low_side=0, margin=0.2, tilts={0: 0.2, 1: -0.2}, axis=-1,
        bows={0: 0.30, 1: -0.10},
    )
    assert not one_sided.bowed

    both = ig.IlluminationGeometry(
        low_side=0, margin=0.2, tilts={0: 0.2, 1: -0.2}, axis=-1,
        bows={0: 0.30, 1: 0.28},
    )
    assert both.bowed
    assert both.bow == pytest.approx(0.29)


def test_the_side_measurement_still_works_on_a_bowed_sheet():
    """The two measurements must not interfere.

    A real sheet does both at once: it decays end-to-end AND has a waist. The
    tilt has to survive the bow, or adding this would break the
    `illumination_low_side` check that split/blend depend on.
    """
    n = SHAPE[-1]
    waist = _waist_sigmas(n, centre=(n - 1) / 2, max_sigma=2.0)
    # Side 0 enters at the low end: decay added on top, worsening with x.
    side0 = _blur_columns(_speckle(SHAPE, seed=3), waist + np.linspace(0, 2.0, n))
    # Side 1 enters at the high end: same waist, decay mirrored.
    side1 = _blur_columns(_speckle(SHAPE, seed=3), waist + np.linspace(2.0, 0, n))

    g = ig.measure({0: side0, 1: side1}, axis=-1)
    assert g is not None
    assert g.confident, g.describe()
    assert g.low_side == 0, g.describe()
    assert g.bows and set(g.bows) == {0, 1}
    assert g.profiles and len(g.profiles[0]) == ig._N_BANDS


def _rows(bows_per_tile):
    return [
        (f"tile{i}", 0, 0, 0.2, {0: 0.2, 1: -0.2}, b)
        for i, b in enumerate(bows_per_tile)
    ]


def test_a_repeatable_bow_is_called_a_light_sheet_waist():
    """Consistent across tiles => the sheet, because the waist is tile-fixed.

    The illumination optics are rigid with respect to the detection objective
    and the sample is scanned, so the beam waist sits at the same place inside
    every tile no matter where that tile is in the mosaic.
    """
    r = ig.AcquisitionGeometry(
        low_side=0, votes={0: 4},
        rows=_rows([{0: 0.30, 1: 0.29}, {0: 0.31, 1: 0.30}]),
    )
    assert r.bow_mean == pytest.approx(0.30)
    assert r.bow_spread is not None and r.bow_spread < r.bow_mean
    verdict = r.bow_verdict()
    assert "light-sheet beam waist" in verdict
    assert "CANNOT correct" in verdict
    assert verdict in r.summary()


def test_a_bow_that_moves_between_tiles_is_blamed_on_the_sample():
    """Scattered across tiles => the sample's own shape, not the sheet.

    This is the confound that would otherwise sell a deconvolution to correct
    the fact that a brain is thicker in the middle than at its edges.
    """
    r = ig.AcquisitionGeometry(
        low_side=0, votes={0: 4},
        rows=_rows([{0: 0.45, 1: 0.40}, {0: -0.30, 1: 0.02}]),
    )
    assert r.bow_mean is not None and r.bow_mean > ig._MIN_BOW
    assert r.bow_spread is not None and r.bow_spread >= r.bow_mean
    verdict = r.bow_verdict()
    assert "sample" in verdict and "light-sheet beam waist" not in verdict


def test_no_bow_says_there_is_nothing_to_deconvolve():
    r = ig.AcquisitionGeometry(
        low_side=0, votes={0: 2},
        rows=_rows([{0: 0.00, 1: 0.01}, {0: -0.01, 1: 0.00}]),
    )
    verdict = r.bow_verdict()
    assert "nothing here for an axial" in verdict


def test_the_bow_is_reported_even_when_the_side_is_inconclusive():
    """They are independent measurements, so one must not suppress the other.

    Two sheets can be indistinguishable end-to-end (no usable tilt, so
    `low_side` is None) and still both be sharpest mid-frame.
    """
    r = ig.AcquisitionGeometry(
        low_side=None, votes={},
        rows=_rows([{0: 0.30, 1: 0.29}, {0: 0.31, 1: 0.30}]),
    )
    text = r.summary()
    assert "No tile gave a confident answer" in text
    assert "light-sheet beam waist" in text


def test_an_acquisition_with_no_bow_data_reports_nothing_rather_than_zero():
    """Absent is not the same as measured-and-flat."""
    r = ig.AcquisitionGeometry(low_side=0, votes={0: 1}, rows=[])
    assert r.bow_mean is None
    assert r.bow_spread is None
    assert r.bow_verdict() == ""
