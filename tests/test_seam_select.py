"""Seam shift chosen by cell match (seam_select), and its gate and report column.

Run: QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_seam_select.py -q
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from flamingo_stitcher import registration_report as rr
from flamingo_stitcher import seam_select as S

pytest.importorskip("multiview_stitcher")

VOX = {"z": 10.0, "y": 2.0, "x": 2.0}  # um; the band is 0.5-1.5 Z, 1-4 XY voxels


def _punctate(shape, seed=0, n=1500):
    """Cells on a smooth brightness ramp: the ramp dominates SSIM, the cells
    carry the true shift."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, np.float32)
    idx = tuple(rng.integers(0, s, n) for s in shape)
    vol[idx] = 1.0
    vol = ndimage.gaussian_filter(vol, (0.6, 1.2, 1.2)) * 400
    zz, yy, xx = np.meshgrid(*(np.linspace(0, 1, s) for s in shape), indexing="ij")
    vol += 100 + 60 * np.sin(3 * yy + 2 * xx) + 20 * zz
    vol += rng.normal(0, 3, shape).astype(np.float32)
    return vol


def _pair(shift):
    """Fixed and moving crops of one volume, moving(x + shift) == fixed(x)."""
    big = _punctate((40, 120, 160))
    z0, y0, x0 = 10, 20, 30
    dz, dy, dx = shift
    fixed = big[z0:z0 + 20, y0:y0 + 40, x0:x0 + 96]
    moving = big[z0 - dz:z0 - dz + 20, y0 - dy:y0 - dy + 40, x0 - dx:x0 - dx + 96]
    return types.SimpleNamespace(data=fixed.copy()), types.SimpleNamespace(data=moving.copy())


@pytest.mark.parametrize("shift", [(2, 7, -11), (-3, -9, 5), (0, 15, 0)])
def test_recovers_a_known_shift(shift):
    fixed, moving = _pair(shift)
    log = S.SeamChoiceLog()
    res = S.cell_matched_registration(
        fixed, moving, band_sigmas_px=S.band_sigmas_px(VOX), choice_log=log,
        upsample_factor=1,
    )
    t = np.asarray(res["affine_matrix"])[:3, 3]
    assert np.allclose(t, shift, atol=0.6), t
    (choice,) = log.choices
    assert choice.cell_r > 0.5
    assert choice.quality == res["quality"]
    assert log.by_quality(res["quality"]) is choice


def test_wrong_candidates_score_near_zero():
    fixed, moving = _pair((1, 8, -6))
    sig = S.band_sigmas_px(VOX)
    v = np.ones(fixed.data.shape, bool)
    f0 = S._band_pass(fixed.data, v, *sig)
    f1 = S._band_pass(moving.data, v, *sig)
    right = S.cell_r(f0, f1, v, v, (1, 8, -6))
    flipped = S.cell_r(f0, f1, v, v, (1, -8, -6))
    wrapped = S.cell_r(f0, f1, v, v, (1, 8 - 40, -6))  # one overlap width over
    # The wrapped candidate overlaps only 8 rows here, so its r is noisier;
    # what matters is that the right shift wins by a wide margin.
    assert right > 0.5
    assert abs(flipped) < right / 3 and abs(wrapped) < right / 3


def test_candidate_list_matches_multiview_stitcher_expansion():
    # A peak at +5 on a 40-long axis stands for +5, -5, 35 and -45; all are
    # kept, as multiview-stitcher only drops shifts beyond the LARGEST dimension.
    cands = S._candidates([np.array([0.0, 5.0, 0.0])], (20, 40, 96))
    ys = sorted({c[1] for c in cands})
    assert ys == [-45.0, -5.0, 5.0, 35.0]


def test_coarse_voxels_fall_back_to_ssim():
    assert S.band_sigmas_px({"z": 10, "y": 2.1, "x": 2.1}) is not None
    assert S.band_sigmas_px({"z": 10, "y": 8.0, "x": 8.0}) is None
    assert S.band_sigmas_px({"z": 10}) is None


def test_quality_lookup_refuses_ambiguity():
    log = S.SeamChoiceLog()
    a = S.SeamChoice((0, 1, 2), 0.5, 0.4, 10)
    log.add(a)
    log.add(S.SeamChoice((0, 3, 2), 0.7, 0.2, 10))
    log.add(S.SeamChoice((0, 5, 2), 0.7, 0.3, 10))
    assert log.by_quality(0.5) is a
    assert log.by_quality(np.array([0.5])) is a
    assert log.by_quality(0.7) is None  # two seams share it: neither
    assert log.by_quality(float("nan")) is None


# --- pipeline wiring ---------------------------------------------------------


def _pipe(**cfg):
    from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

    c = StitchingConfig()
    for k, v in cfg.items():
        setattr(c, k, v)
    return StitchingPipeline(c)


def test_reject_drops_a_seam_whose_cells_do_not_match():
    p = _pipe(registration_min_cell_r=0.1)
    log = S.SeamChoiceLog()
    log.add(S.SeamChoice((0, 1, 2), 0.81, 0.03, 96))
    log.add(S.SeamChoice((0, 1, 2), 0.62, 0.45, 96))
    reject = p._cell_match_reject(lambda a, b, d: None, log)
    assert "cells r 0.03" in reject(0, 1, {"quality": 0.81})
    assert reject(0, 1, {"quality": 0.62}) is None
    assert reject(0, 1, {"quality": 0.99}) is None  # unknown seam: no guess


def test_geometric_reason_wins_and_floor_zero_disables():
    p = _pipe(registration_min_cell_r=0.1)
    log = S.SeamChoiceLog()
    log.add(S.SeamChoice((0, 1, 2), 0.81, 0.03, 96))
    geo = p._cell_match_reject(lambda a, b, d: "too far", log)
    assert geo(0, 1, {"quality": 0.81}) == "too far"
    off = _pipe(registration_min_cell_r=0.0)
    base = lambda a, b, d: None  # noqa: E731
    assert off._cell_match_reject(base, log) is base


def test_setup_falls_back_and_converts_band_to_registration_voxels():
    p = _pipe(registration_seam_choice="cells")
    func, kw, log = p._seam_choice_setup({"z": 2.5, "y": 2.1, "x": 2.1}, {"z": 4, "y": 1, "x": 1})
    assert func is S.cell_matched_registration and log is kw["choice_log"]
    inner, outer = kw["band_sigmas_px"]
    assert inner[0] == pytest.approx(0.5) and outer[2] == pytest.approx(8.0 / 2.1)
    # XY binning 4 on a 2.1 um voxel = 8.4 um: cells unresolved -> SSIM.
    assert p._seam_choice_setup({"z": 2.5, "y": 2.1, "x": 2.1}, {"z": 4, "y": 4, "x": 4}) == (None, None, None)
    q = _pipe(registration_seam_choice="ssim")
    assert q._seam_choice_setup({"z": 2.5, "y": 2.1, "x": 2.1}, {}) == (None, None, None)


def _seam(**kw):
    base = dict(tile_a="a", tile_b="b", index_a=0, index_b=1, axis="y",
                status=rr.STATUS_REGISTERED, quality=0.62)
    base.update(kw)
    return rr.SeamResult(**base)


def test_cell_scores_reach_report_rows():
    from flamingo_stitcher.pipeline import StitchingPipeline

    log = S.SeamChoiceLog()
    log.add(S.SeamChoice((0, 1, 2), 0.62, 0.45, 96))
    rows = [_seam(), _seam(quality=0.3), _seam(quality=None)]
    StitchingPipeline._attach_cell_scores(rows, log)
    assert [r.cell_r for r in rows] == [0.45, None, None]


def test_verbose_table_shows_cells_column_only_when_measured():
    report = rr.RegistrationReport(ran=True, applied=True, seams=[_seam()])
    assert not any("cells r" in ln for ln in rr.format_verbose_alignment(report))
    report.seams[0].cell_r = 0.45
    lines = rr.format_verbose_alignment(report)
    row = next(ln for ln in lines if ln.startswith("  a "))
    assert "0.45" in row
    assert "cell_r" in rr.SEAM_CSV_HEADER
    assert "0.45" in rr.seam_rows_csv(report)
