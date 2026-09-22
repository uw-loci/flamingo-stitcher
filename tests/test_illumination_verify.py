"""The pre-flight that checks `illumination_low_side` against the data.

Split/blend keep one sheet per half of the frame. With `low_side` backwards
they keep the FAR half of each sheet, and the output still looks smooth and
plausible -- there is no downstream symptom, only a worse image. This is the
check that turns that into a message before the run instead of a surprise
after it.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from flamingo_stitcher import illumination_geometry as ig
from flamingo_stitcher.pipeline import StitchingPipeline


NZ, NY, NX = 16, 160, 160


def _sheets(low_side=1):
    rng = np.random.default_rng(0)
    v = gaussian_filter(rng.normal(0, 1, (NZ, NY, NX)), (1.0, 3, 3))
    v -= v.min()
    v /= v.max()
    tile = ((v**2) * 3000 + 200).astype(np.float32)
    out = {}
    for side, near in ((low_side, 0), (1 - low_side, NX - 1)):
        d = (np.abs(np.arange(NX) - near) / (NX - 1)).reshape(1, 1, NX)
        blurred = gaussian_filter(tile, (0, 5, 5))
        out[side] = (
            ((1 - d) * tile + d * blurred) * (0.25 + 0.75 * np.exp(-2.2 * d))
        ).astype(np.float32)
    return out


def _pipeline(**cfg):
    base = dict(
        illumination_fusion="split",
        illumination_low_side=1,
        illumination_verify=True,
        illumination_axis="auto",
    )
    base.update(cfg)
    p = StitchingPipeline.__new__(StitchingPipeline)
    p.config = SimpleNamespace(**base)
    p.logger = mock.MagicMock()
    p._resolve_illumination_axis = lambda vol: -1
    return p


def _tiles(volumes, n=1):
    return [
        SimpleNamespace(
            raw_files={3: {s: f"side{s}.raw" for s in volumes}},
            n_planes=NZ,
            frame_width=NX,
            frame_height=NY,
        )
        for _ in range(n)
    ]


def _run(pipeline, volumes, n_tiles=1):
    with mock.patch(
        "flamingo_stitcher.pipeline.load_tile_volume",
        side_effect=lambda path, *a, **k: volumes[int(str(path)[4])],
    ):
        pipeline._verify_illumination_geometry(_tiles(volumes, n_tiles))


def test_a_matching_setting_passes_and_says_so():
    vols = _sheets(low_side=1)
    p = _pipeline(illumination_low_side=1)
    _run(p, vols)
    said = " ".join(str(c) for c in p.logger.info.call_args_list)
    assert "confirmed" in said


def test_a_backwards_setting_stops_the_run():
    vols = _sheets(low_side=1)
    p = _pipeline(illumination_low_side=0)
    with pytest.raises(ValueError) as exc:
        _run(p, vols)
    msg = str(exc.value)
    # The user has to be told WHERE to fix it, WHAT to set it to, and how to
    # override -- with the config key spelled as its real YAML path, since
    # `illumination_verify` is the dataclass field and not something anyone can
    # type into a config file.
    assert "Options tab" in msg
    assert "Measure from data" in msg, "the rig has no CLI; name the button"
    assert "to 1" in msg
    assert "illumination.verify" in msg


def test_max_fusion_is_never_checked():
    """max uses both sheets everywhere; the geometry is irrelevant to it."""
    p = _pipeline(illumination_fusion="max", illumination_low_side=0)
    with mock.patch("flamingo_stitcher.pipeline.load_tile_volume") as loader:
        p._verify_illumination_geometry(_tiles(_sheets()))
    loader.assert_not_called()


def test_an_unset_low_side_is_left_to_the_existing_preflight():
    """`_check_illumination_fusion_is_usable` already refuses low_side < 0."""
    p = _pipeline(illumination_low_side=-1)
    with mock.patch("flamingo_stitcher.pipeline.load_tile_volume") as loader:
        p._verify_illumination_geometry(_tiles(_sheets()))
    loader.assert_not_called()


def test_the_check_can_be_switched_off():
    vols = _sheets(low_side=1)
    p = _pipeline(illumination_low_side=0, illumination_verify=False)
    _run(p, vols)  # backwards, but must not raise
    said = " ".join(str(c) for c in p.logger.info.call_args_list)
    assert "skipped" in said


def test_an_unmeasurable_acquisition_continues():
    """Empty medium cannot answer. That must not block a run."""
    rng = np.random.default_rng(1)
    flat = rng.normal(300, 8, (NZ, NY, NX)).astype(np.float32)
    p = _pipeline(illumination_low_side=0)
    _run(p, {0: flat, 1: flat.copy()})
    said = " ".join(str(c) for c in p.logger.info.call_args_list)
    assert "could not be measured" in said


def test_it_keeps_trying_tiles_until_one_can_answer():
    """Rim tiles are usually empty medium; a later tile has the sample."""
    rng = np.random.default_rng(2)
    flat = rng.normal(300, 8, (NZ, NY, NX)).astype(np.float32)
    good = _sheets(low_side=1)
    calls = {"n": 0}

    def loader(path, *a, **k):
        side = int(str(path)[4])
        calls["n"] += 1
        # First two tiles (4 reads) are empty; then the real thing.
        return flat if calls["n"] <= 4 else good[side]

    p = _pipeline(illumination_low_side=1)
    with mock.patch("flamingo_stitcher.pipeline.load_tile_volume", side_effect=loader):
        p._verify_illumination_geometry(_tiles(good, n=4))
    said = " ".join(str(c) for c in p.logger.info.call_args_list)
    assert "confirmed" in said


def test_a_single_sided_acquisition_is_not_probed():
    p = _pipeline()
    tiles = [
        SimpleNamespace(
            raw_files={3: {0: "side0.raw"}},
            n_planes=NZ, frame_width=NX, frame_height=NY,
        )
    ]
    with mock.patch("flamingo_stitcher.pipeline.load_tile_volume") as loader:
        p._verify_illumination_geometry(tiles)
    loader.assert_not_called()


# --------------------------------------------------------------------------
# Per-seam registration progress
# --------------------------------------------------------------------------


def test_registration_reports_progress_per_seam():
    """Without this the whole phase is one emit and the ETA extrapolates from
    a frozen fraction -- on the 2026-09-18 run, for 6h41m."""
    seen = []
    p = StitchingPipeline.__new__(StitchingPipeline)
    p._progress_fn = lambda pct, msg: seen.append(pct)

    def stub(**kw):
        return {"affine_matrix": None, "quality": 0.8}

    with mock.patch(
        "multiview_stitcher.registration.phase_correlation_registration", stub
    ):
        func = p._seam_progress_func(8)
        for _ in range(8):
            func(fixed_data=None, moving_data=None)

    assert seen == sorted(seen), "progress must never go backwards"
    assert 45 < seen[0] and seen[-1] == 70, seen


def test_the_wrapper_keeps_the_signature_multiview_stitcher_inspects():
    """multiview-stitcher picks arguments with `has_keyword`; a bare
    *args/**kwargs wrapper advertises none and silently changes the call."""
    from dask.utils import has_keyword
    from multiview_stitcher import registration as mvs_reg

    p = StitchingPipeline.__new__(StitchingPipeline)
    p._progress_fn = lambda *a: None
    wrapped = p._seam_progress_func(1)
    for kw in ("fixed_data", "moving_data", "initial_affine",
               "fixed_origin", "moving_spacing"):
        assert has_keyword(wrapped, kw) == has_keyword(
            mvs_reg.phase_correlation_registration, kw
        ), kw
