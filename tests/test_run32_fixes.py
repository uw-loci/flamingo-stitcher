"""Fixes from Ket_Trial1 run 32 (2026-10-27): the live ETA, and fast destripe.

Run 32's live ETA fell from a 12:10 prior to 7:55 and then 5:09 within the
first 12 s (actual 11:13): the flat-field step reported 45%, the end of the
preprocessing band, which the estimator's monotone fraction kept. Its phases
also logged out of order. Fast destripe used the full-resolution sigma and
level on the downsampled frame.
"""

import logging
from types import SimpleNamespace

import numpy as np

import flamingo_stitcher.pipeline as P
from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline


def test_run32_status_messages_map_to_the_right_phase():
    pipe = StitchingPipeline(StitchingConfig())
    assert pipe._classify_phase("Loading reference channel 3 for registration...") == "preprocess"
    assert pipe._classify_phase(
        "Materializing 42 tiles for channel 3 (246.0 GB temp on disk, 1 worker)"
    ) == "preprocess"
    assert pipe._classify_phase("Registering tiles (channel 3)...") == "register"
    assert pipe._classify_phase(
        "Fusing channel 3 (1/1) [materializing tiles]..."
    ) == "fuse"


def test_flat_field_estimation_reports_its_own_percent(monkeypatch):
    import flamingo_stitcher.flat_field as ff

    seen = []

    def fake_estimate(ch_data, progress_fn=None):
        progress_fn("Estimating flat-fields via isolated environment...")
        return {}

    monkeypatch.setattr(ff, "is_available", lambda: True)
    monkeypatch.setattr(ff, "estimate_flat_fields", fake_estimate)
    monkeypatch.setattr(
        P, "load_tile_volume", lambda *a, **k: np.zeros((3, 4, 4), np.uint16)
    )
    pipe = StitchingPipeline(StitchingConfig())
    pipe._progress_fn = lambda pct, msg: seen.append(pct)
    tile = SimpleNamespace(
        raw_files={3: {0: "a.raw"}}, n_planes=3, frame_width=4, frame_height=4
    )
    pipe._estimate_flatfield_models([tile], [3])
    assert seen == [P._PCT_FLATFIELD]
    assert P._PCT_FLATFIELD < 5 < P._PCT_REGISTER


def _plane(n, seed=0):
    rng = np.random.default_rng(seed)
    img = 400 + 50 * rng.random((n, n))
    img[:, :] += 80 * np.sin(np.arange(n) / 3.0)[:, None]  # horizontal stripes
    return img.astype(np.uint16)[None]


def test_fast_destripe_converts_sigma_and_level(caplog):
    params = {"sigma_foreground": 228.0, "sigma_background": 456.0, "level": 7,
              "wavelet": "db3", "threshold": 110.0, "crossover": 10.0}
    with caplog.at_level(logging.INFO):
        P.destripe_volume(_plane(256), direction="horizontal", params=params,
                          downsample_xy=2)
    assert "sigma 228/456 -> 114/228, level 7 -> 6" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO):
        P.destripe_volume(_plane(256), direction="horizontal", params=params)
    assert "downsampled frame" not in caplog.text


def test_fast_destripe_converted_settings_spare_real_structure_like_native():
    """Short horizontal bars (real structure, 6x longer than thick) take about
    the native damage with converted settings; unconverted, the downsampled
    frame treats them as stripes and loses more."""
    from flamingo_stitcher._pystripe_core import filter_streaks

    rng = np.random.default_rng(3)
    native = np.full((512, 512), 400.0)
    for _ in range(60):
        y, x = rng.integers(0, 504), rng.integers(0, 464)
        native[y:y + 8, x:x + 48] += 300
    small = native.reshape(256, 2, 256, 2).mean((1, 3))
    kw = dict(wavelet="db3", crossover=10, threshold=110)

    def damage(a, b):
        return np.abs(a - b).mean()

    d_native = damage(native, filter_streaks(native, sigma=[64, 64], level=4, **kw))
    d_conv = damage(small, filter_streaks(small, sigma=[32, 32], level=3, **kw))
    d_raw = damage(small, filter_streaks(small, sigma=[64, 64], level=4, **kw))
    assert d_raw > 1.2 * d_native
    assert abs(d_conv - d_native) < abs(d_raw - d_native)
