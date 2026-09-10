"""Learning a tile's SHAPE must not cost a full preprocessing pass.

Two call sites want only `.shape` before materializing a channel, and both ran
the entire chain to get it and threw the pixels away. On the 2026-09-09 run
that showed up plainly in the log: tile 1 was destriped twice, 93 s of it for a
tuple.

    21:51  Destriping 643 planes ... 93.2s     <- the shape probe
    21:52  Materializing 49 tiles ...
    21:53  Destriping 643 planes ... 90.1s     <- the one that counts

Only `downsample_volume` changes shape. Destripe, deconvolution and depth
attenuation are filters, so `shape_only` skips them and the answer is
identical.

Run: python -m pytest tests/test_shape_probe.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _synth_acq import write_synth_acquisition  # noqa: E402

from flamingo_stitcher.pipeline import (  # noqa: E402
    StitchingConfig,
    StitchingPipeline,
    discover_tiles,
)


@pytest.fixture
def tiles(tmp_path):
    acq = write_synth_acquisition(
        tmp_path / "acq",
        grid=(2, 1),
        n_planes=8,
        channels=(1,),
        illum_sides=(0,),
        frame_size=(32, 32),
    )
    return discover_tiles(acq)


def _pipe(**kwargs):
    return StitchingPipeline(StitchingConfig(**kwargs))


class TestTheShapeIsTheSame:
    def test_plain_config(self, tiles):
        pipe = _pipe()
        full = pipe._preprocess_single_tile(tiles[0], 1).shape
        probe = pipe._preprocess_single_tile(tiles[0], 1, shape_only=True).shape
        assert probe == full

    def test_with_downsample_which_DOES_change_shape(self, tiles):
        # The one stage the probe must keep.
        pipe = _pipe(downsample_xy=2, downsample_z=2)
        full = pipe._preprocess_single_tile(tiles[0], 1).shape
        probe = pipe._preprocess_single_tile(tiles[0], 1, shape_only=True).shape
        assert probe == full
        plain = _pipe()._preprocess_single_tile(tiles[0], 1, shape_only=True).shape
        assert probe != plain, "downsample was skipped, so the shape is wrong"

    def test_with_depth_attenuation(self, tiles):
        pipe = _pipe(depth_attenuation=True, depth_attenuation_mu=0.002)
        full = pipe._preprocess_single_tile(tiles[0], 1).shape
        probe = pipe._preprocess_single_tile(tiles[0], 1, shape_only=True).shape
        assert probe == full


class TestTheExpensiveStagesAreSkipped:
    def _count_destripes(self, monkeypatch):
        import flamingo_stitcher.pipeline as mod

        calls = []
        real = mod.destripe_volume

        def counting(volume, *a, **k):
            calls.append(1)
            return real(volume, *a, **k)

        monkeypatch.setattr(mod, "destripe_volume", counting)
        return calls

    def test_the_probe_does_not_destripe(self, tiles, monkeypatch):
        calls = self._count_destripes(monkeypatch)
        _pipe(destripe=True)._preprocess_single_tile(tiles[0], 1, shape_only=True)
        assert calls == [], "the shape probe destriped the tile"

    def test_the_probe_does_not_destripe_in_fast_mode_either(
        self, tiles, monkeypatch
    ):
        calls = self._count_destripes(monkeypatch)
        _pipe(destripe=True, destripe_fast=True)._preprocess_single_tile(
            tiles[0], 1, shape_only=True
        )
        assert calls == []

    def test_a_real_preprocess_still_destripes(self, tiles, monkeypatch):
        # Guard the guard: shape_only must not have switched destriping off.
        calls = self._count_destripes(monkeypatch)
        _pipe(destripe=True)._preprocess_single_tile(tiles[0], 1)
        assert calls, "destripe stopped running on a normal preprocess"

    def test_the_probe_does_not_deconvolve(self, tiles, monkeypatch):
        pipe = _pipe(deconvolution_enabled=True)
        called = []
        monkeypatch.setattr(
            pipe, "_deconvolve_tile", lambda v, t: called.append(1) or v
        )
        pipe._preprocess_single_tile(tiles[0], 1, shape_only=True)
        assert called == []

    def test_the_shape_still_matches_with_destripe_on(self, tiles):
        pipe = _pipe(destripe=True)
        full = pipe._preprocess_single_tile(tiles[0], 1).shape
        probe = pipe._preprocess_single_tile(tiles[0], 1, shape_only=True).shape
        assert probe == full
