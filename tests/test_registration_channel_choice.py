"""Which channel does registration run on?

Registration runs on exactly ONE channel and costs hours. Before this, the
channel was `reg_channel` if it happened to name a real channel and otherwise
the LOWEST channel ID -- an ordering with nothing to do with which channel
carries structure. There was also no way to say otherwise from the GUI: the
setting existed only in the YAML and on the CLI. On a multichannel acquisition
that is a coin flip between spending the whole budget on a dense channel and
spending it on a sparse marker whose seams then fail the quality gate.

Run: python -m pytest tests/test_registration_channel_choice.py -q
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

RNG = np.random.default_rng(7)
SHAPE = (24, 96, 96)


def sparse():
    """A marker channel: mostly empty, a bright constant plus noise."""
    return RNG.normal(3000, 40, SHAPE).astype(np.float32)


def dense():
    """An autofluorescence channel: structure at cellular scale."""
    seed = RNG.random(SHAPE).astype(np.float32)
    blobs = ndimage.gaussian_filter(seed, sigma=3.0)
    blobs = (blobs - blobs.min()) / max(1e-9, float(np.ptp(blobs)))
    return (800.0 + 600.0 * blobs + RNG.normal(0, 15, SHAPE)).astype(np.float32)


def _tiles(n=4):
    return [SimpleNamespace(x_mm=float(i), y_mm=0.0) for i in range(n)]


def _pipe(**kwargs):
    return StitchingPipeline(StitchingConfig(**kwargs))


class TestItPicksByContentNotByID:
    def test_the_dense_channel_wins_over_the_lower_numbered_sparse_one(self):
        pipe = _pipe()
        loaded = {
            0: [(sparse(), None)] * 3,   # lower ID, nothing to register on
            1: [(dense(), None)] * 3,
        }
        assert pipe._resolve_reg_channel([0, 1], loaded=loaded) == 1

    def test_the_old_behaviour_would_have_taken_channel_zero(self):
        # The regression this exists to prevent: ID order is not content order.
        pipe = _pipe()
        loaded = {0: [(sparse(), None)] * 3, 1: [(dense(), None)] * 3}
        assert pipe.config.reg_channel == 0
        assert pipe._resolve_reg_channel([0, 1], loaded=loaded) != 0

    def test_it_probes_tiles_when_no_data_is_loaded(self):
        pipe = _pipe()
        seen = []

        def fake(tile, ch, illum_side=None):
            seen.append(ch)
            return dense() if ch == 2 else sparse()

        pipe._preprocess_single_tile = fake
        assert pipe._resolve_reg_channel([1, 2], tiles=_tiles(9)) == 2
        assert set(seen) == {1, 2}

    def test_it_samples_tiles_spread_through_the_mosaic_not_the_first_few(self):
        # The first tiles of a mosaic are a corner, and a corner is exactly
        # where the sample is not.
        pipe = _pipe()
        seen = []
        pipe._preprocess_single_tile = lambda tile, ch, illum_side=None: (
            seen.append(tile.x_mm) or dense()
        )
        pipe._resolve_reg_channel([0, 1], tiles=_tiles(9))
        assert max(seen) >= 8.0, f"only sampled the front of the list: {seen}"

    def test_loaded_data_is_sampled_across_the_mosaic_too(self):
        # The in-memory path takes the same care as the probing path.
        pipe = _pipe()
        # Dense only at the END of the list; a front-loaded sample misses it.
        vols = [sparse()] * 8 + [dense()]
        loaded = {0: [(v, None) for v in vols], 1: [(sparse(), None)] * 9}
        assert pipe._resolve_reg_channel([0, 1], loaded=loaded) == 0


class TestTheOverride:
    def test_auto_off_uses_the_configured_channel(self):
        pipe = _pipe(registration_channel_auto=False, reg_channel=1)
        loaded = {0: [(dense(), None)], 1: [(sparse(), None)]}
        assert pipe._resolve_reg_channel([0, 1], loaded=loaded) == 1

    def test_auto_off_falls_back_when_the_channel_is_absent(self):
        pipe = _pipe(registration_channel_auto=False, reg_channel=9)
        assert pipe._resolve_reg_channel([2, 3]) == 2

    def test_a_single_channel_run_never_probes(self):
        pipe = _pipe()
        pipe._preprocess_single_tile = lambda *a, **k: pytest.fail(
            "probed a channel on a single-channel acquisition"
        )
        assert pipe._resolve_reg_channel([3], tiles=_tiles()) == 3


class TestItDecidesOnce:
    def test_the_choice_is_cached_across_calls(self):
        pipe = _pipe()
        calls = []
        pipe._preprocess_single_tile = lambda tile, ch, illum_side=None: (
            calls.append(ch) or dense()
        )
        first = pipe._resolve_reg_channel([0, 1], tiles=_tiles())
        n = len(calls)
        again = pipe._resolve_reg_channel([0, 1], tiles=_tiles())
        assert again == first
        assert len(calls) == n, "re-probed a decision it had already made"


class TestItNeverFailsTheRun:
    def test_an_unmeasurable_channel_set_falls_back_to_the_first(self):
        pipe = _pipe()
        pipe._preprocess_single_tile = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("no such channel")
        )
        assert pipe._resolve_reg_channel([4, 5], tiles=_tiles()) == 4

    def test_no_channels_at_all_returns_the_configured_value(self):
        pipe = _pipe(reg_channel=2)
        assert pipe._resolve_reg_channel([]) == 2


class TestItSaysWhatItChose:
    def test_the_log_names_the_channel_and_every_score(self, caplog):
        pipe = _pipe()
        loaded = {0: [(sparse(), None)] * 2, 1: [(dense(), None)] * 2}
        with caplog.at_level(logging.INFO):
            pipe._resolve_reg_channel([0, 1], loaded=loaded)
        text = caplog.text
        assert "Registration channel: 1" in text
        assert "ch0=" in text and "ch1=" in text
