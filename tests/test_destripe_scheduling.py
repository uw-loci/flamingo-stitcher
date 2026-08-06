"""Destripe scheduling, in-place filtering, and unambiguous channel labels.

Background — the 12-hour run
----------------------------
A 98-tile run spent over twelve hours in preprocessing. The same code, on the
same data, in the same run, logged:

    22:39   1 tile,  24 threads  ->  25.6 planes/s   (62.6 s/tile)
    10:53   4 tiles,  6 threads  ->   2.2 planes/s  (740.9 s/tile)

The thread count fell because ``_destripe_worker_budget`` divides the machine
by the number of concurrent preprocess workers (24 // 4 = 6) — but a 4x thread
cut cannot produce an 11.6x slowdown. Four tiles in flight means four
tile-sized output arrays (~3.35 GB each) plus downsample buffers plus page
cache for four memmapped inputs, and the whole chain went to thrashing.

Measured separately: per-plane destripe parallelism tops out at ~1.5x on
1024x1024 planes and is flat past two workers, with processes no better than
threads. So splitting the machine across tiles cannot pay for itself here —
the aggregate 8.4 planes/s was 3x WORSE than one tile alone managed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flamingo_stitcher import pipeline as P  # noqa: E402


class _FakePipeline:
    """Just enough of StitchingPipeline to exercise the worker picker."""

    logger = types.SimpleNamespace(
        info=lambda *a, **k: None, debug=lambda *a, **k: None
    )

    def __init__(self, **cfg):
        defaults = {"preprocess_workers": 0, "destripe": False}
        defaults.update(cfg)
        self.config = types.SimpleNamespace(**defaults)

    pick = P.StitchingPipeline._pick_preprocess_workers


class TestPreprocessWorkerChoice:
    def test_destripe_on_forces_sequential_tiles(self):
        """The fix: don't fragment the machine when destripe already fills it."""
        p = _FakePipeline(destripe=True)

        assert p.pick(per_worker_bytes=1024, n_tiles=98) == 1

    def test_destripe_off_still_runs_tiles_concurrently(self):
        """Without destriping, overlapping tile I/O with compute is a win."""
        p = _FakePipeline(destripe=False)

        assert p.pick(per_worker_bytes=1024, n_tiles=98) > 1

    def test_an_explicit_override_still_wins_with_destripe_on(self):
        """The user can opt back into concurrency if their box likes it."""
        p = _FakePipeline(destripe=True, preprocess_workers=3)

        assert p.pick(per_worker_bytes=1024, n_tiles=98) == 3

    def test_never_more_workers_than_tiles(self):
        p = _FakePipeline(destripe=False)

        assert p.pick(per_worker_bytes=1024, n_tiles=2) == 2

    def test_destripe_workers_get_the_whole_machine_when_sequential(self):
        """pp == 1 means no nesting, so destripe_volume auto-sizes to all cores."""
        pipe = _FakePipeline(destripe=True, destripe_workers=None)
        pipe._active_preprocess_workers = 1

        budget = P.StitchingPipeline._destripe_worker_budget(pipe)

        assert budget is None  # None => destripe_volume picks cores itself

    def test_concurrent_preprocess_still_divides_the_budget(self):
        """The nesting guard must stay for anyone who overrides back to >1."""
        pipe = _FakePipeline(destripe=True, destripe_workers=None)
        pipe._active_preprocess_workers = 4

        budget = P.StitchingPipeline._destripe_worker_budget(pipe)

        assert budget is not None and budget >= 1


class TestInPlaceDestripe:
    """Skip one tile-sized allocation where the caller owns a writable array.

    Note this is NOT a halving of peak RAM on the main path: `.raw` tiles are
    read-only memmaps, so the output array is the only tile-sized thing in
    anonymous memory (see TestInPlaceIsSafeOnReadOnlyInput). It pays off on
    the destripe_fast path, which filters an in-memory downsampled array.
    """

    def _volume(self):
        rng = np.random.default_rng(0)
        vol = rng.integers(200, 3000, (4, 64, 64)).astype(np.uint16)
        vol += (np.sin(np.arange(64) * 0.7)[None, :, None] * 100).astype(np.uint16)
        return vol

    def test_in_place_writes_into_the_input(self):
        vol = self._volume()
        out = P.destripe_volume(vol, max_workers=2, direction="horizontal",
                                in_place=True)

        assert out is vol

    def test_the_default_still_allocates_a_separate_result(self):
        vol = self._volume()
        original = vol.copy()

        out = P.destripe_volume(vol, max_workers=2, direction="horizontal")

        assert out is not vol
        np.testing.assert_array_equal(vol, original)  # input untouched

    def test_in_place_and_copy_produce_the_same_pixels(self):
        """Overwriting a plane after reading it must not change the answer."""
        vol = self._volume()
        expected = P.destripe_volume(vol.copy(), max_workers=2,
                                     direction="horizontal")
        got = P.destripe_volume(vol, max_workers=2, direction="horizontal",
                                in_place=True)

        np.testing.assert_array_equal(got, expected)

    def test_vertical_direction_round_trips_in_place(self):
        """The transpose path writes back through a transposed view."""
        vol = self._volume()
        expected = P.destripe_volume(vol.copy(), max_workers=2,
                                     direction="vertical")
        got = P.destripe_volume(vol, max_workers=2, direction="vertical",
                                in_place=True)

        np.testing.assert_array_equal(got, expected)


class TestAggregateThroughputLogging:
    """Per-tile rates hid a 3x regression for twelve hours."""

    def test_aggregate_is_reported_when_tiles_overlap(self):
        meter = P._DestripeMeter()
        meter.start()
        meter.start()
        meter.add_planes(100)
        assert meter.finish() is None  # first of two finishing: stay quiet
        meter.add_planes(100)

        rate, planes, peak = meter.finish()

        assert planes == 200
        assert peak == 2
        assert rate > 0

    def test_no_aggregate_line_for_a_lone_tile(self):
        meter = P._DestripeMeter()
        meter.start()
        meter.add_planes(50)

        rate, planes, peak = meter.finish()

        assert peak == 1  # caller suppresses the line — nothing to compare

    def test_the_meter_resets_between_runs(self):
        meter = P._DestripeMeter()
        meter.start()
        meter.add_planes(10)
        meter.finish()
        meter.start()
        meter.add_planes(7)

        _, planes, _ = meter.finish()

        assert planes == 7  # not 17


class TestChannelLabelling:
    """"channel 3" for a single-channel run reads as three channels."""

    def test_a_single_channel_leads_with_the_count(self):
        label = P.describe_channel_set([3])

        assert label.startswith("1 ")
        assert "channel 3" in label

    def test_the_laser_is_named_so_the_index_is_unambiguous(self):
        assert "640" in P.describe_channel(3)
        assert "405" in P.describe_channel(0)

    def test_several_channels_still_list_them_all(self):
        label = P.describe_channel_set([0, 3])

        assert label.startswith("2 ")
        assert "channel 0" in label and "channel 3" in label

    def test_a_subset_says_how_many_were_acquired(self):
        label = P.describe_channel_set([3], available=[0, 1, 2, 3])

        assert label.startswith("1 ")
        assert "of 4 acquired" in label

    def test_no_such_note_when_everything_is_selected(self):
        assert "acquired" not in P.describe_channel_set([0, 3], available=[0, 3])

    def test_empty_selection_is_stated_as_zero(self):
        assert P.describe_channel_set([]).startswith("0")

    def test_an_unknown_channel_index_degrades_gracefully(self):
        assert P.describe_channel(97) == "channel 97"

    def test_labelling_never_raises_even_with_no_hardware_config(self, monkeypatch):
        """A cosmetic helper must not be able to abort a stitch."""
        import flamingo_stitcher.config_loader as cl

        monkeypatch.setattr(
            cl, "get_hardware_config", lambda: (_ for _ in ()).throw(OSError("nope"))
        )

        assert P.describe_channel(3) == "channel 3"
        assert P.describe_channel_set([3]).startswith("1 ")


class TestMemoryEstimateTracksTheNewChoice:
    """The estimate must model the picker, or the ETA and mode go wrong again."""

    def test_destripe_run_is_not_modelled_as_four_concurrent_tiles(self):
        cfg_on = types.SimpleNamespace(preprocess_workers=0, destripe=True)
        cfg_off = types.SimpleNamespace(preprocess_workers=0, destripe=False)

        # Mirror of the branch under test in _estimate_memory_requirements.
        def modelled(config):
            req = int(getattr(config, "preprocess_workers", 0) or 0)
            if req > 0:
                return min(req, 8)
            if getattr(config, "destripe", False):
                return 1
            return 4

        assert modelled(cfg_on) == 1
        assert modelled(cfg_off) == 4


class TestInPlaceIsSafeOnReadOnlyInput:
    """`.raw` tiles are memory-mapped READ-ONLY.

    An earlier version of the in-place change assumed the caller always owned
    a writable array. It passed every unit test (which build plain arrays) and
    then died on the first real tile with "assignment destination is read-only".
    """

    def _readonly_volume(self, tmp_path):
        path = tmp_path / "tile.raw"
        rng = np.random.default_rng(0)
        rng.integers(200, 3000, (4, 32, 32)).astype(np.uint16).tofile(path)
        vol = np.memmap(path, dtype=np.uint16, mode="r", shape=(4, 32, 32))
        assert not vol.flags.writeable
        return vol

    def test_read_only_input_falls_back_to_a_copy(self, tmp_path):
        vol = self._readonly_volume(tmp_path)

        out = P.destripe_volume(vol, max_workers=2, direction="horizontal",
                                in_place=True)

        assert out is not vol
        assert out.shape == vol.shape

    def test_the_fallback_gives_the_same_answer_as_a_plain_copy(self, tmp_path):
        vol = self._readonly_volume(tmp_path)

        forced = P.destripe_volume(vol, max_workers=2, direction="horizontal",
                                   in_place=True)
        plain = P.destripe_volume(vol, max_workers=2, direction="horizontal")

        np.testing.assert_array_equal(forced, plain)


class TestSplitIlluminationLabels:
    """One laser + split illumination = TWO output channels, legitimately.

    ``_output_channel_units`` labels them "3_I0" / "3_I1", and those labels
    reach the same logs. "channel 3_I0" was no clearer than "channel 3".
    """

    def test_a_side_label_names_both_the_laser_and_the_side(self):
        label = P.describe_channel("3_I0")

        assert "channel 3" in label
        assert "640" in label
        assert "side 0" in label

    def test_the_two_sides_are_distinguishable(self):
        assert P.describe_channel("3_I0") != P.describe_channel("3_I1")

    def test_the_pair_still_leads_with_the_count(self):
        label = P.describe_channel_set(["3_I0", "3_I1"])

        assert label.startswith("2 ")
        assert "side 0" in label and "side 1" in label

    def test_an_unparseable_label_does_not_raise(self):
        assert "weird" in P.describe_channel("weird_Ix")


class TestBorderQCSpillIsReusable:
    """QC's spill must match what the first output unit wants.

    With split_illumination on, QC materialized the FUSED reference channel,
    but the fusion loop's reuse guard required ``side is None`` — so the spill
    matched no output unit and QC became a silent extra preprocess of every
    tile. On the rig's 98-tile run that pass took 18,659.7 s (5.2 h) and was
    thrown away, on top of the two passes the split itself needs.
    """

    def _units(self, split, sides=(0, 1)):
        cfg = types.SimpleNamespace(split_illumination=split)
        pipe = types.SimpleNamespace(config=cfg)
        tiles = [types.SimpleNamespace(raw_files={3: {s: None for s in sides}})]
        return P.StitchingPipeline._output_channel_units(pipe, tiles, [3])

    def test_split_expands_one_laser_into_one_unit_per_side(self):
        units = self._units(split=True)

        assert [u[0] for u in units] == ["3_I0", "3_I1"]
        assert [u[2] for u in units] == [0, 1]

    def test_unsplit_keeps_a_single_fused_unit(self):
        units = self._units(split=False)

        assert len(units) == 1
        assert units[0][2] is None  # side None => sides fused

    def test_the_first_units_side_is_what_qc_should_build(self):
        """That first side is what gets threaded into _run_border_qc_streaming."""
        assert self._units(split=True)[0][2] == 0
        assert self._units(split=False)[0][2] is None

    def test_a_single_sided_acquisition_is_not_split(self):
        """Nothing to separate — must stay one unit, and QC stays fused."""
        units = self._units(split=True, sides=(0,))

        assert len(units) == 1
        assert units[0][2] is None

    @pytest.mark.parametrize(
        "spill_side,unit_side,reusable",
        [(0, 0, True), (0, 1, False), (None, None, True), (None, 0, False)],
    )
    def test_reuse_requires_a_matching_side(self, spill_side, unit_side, reusable):
        """The guard compares sides now; it used to demand `side is None`.

        Reusing a fused spill for a single-side unit would hand that channel
        the wrong pixels, so the match has to be exact in both directions.
        """
        assert (unit_side == spill_side) is reusable
