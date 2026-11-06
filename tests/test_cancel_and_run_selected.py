"""Cancel stops the running acquisition; 'Run Selected' runs only the selected.

User, 2026-11-06: "if there is something in a queue after the currently
stitching, you can only cancel the pending, not the currently stitching".
Cancel did mark the pending rows at once, but the running stitch checked for it
only between tiles, and a full-size tile took about 17 minutes (run 37), most
of it merging the two light paths. The merge now checks every 16 planes.

Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/test_cancel_and_run_selected.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flamingo_stitcher.pipeline import (  # noqa: E402
    PipelineCancelled,
    fuse_illumination_sides,
)


def _sides(n=64):
    rng = np.random.default_rng(0)
    a = rng.integers(100, 4000, (n, 24, 24), dtype=np.uint16)
    b = rng.integers(100, 4000, (n, 24, 24), dtype=np.uint16)
    return {0: a, 1: b}


class _CancelAfter:
    """True from the ``n``-th call on: a cancel arriving part-way through."""

    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls >= self.n


@pytest.mark.parametrize("method", ["max", "mean", "content"])
def test_merge_stops_part_way_when_cancelled(method):
    cancel = _CancelAfter(2)
    with pytest.raises(PipelineCancelled):
        fuse_illumination_sides(_sides(), method=method, cancel_fn=cancel)
    assert cancel.calls == 2  # checked every 16 planes, stopped at the second check


@pytest.mark.parametrize("method", ["max", "mean", "content"])
def test_merge_is_unchanged_when_not_cancelled(method):
    v = _sides()
    plain = fuse_illumination_sides(v, method=method)
    checked = fuse_illumination_sides(v, method=method, cancel_fn=lambda: False)
    np.testing.assert_array_equal(plain, checked)


def test_cancel_during_preprocess_ends_the_run_and_removes_its_scratch(tmp_path):
    pytest.importorskip("multiview_stitcher")
    sys.path.insert(0, str(Path(__file__).parent))
    from _synth_acq import write_synth_acquisition

    from flamingo_stitcher.pipeline import (
        SCRATCH_FOLDER,
        StitchingConfig,
        StitchingPipeline,
    )

    acq = write_synth_acquisition(tmp_path / "acq", grid=(2, 2), n_planes=40,
                                  illum_sides=(0, 1), frame_size=(64, 64))
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.streaming_mode = True
    cfg.skip_registration = False
    cfg.output_format = "ome-tiff"
    cfg.resource_guard_enabled = False
    cfg.illumination_fusion = "max"
    cfg.scratch_dir = str(tmp_path / "scratch")
    cancel = _CancelAfter(3)
    pipe = StitchingPipeline(cfg, cancelled_fn=cancel, progress_fn=lambda p, m: None)
    out = pipe.run(acq, tmp_path / "out")
    assert out == tmp_path / "out"
    assert not list((tmp_path / "out").rglob("*.ome.tif"))  # stopped, nothing written
    assert not (tmp_path / "scratch" / SCRATCH_FOLDER).exists()


# --- 'Run Selected' ----------------------------------------------------------

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from flamingo_stitcher import run_lock  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp, monkeypatch, tmp_path):
    from flamingo_stitcher import scope_profiles
    from flamingo_stitcher.gui.stitching_dialog import NativeStitchingDialog

    monkeypatch.setattr(scope_profiles, "resolve_for_acquisition",
                        lambda p: ("n7", "6.2", {}, "test"))
    monkeypatch.setattr(scope_profiles, "seen_path", lambda: tmp_path / "seen.json")
    d = NativeStitchingDialog()
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
        d._add_path_to_queue(tmp_path / name)
    for it in d._queue:
        it["tiles"] = ["fake"]
    yield d
    run_lock.release(d)
    d.deleteLater()


def _select(d, *rows):
    d._queue_table.clearSelection()
    for r in rows:
        for c in range(d._queue_table.columnCount()):
            d._queue_table.item(r, c).setSelected(True)


def test_run_selected_button_follows_the_selection(dialog):
    d = dialog
    _select(d, 1)
    assert d._run_selected_btn.isEnabled()
    d._queue[1]["status"] = "done"
    d._update_queue_table()
    _select(d, 1)
    assert not d._run_selected_btn.isEnabled()


def test_a_selected_run_takes_only_the_selected_and_cancel_spares_the_rest(
        dialog, monkeypatch):
    d = dialog
    started = []
    monkeypatch.setattr(d, "_resolve_existing_output", lambda item, out: out)
    monkeypatch.setattr(d, "_launch_item_when_memory_ready",
                        lambda item, out, elapsed=0.0: started.append(item))
    d._output_dir_edit.setText(str(Path(d._queue[0]["path"]).parent / "out"))
    d._batch_only = {id(d._queue[1])}
    d._batch_running = True
    d._advance_queue()
    assert started == [d._queue[1]]
    d._worker = None
    monkeypatch.setattr(d, "_on_batch_complete", lambda: None)
    d._on_cancel()
    assert d._queue[0]["status"] == "pending"
    assert d._queue[2]["status"] == "pending"
    d._batch_running = False


# --- Queue time while a run is going -----------------------------------------
# User, 2026-11-06: "I think we only show the CURRENT". The progress line's
# 'overall' time is the running acquisition only; the queue total is added.


def test_progress_line_adds_the_acquisitions_still_waiting(dialog, monkeypatch):
    import flamingo_stitcher.pipeline as pl
    from flamingo_stitcher.timing_cache import StitchingTimingCache

    d = dialog
    monkeypatch.setattr(pl, "build_timing_key", lambda *a, **k: "k")
    monkeypatch.setattr(pl, "rough_run_seconds", lambda tiles, cfg: 3600.0)
    monkeypatch.setattr(StitchingTimingCache, "get_total_s", lambda self, key: None)
    d._batch_running = True
    d._queue_index = 0
    d._queue[0]["status"] = "stitching"
    d._on_eta_seconds(1800.0)
    html = d._queue_eta_html()
    assert "queue: 2 more after this one" in html
    assert "about 2:30:00 left" in html or "2:30:00" in html  # 0.5 h + 2 x 1 h
    d._batch_only = {id(d._queue[0]), id(d._queue[2])}  # 'Run Selected'
    assert "queue: 1 more after this one" in d._queue_eta_html()
    d._on_progress(40, "Preprocessing tile 3/24 <ch 3>")
    assert "queue: 1 more" in d._status_label.text()
    assert "&lt;ch 3&gt;" in d._status_label.text()  # plain part escaped
    d._batch_running = False
    d._batch_only = None
    assert d._queue_eta_html() == ""
