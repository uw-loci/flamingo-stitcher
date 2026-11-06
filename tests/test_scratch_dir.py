"""Scratch (temp) directory redirection for streaming spill files.

When config.scratch_dir is set, the `.stitch_tmp` folder (per-tile spill memmaps
+ fused memmap) lives under it instead of next to the output — so temp I/O can
be moved off a slow/network output drive onto a fast local disk. Unset keeps the
default (`<output>/.stitch_tmp`).
"""

import tempfile
from pathlib import Path

from flamingo_stitcher.pipeline import (
    StitchingConfig,
    _same_volume,
    _scratch_base_dir,
)


def test_default_is_alongside_output():
    cfg = StitchingConfig()
    assert cfg.scratch_dir is None
    out = Path(tempfile.mkdtemp())
    assert _scratch_base_dir(cfg, out) == out  # -> <output>/.stitch_tmp


def test_scratch_dir_redirects_base():
    cfg = StitchingConfig()
    scratch = Path(tempfile.mkdtemp())
    cfg.scratch_dir = str(scratch)
    out = Path(tempfile.mkdtemp())
    assert _scratch_base_dir(cfg, out) == scratch


def test_same_volume_true_for_same_tree():
    d = Path(tempfile.mkdtemp())
    assert _same_volume(d, d) is True


def test_same_volume_resolves_nonexistent_child():
    # Works before the scratch dir is created (walks up to an existing ancestor).
    d = Path(tempfile.mkdtemp())
    child = d / "not" / "created" / "yet"
    assert _same_volume(child, d) is True


# --- One folder per run (0.14.0) ---------------------------------------------
# Before 0.14.0 every run used `.stitch_tmp` itself: two runs on one scratch drive
# wrote the same reg_chNN / fused.dat files and the first to finish deleted both.

from flamingo_stitcher.pipeline import (  # noqa: E402
    SCRATCH_FOLDER,
    StitchingPipeline,
    _other_scratch_folders,
    _remove_run_scratch,
)


def _pipe(scratch):
    cfg = StitchingConfig()
    cfg.scratch_dir = str(scratch)
    return StitchingPipeline(cfg, progress_fn=lambda p, m: None)


def test_two_runs_on_one_scratch_drive_get_separate_folders():
    scratch = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    a = _pipe(scratch)._run_scratch(out)
    b = _pipe(scratch)._run_scratch(out)
    assert a != b
    assert a.parent == b.parent == scratch / SCRATCH_FOLDER


def test_one_run_keeps_one_folder():
    scratch = Path(tempfile.mkdtemp())
    p = _pipe(scratch)
    out = Path(tempfile.mkdtemp())
    assert p._run_scratch(out) == p._run_scratch(out)


def test_cleanup_leaves_the_other_runs_folder():
    scratch = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    mine, theirs = _pipe(scratch)._run_scratch(out), _pipe(scratch)._run_scratch(out)
    for d in (mine, theirs):
        (d / "reg_ch03").mkdir(parents=True)
        (d / "fused.dat").write_bytes(b"\0" * 1024)
    _remove_run_scratch(mine)
    assert not mine.exists()
    assert (theirs / "fused.dat").exists()
    _remove_run_scratch(theirs)
    assert not (scratch / SCRATCH_FOLDER).exists()  # last one out removes .stitch_tmp


def test_other_folders_are_reported_with_their_size():
    scratch = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    mine, theirs = _pipe(scratch)._run_scratch(out), _pipe(scratch)._run_scratch(out)
    mine.mkdir(parents=True)
    theirs.mkdir(parents=True)
    (theirs / "fused.dat").write_bytes(b"\0" * 2**20)
    (folder, gb), = _other_scratch_folders(mine)
    assert folder == theirs
    assert abs(gb - 2**20 / 1024**3) < 1e-9
