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
