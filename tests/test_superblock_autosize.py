"""Auto-sizing of super-block fusion regions (resolve_superblock_chunks).

Pure logic (no heavy deps): given the output extent and mode, decide the
super-block region size that bounds streaming fuse-graph memory. The runtime
and the memory estimate call this same resolver so they never disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flamingo_stitcher.pipeline import resolve_superblock_chunks  # noqa: E402


class _Cfg:
    def __init__(self, chunks=0, target=4.0, cs=None):
        self.fusion_superblock_chunks = chunks
        self.fusion_superblock_target_gb = target
        self.output_chunksize = cs or {"z": 128, "y": 256, "x": 256}


# A large streaming output (the 623 GB job that ballooned to 127 GB) must
# auto-enable super-block; the region output should land near the target.
def test_large_streaming_autoenables():
    n = resolve_superblock_chunks(_Cfg(), 2118, 15017, 10526, use_streaming=True)
    assert n > 0
    region_gb = (n**3) * 128 * 256 * 256 * 2 / (1024**3)
    assert 1.0 <= region_gb <= 8.0  # ~target, bounded


def test_in_memory_never_superblocks():
    assert resolve_superblock_chunks(_Cfg(), 2118, 15017, 10526, False) == 0


def test_small_output_stays_whole():
    # Below 2x target: whole-output graph is already bounded.
    assert resolve_superblock_chunks(_Cfg(), 128, 256, 256, True) == 0


def test_explicit_override_wins():
    assert resolve_superblock_chunks(_Cfg(chunks=8), 2118, 15017, 10526, True) == 8
    # even in-memory / small, an explicit request is honoured
    assert resolve_superblock_chunks(_Cfg(chunks=3), 128, 256, 256, False) == 3


def test_target_zero_disables_auto():
    assert resolve_superblock_chunks(_Cfg(target=0), 2118, 15017, 10526, True) == 0


def test_bigger_target_gives_bigger_regions():
    small = resolve_superblock_chunks(_Cfg(target=2.0), 4000, 20000, 20000, True)
    big = resolve_superblock_chunks(_Cfg(target=16.0), 4000, 20000, 20000, True)
    assert big >= small > 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
