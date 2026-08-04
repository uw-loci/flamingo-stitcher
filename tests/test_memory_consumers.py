"""`top_memory_consumers` — makes a "waiting for memory" stall actionable.

The batch memory gate used to log the same "waiting…" line every 5 s with no
indication of WHAT was holding the RAM, so a user who wasn't watching the log
never learned that (e.g.) Fiji was sitting on 21 GB they could have freed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from flamingo_stitcher.memory_monitor import (  # noqa: E402
    format_memory_consumers,
    top_memory_consumers,
)


def test_returns_sorted_aggregated_rows():
    rows = top_memory_consumers(limit=5, min_gb=0.0)
    assert isinstance(rows, list)
    assert len(rows) <= 5
    for name, gb, count in rows:
        assert isinstance(name, str) and name
        assert gb >= 0
        assert count >= 1  # aggregated by name, so >=1 process per row
    # descending by size — the biggest hog must be first to be useful
    assert [r[1] for r in rows] == sorted([r[1] for r in rows], reverse=True)


def test_min_gb_filters():
    # An absurd floor must exclude everything rather than raise.
    assert top_memory_consumers(limit=5, min_gb=10_000.0) == []


def test_excludes_own_process():
    """Telling the user to close the stitcher itself would be nonsense.

    Other processes can share our name (many `python3`s on a dev box), so the
    invariant is the DIFFERENCE: dropping ourselves must remove ~our own RSS
    from that name's total, not that the total is below it.
    """
    import psutil

    me = psutil.Process()
    my_name = me.name()
    my_rss_gb = me.memory_info().rss / (1024**3)

    def _total(name, exclude_self):
        rows = top_memory_consumers(limit=500, min_gb=0.0, exclude_self=exclude_self)
        return next((gb for n, gb, _ in rows if n == name), 0.0)

    included = _total(my_name, exclude_self=False)
    excluded = _total(my_name, exclude_self=True)
    # Allow slack for concurrent processes shifting between the two scans.
    assert included - excluded >= my_rss_gb * 0.5


def test_format_is_readable():
    out = format_memory_consumers([("fiji.exe", 21.3, 1), ("chrome.exe", 12.0, 28)])
    assert "fiji.exe — 21.3 GB" in out
    assert "chrome.exe — 12.0 GB (28 processes)" in out
    assert format_memory_consumers([]) == ""
