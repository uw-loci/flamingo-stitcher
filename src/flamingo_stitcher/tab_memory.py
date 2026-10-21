"""Which tab the standalone window opens on. Qt-free so it can be tested
without PyQt (see the GUI-tests-never-ran lesson: logic lives outside Qt).

Requested 2026-10-21: most runs are Single Workflow, yet it was the second
tab, and the window always reopened on Multi-Acquisition.
"""

from __future__ import annotations

from typing import Sequence

# Tab title -> the key saved for it.
TAB_KEYS = {
    "Single Workflow": "single",
    "Multi-Acquisition": "multi",
    "Multi-View": "multiview",
    "Options": "options",
    "Updates": "updates",
}
# Updates is left out: the app jumps there itself for "Update now", and
# reopening after an upgrade on the Updates tab is not where anyone works.
REMEMBERED_TABS = ("single", "multi", "multiview", "options")
KEY_LAST_TAB = "main_window/last_tab"


def initial_tab_index(tab_keys: Sequence[str], saved: str) -> int:
    """Index of the tab to open on: the saved one if it is still there and
    remembered, otherwise the first tab (Single Workflow)."""
    if saved in REMEMBERED_TABS and saved in tab_keys:
        return tab_keys.index(saved)
    return 0
