"""The standalone window opens on Single Workflow, then on the tab used last.

Requested 2026-10-21: most runs are Single Workflow, yet it was the second
tab, and the window always reopened on Multi-Acquisition.
"""

from __future__ import annotations

import re
from pathlib import Path

from flamingo_stitcher.tab_memory import TAB_KEYS, initial_tab_index

KEYS = ["single", "multi", "multiview", "options", "updates"]


def test_single_workflow_is_the_first_tab():
    src = (Path(__file__).parents[1] / "src/flamingo_stitcher/gui/app.py").read_text("utf-8")
    titles = re.findall(r'tabs\.addTab\([^,]+,\s*"([^"]+)"\)', src)
    assert titles[0] == "Single Workflow"
    assert set(titles) == set(TAB_KEYS)          # every tab has a saved key


def test_first_launch_opens_on_the_first_tab():
    assert initial_tab_index(KEYS, "") == 0


def test_it_reopens_where_it_was_left():
    assert initial_tab_index(KEYS, "multiview") == 2
    assert initial_tab_index(KEYS, "options") == 3


def test_updates_and_unknown_keys_fall_back_to_the_first_tab():
    assert initial_tab_index(KEYS, "updates") == 0
    assert initial_tab_index(KEYS, "a tab that no longer exists") == 0


def test_orientations_read_as_words_in_the_ui_and_logs():
    """Requested 2026-10-22: the Orientation Preview showed 'flip_h' and the like."""
    from flamingo_stitcher.orientation import MosaicOrientation, describe_orientation, orientation_label

    assert set(MosaicOrientation.LABELS) == set(MosaicOrientation.NAMES)
    assert orientation_label("flip_h") == "Mirror left–right"
    assert orientation_label("") == "As captured (no change)"
    assert describe_orientation("identity", True, False) == (
        "As captured (no change) (identity), tile order reversed in X"
    )
    assert describe_orientation("rot270", with_name=False) == "Rotate 90° clockwise"
