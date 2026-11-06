"""Settings sets: the letters in the queue's 'Settings' column (0.14.0).

Rules agreed with the user 2026-11-05: first queued = A; an acquisition from a
second microscope (or objective) starts B automatically; editing part of a set
splits those rows into a new letter; identical sets merge into the earlier
letter; rows from different microscopes cannot be edited together.
"""

from flamingo_stitcher.settings_sets import (
    Row,
    apply_edit,
    canonical,
    edit_note,
    next_letter,
    place_new,
    selection_state,
    summary,
)

N7 = "n7, 6.2x"
LIARA = "liara, 20x"
DEFAULT = canonical({"downsample_xy": 4})
X8 = canonical({"downsample_xy": 8})


def _queue(*scopes):
    """Add acquisitions one by one, nothing selected, as the dialog would."""
    rows = []
    for scope in scopes:
        letter, key = place_new(rows, scope, DEFAULT)
        rows.append(Row(scope, key, letter))
    return rows


def test_first_microscope_is_a_second_is_b():
    rows = _queue(N7, N7, LIARA, N7, LIARA)
    assert [r.letter for r in rows] == ["A", "A", "B", "A", "B"]


def test_new_acquisition_joins_the_selected_set_when_it_matches():
    rows = _queue(N7, N7)
    apply_edit(rows, [1], X8)  # row 1 split off into B
    assert [r.letter for r in rows] == ["A", "B"]
    letter, key = place_new(rows, N7, DEFAULT, selected_letter="A")
    assert (letter, key) == ("A", DEFAULT)
    # nothing selected: the most recently queued matching row's set
    assert place_new(rows, N7, DEFAULT) == ("B", X8)
    # a selected set from another microscope does not capture it
    assert place_new(rows, LIARA, DEFAULT, selected_letter="A")[0] == "C"


def test_editing_a_whole_set_keeps_its_letter():
    rows = _queue(N7, N7, LIARA)
    apply_edit(rows, [0, 1], X8)
    assert [r.letter for r in rows] == ["A", "A", "B"]
    assert rows[0].key == rows[1].key == X8


def test_editing_part_of_a_set_splits_it():
    rows = _queue(N7, N7, N7, N7, LIARA)
    apply_edit(rows, [2, 3], X8)
    assert [r.letter for r in rows] == ["A", "A", "C", "C", "B"]


def test_split_rows_join_an_existing_set_with_the_same_settings():
    rows = _queue(N7, N7, N7)
    apply_edit(rows, [2], X8)  # the first free letter is B
    assert [r.letter for r in rows] == ["A", "A", "B"]
    apply_edit(rows, [1], X8)  # same scope and settings as B: joins B
    assert [r.letter for r in rows] == ["A", "B", "B"]


def test_sets_that_become_identical_merge_into_the_earlier_letter():
    rows = _queue(N7, N7)
    apply_edit(rows, [1], X8)
    assert [r.letter for r in rows] == ["A", "B"]
    apply_edit(rows, [0], X8)  # whole Set A edited to match B
    assert [r.letter for r in rows] == ["A", "A"]


def test_same_settings_on_two_microscopes_stay_two_sets():
    rows = _queue(N7, LIARA)
    assert rows[0].key == rows[1].key
    assert [r.letter for r in rows] == ["A", "B"]


def test_selection_states():
    rows = _queue(N7, N7, LIARA)
    assert selection_state(rows, []) == "none"
    assert selection_state(rows, [0, 1]) == "editable"
    assert selection_state(rows, [1, 2]) == "mixed"
    rows[0].editable = False  # running
    assert selection_state(rows, [0]) == "locked"
    assert selection_state(rows, [0, 2]) == "mixed"  # mixed wins: name the reason


def test_edit_notes_say_what_a_change_will_do():
    rows = _queue(N7, N7, N7, LIARA)
    assert edit_note(rows, [0, 1, 2]) == "Editing Set A: 3 acquisitions (n7, 6.2x). A change applies to all 3."
    assert "moves them into another set" in edit_note(rows, [1, 2])
    assert "moves it into another set" in edit_note(rows, [2])
    assert edit_note(rows, [3]) == "Editing Set B (liara, 20x)."
    assert "different microscopes or objectives (liara, 20x and n7, 6.2x)" in edit_note(rows, [2, 3])
    apply_edit(rows, [2], X8)
    assert "from Sets A and C" in edit_note(rows, [1, 2])


def test_summary_and_letters():
    rows = _queue(N7, N7, LIARA)
    apply_edit(rows, [1], X8)
    assert summary(rows) == "3 settings sets: A = n7, 6.2x (1), B = liara, 20x (1), C = n7, 6.2x (1)"
    assert summary(rows, lambda letter: "at XY 8x" if letter == "C" else "").endswith(
        "C = n7, 6.2x at XY 8x (1)"
    )
    assert next_letter("ABC") == "D"
    assert next_letter("ACD") == "B"
    assert next_letter(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) == "AA"
