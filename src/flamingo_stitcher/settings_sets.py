"""Settings sets: which queued acquisitions share one set of settings.

From 0.14.0 every acquisition in a queue keeps its own settings, so a pending
one can be changed while another runs. Acquisitions whose settings are the same
share a letter, shown in the queue's 'Settings' column:

* The first acquisitions queued are Set A.
* A new acquisition from the same microscope and objective as an existing set
  joins it: the selected row's set when that matches, otherwise the most
  recently queued matching row's set. One from a different microscope or
  objective starts the next letter, from that microscope's own defaults.
* Editing every row of one set changes the set and keeps its letter.
* Editing some rows of a set (or rows from several sets) moves those rows into
  one set: an existing set with the same microscope and settings if there is
  one, otherwise a new letter.
* Two sets that end up the same merge into the earlier letter.
* Rows from different microscopes or objectives cannot be edited together:
  settings are tuned per microscope and objective.

Qt-free so the rules can be tested on their own. A row's settings are compared
through ``key``, a canonical string of the settings dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def canonical(settings: Dict[str, Any]) -> str:
    """A string that is equal exactly when two settings dicts are."""
    return json.dumps(settings, sort_keys=True, default=str)


@dataclass
class Row:
    """One queued acquisition as far as settings sets are concerned."""

    scope: str  # microscope and objective, e.g. "n7, 6.2x"
    key: str  # canonical(settings)
    letter: str = ""
    editable: bool = True  # False once it is running or finished
    extra: Dict[str, Any] = field(default_factory=dict)


def _order(letter: str) -> tuple:
    return (len(letter), letter)


def next_letter(used: Iterable[str]) -> str:
    """The first letter not in use: A, B, ... Z, then AA, AB, ..."""
    used = set(used)
    for letter in LETTERS:
        if letter not in used:
            return letter
    n = 0
    while True:
        letter = LETTERS[n // 26] + LETTERS[n % 26]
        if letter not in used:
            return letter
        n += 1


def place_new(
    rows: Sequence[Row], scope: str, default_key: str, selected_letter: Optional[str] = None
) -> tuple:
    """``(letter, key)`` for a newly added acquisition from ``scope``.

    ``default_key`` is used only when it starts a new set (the microscope's
    own defaults).
    """
    if selected_letter:
        for r in rows:
            if r.letter == selected_letter and r.scope == scope:
                return r.letter, r.key
    for r in reversed(list(rows)):
        if r.scope == scope and r.letter:
            return r.letter, r.key
    return next_letter(r.letter for r in rows), default_key


def merge(rows: Sequence[Row]) -> None:
    """Give sets with the same microscope and settings the earlier letter."""
    first: Dict[tuple, str] = {}
    for r in sorted(rows, key=lambda r: _order(r.letter)):
        first.setdefault((r.scope, r.key), r.letter)
    for r in rows:
        r.letter = first[(r.scope, r.key)]


def apply_edit(rows: Sequence[Row], selected: Sequence[int], new_key: str) -> None:
    """Rows ``selected`` now use ``new_key``; re-letter them by the rules above.

    The caller checks :func:`selection_state` first: every selected row is
    editable and from one microscope and objective.
    """
    sel = set(selected)
    if not sel:
        return
    letters = {rows[i].letter for i in sel}
    whole_set = len(letters) == 1 and all(
        i in sel for i, r in enumerate(rows) if r.letter in letters
    )
    scope = rows[next(iter(sel))].scope
    for i in sel:
        rows[i].key = new_key
    if not whole_set:
        same = [
            r.letter
            for i, r in enumerate(rows)
            if i not in sel and r.scope == scope and r.key == new_key
        ]
        letter = min(same, key=_order) if same else next_letter(
            r.letter for i, r in enumerate(rows) if i not in sel
        )
        for i in sel:
            rows[i].letter = letter
    merge(rows)


def selection_state(rows: Sequence[Row], selected: Sequence[int]) -> str:
    """'none', 'locked' (running or finished), 'mixed' (several microscopes or
    objectives) or 'editable'."""
    sel = [rows[i] for i in selected if 0 <= i < len(rows)]
    if not sel:
        return "none"
    if len({r.scope for r in sel}) > 1:
        return "mixed"
    if not all(r.editable for r in sel):
        return "locked"
    return "editable"


def _join(letters: List[str]) -> str:
    return letters[0] if len(letters) == 1 else ", ".join(letters[:-1]) + " and " + letters[-1]


def edit_note(rows: Sequence[Row], selected: Sequence[int]) -> str:
    """The line under the queue: what a change to the settings will do."""
    state = selection_state(rows, selected)
    sel = [rows[i] for i in selected if 0 <= i < len(rows)]
    if state == "none":
        return ""
    if state == "mixed":
        scopes = sorted({r.scope for r in sel})
        return (
            f"{len(sel)} acquisitions selected from different microscopes or objectives "
            f"({_join(scopes)}). Settings are tuned per microscope and objective, so they "
            "are grayed out. Select acquisitions from one microscope and objective to "
            "change them together."
        )
    if state == "locked":
        return ""  # the dialog says why (running or finished)
    letters = sorted({r.letter for r in sel}, key=_order)
    n_in_sets = sum(1 for r in rows if r.letter in letters)
    scope = sel[0].scope
    if len(sel) == 1 and n_in_sets == 1:
        return f"Editing Set {letters[0]} ({scope})."
    if len(letters) == 1 and len(sel) == n_in_sets:
        both = "both" if len(sel) == 2 else f"all {len(sel)}"
        return f"Editing Set {letters[0]}: {len(sel)} acquisitions ({scope}). A change applies to {both}."
    if len(letters) == 1:
        what = "it" if len(sel) == 1 else "them"
        return (
            f"Editing {len(sel)} of the {n_in_sets} acquisitions in Set {letters[0]} "
            f"({scope}). A change moves {what} into another set."
        )
    return (
        f"Editing {len(sel)} acquisitions from Sets {_join(letters)} ({scope}). A change "
        f"puts all {len(sel)} in one set."
    )


def summary(rows: Sequence[Row], describe: Optional[Callable[[str], str]] = None) -> str:
    """'2 settings sets: A = n7, 6.2x (6), B = liara, 20x (2)'.

    ``describe(letter)`` may add how a set differs from the first set of its
    microscope, e.g. 'at XY 8x'.
    """
    counts: Dict[str, int] = {}
    scope_of: Dict[str, str] = {}
    for r in rows:
        counts[r.letter] = counts.get(r.letter, 0) + 1
        scope_of[r.letter] = r.scope
    if not counts:
        return ""
    parts = []
    for letter in sorted(counts, key=_order):
        extra = describe(letter) if describe else ""
        parts.append(
            f"{letter} = {scope_of[letter]}" + (f" {extra}" if extra else "") + f" ({counts[letter]})"
        )
    n = len(parts)
    return f"{n} settings set{'s' if n != 1 else ''}: " + ", ".join(parts)
