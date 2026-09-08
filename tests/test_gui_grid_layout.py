"""No two widgets may be placed in the same grid cell.

v0.12.2 shipped "Fusion region size" into QGridLayout row 6 of Processing
Options — a row already fully occupied by the border-QC controls. The label and
combo were drawn on top of the QC row, and the QC widgets took the clicks, so
the new combo was unusable.

Nothing caught it. The tests asserted the widget existed, that the config
plumbing read it, and that QSettings persisted it — all true, and all blind to
WHERE it was put. PyQt5 is not installed in the dev environment either, so it
was never rendered before release.

This reads the placements straight out of the source, so it needs no Qt and
fails on the collision rather than on a screenshot someone happens to look at.

Run: python -m pytest tests/test_gui_grid_layout.py -q
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "flamingo_stitcher" / "gui"
FILES = sorted(SRC.glob("*.py"))


def _layout_key(node):
    """`grid`, or `self._grid` — anything a placement can be called on."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _rebindings(tree):
    """{layout key: [lineno, ...]} where the name is (re)assigned.

    A grid variable is routinely reused for the next group — stitching_dialog
    rebinds `settings_layout` for its second QGridLayout — so placements have to
    be split at each rebinding or two unrelated grids look like one and every
    row reads as a collision.
    """
    out = defaultdict(list)
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            key = _layout_key(target)
            if key:
                out[key].append(node.lineno)
    return {k: sorted(v) for k, v in out.items()}


def _own_nodes(scope):
    """Every node in `scope`, not descending into nested functions.

    A nested def is its own rebuild path; counting its placements in the
    enclosing scope too would invent collisions.
    """
    for node in ast.iter_child_nodes(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield node
        yield from _own_nodes(node)


def _scopes(tree):
    """Every function body, plus module level, as separate placement scopes.

    Collisions are only meaningful WITHIN one function. A grid populated from
    two methods is a rebuild — stitching_dialog's channel panel has one method
    render a placeholder and another render the rows, each calling takeAt()
    first — and no static check can tell that from an accumulation. The
    Processing Options collision this file exists for was entirely inside one
    method, so scoping per function still catches that class of bug.
    """
    yield "<module>", tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


def _placements(tree, source, scope=None, scope_name="<module>"):
    """{(layout key, generation): [(row, col, rowspan, colspan, lineno, text)]}.

    Only literal integer positions are read. A computed row cannot be checked
    statically, and guessing at one would produce false failures.
    """
    rebound = _rebindings(tree)
    found = defaultdict(list)
    nodes = _own_nodes(scope) if scope is not None else ast.walk(tree)
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in ("addWidget", "addLayout"):
            continue
        key = _layout_key(func.value)
        if key is None:
            continue
        args = node.args[1:]  # first arg is the widget/layout
        if len(args) < 2:
            continue  # box layout, not a grid
        nums = []
        for arg in args[:4]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                nums.append(arg.value)
            else:
                nums = []
                break
        if len(nums) < 2:
            continue
        row, col = nums[0], nums[1]
        rowspan = nums[2] if len(nums) > 2 else 1
        colspan = nums[3] if len(nums) > 3 else 1
        text = ast.get_source_segment(source, node) or ""
        generation = sum(
            1 for line in rebound.get(key, []) if line <= node.lineno
        )
        found[(key, generation)].append(
            (row, col, rowspan, colspan, node.lineno, text.split("\n")[0])
        )
    return found


def _collisions(placements):
    owner = {}
    clashes = []
    for row, col, rowspan, colspan, lineno, text in placements:
        for r in range(row, row + max(1, rowspan)):
            for c in range(col, col + max(1, colspan)):
                if (r, c) in owner:
                    clashes.append(((r, c), owner[(r, c)], (lineno, text)))
                else:
                    owner[(r, c)] = (lineno, text)
    return clashes


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_grid_cell_is_claimed_twice(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for scope_name, scope in _scopes(tree):
        placed = _placements(tree, source, scope=scope, scope_name=scope_name)
        for (layout, _gen), placements in placed.items():
            clashes = _collisions(placements)
            assert not clashes, "\n".join(
                f"{path.name}: {scope_name}() {layout} cell {cell} claimed by "
                f"line {a[0]} ({a[1]}) and line {b[0]} ({b[1]})"
                for cell, a, b in clashes
            )


class TestTheDetectorItself:
    """A checker that cannot fail is worse than no checker."""

    def test_it_catches_an_exact_overlap(self):
        source = "g = QGridLayout()\ng.addWidget(a, 6, 0)\ng.addWidget(b, 6, 0)\n"
        found = _placements(ast.parse(source), source)
        assert _collisions(found[("g", 1)])

    def test_it_catches_a_span_running_over_a_neighbour(self):
        """The real bug: a 1x4 span landing on a row already holding four."""
        source = "g = QGridLayout()\ng.addWidget(a, 6, 1, 1, 3)\ng.addWidget(b, 6, 3)\n"
        found = _placements(ast.parse(source), source)
        assert _collisions(found[("g", 1)])

    def test_it_allows_adjacent_cells(self):
        source = "g = QGridLayout()\ng.addWidget(a, 5, 0)\ng.addWidget(b, 6, 0)\ng.addWidget(c, 5, 1)\n"
        found = _placements(ast.parse(source), source)
        assert not _collisions(found[("g", 1)])

    def test_it_allows_a_multi_row_span_beside_others(self):
        source = "g = QGridLayout()\ng.addWidget(a, 2, 0, 3, 4)\ng.addWidget(b, 5, 0)\n"
        found = _placements(ast.parse(source), source)
        assert not _collisions(found[("g", 1)])

    def test_a_rebound_name_is_a_separate_grid(self):
        """stitching_dialog reuses `settings_layout` for its second group; the
        two share a name and nothing else."""
        source = (
            "g = QGridLayout()\n"
            "g.addWidget(a, 1, 0)\n"
            "g = QGridLayout()\n"
            "g.addWidget(b, 1, 0)\n"
        )
        found = _placements(ast.parse(source), source)
        assert len(found) == 2
        for placements in found.values():
            assert not _collisions(placements)

    def test_it_ignores_box_layouts(self):
        source = "v.addWidget(a)\nv.addWidget(b)\n"
        found = _placements(ast.parse(source), source)
        assert found == {}
