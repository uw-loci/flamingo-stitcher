"""Which folder "Open Output Folder" should open.

The reported bug: it opened the parent directory holding every stitch ever
written there, leaving the user to find the folder they had just made among a
list of them.

`StitchingPipeline.run` returns the output DIRECTORY, and the worker emits
exactly that. The dialog took `.parent` of it -- correct if the report were the
store, one level too high for a directory. `tests/test_open_output_folder_after_rename.py`
fed it a store path in every case, so it agreed with the code and missed this;
it also needs PyQt5, which is installed neither here nor in the release
workflow, so it has never actually run. Hence this file: same rule, no Qt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flamingo_stitcher.pipeline import run_output_folder


OUT = "G:/ScratchTest"
RUN = f"{OUT}/20260903_noelleLargeTest1_stitched"


def test_a_reported_directory_is_already_the_answer():
    """What the worker actually emits. This is the bug."""
    assert run_output_folder(RUN) == Path(RUN)


def test_it_does_not_climb_to_the_parent_of_the_run():
    assert run_output_folder(RUN) != Path(OUT)


@pytest.mark.parametrize(
    "store",
    [
        "stitched.ims",
        "stitched.ome.zarr",
        "stitched.zarr",
        "stitched.ozx",
        "stitched.ome.tif",
        "stitched.tiff",
        "stitched.btf",
    ],
)
def test_a_reported_store_resolves_to_the_folder_holding_it(store):
    assert run_output_folder(f"{RUN}/{store}") == Path(RUN)


def test_case_does_not_matter():
    assert run_output_folder(f"{RUN}/STITCHED.IMS") == Path(RUN)


def test_a_uniquified_folder_is_preserved():
    """A run answered "New folder" writes to a sibling; that is the answer."""
    renamed = f"{OUT}/MyDataset_stitched_2"
    assert run_output_folder(renamed) == Path(renamed)
    assert run_output_folder(f"{renamed}/MyDataset.ome.zarr") == Path(renamed)


def test_it_does_not_touch_the_disk():
    """A moved or deleted result must still resolve, so the rule cannot be
    "is this a directory?"."""
    gone = "/nowhere/at/all/acq_stitched"
    assert not Path(gone).exists()
    assert run_output_folder(gone) == Path(gone)
    assert run_output_folder(gone + "/x.ims") == Path(gone)


def test_a_folder_whose_name_contains_a_dot_is_not_mistaken_for_a_store():
    dotted = f"{OUT}/2026-09-03_sample_1.5x_stitched"
    assert run_output_folder(dotted) == Path(dotted)


def test_the_dialog_delegates_rather_than_reimplementing():
    """Two copies of this rule is how they drift apart."""
    import ast

    src = Path(__file__).resolve().parents[1] / (
        "src/flamingo_stitcher/gui/stitching_dialog.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_run_output_folder"
    )
    body = ast.dump(fn)
    assert "run_output_folder" in body
    assert "parent" not in body, "the dialog must not re-derive the rule"


def test_the_summary_and_the_completion_log_agree():
    """The log said "Completed: ScratchTest" -- the parent's name -- on every
    run, for the same reason the button opened the parent."""
    import ast

    src = Path(__file__).resolve().parents[1] / (
        "src/flamingo_stitcher/gui/stitching_dialog.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "Path(output_path).parent.name" not in text
    assert "self._run_output_folder(output_path).name" in text
