"""A failing isolated task says why: the worker reports on stdout, not stderr.

Run: python3 -m pytest tests/test_isolated_worker_errors.py -q
"""

import json

from flamingo_stitcher.isolated_service import _worker_failure_message


def test_the_workers_own_report_is_the_reason():
    out = json.dumps({"status": "error", "message": "need at least 2 images", "type": "ValueError"})
    msg = _worker_failure_message("flatfield", 1, out, "")
    assert msg == (
        "Isolated worker failed (exit 1) on task 'flatfield': ValueError: need at least 2 images"
    )


def test_a_crash_before_the_handler_keeps_stderr_and_stray_stdout():
    msg = _worker_failure_message("flatfield", 3, "partial", "Traceback: ImportError basicpy")
    assert "stdout: partial" in msg and "stderr: Traceback: ImportError basicpy" in msg


def test_no_output_at_all_is_said_plainly():
    assert _worker_failure_message("x", 1, "", "").endswith("the worker gave no reason (no output at all)")
