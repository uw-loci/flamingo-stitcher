"""_fmt_gb: adaptive MB/GB/TB size display.

Regression for the "0 GB output" bug — a 0.4 GB stitched output was printed as
``~0 GB`` because the estimate readout used a fixed ``{:.0f} GB``. The formatter
must scale the unit so sub-GB sizes show MB (and huge ones TB).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from flamingo_stitcher.gui.stitching_dialog import _fmt_gb  # noqa: E402


class TestFmtGb:
    def test_sub_gb_shows_mb_not_zero(self):
        # The exact bug: ~0.4 GB output must not collapse to "0 GB".
        assert _fmt_gb(0.39) == "399 MB"
        assert _fmt_gb(0.6) == "614 MB"

    def test_gb_range(self):
        assert _fmt_gb(1.0) == "1.0 GB"
        assert _fmt_gb(1.44) == "1.4 GB"
        assert _fmt_gb(84.0) == "84.0 GB"

    def test_tb_range(self):
        assert _fmt_gb(1024.0) == "1.0 TB"
        assert _fmt_gb(6739.0) == "6.6 TB"

    def test_tiny_and_zero(self):
        assert _fmt_gb(0.0) == "0 MB"
        assert _fmt_gb(0.0004) == "419 KB"  # ~0.4 MB → KB

    def test_bad_input_is_safe(self):
        assert _fmt_gb(None) == "? GB"
        assert _fmt_gb(-5.0) == "0 MB"
