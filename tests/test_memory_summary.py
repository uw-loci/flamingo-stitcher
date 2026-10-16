"""The end-of-run memory summary: projections get checked against the machine.

2026-10-16: a full-resolution Imaris run was refused before it started, by a
writer allowance (25% of the output) that had never been measured. The
watchdog already sampled every phase's peak and threw it away at the end.
"""

from __future__ import annotations

import logging

from flamingo_stitcher.pipeline import (
    IMARIS_WRITER_RAM_FRACTION_ASSUMED,
    StitchingConfig,
    StitchingPipeline,
)

GIB = 1024**3


class _Monitor:
    peak_mapped_bytes = 3 * GIB

    def __init__(self, phases):
        self._phases = phases

    def phase_peaks_delta(self):
        return self._phases

    @property
    def peak_delta_bytes(self):
        return max(self._phases.values(), default=0)


def _summary(fmt, phases, caplog, output_gb=100.0):
    cfg = StitchingConfig()
    cfg.output_format = fmt
    pipe = StitchingPipeline(cfg)
    pipe._watchdog_projection = {
        "projected_gb": 58.6, "output_gb": output_gb, "mode": "streaming",
    }
    with caplog.at_level(logging.INFO):
        pipe._log_memory_summary(_Monitor(phases))
    return [r.getMessage() for r in caplog.records]


def test_every_phase_peak_is_logged_against_the_projection(caplog):
    lines = _summary(
        "ome-tiff", {"write": 2 * GIB, "preprocess": 40 * GIB, "fuse": 50 * GIB}, caplog
    )
    line = next(l for l in lines if l.startswith("Memory used"))
    # In pipeline order, not dict order.
    assert line.index("preprocess 40.0") < line.index("fuse 50.0") < line.index("write 2.0")
    assert "run peak 50.0 vs projected 58.6 (streaming)" in line
    assert not any("Imaris writer, measured" in l for l in lines)


def test_an_imaris_run_reports_the_measured_writer_fraction(caplog):
    lines = _summary("imaris", {"fuse": 50 * GIB, "write": 12 * GIB}, caplog)
    line = next(l for l in lines if l.startswith("Imaris writer, measured"))
    assert "12.0 GB for a 100.0 GB output = 12.0%" in line
    assert f"assumes {IMARIS_WRITER_RAM_FRACTION_ASSUMED:.0%}" in line


def test_no_samples_means_no_summary(caplog):
    assert not _summary("imaris", {}, caplog)
