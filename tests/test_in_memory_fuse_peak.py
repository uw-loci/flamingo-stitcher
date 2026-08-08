"""The in-memory fuse path must route its uint16 conversion through the graph.

Companion to ``test_fused_uint16_conversion.py``: that file pins the
``lazy_uint16`` helper in isolation (and measures that it really does save a
full copy). This one pins that the in-memory pipeline actually USES it, which is
the half that regressed — the streaming and preview paths converted lazily all
along, only the in-memory path did::

    vol = np.asarray(fused_sim.data.compute())
    vol = np.clip(vol, 0, 65535).astype(np.uint16)

holding the computed array, the clip temporary and the astype result at once.

Why a call-site probe rather than an end-to-end peak measurement: at any dataset
size small enough for the suite, preprocessing dominates the peak and the extra
fused copy hides inside it (measured: peak/(tiles+output) is 2.57 at 4 MB output
and 2.23 at 15 MB, so no fixed ratio discriminates). The dataset would have to be
minutes-long before the fused volume dominated. Patching the seam is exact and
takes seconds.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent / "src") not in sys.path:
    sys.path.insert(0, str(_HERE.parent / "src"))

pytest.importorskip("multiview_stitcher")

from _synth_acq import write_synth_acquisition  # noqa: E402

from flamingo_stitcher import pipeline as pipeline_mod  # noqa: E402
from flamingo_stitcher.pipeline import (  # noqa: E402
    StitchingConfig,
    StitchingPipeline,
)


def _tiny_config() -> StitchingConfig:
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.skip_registration = True
    cfg.streaming_mode = False  # <- the path under test
    cfg.output_format = "ome-tiff"
    cfg.resource_guard_enabled = False
    cfg.output_chunksize = {"z": 4, "y": 16, "x": 16}
    return cfg


def test_in_memory_path_converts_inside_the_dask_graph(monkeypatch):
    calls = []
    real = pipeline_mod.lazy_uint16

    def spy(darr):
        out = real(darr)
        calls.append((darr.dtype, out.dtype, out.ndim))
        return out

    monkeypatch.setattr(pipeline_mod, "lazy_uint16", spy)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        acq = write_synth_acquisition(
            d / "acq", grid=(2, 2), n_planes=8, frame_size=(32, 32), overlap=0.15
        )
        StitchingPipeline(_tiny_config(), progress_fn=lambda *_: None).run(
            acq, d / "out"
        )

    assert calls, (
        "the in-memory fuse path never called lazy_uint16 — has the uint16 "
        "conversion moved back out of the dask graph and onto the materialized "
        "volume? That costs one extra full copy of the fused output."
    )
    for _in_dtype, out_dtype, out_ndim in calls:
        assert str(out_dtype) == "uint16"
        assert out_ndim == 3


def test_streaming_path_uses_the_same_helper(monkeypatch):
    """Both modes share one conversion, so they cannot drift apart again."""
    calls = []
    real = pipeline_mod.lazy_uint16

    def spy(darr):
        calls.append(darr.dtype)
        return real(darr)

    monkeypatch.setattr(pipeline_mod, "lazy_uint16", spy)

    cfg = _tiny_config()
    cfg.streaming_mode = True

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        acq = write_synth_acquisition(
            d / "acq", grid=(2, 2), n_planes=8, frame_size=(32, 32), overlap=0.15
        )
        StitchingPipeline(cfg, progress_fn=lambda *_: None).run(acq, d / "out")

    assert calls, "the streaming fuse path never called lazy_uint16"


def test_background_zeroing_is_applied_in_memory_too():
    """In-memory mode used to ignore the setting entirely.

    ``background_zero_enabled`` was only honoured in the streaming ``_finalize``,
    so the same acquisition came out different depending on which mode
    auto-select happened to pick. Threshold above the phantom's whole range, so
    a mode that applies it returns all zeros and one that doesn't cannot.
    """
    tifffile = pytest.importorskip("tifffile")

    cfg = _tiny_config()
    cfg.background_zero_enabled = True
    cfg.background_zero_thresholds = {1: 65534}

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        acq = write_synth_acquisition(
            d / "acq", grid=(2, 2), n_planes=8, frame_size=(32, 32), overlap=0.15
        )
        out_dir = d / "out"
        StitchingPipeline(cfg, progress_fn=lambda *_: None).run(acq, out_dir)

        written = sorted(out_dir.rglob("*.tif")) + sorted(out_dir.rglob("*.tiff"))
        assert written, f"no OME-TIFF written to {out_dir}"
        data = tifffile.imread(str(written[0]))

    assert data.max() == 0, (
        "background zeroing below 65534 left non-zero voxels in the in-memory "
        f"output (max={data.max()}) — is the threshold still only applied in "
        "the streaming path?"
    )
