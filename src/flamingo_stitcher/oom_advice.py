"""Step-aware out-of-memory advice.

When a stitch fails with a memory error, a bare ``MemoryError`` /
"Unable to allocate N GiB" tells the user *nothing* about what to change.
Most tools' only answer is "use less data" (downsample). But the pipeline
has several memory levers that preserve the chosen resolution — the right
one depends on *which step* ran out of memory:

  * write  — the multi-resolution pyramid builder is the single most
             memory-hungry final step. OME-Zarr streams it chunk-by-chunk;
             OME-TIFF/Imaris are heavier. Switching format keeps full res.
  * fuse /
    preprocess — peak RAM is roughly ``workers x tile`` plus per-tile
             float buffers added by deconvolution / depth attenuation /
             non-fast destripe / content-based blending. Each is optional.
  * register — builds a multiscale image for every tile at once; skipping
             registration (trust stage positions) removes it entirely.

This module is pure (no Qt, no numpy) so it is trivially unit-testable and
callable from both the GUI failure handler and the CLI. It returns a list
of prioritised advice lines — resolution-preserving levers first, an
explicit downsample fallback last (clearly flagged as the only one that
lowers resolution).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Substrings that identify a memory-exhaustion failure across the stack:
# numpy ("Unable to allocate"), the C++/BLAS allocators ("bad_alloc"),
# the OS ("Cannot allocate memory" / errno ENOMEM), and Python's own
# ``MemoryError`` repr.
_MEMORY_ERROR_MARKERS = (
    "unable to allocate",
    "memoryerror",
    "out of memory",
    "cannot allocate memory",
    "bad_alloc",
    "std::bad_alloc",
    "allocate array",
    "array is too big",
)


def is_memory_error(error_msg: str) -> bool:
    """True if ``error_msg`` looks like an out-of-memory failure.

    Matches numpy, C++/BLAS, OS-level and Python memory errors. Deliberately
    broad — a false positive just appends advice the user can ignore, whereas
    a false negative hides the one thing that would have helped.
    """
    if not error_msg:
        return False
    low = error_msg.lower()
    return any(marker in low for marker in _MEMORY_ERROR_MARKERS)


# Canonical step keys, mirroring the dialog's step pills.
_STEP_TITLES = {
    "discover": "tile discovery",
    "register": "registration",
    "preprocess": "tile preprocessing",
    "fuse": "channel fusion",
    "write": "writing the output",
    "metadata": "writing metadata",
}


def _truthy(settings: Dict[str, Any], key: str) -> bool:
    return bool(settings.get(key))


def _write_step_advice(settings: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    fmt = str(settings.get("output_format", "") or "").lower()
    heavy_pyramid = (
        "tiff" in fmt or "imaris" in fmt or fmt == "both" or "ims" in fmt
    )
    if heavy_pyramid:
        lines.append(
            "Switch Output format -> OME-Zarr (sharded). It builds the "
            "multi-resolution pyramid chunk-by-chunk and never holds a whole "
            "pyramid level in RAM; OME-TIFF/Imaris pyramids are the heaviest "
            "final step at TB scale. Same resolution, far less peak memory."
        )
        lines.append(
            "Already on the latest version? The in-RAM OME-TIFF pyramid build "
            "was fixed in v0.5.0 (it now spills each level to disk). Check the "
            "version banner at the top of the log and update if it is older."
        )
    else:
        lines.append(
            "The pyramid builder is the memory peak. Update to the latest "
            "version (v0.5.0+ spills each pyramid level to disk instead of "
            "holding it in RAM), or reduce the number of pyramid levels in "
            "Processing Options."
        )
    return lines


def _preprocess_fuse_advice(settings: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    pp = int(settings.get("preprocess_workers", 0) or 0)
    fw = int(settings.get("fuse_workers", 0) or 0)
    eff = max(pp, fw) or 4  # 0 => auto => 4
    if eff > 1:
        lines.append(
            f"Lower the preprocess/fuse worker count (currently ~{eff}) to 1-2 "
            "in Processing Options. Each worker holds one whole tile in RAM, so "
            "workers x tile-size is most of the streaming working set."
        )

    if _truthy(settings, "content_based_fusion"):
        lines.append(
            "Turn off Content-based blending. It adds per-block halos and "
            "float working buffers (and is 5-10x slower); plain cosine/max "
            "blending gives near-identical seams at a fraction of the memory."
        )

    heavy = []
    if _truthy(settings, "deconvolution_enabled"):
        heavy.append("Deconvolution")
    if _truthy(settings, "depth_attenuation"):
        heavy.append("Depth attenuation")
    if _truthy(settings, "destripe") and not _truthy(settings, "destripe_fast"):
        heavy.append("Destripe (non-fast)")
    if _truthy(settings, "flat_field_correction"):
        heavy.append("Flat-field")
    if heavy:
        lines.append(
            "These each allocate an extra whole-tile float buffer: "
            + ", ".join(heavy)
            + ". Disable the ones you can spare, or set Destripe -> Fast "
            "(destripe after downsample)."
        )

    if str(settings.get("illumination_fusion", "") or "").lower() == "leonardo":
        lines.append(
            "Illumination fusion is set to Leonardo, which materialises both "
            "sides in a heavier host/GPU buffer. Switch to Max or Mean unless "
            "you specifically need Leonardo."
        )

    lines.append(
        "Make sure the scratch/spill directory is on a fast drive with room — "
        "in streaming mode tiles are written there and read back during fusion; "
        "a slow or full scratch disk forces larger in-RAM buffers."
    )
    return lines


def _register_advice(settings: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if not _truthy(settings, "skip_registration"):
        lines.append(
            "Registration builds a multiscale image for every tile at once. If "
            "the stage positions are already accurate, tick 'Skip registration "
            "(use stage positions)' — it removes the registration memory load "
            "entirely and keeps full resolution."
        )
    lines.append(
        "Confirm pandas is installed: without it, registration silently falls "
        "back to metadata positions (v0.5.0+ bundles it)."
    )
    return lines


def oom_advice(
    step_key: Optional[str],
    settings: Optional[Dict[str, Any]] = None,
    *,
    use_streaming: Optional[bool] = None,
) -> List[str]:
    """Return prioritised, resolution-preserving OOM advice.

    Args:
        step_key: the pipeline step that failed (``discover``/``register``/
            ``preprocess``/``fuse``/``write``/``metadata``), or None if unknown.
        settings: current config flags. Recognised keys (all optional):
            ``output_format``, ``content_based_fusion``, ``deconvolution_enabled``,
            ``depth_attenuation``, ``destripe``, ``destripe_fast``,
            ``flat_field_correction``, ``illumination_fusion``,
            ``preprocess_workers``, ``fuse_workers``, ``skip_registration``.
        use_streaming: resolved memory mode. If ``False`` (in-memory), enabling
            streaming is offered first — it is the single biggest lever.

    Returns:
        Ordered advice lines. Every line preserves the chosen resolution except
        the final downsample fallback, which is explicitly flagged as such.
    """
    settings = settings or {}
    lines: List[str] = []

    # Streaming is the biggest single lever and applies to every compute step.
    if use_streaming is False and step_key in (
        None,
        "preprocess",
        "fuse",
        "register",
        "write",
    ):
        lines.append(
            "Enable Streaming mode (Processing Options -> Memory mode = "
            "Streaming). In-memory mode holds the whole fused volume plus all "
            "tiles in RAM; streaming bounds RAM to a small working set and "
            "spills to disk — same resolution."
        )

    if step_key == "write":
        lines += _write_step_advice(settings)
    elif step_key in ("preprocess", "fuse"):
        lines += _preprocess_fuse_advice(settings)
    elif step_key == "register":
        lines += _register_advice(settings)
    else:
        # Unknown / discover / metadata: offer the broadly-applicable levers.
        lines += _preprocess_fuse_advice(settings)
        lines += _write_step_advice(settings)

    # Universal closers.
    lines.append(
        "Close other big memory users (browsers, image viewers) to free RAM "
        "for the run."
    )
    lines.append(
        "Last resort (this is the only lever that lowers resolution): raise "
        "the XY or Z downsample factor by one step. Everything above keeps "
        "full resolution."
    )
    return lines


def format_oom_advice(
    step_key: Optional[str],
    settings: Optional[Dict[str, Any]] = None,
    *,
    use_streaming: Optional[bool] = None,
) -> str:
    """Render :func:`oom_advice` as a numbered, log-friendly block."""
    step_title = _STEP_TITLES.get(step_key or "", None)
    where = f" while {step_title}" if step_title else ""
    header = (
        f"Ran out of memory{where}. You can likely finish at the SAME "
        "resolution by changing one of these (most effective first):"
    )
    body = oom_advice(step_key, settings, use_streaming=use_streaming)
    numbered = "\n".join(f"  {i}. {line}" for i, line in enumerate(body, 1))
    return f"{header}\n{numbered}"
