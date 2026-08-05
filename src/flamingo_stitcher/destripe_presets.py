"""Per-microscope destripe settings, keyed on the acquisition's own metadata.

Destripe tuning does NOT transfer between scopes, and two of the knobs are the
reason why: ``threshold`` (the foreground/background split) and ``crossover``
(the blend width) are **absolute intensity values in raw counts**. A scope with
different laser power, exposure, gain or bit scaling puts its tissue somewhere
else entirely, so a threshold tuned on one instrument can land above everything
(whole image treated as background) or below everything (all foreground) on
another. That silently changes what the filter does rather than merely
detuning it. ``sigma`` / ``level`` / ``wavelet`` travel better, but the set is
stored whole so a scope's tuning is reproduced exactly.

This mirrors how tile orientation is already handled (``orientation.py``):
presets live in a small JSON file keyed by the "Microscope name" read out of
the acquisition's ``ScopeSettings.txt`` / ``FlamingoMetaData*.txt``, so the
right settings follow the data rather than the GUI session.

An acquisition from an unrecognised scope is NOT blocked — the caller keeps
whatever settings were last used and warns, so a first run on a new instrument
still goes through.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Sits beside orientation_presets.json, same convention.
_PRESETS_FILE = "destripe_presets.json"


def _presets_path() -> Path:
    return Path.home() / ".flamingo_stitcher" / _PRESETS_FILE


def _read_presets() -> Dict[str, dict]:
    try:
        p = _presets_path()
        if p.is_file():
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                return data
    except Exception as e:  # noqa: BLE001 - best-effort
        logger.debug("Could not read destripe presets: %s", e)
    return {}


def _norm(microscope_name: Optional[str]) -> str:
    return str(microscope_name or "").strip().lower()


def load_destripe_preset(microscope_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Saved ``{"params": {...}, "direction": str}`` for a scope, or None."""
    key = _norm(microscope_name)
    if not key:
        return None
    entry = _read_presets().get(key)
    if not isinstance(entry, dict):
        return None
    params = entry.get("params")
    if not isinstance(params, dict):
        return None
    return {
        "params": dict(params),
        "direction": str(entry.get("direction") or "auto"),
    }


def save_destripe_preset(
    microscope_name: Optional[str],
    params: Dict[str, Any],
    direction: str = "auto",
) -> bool:
    """Persist this scope's destripe settings. Returns True if written.

    Best-effort: a failure to save must never interrupt a run, so this reports
    rather than raises.
    """
    key = _norm(microscope_name)
    if not key:
        logger.info("No microscope name — destripe settings not saved per scope")
        return False
    try:
        presets = _read_presets()
        presets[key] = {
            "params": dict(params or {}),
            "direction": str(direction or "auto"),
        }
        path = _presets_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(presets, indent=2))
        logger.info(f"Saved destripe settings for microscope '{microscope_name}'")
        return True
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        logger.warning(f"Could not save destripe settings for {microscope_name}: {e}")
        return False


def microscope_for_acquisition(acquisition_dir: Path) -> Optional[str]:
    """The acquisition's "Microscope name", or None when unreadable."""
    try:
        from flamingo_stitcher.orientation import read_microscope_name

        return read_microscope_name(Path(acquisition_dir))
    except Exception as e:  # noqa: BLE001 - never break a run on metadata
        logger.debug("Could not read microscope name: %s", e)
        return None


def resolve_for_acquisition(
    acquisition_dir: Path,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """``(microscope_name, preset_or_None)`` for one acquisition."""
    scope = microscope_for_acquisition(acquisition_dir)
    return scope, load_destripe_preset(scope)


def describe_resolution(
    scope: Optional[str], preset: Optional[Dict[str, Any]]
) -> Tuple[bool, str]:
    """``(applied, message)`` describing what a caller should do and say.

    The unknown-scope message is deliberately explicit about *why* carrying
    settings over is risky, since the failure is silent in the output.
    """
    if preset:
        return True, f"Destripe settings loaded for microscope '{scope}'."
    if scope:
        return False, (
            f"No destripe settings saved for microscope '{scope}' — continuing "
            "with the last-used settings. Threshold and crossover are absolute "
            "intensity values, so settings tuned on another scope may not "
            "transfer. Check them in Destripe → Preview…, then re-save."
        )
    return False, (
        "Could not read a microscope name for this acquisition (no "
        "ScopeSettings.txt / FlamingoMetaData) — continuing with the last-used "
        "destripe settings, which may have been tuned on another scope."
    )
