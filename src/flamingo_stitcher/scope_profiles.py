"""Per-microscope, per-objective tuning values for the stitching pipeline.

Some pipeline settings are judgements about an instrument and a sample, not
universal constants — how much tile overlap is enough for phase correlation to
mean anything, how much of a mosaic has to register before the result is worth
believing, how far a tile can plausibly have been mis-placed by the stage. Bake
those in and the first scope they do not fit gets a silently wrong answer and no
way to say so; expose them as global settings and they get tuned for whichever
scope was used last.

So they are stored per **microscope and objective**, keyed on the acquisition's
own metadata, exactly as tile orientation (``orientation.py``) and destripe
tuning (``destripe_presets.py``) already are. The settings follow the data
rather than the GUI session, which is what makes a batch that mixes instruments
come out right.

Objective matters as well as scope because these values are expressed against a
field of view: a bound in µm that is a quarter of the overlap at 17x is most of
the overlap at 4x. Today each rig here has one objective, so a scope-wide entry
(objective ``*``) is the common case and an objective-specific entry overrides
it when there is one.

Resolution order, widest to narrowest — each layer only overrides what it
actually sets:

    dataclass defaults  ->  stitching_config.yaml  ->  scope profile  ->
    an explicit CLI flag or GUI control for this run

Nothing here raises. An unreadable profile, an unknown scope or a corrupt file
leaves the caller with whatever it already had, because refusing to stitch
because a preferences file is malformed would be a worse failure than the one it
guards against.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROFILES_FILE = "scope_profiles.json"
# Microscopes an acquisition has been stitched from, so the Options tab can
# offer them without the user retyping a name that has to match exactly.
_SEEN_FILE = "seen_microscopes.json"

# Objective key meaning "every objective on this microscope".
ANY_OBJECTIVE = "*"

PROFILE_VERSION = 1


@dataclass(frozen=True)
class Tunable:
    """One setting a scope profile may carry.

    ``field`` is the :class:`~flamingo_stitcher.pipeline.StitchingConfig`
    attribute name — the profile stores config fields directly so there is no
    second vocabulary to keep in sync.
    """

    field: str
    label: str
    kind: str  # "float" | "int" | "bool" | "fraction"
    minimum: float
    maximum: float
    step: float
    help: str
    decimals: int = 2
    suffix: str = ""


# The whitelist. A profile may only carry these, so a stale or hand-edited file
# can never reach into unrelated pipeline behaviour — and every entry here is
# something a person can be expected to reason about from the registration
# report, which is the test for whether a knob belongs in the UI at all.
TUNABLES: Tuple[Tunable, ...] = (
    Tunable(
        field="min_tile_structure",
        label="Minimum tile structure to register",
        kind="float",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        decimals=2,
        help=(
            "Tiles below this are held out of the registration and placed with "
            "the mosaic. The score is texture relative to noise, so it does NOT "
            "track brightness — bright, featureless agarose scores the same as "
            "empty air, which is why an intensity threshold cannot do this job. "
            "Featureless material sits at about 0.09; real structure crosses it "
            "at a contrast-to-noise ratio near 1. Lower it if dim sample is "
            "being held out; the per-tile scores are in the registration report."
        ),
    ),
    Tunable(
        field="illumination_low_side",
        label="Illumination side lighting the LOW end of the frame",
        kind="int",
        minimum=-1,
        maximum=7,
        step=1,
        decimals=0,
        help=(
            "Which illumination side (the I number in the file name) lights the "
            "LOW-numbered end of the illumination axis — the left of the frame "
            "when the sheets run across X. Only the Split and Blend fusion "
            "modes use it, and without it they fall back to Max rather than "
            "guess: choosing the wrong sheet everywhere is silent, because the "
            "output stays smooth and plausible while being built from the worse "
            "half of every frame. -1 means not set. On n7 this is 1."
        ),
    ),
    Tunable(
        field="illumination_pure_frac",
        label="Share of the frame each sheet owns outright (Blend)",
        kind="fraction",
        minimum=0.0,
        maximum=0.5,
        step=0.05,
        decimals=2,
        suffix=" of the frame",
        help=(
            "Under Blend, how much of the frame nearest each sheet is taken "
            "from that sheet alone; the rest is a smoothstep handover. 0.5 is a "
            "hard split with no handover at all (the same as Split). Lower it "
            "if the middle of the frame looks abruptly different; raise it if "
            "the handover band itself looks soft."
        ),
    ),
    Tunable(
        field="registration_channel_auto",
        label="Choose the registration channel automatically",
        kind="bool",
        minimum=0,
        maximum=1,
        step=1,
        decimals=0,
        help=(
            "Registration runs on ONE channel and costs hours, so which channel "
            "it uses matters as much as any threshold here. On measures the "
            "structure in a few tiles of each channel and registers on the "
            "densest. Off uses the channel number below, which is what every "
            "run before this did — and that number defaulted to the lowest "
            "channel ID, an ordering with nothing to do with which channel "
            "carries structure. The chosen channel and every channel's score "
            "are written to the run log."
        ),
    ),
    Tunable(
        field="reg_channel",
        label="Registration channel (when not automatic)",
        kind="int",
        minimum=0,
        maximum=15,
        step=1,
        decimals=0,
        help=(
            "The channel to register on when the automatic choice above is off. "
            "If this channel is not present in an acquisition, the first one "
            "that is gets used instead."
        ),
    ),
    Tunable(
        field="registration_seam_content_gate",
        label="Skip seams whose overlap is empty",
        kind="bool",
        minimum=0,
        maximum=1,
        step=1,
        decimals=0,
        help=(
            "Two tiles can both contain plenty of sample while the strip they "
            "actually SHARE is empty medium — the sample ends partway across "
            "the mosaic. Phase correlation on such a strip does not fail "
            "loudly; it returns a confident peak drawn from noise, which then "
            "has to be caught by the quality threshold or the shift bound. On, "
            "the pair is never scheduled and is reported as 'empty_overlap' "
            "rather than counted as a failed seam. Turn it off if the seam "
            "table shows real seams being skipped."
        ),
    ),
    Tunable(
        field="quality_threshold",
        label="Seam quality threshold",
        kind="float",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        decimals=2,
        help=(
            "Minimum correlation for a seam to be used. The score is a Spearman "
            "rank correlation over the overlap, so a sparse or low-contrast "
            "sample scores low even when the alignment is right. Raise it if the "
            "seam table shows seams passing with implausible shifts; lower it if "
            "good-looking seams are being rejected."
        ),
    ),
    Tunable(
        field="min_registration_overlap_frac",
        label="Minimum tile overlap to attempt registration",
        kind="fraction",
        minimum=0.0,
        maximum=0.5,
        step=0.01,
        decimals=2,
        suffix=" of a frame",
        help=(
            "Below this measured overlap, registration is not attempted at all. "
            "Phase correlation on a sliver of shared content does not fail "
            "loudly — it returns a confident wrong shift. 0 disables the check."
        ),
    ),
    Tunable(
        field="min_registered_seam_frac",
        label="Minimum share of seams that must register",
        kind="fraction",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        decimals=2,
        suffix=" of expected seams",
        help=(
            "Below this, the registration is not applied and tiles are placed by "
            "stage position. Registration routinely succeeds on the part of a "
            "mosaic that contains sample and fails on the rest; when only a few "
            "seams agree, those few are as likely to be a confident wrong "
            "correlation peak as a measurement. Lower it for a sample that only "
            "ever covers part of the grid. 0 disables the check."
        ),
    ),
    Tunable(
        field="max_registration_shift_um",
        label="Maximum lateral correction",
        kind="float",
        minimum=0.0,
        maximum=2000.0,
        step=5.0,
        decimals=1,
        suffix=" µm",
        help=(
            "A tile whose X/Y correction exceeds this keeps its stage position. "
            "0 derives the bound from the measured tile overlap. Set it to what "
            "this stage can actually be wrong by — the automatic bound is one "
            "whole overlap, which is generous enough to admit a garbage peak."
        ),
    ),
    Tunable(
        field="max_registration_shift_z_um",
        label="Maximum axial correction",
        kind="float",
        minimum=0.0,
        maximum=2000.0,
        step=5.0,
        decimals=1,
        suffix=" µm",
        help=(
            "As above, for Z. 0 derives a bound from the Z step and stack depth. "
            "On an XY mosaic there is no Z overlap to measure a bound from, so "
            "this is the honest place to state what the stage can do."
        ),
    ),
)

TUNABLE_FIELDS: Tuple[str, ...] = tuple(t.field for t in TUNABLES)
TUNABLE_BY_FIELD: Dict[str, Tunable] = {t.field: t for t in TUNABLES}


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


def _norm_scope(microscope_name: Optional[str]) -> str:
    return str(microscope_name or "").strip().lower()


def normalize_objective(objective: Any) -> str:
    """A stable key for an objective magnification.

    Rounded to one decimal so 17.0 and 17.000 are the same objective, and
    formatted rather than left as a float so the JSON key is readable and
    round-trips exactly.
    """
    if objective is None:
        return ANY_OBJECTIVE
    if isinstance(objective, str):
        text = objective.strip().lower()
        if not text or text == ANY_OBJECTIVE:
            return ANY_OBJECTIVE
        text = text.rstrip("x")
        try:
            objective = float(text)
        except ValueError:
            return ANY_OBJECTIVE
    try:
        value = float(objective)
    except (TypeError, ValueError):
        return ANY_OBJECTIVE
    if value <= 0:
        return ANY_OBJECTIVE
    return f"{value:.1f}x"


def profile_key(microscope_name: Optional[str], objective: Any) -> str:
    """``"scope|17.0x"``. Empty when there is no microscope to key on."""
    scope = _norm_scope(microscope_name)
    if not scope:
        return ""
    return f"{scope}|{normalize_objective(objective)}"


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def profiles_path() -> Path:
    return Path.home() / ".flamingo_stitcher" / _PROFILES_FILE


def seen_path() -> Path:
    """Where the set of microscopes we have ever stitched from is kept.

    A separate file from the profiles, deliberately: a name here means only
    "an acquisition from this instrument has been through the app", which is
    not the same claim as "this instrument has tuned settings". Mixing the two
    would make an untuned microscope look configured.
    """
    return Path.home() / ".flamingo_stitcher" / _SEEN_FILE


def _read_seen() -> set:
    try:
        path = seen_path()
        if not path.is_file():
            return set()
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - a convenience list must never fail
        logger.debug("Could not read seen microscopes: %s", e)
        return set()
    if isinstance(data, dict):  # tolerate {"microscopes": [...]}
        data = data.get("microscopes", [])
    if not isinstance(data, list):
        return set()
    return {n for n in (_norm_scope(x) for x in data) if n}


def remember_microscope(microscope_name: Optional[str]) -> bool:
    """Record that an acquisition from this microscope has been seen.

    Returns True when the name was new. A SET by construction: names are
    normalised the same way lookups are (`_norm_scope`, lower-cased and
    stripped), so the same instrument cannot land in the list twice under
    different spacing or case.

    Never raises. Failing to remember a name for the Options dropdown must not
    be able to affect a stitching run.
    """
    name = _norm_scope(microscope_name)
    if not name:
        return False
    try:
        seen = _read_seen()
        if name in seen:
            return False
        seen.add(name)
        path = seen_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(sorted(seen), indent=2), encoding="utf-8"
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not remember microscope %r: %s", microscope_name, e)
        return False


def known_microscopes() -> List[str]:
    """Every microscope the Options tab should offer, sorted.

    The union of instruments with SAVED settings and instruments merely seen
    during a run, so a microscope becomes selectable the first time one of its
    acquisitions is stitched rather than only after someone types its name in
    by hand.
    """
    names = {
        _norm_scope(key.split("|", 1)[0]) for key in _read_all()
    }
    names |= _read_seen()
    return sorted(n for n in names if n)


def _read_all() -> Dict[str, dict]:
    try:
        p = profiles_path()
        if p.is_file():
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                entries = data.get("profiles")
                if isinstance(entries, dict):
                    return entries
                # A bare mapping is the pre-version layout; accept it.
                if "version" not in data:
                    return data
    except Exception as e:  # noqa: BLE001 - best-effort
        logger.debug("Could not read scope profiles: %s", e)
    return {}


def _write_all(entries: Dict[str, dict]) -> bool:
    try:
        p = profiles_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"version": PROFILE_VERSION, "profiles": entries},
                indent=2,
                sort_keys=True,
            )
        )
        return True
    except Exception as e:  # noqa: BLE001 - never interrupt a run
        logger.warning("Could not save scope profiles: %s", e)
        return False


def _clean(values: Any) -> Dict[str, Any]:
    """Only whitelisted fields, only values of the right shape."""
    if not isinstance(values, dict):
        return {}
    out: Dict[str, Any] = {}
    for field, value in values.items():
        spec = TUNABLE_BY_FIELD.get(field)
        if spec is None:
            continue
        try:
            if spec.kind == "bool":
                out[field] = bool(value)
            elif spec.kind == "int":
                out[field] = int(value)
            else:
                number = float(value)
                # A value outside the control's own range is a corrupt or
                # hand-edited file; drop it rather than feed it to the pipeline.
                if not (spec.minimum <= number <= spec.maximum):
                    logger.warning(
                        "Ignoring out-of-range %s=%s in scope profile "
                        "(expected %s..%s)",
                        field,
                        number,
                        spec.minimum,
                        spec.maximum,
                    )
                    continue
                out[field] = number
        except (TypeError, ValueError):
            continue
    return out


def list_profiles() -> Dict[str, Dict[str, Any]]:
    """Every saved profile, keyed ``"scope|objective"``, cleaned."""
    return {key: _clean(entry) for key, entry in _read_all().items()}


def load_profile(
    microscope_name: Optional[str], objective: Any = None
) -> Tuple[Dict[str, Any], str]:
    """``(values, source_key)`` for a scope + objective.

    An objective-specific entry wins over a scope-wide one; the two are NOT
    merged. A half-inherited profile would be the hardest kind to reason about
    from a report — "which of these two entries produced this number?" — so an
    objective entry, once it exists, is the whole answer for that objective.
    """
    scope = _norm_scope(microscope_name)
    if not scope:
        return {}, ""
    entries = _read_all()
    for key in (profile_key(scope, objective), f"{scope}|{ANY_OBJECTIVE}"):
        if key and key in entries:
            values = _clean(entries[key])
            if values:
                return values, key
    return {}, ""


def save_profile(
    microscope_name: Optional[str],
    objective: Any,
    values: Dict[str, Any],
) -> bool:
    """Persist a profile. Returns True if written.

    Pass ``objective=None`` (or ``"*"``) to save one entry for every objective
    on this microscope, which is the right choice when a rig has only one.
    """
    key = profile_key(microscope_name, objective)
    if not key:
        logger.warning("Cannot save a scope profile without a microscope name")
        return False
    cleaned = _clean(values)
    entries = _read_all()
    if cleaned:
        entries[key] = cleaned
    else:
        entries.pop(key, None)
    return _write_all(entries)


def delete_profile(microscope_name: Optional[str], objective: Any) -> bool:
    key = profile_key(microscope_name, objective)
    entries = _read_all()
    if key not in entries:
        return False
    entries.pop(key)
    return _write_all(entries)


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #


def resolve_for_acquisition(
    acquisition_dir: Path,
) -> Tuple[Optional[str], Optional[float], Dict[str, Any], str]:
    """``(microscope, objective, values, source_key)`` for one acquisition."""
    scope: Optional[str] = None
    objective: Optional[float] = None
    try:
        from flamingo_stitcher.orientation import read_microscope_name

        scope = read_microscope_name(Path(acquisition_dir))
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not read microscope name: %s", e)
    try:
        from flamingo_stitcher.pipeline import read_objective_magnification

        objective = read_objective_magnification(Path(acquisition_dir))
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not read objective magnification: %s", e)
    values, source = load_profile(scope, objective)
    return scope, objective, values, source


def apply_profile(config: Any, values: Dict[str, Any]) -> List[str]:
    """Set whitelisted fields on a config. Returns human-readable changes.

    Only fields the profile actually carries are touched, so a profile that
    tunes one threshold does not quietly reset the others to their defaults.
    """
    changed: List[str] = []
    for field, value in _clean(values).items():
        before = getattr(config, field, None)
        if before == value:
            continue
        setattr(config, field, value)
        spec = TUNABLE_BY_FIELD[field]
        changed.append(f"{spec.label}: {before} -> {value}")
    return changed


def describe_resolution(
    scope: Optional[str],
    objective: Optional[float],
    values: Dict[str, Any],
    source: str,
) -> Tuple[bool, str]:
    """``(applied, message)`` — what a caller should do and say about it."""
    where = f"'{scope}'" if scope else "an unnamed microscope"
    lens = f" at {normalize_objective(objective)}" if objective else ""
    if values:
        scope_wide = source.endswith(f"|{ANY_OBJECTIVE}")
        which = "all objectives" if scope_wide else f"objective {source.split('|')[-1]}"
        return True, (
            f"Stitching options loaded for {where}{lens} (saved for {which})."
        )
    if scope:
        return False, (
            f"No saved stitching options for {where}{lens} — using the defaults. "
            f"Tune them once in the Options tab and every run from this "
            f"microscope will use them."
        )
    return False, (
        "Could not read a microscope name for this acquisition (no "
        "ScopeSettings.txt / FlamingoMetaData), so no per-scope stitching "
        "options could be applied; using the defaults."
    )
