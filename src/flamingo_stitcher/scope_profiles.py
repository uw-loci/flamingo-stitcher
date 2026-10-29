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
    # (field, value): the control is grayed out unless that other tunable
    # (a bool) has this value, because the setting is not used otherwise.
    enabled_when: Optional[Tuple[str, bool]] = None


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
            "Tiles scoring below this are not registered: they keep their stage "
            "position and move with the rest of the mosaic. The score measures "
            "texture, not brightness. Empty gel, medium or air scores about 0.09 "
            "(default 0.15). Lower it if tiles with dim but real sample are left out "
            "(the run log lists each one with its score). This also sets what 'Skip "
            "seams whose overlap is empty' counts as empty."
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
            "Which illumination side (the I number in the tile file names) lights the "
            "low end of the illumination axis, that is, the left side of each raw "
            "camera image when the sheets run across X. Needed only when 'Illum. "
            "fusion' is 'Split (each sheet owns its half)' or 'Blend (split with a "
            "soft handover)'. Those will not run while this is -1 (not set). Use "
            "'Measure from data…' rather than guessing. If the data clearly disagrees "
            "with this value, the run stops and says so."
        ),
    ),
    Tunable(
        field="illumination_pure_frac",
        label="Share of the mosaic each sheet owns outright (Blend)",
        kind="fraction",
        minimum=0.0,
        maximum=0.5,
        step=0.05,
        decimals=2,
        suffix=" of the mosaic",
        help=(
            "Used only when 'Illum. fusion' is 'Blend (split with a soft handover)'. "
            "How much of the mosaic, measured in from each end along the illumination "
            "axis, comes from the nearer sheet alone. The strip in between is a "
            "gradual handover. Default 0.35. 0.5 gives a hard cut, the same as "
            "'Split (each sheet owns its half)'. Lower it if there is a visible step "
            "across the middle of the mosaic; "
            "raise it if the handover strip looks blurred."
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
            "On (default): before registering, the app scores the structure in 3 tiles "
            "of each channel and registers on the channel with the most. Off: "
            "registration uses 'Registration channel (when not automatic)'. Turn it "
            "off if you know a different channel aligns better than the one chosen. "
            "The run log lists the chosen channel and every channel's score."
        ),
    ),
    Tunable(
        field="reg_channel",
        label="Registration channel (when not automatic)",
        enabled_when=("registration_channel_auto", False),
        kind="int",
        minimum=0,
        maximum=15,
        step=1,
        decimals=0,
        help=(
            "The channel number to register on when 'Choose the registration channel "
            "automatically' is off. Pick the channel with the densest, sharpest "
            "structure. If an acquisition does not have this channel, the first "
            "channel being stitched is used instead."
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
            "On (default): two tiles whose shared strip has no structure (the sample "
            "ends before the edge) are not registered against each other. The pair is "
            "listed as 'empty_overlap' in the seam table (registration_seams.csv) and "
            "does not count against 'Minimum share of seams that must register'. Turn "
            "it off if seams with visible sample are being skipped. 'Minimum tile "
            "structure to register' sets what counts as empty."
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
            "Seams whose match score (0 to 1) is below this are not used. Default 0.4. "
            "Sparse or low-contrast samples score low even when they are aligned. "
            "Raise it if the seam table (registration_seams.csv) shows seams passing "
            "with implausible shifts; lower it if seams that look well matched are "
            "being rejected."
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
            "If the measured tile overlap (as a share of the frame, on the thinner of "
            "X and Y) is below this, registration is skipped and tiles are placed by "
            "stage position. Default 0.05 (5%). Lower it only if your overlap really "
            "is that small and you want registration to try anyway. 0 turns the check "
            "off."
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
            "If fewer than this share of seams register, the registration is discarded "
            "and every tile is placed by stage position. Default 0.5. Seams skipped as "
            "empty do not count. Lower it for a sample that covers only part of the "
            "grid. 0 turns the check off."
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
            "The largest X/Y shift registration may apply at one seam. A seam that "
            "needs more is discarded before the tiles are placed, and a tile left "
            "farther than this from all its neighbors is moved back in line with them. "
            "0 = Auto: the measured tile overlap width. Set it to how far this stage "
            "can really misplace a tile."
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
            "The largest Z shift registration may apply at one seam. A seam that needs "
            "more is discarded before the tiles are placed, and a tile left farther "
            "than this in Z from all its neighbors is moved back in line with them. 0 "
            "= Auto: at least 8 Z steps and at least 25 µm, but never more than a "
            "quarter of the stack depth. Set it to how far this stage or focus can "
            "really drift between tiles."
        ),
    ),
    # Intensity adjustment (BigStitcher's parameters and defaults). Used only
    # when the main dialog's "Intensity adjustment" is Measure or Apply.
    Tunable(
        field="intensity_adjust_downsampling",
        label="Intensity adjustment: sample spacing",
        kind="float",
        minimum=1.0,
        maximum=64.0,
        step=1.0,
        decimals=1,
        suffix=" native px",
        help=(
            "Used only when 'Intensity adjustment' is 'Measure only' or 'Apply'. "
            "Overlapping tiles are compared at points spaced this many native XY "
            "pixels apart, in every direction. Default 10. Larger is faster but uses "
            "fewer points. Raise it if the fit takes too long (the run log gives its "
            "time)."
        ),
    ),
    Tunable(
        field="intensity_adjust_max_matches",
        label="Intensity adjustment: samples kept per tile pair",
        kind="int",
        minimum=10,
        maximum=1000000,
        step=1000,
        help=(
            "Used only when 'Intensity adjustment' is 'Measure only' or 'Apply'. The "
            "most comparison points kept for each pair of overlapping tiles. Extra "
            "points are dropped at random, the same way on every run, so results "
            "repeat. Default 10000. Lower it to make the fit faster."
        ),
    ),
    Tunable(
        field="intensity_adjust_affine",
        label="Intensity adjustment: fit scale as well as offset",
        kind="bool",
        minimum=0.0,
        maximum=1.0,
        step=1.0,
        help=(
            "Used only when 'Intensity adjustment' is 'Measure only' or 'Apply'. On "
            "(default): each tile's brightness is both scaled and shifted to match its "
            "neighbors. Off: only shifted up or down, so each tile keeps its contrast. "
            "Turn it off if you want to remove brightness offsets between tiles "
            "without changing their contrast."
        ),
    ),
    Tunable(
        field="intensity_adjust_offset_regularization",
        label="Intensity adjustment: pull toward offset-only",
        enabled_when=("intensity_adjust_affine", True),
        kind="fraction",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        decimals=2,
        help=(
            "Used only when 'Intensity adjustment: fit scale as well as offset' is on. "
            "Moves the fit between scale + offset (0) and offset only (1). Default "
            "0.1. Raise it if tiles come out with visibly different contrast after "
            "adjustment."
        ),
    ),
    Tunable(
        field="intensity_adjust_identity_regularization",
        label="Intensity adjustment: pull toward unchanged",
        kind="fraction",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        decimals=2,
        help=(
            "Used only when 'Intensity adjustment' is 'Measure only' or 'Apply'. How "
            "strongly every tile is pulled toward no change (0 = no pull, 1 = tiles "
            "left as acquired). Default 0.1. Raise it if tiles are changed more than "
            "the seams need; lower it if brightness steps remain at the seams."
        ),
    ),
    Tunable(
        field="intensity_adjust_keep_range",
        label="Intensity adjustment: keep the input brightness range",
        kind="bool",
        minimum=0.0,
        maximum=1.0,
        step=1.0,
        help=(
            "Used only when 'Intensity adjustment' is 'Measure only' or 'Apply' "
            "('Measure only' logs the values it would apply). On (default): only the "
            "brightness differences between tiles are corrected, and the typical tile "
            "keeps its acquired brightness. Off: the raw fit is applied unchanged, "
            "which on noisy overlaps can darken the whole mosaic. Seams match equally "
            "well either way. Leave it on unless you need numbers identical to the raw "
            "fit."
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
