"""Whole-mosaic display orientation + a MIP-based orientation preview.

Each tile the camera captures is already correctly oriented on its own; what
differs between microscope systems is how the assembled *mosaic* should be
framed (which stage direction ends up left/right/up/down, and whether the grid
reads rotated). Rotating or mirroring the fully assembled mosaic is
mathematically identical to applying that same transform to every tile's pixels
**and** their placement together, so this is a safe, purely-cosmetic transform
of the FINAL fused output — it never touches registration or seam alignment.

There are exactly eight such orientations (the dihedral group of the square:
4 rotations × an optional mirror), enumerated by :class:`MosaicOrientation`.

Because beads (and many samples) make it hard to eyeball the right orientation,
:func:`build_mip_mosaic` assembles a fast, low-res preview from the per-tile
``*_MP.tif`` max-projection files so a user can see all eight candidates and
pick the correct one — the same transform then applies to the real run.

Orientation is not stored in the acquisition; the only per-scope identifier on
disk is the "Microscope name" (ScopeSettings.txt / FlamingoMetaData.txt), so a
named preset (see the ``microscopes:`` block of ``microscope_hardware.yaml``)
lets a user pick their system once instead of choosing every run.

Kept dependency-light (numpy + tifffile, both already required) so it is cheap
to import and the GUI can consume the preview arrays directly as QImages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OrientationUnknownError(RuntimeError):
    """Raised when an acquisition's microscope has no chosen tile orientation.

    A new microscope must have its orientation selected once (via the preview /
    a preset) — the pipeline refuses to stitch with a guessed orientation, which
    would silently mis-place tiles.
    """


# --------------------------------------------------------------------------- #
# The eight whole-mosaic orientations (dihedral group of the square)
# --------------------------------------------------------------------------- #
class MosaicOrientation:
    """One of eight rotation×mirror orientations applied to an assembled mosaic.

    Identified by a short name (see :attr:`NAMES`). ``apply2d`` transforms a
    ``(Y, X)`` image; ``apply_volume_xy`` transforms the trailing two axes of a
    ``(Z, Y, X)`` volume the same way. Both return views where possible (no
    copy). ``identity`` is the historical, unchanged output.
    """

    NAMES: Tuple[str, ...] = (
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "flip_h",
        "flip_v",
        "transpose",
        "anti_transpose",
    )

    # Human-friendly descriptions for UI labels.
    LABELS: Dict[str, str] = {
        "identity": "Identity (unchanged)",
        "rot90": "Rotate 90° CCW",
        "rot180": "Rotate 180°",
        "rot270": "Rotate 270° CCW",
        "flip_h": "Mirror left–right",
        "flip_v": "Mirror up–down",
        "transpose": "Transpose (main diagonal)",
        "anti_transpose": "Anti-transpose (anti-diagonal)",
    }

    def __init__(self, name: str = "identity") -> None:
        key = (name or "identity").strip().lower()
        if key not in self.NAMES:
            raise ValueError(
                f"Unknown mosaic orientation {name!r}; expected one of "
                f"{', '.join(self.NAMES)}"
            )
        self.name = key

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"MosaicOrientation({self.name!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, MosaicOrientation) and other.name == self.name

    @property
    def label(self) -> str:
        return self.LABELS[self.name]

    @property
    def is_identity(self) -> bool:
        return self.name == "identity"

    @classmethod
    def from_name(cls, name: Optional[str]) -> "MosaicOrientation":
        """Build from a name, tolerating None/unknown → identity (logged)."""
        try:
            return cls(name or "identity")
        except ValueError:
            logger.warning(
                "Unknown output_orientation %r; using 'identity'", name
            )
            return cls("identity")

    def apply2d(self, image: np.ndarray) -> np.ndarray:
        """Transform a 2-D ``(Y, X)`` array."""
        n = self.name
        if n == "identity":
            return image
        if n == "rot90":
            return np.rot90(image, 1)
        if n == "rot180":
            return np.rot90(image, 2)
        if n == "rot270":
            return np.rot90(image, 3)
        if n == "flip_h":
            return image[:, ::-1]
        if n == "flip_v":
            return image[::-1, :]
        if n == "transpose":
            return image.T
        # anti_transpose: reflect across the anti-diagonal
        return image[::-1, ::-1].T

    def apply_volume_xy(self, volume: np.ndarray) -> np.ndarray:
        """Transform the trailing ``(Y, X)`` axes of a ``(Z, Y, X)`` volume."""
        n = self.name
        if n == "identity":
            return volume
        if n == "rot90":
            return np.rot90(volume, 1, axes=(1, 2))
        if n == "rot180":
            return np.rot90(volume, 2, axes=(1, 2))
        if n == "rot270":
            return np.rot90(volume, 3, axes=(1, 2))
        if n == "flip_h":
            return volume[:, :, ::-1]
        if n == "flip_v":
            return volume[:, ::-1, :]
        if n == "transpose":
            return volume.transpose(0, 2, 1)
        return volume[:, ::-1, ::-1].transpose(0, 2, 1)


# --------------------------------------------------------------------------- #
# Microscope-name presets (output_orientation) + name lookup
# --------------------------------------------------------------------------- #
def load_orientation_presets() -> Dict[str, str]:
    """``{microscope_name_lower: tile_orientation_name}`` from the hardware YAML.

    Name-only view of the bundled ``microscopes:`` presets (drops the reverse
    flags — see :func:`_load_yaml_preset_objects` for the full form). Never
    raises — a missing/broken block yields an empty registry.
    """
    return {k: v.name for k, v in _load_yaml_preset_objects().items()}


def preset_for_microscope(name: Optional[str]) -> Optional[str]:
    """Return the ``output_orientation`` for a microscope name, or None."""
    if not name:
        return None
    return load_orientation_presets().get(name.strip().lower())


# Horizontal whitespace only around "=", and the value must stay on the SAME
# line. With a plain \s* the newline after an empty "Microscope name =" was
# consumed and (.+) captured the next line — a blank field resolved to
# "</Scope Settings>", which then became the preset key. Every scope with a
# blank name would have shared that one bogus key, for destripe settings AND
# tile orientation.
_NAME_RE = re.compile(r"Microscope name[^\S\r\n]*=[^\S\r\n]*([^\r\n]+)")


def read_microscope_name(acquisition_dir: Path) -> Optional[str]:
    """Read "Microscope name" from an acquisition's metadata, or None.

    Checks ScopeSettings.txt then FlamingoMetaData*.txt, in the acquisition
    root and one level down (dated subfolders). Read errors are swallowed (a
    corrupt metadata file must never break orientation lookup).
    """
    acquisition_dir = Path(acquisition_dir)
    bases = [acquisition_dir, *_child_dirs(acquisition_dir)]
    candidates: List[Path] = []
    for base in bases:
        candidates.append(base / "ScopeSettings.txt")
        try:
            candidates.extend(sorted(base.glob("FlamingoMetaData*.txt")))
        except OSError:
            pass
    for f in candidates:
        try:
            if not f.is_file():
                continue
            m = _NAME_RE.search(f.read_text(errors="replace"))
            if m and m.group(1).strip():
                return m.group(1).strip()
        except OSError:
            continue
    return None


def _child_dirs(d: Path) -> List[Path]:
    try:
        return [c for c in sorted(d.iterdir()) if c.is_dir()]
    except OSError:
        return []


def resolve_output_orientation(acquisition_dir: Path) -> Optional[str]:
    """Best-effort ``tile_orientation`` name for an acquisition via microscope name.

    Returns the matching preset name, or None when the microscope name is
    unknown / has no preset (caller keeps the configured/default orientation).
    """
    name = read_microscope_name(acquisition_dir)
    if not name:
        return None
    ori = preset_for_microscope(name)
    if ori:
        logger.info("Tile-orientation preset '%s' → %s", name, ori)
    return ori


# --------------------------------------------------------------------------- #
# User-saved per-microscope orientation (chosen from the preview) — full
# (tile_orientation, reverse_x, reverse_y), persisted to a user JSON file so it
# survives restarts and so CLI + GUI share it. This is separate from the bundled
# YAML presets (which ship a name only).
# --------------------------------------------------------------------------- #
@dataclass
class TileOrientation:
    """A complete tile-placement choice: pixel orientation + tile-order signs."""

    name: str = ""  # one of MosaicOrientation.NAMES ("" = leave default)
    reverse_x: bool = False
    reverse_y: bool = False


def _user_presets_path() -> Path:
    return Path.home() / ".flamingo_stitcher" / "orientation_presets.json"


def _read_user_presets() -> Dict[str, Dict]:
    try:
        import json

        p = _user_presets_path()
        if p.is_file():
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                return data
    except Exception as e:  # noqa: BLE001 - best-effort
        logger.debug("Could not read user orientation presets: %s", e)
    return {}


def save_microscope_orientation(
    microscope_name: str, name: str, reverse_x: bool, reverse_y: bool
) -> None:
    """Persist the chosen orientation for a microscope (best-effort)."""
    if not microscope_name:
        return
    try:
        import json

        presets = _read_user_presets()
        presets[str(microscope_name).strip().lower()] = {
            "tile_orientation": name,
            "reverse_x": bool(reverse_x),
            "reverse_y": bool(reverse_y),
        }
        path = _user_presets_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(presets, indent=2))
        logger.info(
            "Saved tile orientation for '%s': %s (revX=%s revY=%s)",
            microscope_name,
            name,
            reverse_x,
            reverse_y,
        )
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        logger.warning("Could not save orientation for %s: %s", microscope_name, e)


def _load_yaml_preset_objects() -> Dict[str, "TileOrientation"]:
    """Full per-microscope presets from the bundled hardware YAML.

    Reads the optional ``microscopes:`` block: each entry's ``tile_orientation``
    (one of the 8 dihedral names; ``output_orientation`` accepted as a
    deprecated alias) plus optional ``reverse_x`` / ``reverse_y`` tile-order
    flags. Never raises — a missing/broken block yields an empty registry.
    """
    try:
        import yaml

        from flamingo_stitcher.config_loader import _CONFIGS_DIR

        yaml_path = _CONFIGS_DIR / "microscope_hardware.yaml"
        if not yaml_path.is_file():
            return {}
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        scopes = (raw.get("microscopes") or {}) if isinstance(raw, dict) else {}
        out: Dict[str, TileOrientation] = {}
        for name, entry in scopes.items():
            if not isinstance(entry, dict):
                continue
            key = entry.get("tile_orientation") or entry.get("output_orientation")
            if not key:
                continue
            ori = str(key).strip().lower()
            if ori not in MosaicOrientation.NAMES:
                continue
            out[str(name).strip().lower()] = TileOrientation(
                name=ori,
                reverse_x=bool(entry.get("reverse_x", False)),
                reverse_y=bool(entry.get("reverse_y", False)),
            )
        return out
    except Exception as e:  # noqa: BLE001 - presets are best-effort
        logger.debug("Could not load orientation presets: %s", e)
        return {}


# The four orientations that swap the image axes. A stripe running one way in
# the camera frame comes out running the other way in the stitched output.
_AXIS_SWAPPING = frozenset({"rot90", "rot270", "transpose", "anti_transpose"})


def stripe_axis_in_camera_frame(
    orientation_name: str, output_axis: str = "horizontal"
) -> str:
    """Stripe axis in the RAW CAMERA FRAME, given the axis in the OUTPUT frame.

    Stripe direction is not a property of the image content — it is set by the
    light-sheet propagation direction, and the stitched output is stage-aligned,
    so the axis in the OUTPUT frame is a known constant per microscope
    (horizontal for the Flamingo). Destriping, however, runs in the raw camera
    frame, before the per-tile rotation/flip. This maps the known output-frame
    axis back through that transform.

    Deriving it beats detecting it from pixels: a detector has to guess from
    mean-profile spectra, which is unreliable on background-dominated tiles and
    meaningless on the blank tiles a blind acquisition can easily produce.

    Verified empirically against ``MosaicOrientation.apply2d`` for all eight
    orientations: only the axis-swapping four flip the answer.
    """
    axis = (output_axis or "horizontal").strip().lower()
    if axis not in ("horizontal", "vertical"):
        raise ValueError(
            f"output_axis must be 'horizontal' or 'vertical', got {output_axis!r}"
        )
    name = (orientation_name or "identity").strip().lower()
    if name not in MosaicOrientation.NAMES:
        name = "identity"
    if name in _AXIS_SWAPPING:
        return "vertical" if axis == "horizontal" else "horizontal"
    return axis


def resolve_tile_orientation(acquisition_dir: Path) -> Optional["TileOrientation"]:
    """Full orientation for an acquisition, by microscope name.

    Precedence: user-saved preset (name + reverse flags) > bundled YAML preset
    (name + reverse flags) > None (caller keeps the configured default). None
    means "no saved choice for this system".
    """
    name = read_microscope_name(acquisition_dir)
    if not name:
        return None
    key = name.strip().lower()
    user = _read_user_presets().get(key)
    if isinstance(user, dict) and user.get("tile_orientation"):
        return TileOrientation(
            name=str(user["tile_orientation"]).strip().lower(),
            reverse_x=bool(user.get("reverse_x", False)),
            reverse_y=bool(user.get("reverse_y", False)),
        )
    return _load_yaml_preset_objects().get(key)


# --------------------------------------------------------------------------- #
# MIP-mosaic preview
# --------------------------------------------------------------------------- #
def build_mip_mosaic(
    acquisition_dir: Path,
    *,
    pixel_size_um: Optional[float] = None,
    target_long_px: int = 1000,
    channel: Optional[int] = None,
    z_range: Optional[Tuple[float, float]] = None,
    label_tiles: bool = True,
    tile_transform: Optional["MosaicOrientation"] = None,
    reverse_x: bool = False,
    reverse_y: bool = False,
) -> Optional[np.ndarray]:
    """Assemble a fast low-res MIP mosaic for preview.

    ``tile_transform`` (a MosaicOrientation) is applied to EACH TILE before it is
    placed at its stage position — the per-tile reorientation that actually makes
    tiles connect. ``None`` composites tiles as-is.

    Places each tile's max-projection at its stage position, so a user can judge
    the mosaic layout. Returns a 2-D ``float32`` array normalised to [0, 1], or
    ``None`` if no tiles / MIPs could be read. Placement is by stage centre and
    is intentionally approximate.

    ``z_range`` (lo, hi) as fractions in [0, 1] restricts the projection to a
    sub-range of Z planes — e.g. ``(0.0, 0.25)`` projects only the bottom
    quarter, revealing structure that the full-stack projection buries under
    scattered beads. When ``None`` (full stack), the fast per-tile ``*_MP.tif``
    companion is used; a sub-range reads the raw and projects just those planes.

    ``label_tiles`` burns each tile's grid index (e.g. ``X0Y0``) into the mosaic
    so it rotates/flips with the tiles — the label shows where each tile lands
    in every orientation, for reference.
    """
    loaded, pixel_size_um = _load_tile_mips(
        acquisition_dir, pixel_size_um, channel, z_range
    )
    if not loaded:
        return None
    return _composite_tiles(
        loaded,
        pixel_size_um,
        tile_transform,
        target_long_px,
        label_tiles,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
    )


def _load_tile_mips(
    acquisition_dir: Path,
    pixel_size_um: Optional[float],
    channel: Optional[int],
    z_range: Optional[Tuple[float, float]],
    illum_side: Optional[int] = None,
) -> "Tuple[List[Tuple[float, float, np.ndarray, str]], float]":
    """Read each tile's MIP once: ``([(x_mm, y_mm, mip, label), ...], pixel_um)``."""
    from flamingo_stitcher.pipeline import (  # lazy: avoid import cycle
        discover_flat_tiles,
        discover_tiles,
    )

    acquisition_dir = Path(acquisition_dir)
    tiles = discover_tiles(acquisition_dir)
    if not tiles:
        tiles = discover_flat_tiles(acquisition_dir)
    if not tiles:
        logger.warning("Orientation preview: no tiles in %s", acquisition_dir)
        return [], pixel_size_um or 0.406

    if pixel_size_um is None:
        pixel_size_um = _guess_pixel_size_um(acquisition_dir)

    loaded: List[Tuple[float, float, np.ndarray, str]] = []
    for t in tiles:
        mip = _tile_mip(t, channel, z_range, illum_side)
        if mip is None:
            continue
        loaded.append((float(t.x_mm), float(t.y_mm), mip, _tile_label(t)))
    if not loaded:
        logger.warning("Orientation preview: no MIPs could be read")
    return loaded, float(pixel_size_um)


def _composite_tiles(
    loaded: "List[Tuple[float, float, np.ndarray, str]]",
    pixel_size_um: float,
    tile_transform: Optional["MosaicOrientation"],
    target_long_px: int,
    label_tiles: bool,
    reverse_x: bool = False,
    reverse_y: bool = False,
) -> np.ndarray:
    """Composite tiles at their stage positions, transforming EACH TILE first.

    ``tile_transform`` is applied to every tile's pixels BEFORE placement — this
    reorients each tile's CONTENT so adjacent tiles connect. ``reverse_x`` /
    ``reverse_y`` reverse the tile ORDER along that stage axis (place at the
    mirrored position) WITHOUT touching the tile pixels — the separate degree of
    freedom for a stage whose axis sign is inverted (X3 X2 X1 X0 instead of X0…
    X3). Content orientation and tile order are independent: a system can need a
    per-tile flip in X but an order reversal in Y. Tile labels are drawn upright
    at the tile centres so they stay readable.
    """
    frame_px = max((max(m.shape) for _, _, m, _ in loaded), default=2048)
    fov_mm = frame_px * float(pixel_size_um) / 1000.0

    xs = [x for x, _, _, _ in loaded]
    ys = [y for _, y, _, _ in loaded]
    x_min, x_max = min(xs) - fov_mm / 2, max(xs) + fov_mm / 2
    y_min, y_max = min(ys) - fov_mm / 2, max(ys) + fov_mm / 2
    span_mm = max(x_max - x_min, y_max - y_min, 1e-6)
    mm_per_px = span_mm / float(max(target_long_px, 64))

    W = max(1, int(round((x_max - x_min) / mm_per_px)))
    H = max(1, int(round((y_max - y_min) / mm_per_px)))
    canvas = np.zeros((H, W), dtype=np.float32)

    tile_px = max(1, int(round(fov_mm / mm_per_px)))
    label_spots: List[Tuple[str, int, int]] = []
    for x_mm, y_mm, mip, label in loaded:
        img = mip.astype(np.float32)
        if tile_transform is not None:
            img = tile_transform.apply2d(img)  # per-tile reorientation
        small = _resize(img, tile_px, tile_px)
        # Reverse the tile ORDER along an axis (mirror the placement) without
        # touching the tile pixels — the stage-sign degree of freedom.
        cx = ((x_max - x_mm) if reverse_x else (x_mm - x_min)) / mm_per_px
        cy = ((y_max - y_mm) if reverse_y else (y_mm - y_min)) / mm_per_px
        r0 = int(round(cy - small.shape[0] / 2))
        c0 = int(round(cx - small.shape[1] / 2))
        _blit_max(canvas, small, r0, c0)
        label_spots.append((label, int(round(cy)), int(round(cx))))

    hi = float(np.percentile(canvas, 99.5)) if canvas.any() else 1.0
    if hi <= 0:
        hi = 1.0
    canvas = np.clip(canvas / hi, 0.0, 1.0)

    if label_tiles:
        scale = max(2, tile_px // 40)
        for label, cy, cx in label_spots:
            if label:
                _draw_label(canvas, label, cy, cx, scale)
    return canvas


def orientation_previews(
    acquisition_dir: Path,
    *,
    pixel_size_um: Optional[float] = None,
    target_long_px: int = 1000,
    channel: Optional[int] = None,
    z_range: Optional[Tuple[float, float]] = None,
    label_tiles: bool = True,
    reverse_x: bool = False,
    reverse_y: bool = False,
    illum_side: Optional[int] = None,
) -> "Dict[str, np.ndarray]":
    """Build one mosaic per per-tile orientation (8), under the given tile order.

    Returns ``{orientation_name: mosaic}`` for all eight. Each mosaic reorients
    every tile's pixels by that orientation then re-composites at the stage grid,
    so the panels differ in how tiles CONNECT. ``reverse_x`` / ``reverse_y``
    reverse the tile ORDER along that axis for all panels (the separate stage-
    sign control) — pick the pixel orientation AND the order that together make
    the tissue continuous across the seams. Tiles are read once, composited
    eight times.

    ``channel`` / ``illum_side`` of None fall back to the lowest index present.
    Both used to be unreachable from the GUI, so a dual-sided acquisition was
    always previewed from side 0 with nothing saying so — and orientation is a
    per-side question.
    """
    loaded, px = _load_tile_mips(
        acquisition_dir, pixel_size_um, channel, z_range, illum_side
    )
    if not loaded:
        return {}
    return {
        name: _composite_tiles(
            loaded,
            px,
            MosaicOrientation(name),
            target_long_px,
            label_tiles,
            reverse_x=reverse_x,
            reverse_y=reverse_y,
        )
        for name in MosaicOrientation.NAMES
    }


def _is_full_range(z_range: Optional[Tuple[float, float]]) -> bool:
    return z_range is None or (z_range[0] <= 0.0 and z_range[1] >= 1.0)


def _tile_mip(
    tile,
    channel: Optional[int],
    z_range: Optional[Tuple[float, float]] = None,
    illum_side: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Load a tile's max-projection.

    Full Z (default): use the fast per-tile ``*_MP.tif`` companion. A Z
    sub-range: project just those planes from the raw/TIFF stack, so structure
    confined to part of the stack isn't buried by the whole-stack projection.
    """
    raw = _representative_raw(tile, channel, illum_side)
    if raw is None:
        return None
    if _is_full_range(z_range):
        mip = _read_mip_companion(raw)
        if mip is not None:
            return mip
    projected = _mip_from_stack(raw, tile, z_range)
    if projected is not None:
        return projected
    # A Z sub-range was asked for but the stack is missing/unreadable. The
    # whole-stack companion is the wrong range, but it beats dropping the tile
    # out of the mosaic entirely (which reads as a hole in the preview).
    return _read_mip_companion(raw)


def _read_mip_companion(raw: Path) -> Optional[np.ndarray]:
    """Read the tile's pre-computed ``*_MP.tif``, or None if absent/unreadable."""
    for suffix in ("_MP.tif", "_MP.tiff"):
        mp = raw.with_name(raw.stem + suffix)
        if not mp.is_file():
            continue
        try:
            import tifffile

            return np.asarray(tifffile.imread(str(mp)))
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not read MIP %s: %s", mp.name, e)
    return None


def _representative_raw(
    tile, channel: Optional[int], illum_side: Optional[int] = None
) -> Optional[Path]:
    """The file to preview for this tile.

    ``channel``/``illum_side`` of None fall back to the lowest index, which is
    what this used to do unconditionally — so a dual-sided acquisition was
    always previewed from side 0 with no way to see the other, and no
    indication that a choice had been made. Orientation is a per-side
    question: the two light paths can differ, and picking one silently is how
    a hardcoded assumption survives.
    """
    rf = getattr(tile, "raw_files", None) or {}
    if not rf:
        return None
    if channel is not None and channel in rf:
        by_illum = rf[channel]
    else:
        by_illum = rf[sorted(rf)[0]]
    if not by_illum:
        return None
    if illum_side is not None and illum_side in by_illum:
        return by_illum[illum_side]
    return by_illum[sorted(by_illum)[0]]


def available_channels_and_sides(acquisition_dir: Path):
    """``(channels, sides)`` present in an acquisition, for populating pickers."""
    from flamingo_stitcher.pipeline import discover_flat_tiles, discover_tiles

    tiles = []
    for scan in (discover_tiles, discover_flat_tiles):
        try:
            tiles = scan(Path(acquisition_dir))
        except Exception as e:  # noqa: BLE001 - try the other layout
            logger.debug("%s failed: %s", scan.__name__, e)
            continue
        if tiles:
            break
    channels = sorted({c for t in tiles for c in (getattr(t, "raw_files", {}) or {})})
    sides = sorted(
        {
            s
            for t in tiles
            for m in (getattr(t, "raw_files", {}) or {}).values()
            for s in (m or {})
        }
    )
    return channels, sides


def has_orientation_preview_data(acquisition_dir: Path) -> bool:
    """True if the orientation preview can build from this acquisition.

    The preview needs at least one tile's max-projection — a per-tile
    ``*_MP.tif`` companion (fast path) or a readable raw/TIFF stack to project
    from. Returns False for a metadata-only / unreadable acquisition, where the
    orientation can't be determined and the user needs a dataset that includes
    MIPs for that microscope.
    """
    try:
        from flamingo_stitcher.pipeline import discover_flat_tiles, discover_tiles

        tiles = discover_tiles(acquisition_dir)
        if not tiles:
            tiles = discover_flat_tiles(acquisition_dir)
        for t in tiles:
            raw = _representative_raw(t, None)
            if raw is None:
                continue
            if raw.is_file():
                return True
            if raw.with_name(raw.stem + "_MP.tif").is_file():
                return True
        return False
    except Exception as e:  # noqa: BLE001 - best-effort availability probe
        logger.debug("has_orientation_preview_data failed: %s", e)
        return False


def _plane_bounds(
    n: int, z_range: Optional[Tuple[float, float]]
) -> Tuple[int, int, int]:
    """(z0, z1, step) plane indices for a Z sub-range, capped at ~24 samples."""
    if _is_full_range(z_range) or n <= 0:
        z0, z1 = 0, n
    else:
        lo = min(max(z_range[0], 0.0), 1.0)
        hi = min(max(z_range[1], 0.0), 1.0)
        if hi < lo:
            lo, hi = hi, lo
        z0 = int(lo * n)
        z1 = max(z0 + 1, int(round(hi * n)))
        z1 = min(z1, n)
    step = max(1, (z1 - z0) // 24)  # ≤ ~24 planes read per tile
    return z0, z1, step


def _mip_from_stack(
    raw: Path, tile, z_range: Optional[Tuple[float, float]]
) -> Optional[np.ndarray]:
    """MIP over a (sub-)range of Z planes from a raw/TIFF tile stack."""
    try:
        n = int(getattr(tile, "n_planes", 0) or 0)
        z0, z1, step = _plane_bounds(n, z_range)
        if raw.suffix.lower() == ".raw":
            fw = int(getattr(tile, "frame_width", 2048) or 2048)
            fh = int(getattr(tile, "frame_height", 2048) or 2048)
            if n <= 0 or fw <= 0 or fh <= 0:
                return None
            mm = np.memmap(raw, dtype=np.uint16, mode="r", shape=(n, fh, fw))
            return np.max(mm[z0:z1:step], axis=0)
        # (Big)TIFF: read only the sampled pages in the range.
        import tifffile

        with tifffile.TiffFile(str(raw)) as tf:
            pages = tf.pages
            npages = len(pages)
            if n <= 0 or n > npages:
                z0, z1, step = _plane_bounds(npages, z_range)
            idxs = list(range(z0, min(z1, npages), step))
            if not idxs:
                return None
            stack = np.stack([np.asarray(pages[i].asarray()) for i in idxs])
            return np.max(stack, axis=0)
    except (OSError, ValueError) as e:
        logger.debug("MIP-from-stack failed for %s: %s", raw.name, e)
        return None


def _tile_label(tile) -> str:
    """A short reference label for a tile — its grid index, e.g. ``X0Y0``."""
    idx = getattr(tile, "tile_index", None)
    if idx and len(idx) == 2:
        return f"X{int(idx[0])}Y{int(idx[1])}"
    return ""


# 3×5 bitmap glyphs (rows top→bottom) for the tile-index labels. Kept tiny and
# dependency-free so labels can be burned into the numpy mosaic (and thus rotate
# with it) without Pillow/Qt.
_FONT_3x5 = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
}


def _draw_label(canvas: np.ndarray, text: str, cy: int, cx: int, scale: int) -> None:
    """Burn ``text`` (3×5 bitmap font) centred near (cy, cx) into ``canvas``.

    Draws a dark plate then bright glyph pixels so the label reads at any mosaic
    brightness. Clipped to the canvas; off-canvas labels are skipped.
    """
    scale = max(1, int(scale))
    gw, gh = 3, 5
    adv = gw + 1  # 1-column gap between glyphs
    text = "".join(ch for ch in text if ch in _FONT_3x5)
    if not text:
        return
    total_w = (len(text) * adv - 1) * scale
    total_h = gh * scale
    top = int(cy - total_h / 2)
    left = int(cx - total_w / 2)
    H, W = canvas.shape
    # Dark backing plate for contrast (with a small margin).
    pad = scale
    r0, r1 = max(0, top - pad), min(H, top + total_h + pad)
    c0, c1 = max(0, left - pad), min(W, left + total_w + pad)
    if r0 >= r1 or c0 >= c1:
        return
    canvas[r0:r1, c0:c1] *= 0.15
    x = left
    for ch in text:
        glyph = _FONT_3x5[ch]
        for gy in range(gh):
            row = glyph[gy]
            for gx in range(gw):
                if row[gx] == "1":
                    yy0 = top + gy * scale
                    xx0 = x + gx * scale
                    yy1, xx1 = yy0 + scale, xx0 + scale
                    ay0, ax0 = max(0, yy0), max(0, xx0)
                    ay1, ax1 = min(H, yy1), min(W, xx1)
                    if ay0 < ay1 and ax0 < ax1:
                        canvas[ay0:ay1, ax0:ax1] = 1.0
        x += adv * scale


def _guess_pixel_size_um(acquisition_dir: Path) -> float:
    try:
        from flamingo_stitcher.pipeline import suggested_pixel_size_um

        px = suggested_pixel_size_um(acquisition_dir)
        if px and px > 0:
            return float(px)
    except Exception:  # noqa: BLE001
        pass
    return 0.406  # legacy default (16×/25.68× objective)


def _resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize ``img`` to ``(out_h, out_w)`` so a tile fills its mosaic footprint.

    Block-max downsamples first when shrinking (preserving bright MIP pixels /
    beads), then nearest-neighbour maps to the exact target — which also handles
    upscaling a small MIP up to its footprint.
    """
    out_h, out_w = max(1, int(out_h)), max(1, int(out_w))
    ih, iw = img.shape
    fy, fx = max(1, ih // out_h), max(1, iw // out_w)
    if fy > 1 or fx > 1:
        h, w = (ih // fy) * fy, (iw // fx) * fx
        if h and w:
            img = img[:h, :w].reshape(h // fy, fy, w // fx, fx).max(axis=(1, 3))
            ih, iw = img.shape
    ys = np.minimum(np.arange(out_h) * ih // out_h, ih - 1)
    xs = np.minimum(np.arange(out_w) * iw // out_w, iw - 1)
    return img[np.ix_(ys, xs)]


def _blit_max(canvas: np.ndarray, tile: np.ndarray, r0: int, c0: int) -> None:
    """Composite ``tile`` into ``canvas`` at (r0, c0) with a max blend (clipped)."""
    H, W = canvas.shape
    th, tw = tile.shape
    r1, c1 = r0 + th, c0 + tw
    cr0, cc0 = max(0, r0), max(0, c0)
    cr1, cc1 = min(H, r1), min(W, c1)
    if cr0 >= cr1 or cc0 >= cc1:
        return
    tr0, tc0 = cr0 - r0, cc0 - c0
    tr1, tc1 = tr0 + (cr1 - cr0), tc0 + (cc1 - cc0)
    region = canvas[cr0:cr1, cc0:cc1]
    np.maximum(region, tile[tr0:tr1, tc0:tc1], out=region)


def render_contact_sheet(
    previews: "Dict[str, np.ndarray]", out_path: Path
) -> Optional[Path]:
    """Write a labelled 2×4 contact sheet of the eight orientations (PIL).

    Returns the path, or None if PIL is unavailable (the GUI does not need
    this — it renders the arrays directly).
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001 - PIL is not a hard runtime dep
        logger.info("Pillow not available; skipping contact-sheet render")
        return None

    names = list(MosaicOrientation.NAMES)
    cell = 320
    pad, label_h = 8, 20
    cols, rows = 4, 2
    sheet_w = cols * cell + (cols + 1) * pad
    sheet_h = rows * (cell + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + label_h + pad)
        arr = previews.get(name)
        if arr is not None and arr.size:
            im = Image.fromarray(
                (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            ).convert("RGB")
            im.thumbnail((cell, cell))
            sheet.paste(im, (x + (cell - im.width) // 2, y))
        draw.text((x + 2, y + cell + 4), f"{i + 1}. {name}", fill=(230, 230, 230))
    out_path = Path(out_path)
    sheet.save(str(out_path))
    return out_path
