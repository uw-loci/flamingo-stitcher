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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
    """``{microscope_name_lower: output_orientation}`` from the hardware YAML.

    Reads the optional ``microscopes:`` block of the bundled
    ``configs/microscope_hardware.yaml``. Never raises — a missing/broken block
    yields an empty registry.
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
        out: Dict[str, str] = {}
        for name, entry in scopes.items():
            if isinstance(entry, dict) and entry.get("output_orientation"):
                ori = str(entry["output_orientation"]).strip().lower()
                if ori in MosaicOrientation.NAMES:
                    out[str(name).strip().lower()] = ori
        return out
    except Exception as e:  # noqa: BLE001 - presets are best-effort
        logger.debug("Could not load orientation presets: %s", e)
        return {}


def preset_for_microscope(name: Optional[str]) -> Optional[str]:
    """Return the ``output_orientation`` for a microscope name, or None."""
    if not name:
        return None
    return load_orientation_presets().get(name.strip().lower())


_NAME_RE = re.compile(r"Microscope name\s*=\s*(.+)")


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
    """Best-effort ``output_orientation`` for an acquisition via microscope name.

    Returns the matching preset name, or None when the microscope name is
    unknown / has no preset (caller keeps the configured/default orientation).
    """
    name = read_microscope_name(acquisition_dir)
    if not name:
        return None
    ori = preset_for_microscope(name)
    if ori:
        logger.info("Output-orientation preset '%s' → %s", name, ori)
    return ori


# --------------------------------------------------------------------------- #
# MIP-mosaic preview
# --------------------------------------------------------------------------- #
def build_mip_mosaic(
    acquisition_dir: Path,
    *,
    pixel_size_um: Optional[float] = None,
    target_long_px: int = 1000,
    channel: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Assemble a fast low-res MIP mosaic (identity orientation) for preview.

    Places each tile's max-projection (the per-tile ``*_MP.tif`` companion, or a
    cheap strided MIP of the raw when absent) at its stage position, so a user
    can judge the mosaic layout. Returns a 2-D ``float32`` array normalised to
    [0, 1], or ``None`` if no tiles / MIPs could be read. This is a *preview*:
    placement is by stage centre and is intentionally approximate.
    """
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
        return None

    if pixel_size_um is None:
        pixel_size_um = _guess_pixel_size_um(acquisition_dir)

    # Per-tile MIPs (native px) + their mm extent.
    loaded: List[Tuple[float, float, np.ndarray]] = []
    frame_px = None
    for t in tiles:
        mip = _tile_mip(t, channel)
        if mip is None:
            continue
        frame_px = max(frame_px or 0, max(mip.shape))
        loaded.append((float(t.x_mm), float(t.y_mm), mip))
    if not loaded:
        logger.warning("Orientation preview: no MIPs could be read")
        return None

    fov_mm = (frame_px or 2048) * float(pixel_size_um) / 1000.0

    xs = [x for x, _, _ in loaded]
    ys = [y for _, y, _ in loaded]
    # mm span of tile CENTRES, padded by half a FOV each side (tiles centred).
    x_min, x_max = min(xs) - fov_mm / 2, max(xs) + fov_mm / 2
    y_min, y_max = min(ys) - fov_mm / 2, max(ys) + fov_mm / 2
    span_mm = max(x_max - x_min, y_max - y_min, 1e-6)
    mm_per_px = span_mm / float(max(target_long_px, 64))

    W = max(1, int(round((x_max - x_min) / mm_per_px)))
    H = max(1, int(round((y_max - y_min) / mm_per_px)))
    canvas = np.zeros((H, W), dtype=np.float32)

    tile_px = max(1, int(round(fov_mm / mm_per_px)))
    for x_mm, y_mm, mip in loaded:
        small = _resize(mip.astype(np.float32), tile_px, tile_px)
        cx = (x_mm - x_min) / mm_per_px
        cy = (y_mm - y_min) / mm_per_px
        r0 = int(round(cy - small.shape[0] / 2))
        c0 = int(round(cx - small.shape[1] / 2))
        _blit_max(canvas, small, r0, c0)

    # Normalise to [0, 1] with a robust upper percentile so a few hot beads
    # don't wash the preview out.
    hi = float(np.percentile(canvas, 99.5)) if canvas.any() else 1.0
    if hi <= 0:
        hi = 1.0
    return np.clip(canvas / hi, 0.0, 1.0)


def orientation_previews(mosaic: np.ndarray) -> "Dict[str, np.ndarray]":
    """Return ``{orientation_name: transformed_mosaic}`` for all eight."""
    return {n: MosaicOrientation(n).apply2d(mosaic) for n in MosaicOrientation.NAMES}


def _tile_mip(tile, channel: Optional[int]) -> Optional[np.ndarray]:
    """Load a tile's max-projection: prefer the ``*_MP.tif`` companion."""
    raw = _representative_raw(tile, channel)
    if raw is None:
        return None
    mp = raw.with_name(raw.stem + "_MP.tif")
    if mp.is_file():
        try:
            import tifffile

            return np.asarray(tifffile.imread(str(mp)))
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not read MIP %s: %s", mp.name, e)
    return _cheap_mip_from_raw(raw, tile)


def _representative_raw(tile, channel: Optional[int]) -> Optional[Path]:
    rf = getattr(tile, "raw_files", None) or {}
    if not rf:
        return None
    if channel is not None and channel in rf:
        by_illum = rf[channel]
    else:
        by_illum = rf[sorted(rf)[0]]
    if not by_illum:
        return None
    return by_illum[sorted(by_illum)[0]]


def _cheap_mip_from_raw(raw: Path, tile) -> Optional[np.ndarray]:
    """A cheap MIP from a raw file: max over a strided subset of Z planes."""
    if raw.suffix.lower() != ".raw":
        return None  # (Big)TIFF tiles: skip the heavy fallback for previews
    try:
        fw = int(getattr(tile, "frame_width", 2048) or 2048)
        fh = int(getattr(tile, "frame_height", 2048) or 2048)
        n = int(getattr(tile, "n_planes", 0) or 0)
        if n <= 0 or fw <= 0 or fh <= 0:
            return None
        mm = np.memmap(raw, dtype=np.uint16, mode="r", shape=(n, fh, fw))
        step = max(1, n // 24)  # sample ~24 planes, not the whole stack
        return np.max(mm[::step], axis=0)
    except OSError as e:
        logger.debug("Cheap MIP failed for %s: %s", raw.name, e)
        return None


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
