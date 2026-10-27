"""A single-image map of where registration put every tile, and how well.

Background: each tile's max projection, placed exactly where it was fused.
Over it, semi-transparent:
  * each tile's own area tinted by its Z correction (blue = moved down,
    orange = moved up; colour-blind-safe diverging scale), with its grid label,
    Z shift and how it was placed written in it;
  * each overlap strip filled by what registration made of that seam
    (agreed with the solve, dropped by the solve, below quality, ...).

Pure drawing: no pipeline state. ``render`` takes plain records so it can be
tested without a stitch. Pillow does the text (it ships with the app as a
matplotlib dependency); the image arithmetic is numpy.

Sizing is derived from the text, not fixed: the tile size is chosen so the
widest tile label fits with room to spare, and everything else scales from
that.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MapTile:
    label: str                                  # "X003 Y002"
    bbox_um: Tuple[float, float, float, float]  # world y0, x0, y1, x1 (as fused)
    dz_um: float = 0.0
    how: str = ""                               # "", "carried", "clamped Z", ...
    mip: Optional[np.ndarray] = None            # 2-D, oriented as stitched
    grid: Optional[Tuple[int, int]] = None      # (ix, iy) grid index


@dataclass
class MapSeam:
    a: int
    b: int
    status: str
    quality: Optional[float] = None
    residual_um: Optional[float] = None
    axis: str = ""                              # "x" or "y": which neighbours
    # What this seam measured minus where the placement put its two tiles
    # relative to each other, (dz, dy, dx) µm. The direction of the
    # disagreement, which a single residual number hides.
    disagree_um: Optional[Tuple[float, float, float]] = None


def dominant(disagree) -> Tuple[str, float]:
    """The largest component of a (dz, dy, dx) disagreement: ("y", -52.0)."""
    comps = list(zip(("z", "y", "x"), disagree))
    return max(comps, key=lambda c: abs(c[1]))


@dataclass
class Boundary:
    """One row-to-row (axis "y") or column-to-column (axis "x") join."""

    axis: str
    a: int                                   # lower grid index
    b: int                                   # upper grid index
    kept: int                                # links the solve used
    measured: int                            # links with a measurement
    total: int
    normal_um: List[float] = field(default_factory=list)  # |disagreement| across the join
    z_um: List[float] = field(default_factory=list)

    @property
    def name(self) -> str:
        p = "Y" if self.axis == "y" else "X"
        return f"{p}{self.a:03d} | {p}{self.b:03d}"

    @property
    def weak(self) -> str:
        """"one" when held by at most one link of several, "few" when fewer
        than half the measured links were kept, else ""."""
        if self.measured >= 2 and self.kept <= 1:
            return "one"
        if self.measured >= 2 and self.kept * 2 < self.measured:
            return "few"
        return ""


def boundaries(tiles: Sequence["MapTile"], seams: Sequence["MapSeam"]) -> List[Boundary]:
    out: Dict[Tuple[str, int, int], Boundary] = {}
    for s in seams:
        ga = tiles[s.a].grid if 0 <= s.a < len(tiles) else None
        gb = tiles[s.b].grid if 0 <= s.b < len(tiles) else None
        if not ga or not gb or s.axis not in ("x", "y"):
            continue
        k = 1 if s.axis == "y" else 0
        lo, hi = sorted((ga[k], gb[k]))
        if lo == hi:
            continue
        key = (s.axis, lo, hi)
        b = out.setdefault(key, Boundary(s.axis, lo, hi, 0, 0, 0))
        b.total += 1
        st = (s.status or "").lower()
        if st in ("registered", "pruned", "implausible_shift"):
            b.measured += 1
        if st == "registered":
            b.kept += 1
        if s.disagree_um is not None and st != "registered":
            dz, dy, dx = s.disagree_um
            b.normal_um.append(abs(dy if s.axis == "y" else dx))
            b.z_um.append(abs(dz))
    return sorted(out.values(), key=lambda b: (b.axis != "y", b.a))


# Seam categories, worst last. Okabe-Ito colours (distinguishable with the
# common colour-vision deficiencies). (key, legend text, RGB)
SEAM_CATEGORIES: List[Tuple[str, str, Tuple[int, int, int]]] = [
    ("agrees", "Registered, agrees with the solve", (0, 158, 115)),
    ("strained", "Registered, but left visibly off", (240, 228, 66)),
    ("pruned", "Measured, dropped by the solve", (230, 159, 0)),
    ("below_quality", "Below quality threshold", (213, 94, 0)),
    ("implausible", "Shift beyond the allowed bound", (204, 121, 167)),
    ("empty", "No structure in the overlap", (150, 150, 150)),
    ("not_measured", "Not measured", (90, 90, 90)),
]
_SEAM_RGB = {k: c for k, _, c in SEAM_CATEGORIES}
# Links the solve did not use: marked with a red X on the map.
BROKEN_CATEGORIES = ("pruned", "below_quality", "implausible")

# Diverging Z scale, blue (negative) - near white - orange (positive); the
# blue/orange pair stays distinct with the common colour-vision deficiencies.
# Ends brightened for a dark background: the first version's (33,102,172) and
# (179,88,6) at 30% over dark tissue were hard to see (user, 2026-10-19).
_Z_NEG = np.array([67, 147, 255], float)
_Z_MID = np.array([247, 247, 247], float)
_Z_POS = np.array([255, 128, 14], float)

# Tint opacity grows with |dz|: an unshifted tile stays readable as image, a
# large correction is unmistakable.
TILE_TINT_ALPHA_MIN = 0.12
TILE_TINT_ALPHA_MAX = 0.60
TILE_TINT_ALPHA = TILE_TINT_ALPHA_MAX  # kept for callers reading the old name

# A tile is TILE_PX_PER_LABEL x its widest label line wide, and at least
# MIN_TILE_PX: room for three lines of text with margin, and no more -- the
# map's memory is a few bytes per pixel and the pixel count grows with tiles.
TILE_PX_PER_LABEL = 2.2
MIN_TILE_PX = 200

# Largest mosaic area drawn (pixels, before the legend): ~40 MB of 8-bit RGB.
MAX_MAP_PIXELS = 14_000_000
SEAM_ALPHA = 0.40


def seam_category(
    status: str,
    residual_um: Optional[float],
    tol_um: Optional[float],
    off_limits_um: Optional[Tuple[float, float]] = None,
    disagree_um: Optional[Tuple[float, float, float]] = None,
) -> str:
    """A kept seam is "strained" when the placement leaves it visibly off.

    With ``off_limits_um`` (Z, XY) and the seam's per-axis disagreement, Z and
    XY are judged separately, each against its own limit; otherwise (older
    callers) against the solve tolerance on the 3-D residual.
    """
    s = (status or "").lower()
    if s == "registered":
        if off_limits_um is not None and disagree_um is not None:
            dz, dy, dx = disagree_um
            z_lim, xy_lim = off_limits_um
            if abs(dz) > z_lim or (dy * dy + dx * dx) ** 0.5 > xy_lim:
                return "strained"
            return "agrees"
        if tol_um and residual_um is not None and residual_um > tol_um:
            return "strained"
        return "agrees"
    if s == "pruned":
        return "pruned"
    if s == "below_quality":
        return "below_quality"
    if s == "implausible_shift":
        return "implausible"
    if s in ("empty_overlap", "no_content"):
        return "empty"
    return "not_measured"


def z_scale(values: Sequence[float]) -> Tuple[float, float]:
    """(min, max) of the run's Z corrections, widened to include 0, so blue
    reaches the most negative tile, orange the most positive, white sits at
    no correction, and each side uses its own full range."""
    vals = [float(v) for v in values] or [0.0]
    return min(0.0, min(vals)), max(0.0, max(vals))


def _z_t(dz: float, scale) -> float:
    """Signed position on the scale, -1 at the minimum, +1 at the maximum,
    square-root compressed so a moderate correction is clearly coloured."""
    lo, hi = scale if isinstance(scale, tuple) else (-abs(scale), abs(scale))
    if dz < 0 and lo < 0:
        t = -min(1.0, dz / lo)
    elif dz > 0 and hi > 0:
        t = min(1.0, dz / hi)
    else:
        t = 0.0
    return math.copysign(math.sqrt(abs(t)), t)


def z_colour(dz: float, scale) -> np.ndarray:
    t = _z_t(dz, scale)
    if t < 0:
        return _Z_MID + (-t) * (_Z_NEG - _Z_MID)
    return _Z_MID + t * (_Z_POS - _Z_MID)


def tint_alpha(dz: float, scale) -> float:
    t = abs(_z_t(dz, scale))
    return TILE_TINT_ALPHA_MIN + (TILE_TINT_ALPHA_MAX - TILE_TINT_ALPHA_MIN) * t


def nice_limit(values: Sequence[float]) -> float:
    m = max((abs(v) for v in values), default=0.0)
    if m <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(m))
    for step in (1, 2, 2.5, 5, 10):
        if step * mag >= m:
            return step * mag
    return 10 * mag


def _font(px: int):
    from PIL import ImageFont

    # DejaVu first: it has "µ", which Pillow's built-in scalable font draws as
    # a missing-glyph box. matplotlib ships it, and matplotlib is bundled
    # because multiview-stitcher imports it.
    try:
        import matplotlib

        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        return ImageFont.truetype(str(ttf), px)
    except Exception:  # noqa: BLE001
        pass
    try:
        return ImageFont.load_default(size=px)  # Pillow >= 10.1, scalable
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _text_w(draw, text, font) -> int:
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l


def shrink_for_map(img: np.ndarray, max_px: int = 512) -> np.ndarray:
    """Block-max a projection down to at most ``max_px`` on its long side, so
    a run holds a few hundred KB per tile rather than a full camera frame."""
    img = np.asarray(img, np.float32)
    f = int(math.ceil(max(img.shape) / float(max_px)))
    if f <= 1:
        return img
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).max(axis=(1, 3))


def _resize_max(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Block-max shrink (keeps bright structure), then nearest to exact size."""
    out_h, out_w = max(1, int(out_h)), max(1, int(out_w))
    ih, iw = img.shape
    fy, fx = max(1, ih // out_h), max(1, iw // out_w)
    if fy > 1 or fx > 1:
        h, w = (ih // fy) * fy, (iw // fx) * fx
        img = img[:h, :w].reshape(h // fy, fy, w // fx, fx).max(axis=(1, 3))
        ih, iw = img.shape
    ys = np.minimum(np.arange(out_h) * ih // out_h, ih - 1)
    xs = np.minimum(np.arange(out_w) * iw // out_w, iw - 1)
    return img[np.ix_(ys, xs)]


class _OffsetDraw:
    """ImageDraw with every coordinate shifted by (ox, oy): draws the mosaic in
    its own coordinates straight onto the final canvas."""

    def __init__(self, draw, ox, oy):
        self._d, self._ox, self._oy = draw, ox, oy

    def _pts(self, xy):
        if xy and isinstance(xy[0], (tuple, list)):
            return [(x + self._ox, y + self._oy) for x, y in xy]
        out = list(xy)
        for k in range(0, len(out), 2):
            out[k] += self._ox
            out[k + 1] += self._oy
        return out

    def text(self, xy, *a, **k):
        return self._d.text((xy[0] + self._ox, xy[1] + self._oy), *a, **k)

    def line(self, xy, *a, **k):
        return self._d.line(self._pts(xy), *a, **k)

    def rectangle(self, xy, *a, **k):
        return self._d.rectangle(self._pts(xy), *a, **k)

    def rounded_rectangle(self, xy, *a, **k):
        return self._d.rounded_rectangle(self._pts(xy), *a, **k)

    def textbbox(self, *a, **k):
        return self._d.textbbox(*a, **k)


def tile_text(t: MapTile) -> List[str]:
    lines = [t.label, f"dz {t.dz_um:+.0f} µm"]
    if t.how:
        lines.append(t.how)
    return lines


def render(
    tiles: Sequence[MapTile],
    seams: Sequence[MapSeam],
    out_path: Path,
    *,
    tol_um: Optional[float] = None,
    off_limits_um: Optional[Tuple[float, float]] = None,
    title: str = "",
    info_lines: Sequence[str] = (),
    font_px: int = 16,
    background_note: str = "",
) -> Optional[Path]:
    """Draw the map to ``out_path`` (PNG). Returns the path, or None."""
    try:
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001
        logger.info("Pillow not available; registration map skipped")
        return None
    if not tiles:
        return None

    font = _font(font_px)
    small = _font(max(11, int(font_px * 0.75)))
    big = _font(int(font_px * 1.25))
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    # --- scale: the widest tile text must fit with margin --------------------
    widest = max(_text_w(probe, s, font) for t in tiles for s in tile_text(t))
    tile_px = max(MIN_TILE_PX, int(math.ceil(TILE_PX_PER_LABEL * widest)))
    # Cap the mosaic at ~MAX_MAP_PIXELS: on a very large grid shrink the tiles
    # and the text with them (never below a 12 px font), rather than write a
    # PNG nobody can open.
    n_side = math.sqrt(max(1, len(tiles)))
    if (tile_px * n_side) ** 2 > MAX_MAP_PIXELS:
        shrink = math.sqrt(MAX_MAP_PIXELS) / (tile_px * n_side)
        font_px = max(12, int(font_px * shrink))
        font, small, big = _font(font_px), _font(max(11, int(font_px * 0.75))), _font(int(font_px * 1.25))
        widest = max(_text_w(probe, s, font) for t in tiles for s in tile_text(t))
        tile_px = max(int(TILE_PX_PER_LABEL * widest), int(tile_px * shrink))
    tile_w_um = float(np.median([t.bbox_um[3] - t.bbox_um[1] for t in tiles]))
    if tile_w_um <= 0:
        return None
    px_per_um = tile_px / tile_w_um

    y0 = min(t.bbox_um[0] for t in tiles)
    x0 = min(t.bbox_um[1] for t in tiles)
    y1 = max(t.bbox_um[2] for t in tiles)
    x1 = max(t.bbox_um[3] for t in tiles)
    H = int(math.ceil((y1 - y0) * px_per_um)) + 1
    W = int(math.ceil((x1 - x0) * px_per_um)) + 1

    def to_px(b):
        return (
            int(round((b[0] - y0) * px_per_um)), int(round((b[1] - x0) * px_per_um)),
            int(round((b[2] - y0) * px_per_um)), int(round((b[3] - x0) * px_per_um)),
        )

    boxes = [to_px(t.bbox_um) for t in tiles]

    # Memory is per output PIXEL, and the canvas grows with the tile count (a
    # tile needs ~240 px for its label), so keep it to a few bytes a pixel:
    # one float32 plane while compositing, then 8-bit RGB. Tile tint and seam
    # strips are computed on each tile's own box, never on whole-canvas masks.
    # (A first version held ~36 B/px and full-canvas masks per tile, and the
    # streaming memory-slope test caught it at 3.3 MB per tile.)

    # --- background: max projections, max-composited --------------------------
    gray = np.zeros((H, W), np.float32)
    for t, (r0, c0, r1, c1) in zip(tiles, boxes):
        if t.mip is None or t.mip.size == 0:
            continue
        small_img = _resize_max(np.asarray(t.mip, np.float32), r1 - r0, c1 - c0)
        rr0, cc0 = max(0, r0), max(0, c0)
        rr1, cc1 = min(H, r1), min(W, c1)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        reg = gray[rr0:rr1, cc0:cc1]
        np.maximum(reg, small_img[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0], out=reg)
    # Contrast from every 4th pixel each way: the full-image percentile copied
    # and sorted every non-zero pixel, the largest allocation in the render.
    sample = gray[::4, ::4]
    nz = sample[sample > 0]
    lo_g, hi_g = (np.percentile(nz, [1.0, 99.7]) if nz.size else (0.0, 1.0))
    del nz, sample
    gray -= lo_g
    gray *= 1.0 / max(float(hi_g - lo_g), 1e-6)
    np.clip(gray, 0, 1, out=gray)
    gray **= 0.8
    gray *= 255.0
    g8 = gray.astype(np.uint8)
    del gray

    # The whole output is ONE uint8 array; the mosaic is a view into it, so
    # nothing below holds the image twice.
    pad = int(font_px * 0.8)
    legend_w = max(380, int(font_px * 22))
    header = [title] if title else []
    header_h = pad + len(header) * int(font_px * 1.6) + pad
    n_bounds = len(boundaries(tiles, seams))
    legend_h = int(font_px * (20 + 1.6 * len(SEAM_CATEGORIES)
                              + 1.2 * (len(info_lines) + 3) + 1.2 * n_bounds))
    full = np.full(
        (max(H, legend_h) + header_h + pad, W + legend_w + 3 * pad, 3), 18, np.uint8
    )
    rgb = full[header_h:header_h + H, pad:pad + W]
    rgb[...] = g8[..., None]
    del g8

    def _blend(r0, c0, r1, c1, colour, alpha, mask=None):
        reg = rgb[r0:r1, c0:c1]
        mixed = (reg.astype(np.float32) * (1 - alpha)
                 + np.asarray(colour, np.float32) * alpha).astype(np.uint8)
        if mask is None:
            reg[...] = mixed
        else:
            reg[mask] = mixed[mask]

    # --- tile tint by dz, on each tile's own (nearest-centre) area -----------
    limit = z_scale([t.dz_um for t in tiles])
    centres = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        rr0, cc0, rr1, cc1 = max(0, r0), max(0, c0), min(H, r1), min(W, c1)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        yy = np.arange(rr0, rr1, dtype=np.float32)[:, None]
        xx = np.arange(cc0, cc1, dtype=np.float32)[None, :]
        cy, cx = centres[i]
        own = (yy - cy) ** 2 + (xx - cx) ** 2
        mine = np.ones(own.shape, bool)
        for j, (q0, p0, q1, p1) in enumerate(boxes):
            if j == i or q1 <= rr0 or q0 >= rr1 or p1 <= cc0 or p0 >= cc1:
                continue
            # only where tile j actually covers this tile's box
            covers = np.zeros(own.shape, bool)
            covers[max(q0, rr0) - rr0:min(q1, rr1) - rr0,
                   max(p0, cc0) - cc0:min(p1, cc1) - cc0] = True
            ky, kx = centres[j]
            other = (yy - ky) ** 2 + (xx - kx) ** 2
            mine &= ~(covers & ((other < own) | ((other == own) & (j < i))))
        t = tiles[i]
        _blend(rr0, cc0, rr1, cc1, z_colour(t.dz_um, limit),
               tint_alpha(t.dz_um, limit), mine)

    # --- seam strips ----------------------------------------------------------
    counts: Dict[str, int] = {}
    seam_boxes = []
    for s in seams:
        if not (0 <= s.a < len(boxes) and 0 <= s.b < len(boxes)):
            continue
        a, b = boxes[s.a], boxes[s.b]
        r0, c0 = max(a[0], b[0], 0), max(a[1], b[1], 0)
        r1, c1 = min(a[2], b[2], H), min(a[3], b[3], W)
        if r0 >= r1 or c0 >= c1:
            continue
        cat = seam_category(s.status, s.residual_um, tol_um, off_limits_um, s.disagree_um)
        counts[cat] = counts.get(cat, 0) + 1
        _blend(r0, c0, r1, c1, _SEAM_RGB[cat], SEAM_ALPHA)
        seam_boxes.append((s, cat, (r0, c0, r1, c1)))

    del rgb
    canvas = Image.fromarray(full, "RGB")
    del full
    draw = _OffsetDraw(ImageDraw.Draw(canvas), pad, header_h)
    # (Drawn first so the X marks and numbers sit on top.)
    # Joins held by too few links: a heavy dashed line along the join, so a
    # row hanging on one seam is visible without reading a single number.
    bounds = boundaries(tiles, seams)
    weak_rgb = {"one": (255, 60, 60), "few": (255, 170, 40)}
    for bd in bounds:
        if not bd.weak:
            continue
        strips = [
            box for s, _c, box in seam_boxes
            if s.axis == bd.axis and tiles[s.a].grid and tiles[s.b].grid
            and sorted((tiles[s.a].grid[1 if bd.axis == "y" else 0],
                        tiles[s.b].grid[1 if bd.axis == "y" else 0])) == [bd.a, bd.b]
        ]
        if not strips:
            continue
        col, w = weak_rgb[bd.weak], max(4, font_px // 4)
        if bd.axis == "y":
            y = int(np.mean([(b[0] + b[2]) / 2 for b in strips]))
            xa, xb = min(b[1] for b in strips), max(b[3] for b in strips)
            for x in range(xa, xb, 3 * font_px):
                draw.line([(x, y), (min(x + 2 * font_px, xb), y)], fill=col, width=w)
        else:
            x = int(np.mean([(b[1] + b[3]) / 2 for b in strips]))
            ya, yb = min(b[0] for b in strips), max(b[2] for b in strips)
            for y in range(ya, yb, 3 * font_px):
                draw.line([(x, y), (x, min(y + 2 * font_px, yb))], fill=col, width=w)

    for s, cat, (r0, c0, r1, c1) in seam_boxes:
        draw.rectangle([c0, r0, c1 - 1, r1 - 1], outline=_SEAM_RGB[cat], width=2)
        cy0, cx0 = (r0 + r1) / 2, (c0 + c1) / 2
        broken = cat in BROKEN_CATEGORIES
        if broken:
            # A red X across a link the solve did not use: the tiles either
            # side are NOT held together here.
            h = font_px * 0.7
            for w, col in ((max(6, font_px // 3), (0, 0, 0)), (max(3, font_px // 6), (235, 30, 30))):
                draw.line([(cx0 - h, cy0 - h), (cx0 + h, cy0 + h)], fill=col, width=w)
                draw.line([(cx0 - h, cy0 + h), (cx0 + h, cy0 - h)], fill=col, width=w)
        if s.quality is not None:
            txt = f"{s.quality:.2f}"
            if cat in ("pruned", "strained"):
                if s.disagree_um is not None:
                    ax, v = dominant(s.disagree_um)
                    txt += f" | Δ{ax} {v:+.0f}"
                elif s.residual_um is not None:
                    txt += f" | {s.residual_um:.0f}µm"
            tw = _text_w(draw, txt, small)
            tx, ty = cx0 - tw / 2, cy0 - font_px * 0.4
            if broken:
                # beside the X, not under it
                if (c1 - c0) >= (r1 - r0):
                    tx = cx0 + font_px * 1.0
                else:
                    ty = cy0 + font_px * 0.9
                    tx = cx0 - tw / 2
            draw.text((tx, ty), txt, font=small, fill=(255, 255, 255), stroke_width=2,
                      stroke_fill=(0, 0, 0))
    for t, (r0, c0, r1, c1) in zip(tiles, boxes):
        lines = tile_text(t)
        line_h = int(font_px * 1.45)
        cy = (r0 + r1) / 2 - line_h * len(lines) / 2
        for k, line in enumerate(lines):
            f = big if k == 0 else font
            tw = _text_w(draw, line, f)
            x = (c0 + c1) / 2 - tw / 2
            y = cy + k * line_h
            if k == 1:
                # The dz line sits on a chip of its own colour at full
                # strength, so the colour reads even over bright tissue.
                col = tuple(int(v) for v in z_colour(t.dz_um, limit))
                pad_x, pad_y = font_px * 0.4, font_px * 0.15
                draw.rounded_rectangle(
                    [x - pad_x, y - pad_y, x + tw + pad_x, y + font_px * 1.15 + pad_y],
                    radius=int(font_px * 0.3), fill=col, outline=(0, 0, 0), width=1,
                )
                lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                ink = (0, 0, 0) if lum > 140 else (255, 255, 255)
                draw.text((x, y), line, font=f, fill=ink)
                continue
            draw.text((x, y), line, font=f,
                      fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))

    # --- frame: title above, legend to the right --------------------------------
    d = ImageDraw.Draw(canvas)
    for k, line in enumerate(header):
        d.text((pad, pad + k * int(font_px * 1.6)), line, font=big, fill=(240, 240, 240))

    lx, ly = W + 2 * pad, header_h
    d.text((lx, ly), "Z correction per tile (vs stage)", font=big, fill=(240, 240, 240))
    ly += int(font_px * 1.8)
    bar_w, bar_h = legend_w - 2 * pad, int(font_px * 1.2)
    lo, hi = limit
    span = max(hi - lo, 1e-9)
    for x in range(bar_w):
        dz = lo + span * x / max(bar_w - 1, 1)
        c = tuple(int(v) for v in z_colour(dz, limit))
        d.line([(lx + x, ly), (lx + x, ly + bar_h)], fill=c)
    ly += bar_h + 4
    ticks = [(lo, f"min {lo:+.0f} µm" if lo else "0")]
    if lo < 0 < hi:
        ticks.append((0.0, "0"))
    ticks.append((hi, f"max {hi:+.0f} µm" if hi else "0"))
    for val, txt in ticks:
        frac = (val - lo) / span
        tw = _text_w(d, txt, small)
        xpos = lx + frac * (bar_w - 1)
        d.line([(xpos, ly - 4), (xpos, ly - 1)], fill=(220, 220, 220))
        d.text((min(max(lx, xpos - tw / 2), lx + bar_w - tw), ly), txt,
               font=small, fill=(220, 220, 220))
    ly += int(font_px * 1.6)
    d.text((lx, ly), "tint {:.0%} at 0 to {:.0%} at the scale ends".format(
        TILE_TINT_ALPHA_MIN, TILE_TINT_ALPHA_MAX), font=small, fill=(170, 170, 170))
    ly += int(font_px * 2.0)

    d.text((lx, ly), "Overlap (seam)", font=big, fill=(240, 240, 240))
    ly += int(font_px * 1.8)
    sw = int(font_px * 1.2)
    for key, text, col in SEAM_CATEGORIES:
        n = counts.get(key, 0)
        d.rectangle([lx, ly, lx + sw, ly + sw], fill=col, outline=(0, 0, 0))
        d.text((lx + sw + pad // 2, ly + 1), f"{text} ({n})", font=small,
               fill=(230, 230, 230) if n else (130, 130, 130))
        ly += int(sw * 1.35)
    ly += pad
    sx = lx + sw // 2
    for w, col in ((5, (0, 0, 0)), (3, (235, 30, 30))):
        d.line([(sx - 7, ly + 2), (sx + 7, ly + 16)], fill=col, width=w)
        d.line([(sx - 7, ly + 16), (sx + 7, ly + 2)], fill=col, width=w)
    d.text((lx + sw + pad // 2, ly + 1), "Red X: link NOT used by the solve", font=small,
           fill=(230, 230, 230))
    ly += int(sw * 1.6)
    for col, text in (((255, 60, 60), "Dashed: row/column held by ONE link"),
                      ((255, 170, 40), "Dashed: held by under half its links")):
        yy = ly + sw // 2
        for x0 in (lx, lx + sw // 2 + 2):
            d.line([(x0, yy), (x0 + sw // 2 - 2, yy)], fill=col, width=4)
        d.text((lx + sw + pad // 2, ly + 1), text, font=small, fill=(230, 230, 230))
        ly += int(sw * 1.4)
    ly += int(sw * 0.3)
    d.text((lx, ly), "On a seam: quality | largest disagreement", font=small, fill=(170, 170, 170))
    ly += int(font_px * 1.2)
    d.text((lx, ly), "(measured minus placed, µm)", font=small, fill=(170, 170, 170))
    ly += int(font_px * 1.6)
    for axis, heading, nrm in (("y", "Row joins", "Δy"), ("x", "Column joins", "Δx")):
        rows = [b for b in bounds if b.axis == axis]
        if not rows:
            continue
        d.text((lx, ly), f"{heading} (kept / measured)", font=big, fill=(240, 240, 240))
        ly += int(font_px * 1.6)
        for bd in rows:
            rng = (
                f"{nrm} {min(bd.normal_um):.0f}-{max(bd.normal_um):.0f}"
                if bd.normal_um else ""
            )
            colr = {"one": (255, 90, 90), "few": (255, 180, 70)}.get(bd.weak, (200, 200, 200))
            mark = {"one": "  held by ONE link", "few": "  held by few"}.get(bd.weak, "")
            d.text((lx, ly), f"{bd.name}: {bd.kept}/{bd.measured}  {rng}{mark}",
                   font=small, fill=colr)
            ly += int(font_px * 1.15)
        ly += int(font_px * 0.6)
    ly += int(font_px * 1.3)
    if off_limits_um is not None:
        d.text((lx, ly), f"Visibly off: Z > {off_limits_um[0]:.1f} µm or XY > "
               f"{off_limits_um[1]:.1f} µm", font=small, fill=(170, 170, 170))
        ly += int(font_px * 1.3)
    if tol_um:
        d.text((lx, ly), f"Solve tolerance {tol_um:.1f} µm", font=small, fill=(170, 170, 170))
        ly += int(font_px * 1.3)
    ly += pad
    for line in list(info_lines) + ([background_note] if background_note else []):
        d.text((lx, ly), line, font=small, fill=(200, 200, 200))
        ly += int(font_px * 1.2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    return out_path
