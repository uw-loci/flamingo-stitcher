"""Read what an acquisition's metadata files say, from either server generation.

Up to the server build of 2026-10-06, each experiment folder had a
``ScopeSettings.txt`` holding ``Microscope name`` and ``Objective lens
magnification``. From that build on there is no ScopeSettings.txt; the folder
gets ``SystemMetaData.txt`` instead: the microscope's whole system document plus
the first stack's settings, written by ``SystemMetaData::saveExperimentCopy``
on the server.

The document uses the same ``<Block>`` ... ``</Block>`` and ``Name = value``
syntax as the old file, nested, under a ``<MetaData>`` root. The two values the
stitcher needs sit at (server ``MetaDataTemplate.cpp``, and the legacy-key
mapping in ``System/030_LegacySettings/010_RUN_ConvertLegacySettings.py``):

  Microscope name               Instrument/Microscope characteristics/Name
  Objective lens magnification  Instrument/Microscope Hardware/Detection/
                                Detection 1/Detection lens/Magnification

The same build added camera binning. ``Workflow.txt`` (and each stack's
``_Settings.txt`` sidecar) now records ``Bit binning width`` and ``Bit binning
height`` in ``<Camera Settings>``, next to ``ROI width``/``ROI height`` (which
replaced ``AOI width``/``AOI height``). The server writes the ROI from the
camera's own image size (``PCO_GetSizes``), so it is the saved image size in
binned pixels, and each saved pixel covers ``binning`` sensor pixels: the
effective pixel size is ``sensor pitch x binning / magnification``.

What is NOT used: the document's ``Camera/Pixel pitch`` and ``Image pixel
size``. A document converted from the old files starts from the server's
template defaults, which carry example values from another instrument (pitch
4.5 µm, image pixel 0.38 µm) that the conversion does not replace. The camera
pitch therefore still comes from the stitcher's hardware config, and the
document's pitch is only reported when it differs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

SYSTEM_METADATA_FILE = "SystemMetaData.txt"

# Document paths below <MetaData>.
_CHARACTERISTICS = ("Instrument", "Microscope characteristics")
_DETECTION_1 = ("Instrument", "Microscope Hardware", "Detection", "Detection 1")
NAME_PATH = _CHARACTERISTICS + ("Name",)
OBJECTIVE_PATH = _DETECTION_1 + ("Detection lens", "Magnification")
CAMERA_PITCH_PATH = _DETECTION_1 + ("Camera", "Pixel pitch")
# The first stack's camera settings, as the experiment copy records them.
# (MetaDataAcquisitionSettings.cpp: MetaData/Experiment/Microscope Settings.)
_CAMERA_SETTINGS_PATHS = (("Experiment", "Microscope Settings", "Camera Settings"),)

# The name a server gives its document before anyone sets one
# (MetaDataTemplate::defaults). Per-microscope settings keyed on it would be
# shared by every unconfigured instrument.
SERVER_DEFAULT_NAME = "Flamingo"

Node = Dict[str, Union[str, "Node"]]

_OPEN = re.compile(r"^<\s*(?P<name>[^/][^>]*)>$")
_CLOSE = re.compile(r"^</\s*(?P<name>[^>]+)>$")
_VALUE = re.compile(r"^(?P<k>[^=]+?)\s*=\s*(?P<v>.*)$")
_BIN_W = re.compile(r"Bit binning width\s*=\s*(\d+)")
_BIN_H = re.compile(r"Bit binning height\s*=\s*(\d+)")


def parse(text: str) -> Node:
    """The document as nested dicts: blocks are dicts, values are strings."""
    root: Node = {}
    stack: List[Node] = [root]
    for raw in (text or "").splitlines():
        line = raw.strip().replace("\x00", "")
        if not line or line.startswith("#"):
            continue
        m = _CLOSE.match(line)
        if m:
            if len(stack) > 1:
                stack.pop()
            continue
        m = _OPEN.match(line)
        if m:
            block: Node = {}
            stack[-1][m.group("name").strip()] = block
            stack.append(block)
            continue
        m = _VALUE.match(line)
        if m:
            stack[-1][m.group("k").strip()] = m.group("v").strip()
    return root


def get(doc: Node, path: Sequence[str]) -> Optional[str]:
    """A value by path below ``<MetaData>``, or None if absent, empty or a block."""
    node: Union[str, Node, None] = doc.get("MetaData", doc)
    for part in path:
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    if isinstance(node, str) and node.strip():
        return node.strip()
    return None


def _positive_float(text: Optional[str]) -> Optional[float]:
    try:
        value = float(text) if text is not None else None
    except ValueError:
        return None
    return value if value and value > 0 else None


def read_document(path: Path) -> Optional[Node]:
    """Parse a SystemMetaData.txt, or None if it cannot be read."""
    try:
        return parse(Path(path).read_text(errors="replace"))
    except OSError:
        return None


def microscope_name(doc: Node) -> Optional[str]:
    return get(doc, NAME_PATH)


def objective_magnification(doc: Node) -> Optional[float]:
    return _positive_float(get(doc, OBJECTIVE_PATH))


def camera_pixel_pitch_um(doc: Node) -> Optional[float]:
    """The document's camera pixel pitch. Reported, never used (module doc)."""
    return _positive_float(get(doc, CAMERA_PITCH_PATH))


def binning_from_text(text: str) -> Optional[Tuple[int, int]]:
    """``(width, height)`` from ``Bit binning width/height`` lines, or None.

    Works on Workflow.txt, a stack's _Settings.txt and SystemMetaData.txt
    alike, since all three write the same two keys. 0 (a camera that did not
    report it) counts as missing.
    """
    w = _BIN_W.search(text or "")
    h = _BIN_H.search(text or "")
    if not (w and h):
        return None
    bw, bh = int(w.group(1)), int(h.group(1))
    if bw < 1 or bh < 1:
        return None
    return bw, bh


def binning_from_document(doc: Node) -> Optional[Tuple[int, int]]:
    for block in _CAMERA_SETTINGS_PATHS:
        w = get(doc, block + ("Bit binning width",))
        h = get(doc, block + ("Bit binning height",))
        if w and h:
            try:
                bw, bh = int(w), int(h)
            except ValueError:
                continue
            if bw >= 1 and bh >= 1:
                return bw, bh
    return None
