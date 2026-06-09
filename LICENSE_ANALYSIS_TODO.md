# License analysis — findings & decision

Status: **analysis complete**, one decision required (PyQt5 → PySide). Goal was the
**most permissive license possible**.

## Resolution (TL;DR)

- **Origin is permissive.** `flamingo-stitcher` is derived from Py2Flamingo /
  Flamingo_Control, which is **MIT** (© 2023 MichaelSNelson). MIT permits
  relicensing/sublicensing, so the only obligation from the origin is to
  **preserve that MIT notice** (done — see `NOTICE`). No relicense obstacle.
- **One copyleft dependency: PyQt5 (GPL-3.0).** Every other dependency is
  permissive (BSD-3-Clause / MIT / Apache-2.0), including the ones that mattered
  most: multiview-stitcher (BSD-3), ngff-zarr (MIT), pystripe (MIT), RedLionfish
  (Apache-2.0), PyImarisWriter (Apache-2.0), napari/vispy/numpy/scipy/dask/
  tifffile (BSD-3), zarr/numcodecs/PyYAML (MIT).
- **The frozen Windows `.exe` BUNDLES PyQt5**, so the distributed binary is a
  combined work → it is **GPL-3.0** unless the GUI uses LGPL Qt instead.

## The decision: PyQt5 vs PySide6

| Path | Top-level license achievable | Work required |
|---|---|---|
| **Keep PyQt5** (GPL-3.0) | The distributed installer must be **GPL-3.0**. | none |
| **Migrate GUI to PySide6** (LGPL-3.0) | **MIT** (matches origin + all other deps) — the permissive goal. | mechanical port + test |

PySide6 is the official Qt for Python (LGPL-3.0). LGPL allows distributing a
binary that *dynamically links* Qt while keeping your own code permissive,
provided users can relink/replace the Qt libraries — satisfied by PyInstaller
(Qt ships as separate DLLs) plus the installer's reinstall path.

### PySide6 migration scope (mostly mechanical)
- `from PyQt5.X import ...` → `from PySide6.X import ...` (QtCore/QtGui/QtWidgets)
- `pyqtSignal` → `Signal`, `pyqtSlot` → `Slot`
- `dialog.exec_()` → `dialog.exec()`; `QApplication.exec_()` → `exec()`
- Scoped-enum strictness (e.g. `Qt.AlignLeft` → `Qt.AlignmentFlag.AlignLeft`) in spots
- `QSettings`, `QByteArray`, geometry save/restore: API-compatible
- Files touched: `gui/stitching_dialog.py`, `gui/background_zero_preview_dialog.py`,
  `gui/_compat.py`, `gui/app.py`, and `worker.py` (QtCore only). napari preview
  works on PySide6 too.
- Also update the in-app side: `py2flamingo` itself uses PyQt5 — note this does
  not have to migrate for `flamingo-stitcher` to be MIT (the embedded dialog runs
  under whatever Qt the host provides), but the *standalone frozen exe* must ship
  PySide6 to be MIT.

## Decision (2026-06-09): keep PyQt5 → GPL-3.0

Chosen: **keep PyQt5, license the project GPL-3.0-or-later** (no code migration).
The PySide6 → MIT path remains documented above as a future option.

## Deliverables produced
- [x] `LICENSE` — verbatim GNU GPL v3.0 text.
- [x] `pyproject.toml` — `license = "GPL-3.0-or-later"` + GPLv3 classifier.
- [x] `README.md` — license section updated.
- [x] `NOTICE` — origin MIT attribution + third-party component licenses + PyQt5 note.
- [x] This findings doc.

### Note on the host app (Flamingo_Control)
Flamingo_Control is MIT and now declares a dependency on this GPL-3.0 package.
This does not change Flamingo_Control's own source license, and it introduces no
new obligation beyond its **pre-existing** direct PyQt5 (GPL-3.0) dependency —
the app already combined with GPL Qt at install/run time. The combined,
distributed application is effectively GPL-3.0, as it already was.
