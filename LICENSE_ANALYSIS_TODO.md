# License analysis TODO (must complete before first public Release)

Goal: choose the **most permissive license possible** for `flamingo-stitcher`,
consistent with what the code is **derived from** and what the frozen Windows
executable **bundles**. Produce: (1) a final `LICENSE`, (2) a `NOTICE` file
aggregating third-party attributions, (3) a decision on the PyQt5 constraint.

Recommended tool: the `license-check` skill.

## 1. Derived-from (carry-forward obligations)
- [ ] Confirm the license of the upstream **Py2Flamingo / Flamingo_Control** repo
      (this code — pipeline + dialog — is derived from it). The new repo's license
      must be compatible with / no more permissive than what that repo allows.

## 2. The blocking constraint: PyQt5 (GUI)
- [ ] **PyQt5 is GPLv3-or-commercial.** Bundling it in a *distributed* frozen
      `.exe` makes the whole distribution effectively **GPLv3** unless we either:
      - (a) migrate the GUI to **PySide6 / PySide2 (LGPL)** — permits a permissive
        top-level license with dynamically-linked Qt; OR
      - (b) obtain a commercial Qt/PyQt license.
- [ ] Decision required: this single choice determines whether "most permissive"
      (BSD-3 / MIT / Apache-2.0) is achievable, or whether we ship GPLv3.
      - Migration cost estimate: the dialog uses `PyQt5.QtWidgets/QtCore/QtGui`
        + `pyqtSignal`/`QSettings`. A PySide port is mostly mechanical
        (`pyqtSignal`→`Signal`, import swaps) but must be tested.
      - The **CLI-only** install path has no Qt and is unaffected — its license
        could be permissive regardless.

## 3. Bundled runtime deps — audit licenses + collect into NOTICE
Core (always bundled in the exe):
- [ ] numpy (BSD-3) · scipy (BSD-3) · dask (BSD-3) · zarr (MIT) · numcodecs (MIT)
- [ ] tifffile (BSD-3) · PyYAML (MIT) · psutil (BSD-3)
- [ ] **multiview-stitcher** — VERIFY license (core registration/fusion dependency)
- [ ] **ngff-zarr** — VERIFY license
Optional / not necessarily bundled:
- [ ] PyImarisWriter — Apache-2.0 (Windows-only)
- [ ] pystripe — VERIFY
- [ ] RedLionfish / pycudadecon — VERIFY (GPU deconvolution)
- [ ] napari (BSD-3), vispy (BSD-3) — only the optional `preview` extra
- [ ] Isolated-env (NOT bundled, but document): basicpy, leonardo-toolset
      (pull jax / torch / open3d — note their licenses for completeness)

## 4. Deliverables
- [ ] Replace `LICENSE` placeholder with the chosen license text.
- [ ] Write `NOTICE` aggregating all third-party license notices.
- [ ] One-line record of the PyQt5 → (PySide | GPLv3 | commercial) decision.
