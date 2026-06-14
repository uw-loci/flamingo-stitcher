# Flamingo Stitcher

Standalone light-sheet tile **stitching** pipeline and GUI for Flamingo T-SPIM data.

This package is extracted from the [Py2Flamingo](https://github.com/MichaelSNelson/Flamingo_Control)
control application so you can stitch acquisitions **on any machine** — no microscope,
no full control software required — while the *same code* continues to power the
"Tile Stitching" menu option inside Py2Flamingo (single source of truth, no drift).

It converts raw acquisition folders into stitched volumes using
[multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher) for
registration and fusion, and writes OME-Zarr, OME-TIFF, or Imaris `.ims`.

## Install (developers / Python users)

```bash
pip install "flamingo-stitcher[gui]"        # CLI + GUI (PyQt5, no napari)
pip install "flamingo-stitcher[gui,preview]" # + napari background-zero preview
```

Optional backends:

```bash
pip install "flamingo-stitcher[imaris]"    # direct .ims output (Windows only)
pip install pystripe==1.3.1 --no-deps      # destriping (avoid resolver backtracking)
pip install "flamingo-stitcher[deconv]"    # RedLionfish GPU deconvolution (OpenCL)
conda install -c conda-forge pycudadecon   # NVIDIA GPU deconvolution
```

## Windows users (no Python)

Download the latest **`FlamingoStitcher-Setup-vX.Y.Z.exe`** from the
[Releases](../../releases) page and run it. (Unsigned for now — Windows SmartScreen
may warn; choose *More info → Run anyway*.)

**Staying up to date.** The app has an **Updates** tab that checks GitHub
Releases on launch (throttled; opt-out from the same tab). When a newer
installer exists it badges the tab and offers a one-click **Install update** —
it downloads the new `FlamingoStitcher-Setup-*.exe`, launches it, and the
installer upgrades over your current copy in place (no uninstall needed; your
settings are preserved). You can always grab installers manually from the
Releases page instead.

## Usage

GUI:

```bash
flamingo-stitch-gui
```

CLI:

```bash
flamingo-stitch /path/to/acquisition -o /path/to/output \
    --pixel-size-um 0.406 --output-format ome-zarr-sharded
flamingo-stitch /path/to/acquisition --dry-run     # list discovered tiles
flamingo-stitch --help
```

## Dependencies

Core (always installed): `numpy`, `scipy`, `dask[array]`, `zarr`, `numcodecs`,
`multiview-stitcher`, `ngff-zarr`, `tifffile`, `psutil`, `PyYAML`.
The `dask` version range excludes 2025.12.0–2026.3.0 (they break ngff-zarr's
OME-Zarr v0.4 writes; see forum.image.sc topic 120480).

## License

**GNU General Public License v3.0 or later** (GPL-3.0-or-later) — see [`LICENSE`](LICENSE).

The GUI uses **PyQt5** (GPL-3.0), which the Windows installer bundles, making the
distributed binary GPL-3.0. Every other dependency is permissive (BSD-3-Clause /
MIT / Apache-2.0) and GPL-compatible. This project is derived from
[Py2Flamingo](https://github.com/MichaelSNelson/Flamingo_Control) (MIT); see
[`NOTICE`](NOTICE) for full third-party attribution and
[`LICENSE_ANALYSIS_TODO.md`](LICENSE_ANALYSIS_TODO.md) for the analysis (incl. the
PySide6/MIT alternative).
