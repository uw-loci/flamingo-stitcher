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

**Pending** — see the license analysis. Note: the GUI currently uses **PyQt5**
(GPLv3-or-commercial), which affects the distributable license. Third-party
attributions will live in `NOTICE`.
