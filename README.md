# Flamingo Stitcher

Standalone light-sheet tile **stitching** pipeline and GUI for Flamingo T-SPIM data.

It turns a folder of raw microscope tiles into a single stitched 3D volume you can
open in Fiji, napari, QuPath, or Imaris. You can run it **on any machine** — no
microscope and no full control software required. The *same code* also powers the
"Tile Stitching" menu inside [Py2Flamingo](https://github.com/MichaelSNelson/Flamingo_Control),
so results match (single source of truth, no drift).

---

## Quick start (Windows, no Python needed)

If you just want to stitch a dataset, this is all you need.

### 1. Install

1. Go to the [Releases](../../releases) page.
2. Download the newest **`FlamingoStitcher-Setup-vX.Y.Z.exe`**.
3. Run it. (It's unsigned for now, so Windows SmartScreen may warn — click
   **More info → Run anyway**.)
4. Launch **Flamingo Stitcher** from the Start menu.

The app updates itself: the **Updates** tab checks for newer versions on launch and
offers a one-click install when one exists.

### 2. Stitch your first dataset

You'll see three tabs: **Multi-Acquisition**, **Single Workflow**, and **Updates**.
Pick the tab that matches your data (see [Which tab?](#which-tab-do-i-use) below — if
you're unsure, just try one). Then:

1. **Add your data.** Click **Add…** and select your acquisition folder. (You can add
   several and stitch them back-to-back.)
2. **Click "Discover Tiles".** The app scans the folder and figures out the details
   for you. A line appears at the top of *"Tell me about your image"* showing what it
   detected — frame/ROI size, pixel size, Z step, channels, and tile count. **Glance at
   it to confirm it looks right.**
3. **Pick an output folder.** Click **Browse…** next to *Output Directory*. Each
   acquisition is saved into its own subfolder there.
4. **Click "Run All".** Progress shows at the bottom. When it finishes, open the
   result (an `..._stitched` folder) in Fiji or napari.

That's it. The default settings are chosen to work for typical Flamingo data — you
usually don't need to change anything in steps 2–3.

### Which tab do I use?

| Your acquisition looks like… | Use this tab |
|---|---|
| One folder full of `.raw` files named like `...X000_Y000...` | **Single Workflow** |
| Each tile in its own subfolder, **or** you want to stitch many acquisitions at once | **Multi-Acquisition** |

Not sure? Pick one and click **Discover Tiles**. If it finds **0 tiles**, switch to the
other tab and try again.

---

## Understanding the settings

The basic controls are organized into three plainly-named boxes. **For a first run you
can leave all of them alone.** Everything more specialized lives in the collapsed
**Processing Options** panel below them.

- **Tell me about your image** — facts about your data: pixel size, Z step, frame
  (camera ROI), and which channels. These are filled in automatically by *Discover
  Tiles*; change one only if you know a detected value is wrong.
- **What kind of processing should we do?** — the choices that affect the output:
  *Downsample* (make the result smaller/faster), how the two light-sheet sides are
  combined, and how overlapping tiles are blended. The defaults are a good start.
- **How should we save it?** — output format, compression, and memory mode. **OME-Zarr
  (Fiji compatible)** is a safe default. The size / time / memory estimates underneath
  update live as you change settings.

> Tip: hover over any control to get a tooltip explaining it. Open the **Log** panel
> (collapsed at the bottom) to see full detail of what was detected and done.

If you hit trouble (out-of-memory, a run that seems stuck, slow stitching), click
**Help / Troubleshooting** in the app, or see
[`stitching_hardware_troubleshooting.md`](src/flamingo_stitcher/docs/stitching_hardware_troubleshooting.md).

---

## For developers / Python users

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

It converts raw acquisition folders into stitched volumes using
[multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher) for
registration and fusion, and writes OME-Zarr, OME-TIFF, or Imaris `.ims`.

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
