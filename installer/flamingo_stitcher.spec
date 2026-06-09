# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone Flamingo Stitcher GUI (Windows).

Build (from the repo root):
    pyinstaller installer/flamingo_stitcher.spec

Produces a one-folder bundle at ``dist/FlamingoStitcher/`` containing
``FlamingoStitcher.exe``. The Inno Setup script (installer/installer.iss)
wraps that folder into a double-click installer.

napari/vispy/matplotlib are deliberately EXCLUDED — the GUI does not need them
(the optional background-zero preview is disabled when napari is absent).
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = []
binaries = []
hiddenimports = []

# The scientific stack ships data files, compiled extensions, and lazily
# imported submodules that PyInstaller's static analysis misses. collect_all
# pulls submodules + data + dylibs for each.
for pkg in (
    "dask",
    "zarr",
    "numcodecs",
    "ngff_zarr",
    "multiview_stitcher",
    "tifffile",
    "scipy",
    "skimage",          # pulled in by multiview-stitcher / dask-image
    "dask_image",
    "xarray",
    "spatial_image",
    "multiscale_spatial_image",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # a missing optional pkg should not break the build
        print(f"[spec] collect_all({pkg!r}) skipped: {exc}")

# Bundle the vendored config YAMLs so config_loader finds them at runtime.
datas += collect_data_files("flamingo_stitcher", includes=["configs/*.yaml"])
# App icon + preprocessing-env setup scripts.
datas += [
    ("src/flamingo_stitcher/gui/flamingo_icon.png", "flamingo_stitcher/gui"),
    ("scripts/create_preprocessing_env.bat", "scripts"),
    ("scripts/create_preprocessing_env.sh", "scripts"),
]

# PyImarisWriter (optional, Windows-only) — include if installed.
try:
    d, b, h = collect_all("PyImarisWriter")
    datas += d
    binaries += b
    hiddenimports += h
except Exception:
    pass

block_cipher = None

a = Analysis(
    ["installer/launcher.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "napari",
        "vispy",
        "matplotlib",
        "IPython",
        "tkinter",
        "pytest",
        "pyqtgraph",
        "notebook",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FlamingoStitcher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app — no console window
    icon="src/flamingo_stitcher/gui/flamingo_icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FlamingoStitcher",
)
