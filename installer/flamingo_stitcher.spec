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

import os
import importlib.metadata

from PyInstaller.utils.hooks import collect_all, collect_data_files

# PyInstaller resolves relative paths in a spec relative to the spec's own
# directory (SPECPATH), NOT the invocation CWD. Anchor everything on the repo
# root (the parent of installer/) so paths work regardless of where the build
# is launched from.
ROOT = os.path.dirname(os.path.abspath(SPECPATH))

datas = []
binaries = []
hiddenimports = []

# The scientific stack ships data files, compiled extensions, and lazily
# imported submodules that PyInstaller's static analysis misses. collect_all
# pulls submodules + data + dylibs for each.
collect_pkgs = [
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
    # ngff-zarr / multiscale-spatial-image generate image pyramids via
    # itkwasm-downsample, which runs a WebAssembly module through the wasmtime
    # runtime. wasmtime ships a native `_wasmtime.dll` (under
    # wasmtime/<platform>/) and the itkwasm-*-wasi packages ship `.wasi`/`.wasm`
    # payloads — all loaded by explicit path at runtime, so PyInstaller's static
    # analysis misses them unless we collect the packages wholesale. Without this
    # a stitch dies at the registration step with
    #   "Failed to load dynlib/dll '...\\wasmtime\\win32-x86_64\\_wasmtime.dll'".
    "wasmtime",
    "itkwasm",
    # certifi ships cacert.pem; the GUI updater's HTTPS call to api.github.com
    # needs it because the frozen build has no system CA store.
    "certifi",
    # multiview-stitcher's registration.py unconditionally imports vis_utils,
    # which imports matplotlib (+ mpl_toolkits) at module load. Excluding
    # matplotlib therefore makes `import multiview_stitcher.registration` raise
    # ImportError and breaks every stitch at Step 3. collect_all pulls in its
    # mpl-data (fonts/styles) too. We never plot — it's an import-time dep only.
    "matplotlib",
]
# Pull in every installed itkwasm* pipeline package (itkwasm-downsample,
# itkwasm-downsample-wasi, etc.) so their wasm payloads ship too.
for dist in importlib.metadata.distributions():
    dist_name = (dist.metadata["Name"] or "").replace("-", "_").lower()
    if dist_name.startswith("itkwasm") and dist_name not in collect_pkgs:
        collect_pkgs.append(dist_name)

for pkg in collect_pkgs:
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
    (os.path.join(ROOT, "src/flamingo_stitcher/gui/flamingo_icon.png"), "flamingo_stitcher/gui"),
    (os.path.join(ROOT, "scripts/create_preprocessing_env.bat"), "scripts"),
    (os.path.join(ROOT, "scripts/create_preprocessing_env.sh"), "scripts"),
]

# PyImarisWriter (optional, Windows-only) — include if installed.
try:
    d, b, h = collect_all("PyImarisWriter")
    datas += d
    binaries += b
    hiddenimports += h
except Exception:
    pass

a = Analysis(
    [os.path.join(ROOT, "installer", "launcher.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "napari",
        "vispy",
        # NOTE: matplotlib is NOT excluded — multiview-stitcher imports it at
        # module load (registration.py -> vis_utils). It is collected above.
        "IPython",
        "tkinter",
        "pytest",
        "pyqtgraph",
        "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

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
    icon=os.path.join(ROOT, "src/flamingo_stitcher/gui/flamingo_icon.ico"),
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
