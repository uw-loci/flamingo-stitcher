"""Flamingo Stitcher — standalone light-sheet tile stitching for Flamingo T-SPIM data.

Converts raw acquisition folders into stitched volumes using
multiview-stitcher for registration and fusion.

Pipeline:
    Raw uint16 → [dual-illum fusion] → [depth attenuation] → [destripe]
    → [deconvolution] → register → stitch → output (OME-Zarr / OME-TIFF / .ims)

This package was extracted from the Py2Flamingo control application so that
stitching can be installed and run standalone (CLI or GUI) on a machine with
no microscope, while the same code continues to power the in-app menu option.

Public API:
    from flamingo_stitcher import StitchingConfig, StitchingPipeline, discover_tiles

Optional dependencies (install separately if needed):
    pip install "flamingo-stitcher[gui]"      # PyQt5 GUI (no napari)
    pip install "flamingo-stitcher[preview]"  # napari background-zero preview
    pip install "flamingo-stitcher[imaris]"   # direct .ims output (Windows)
    pip install "flamingo-stitcher[destripe]" # pystripe destriping
    pip install "flamingo-stitcher[deconv]"   # RedLionfish GPU deconvolution
    conda install -c conda-forge pycudadecon  # NVIDIA GPU deconvolution
"""

from ._version import __version__
from .pipeline import StitchingConfig, StitchingPipeline, discover_tiles

__all__ = ["StitchingConfig", "StitchingPipeline", "discover_tiles", "__version__"]
