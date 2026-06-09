"""PyQt5 GUI for Flamingo Stitcher.

Exposes the two stitching dialogs (multi-acquisition and single-workflow) and
the standalone application entry point. napari is *not* required here — only
the optional background-zero preview uses it, and it degrades gracefully.
"""

from .stitching_dialog import NativeStitchingDialog, StitchingDialog

__all__ = ["StitchingDialog", "NativeStitchingDialog"]
