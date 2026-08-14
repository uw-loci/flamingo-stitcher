"""Single source of truth for the Flamingo Stitcher version.

Kept in a tiny, dependency-free module (no heavy imports) so that:

  * ``flamingo_stitcher.__init__`` can re-export it without import cost,
  * ``pyproject.toml`` can read it statically via setuptools' dynamic
    ``version = {attr = "flamingo_stitcher._version.__version__"}`` (setuptools
    parses this file's AST without importing numpy/scipy/etc.), and
  * the standalone GUI's update checker (``gui/updater.py``) can compare it
    against the latest GitHub Release tag to tell the user whether they're
    behind.

Release ritual — bump this in lockstep with the pushed tag:
  1. Bump ``__version__`` here (full MAJOR.MINOR.PATCH, no trailing-zero
     truncation, e.g. "0.2.0" not "0.2").
  2. Push the matching tag: ``git tag v0.2.0 && git push origin v0.2.0``.
The release CI verifies the tag's version segment equals this string and
fails fast on a mismatch, so an installer can never ship reporting the wrong
installed version to its own updater.
"""

__version__ = "0.11.0"
