"""Pytest bootstrap: put the package `src/` on sys.path so tests can
`import flamingo_stitcher` without an editable install.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
