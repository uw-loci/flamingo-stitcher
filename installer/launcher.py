"""PyInstaller entry point for the frozen Flamingo Stitcher GUI.

PyInstaller freezes a *script*, not a console-script entry point, so this thin
launcher just calls the same ``main()`` used by the ``flamingo-stitch-gui``
console script.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    # Required so the isolated-preprocessing subprocess machinery works in a
    # frozen (PyInstaller) build on Windows.
    multiprocessing.freeze_support()
    from flamingo_stitcher.gui.app import main

    sys.exit(main())
