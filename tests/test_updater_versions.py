"""The in-app updater's version comparison, including pre-releases (0.14.0).

0.14.0 is first released as the GitHub pre-release 0.14.0b1, which the updater
never offers (it reads releases/latest, which skips pre-releases). Someone who
installs the beta by hand must still be offered 0.14.0 when it comes out; the
old int-only parser rejected "0.14.0b1" and so reported nothing as newer.
"""

import pytest

from flamingo_stitcher.gui.updater import _version_is_newer


@pytest.mark.parametrize(
    "candidate, installed, newer",
    [
        ("0.14.0", "0.14.0b1", True),
        ("0.14.0b1", "0.13.25", True),
        ("0.13.25", "0.14.0b1", False),
        ("0.14.0b2", "0.14.0b1", True),
        ("0.14.0rc1", "0.14.0b3", True),
        ("0.2.0", "0.1.44", True),
        ("0.1.44", "0.2.0", False),
        ("0.14.0", "0.14.0", False),
        ("garbage", "0.1.0", False),
        ("", "0.1.0", False),
    ],
)
def test_version_order(candidate, installed, newer):
    assert _version_is_newer(candidate, installed) is newer
