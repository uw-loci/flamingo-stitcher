"""The illumination geometry measurement.

`split` and `blend` keep one sheet per half of the frame; which sheet owns
which half was, before this, a number typed into Options that nothing checked.
Getting it backwards keeps the FAR half of each sheet and is silent in the
output, so these tests pin the measurement that catches it.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from flamingo_stitcher import illumination_geometry as ig


NZ, NY, NX = 20, 192, 192


def _sample(seed=0):
    rng = np.random.default_rng(seed)
    v = gaussian_filter(rng.normal(0, 1, (NZ, NY, NX)), (1.0, 3, 3))
    v -= v.min()
    v /= v.max()
    return ((v**2) * 3000 + 200).astype(np.float32)


def _sheets(tile, axis=-1, low_side=1, atten=2.2, blur=5.0):
    """Two light sheets: `low_side` enters at the low end of `axis`."""
    n = tile.shape[axis]
    high_side = 1 - low_side
    shape = [1] * tile.ndim
    shape[axis] = n
    out = {}
    for side, near in ((low_side, 0), (high_side, n - 1)):
        d = (np.abs(np.arange(n) - near) / (n - 1)).reshape(shape)
        blurred = gaussian_filter(tile, (0, blur, blur))
        v = (1 - d) * tile + d * blurred
        out[side] = (v * (0.25 + 0.75 * np.exp(-atten * d))).astype(np.float32)
    return out


@pytest.mark.parametrize("low_side", [0, 1])
def test_it_finds_the_sheet_that_lights_the_low_end(low_side):
    g = ig.measure(_sheets(_sample(), low_side=low_side), axis=-1)
    assert g.confident
    assert g.low_side == low_side


def test_it_works_on_the_y_axis_too():
    g = ig.measure(_sheets(_sample(), axis=-2, low_side=0), axis=-2)
    assert g.confident and g.low_side == 0


def test_it_still_works_when_the_sheets_are_equally_bright():
    """The answer must come from focus, not brightness.

    A flat-field correction can remove the illumination envelope entirely, and
    a measure that was really reading intensity would then have nothing left.
    """
    sheets = _sheets(_sample(), low_side=1, atten=0.0)
    levels = [float(v.mean()) for v in sheets.values()]
    assert abs(levels[0] - levels[1]) / max(levels) < 0.01, "test setup"
    g = ig.measure(sheets, axis=-1)
    assert g.confident and g.low_side == 1


def test_it_refuses_to_answer_on_featureless_medium():
    """An empty rim tile must produce `None`, not a coin flip.

    Answering confidently from noise would be worse than not answering: the
    pipeline stops the run on a confident disagreement.
    """
    rng = np.random.default_rng(3)
    flat = rng.normal(300, 8, (NZ, NY, NX)).astype(np.float32)
    g = ig.measure({0: flat, 1: flat.copy()}, axis=-1)
    assert g.low_side is None
    assert not g.confident
    assert "undetermined" in g.describe()


def test_a_sample_gradient_does_not_decide_the_answer():
    """Real samples are sharper at one end. That must cancel.

    The tilt is measured per sheet and then differenced, so a sample that is
    genuinely more detailed on one side biases both sheets the same way.
    """
    tile = _sample()
    # Blur one end of the SAMPLE itself, independent of illumination.
    ramp = (np.arange(NX) / (NX - 1)).reshape(1, 1, NX)
    tile = ((1 - ramp) * tile + ramp * gaussian_filter(tile, (0, 4, 4))).astype(
        np.float32
    )
    for low_side in (0, 1):
        g = ig.measure(_sheets(tile, low_side=low_side), axis=-1)
        assert g.confident and g.low_side == low_side


def test_two_sides_are_required():
    one = {0: _sample()}
    assert ig.measure(one, axis=-1) is None
    three = {0: _sample(), 1: _sample(1), 2: _sample(2)}
    assert ig.measure(three, axis=-1) is None


def test_describe_names_the_axis_and_the_side():
    g = ig.measure(_sheets(_sample(), low_side=1), axis=-1)
    text = g.describe()
    assert "side 1" in text and "LOW end" in text and "axis -1" in text
