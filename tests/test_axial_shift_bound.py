"""The Z shift bound must scale with stack DEPTH, not with the Z step.

Stage/focus drift is physical — micrometres — and accumulates with how deep the
stack is. Sizing the bound only from the voxel ("8 steps") made it TIGHTER the
more finely Z was sampled, which is backwards.

Measured on two rig runs of the same instrument:
  * 2026-09-18, 10 um step, 6430 um deep  -> bound 80 um = 1.24% of depth
  * 2026-10-07,  2.5 um step, 7497 um deep -> bound 25 um = 0.33% of depth
Sampling four times more finely bought a tolerance 3.7x tighter, and 23 of 70
seams were rejected as "implausible shift" — twelve at quality >= 0.85, with
shifts in a smooth 30-195 um spread that is what real drift looks like.

2026-10-15: the fraction went from 1.25% to 2.5%. The same three Y-seams between
the last two rows (105/105/140 um on 10-07, 100/110/140 um on 10-14) were
rejected in every stitch of the 2026-10-06 acquisition; a shift that repeats
across three independent stitches is a measurement.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flamingo_stitcher.pipeline import _Z_CLAMP_DEPTH_FRACTION, StitchingPipeline


def _bound(vz, planes, *, binning_z=1, configured=0.0):
    p = StitchingPipeline.__new__(StitchingPipeline)
    # binning_z is the plane factor registration actually uses; expressed as
    # the physical spacing setting that produces it (0 = every plane).
    p.config = SimpleNamespace(
        max_registration_shift_z_um=configured,
        registration_binning={"z": binning_z, "y": 1, "x": 1},
        registration_z_spacing_um=0.0 if binning_z <= 1 else binning_z * vz,
    )
    return p._axial_shift_bound(planes * vz, vz)


# --------------------------------------------------------------------------
# The regression
# --------------------------------------------------------------------------


def test_a_finely_sampled_deep_stack_is_no_longer_punished():
    """The 2026-10-07 run: 2999 planes at 2.5 um."""
    bound, source = _bound(2.5, 2999, binning_z=4)
    assert source == "auto"
    assert bound == pytest.approx(187.4, rel=1e-2)   # 2.5% of 7497 um
    assert bound > 25.0, "the old rule fell through to the 25 um floor here"


def test_it_admits_the_row_to_row_drift_the_rig_actually_measured():
    """The Y-seam Z shifts from the 2026-10-07 8x run, by row pair:

        006->005  45        002->001  65 65 65 70 80 80
        004->003  30        001->000 105 105 140
        003->002  35 40 50

    Six independent columns agreeing on 65-80 um at one row pair is a
    measurement, not a stray correlation peak. So is 001->000: the 2026-10-14
    stitch of the same acquisition measured it again at 100/110/140 um. An
    earlier version of this test asserted that only 8 of these 10 got through,
    which pinned the rejection of exactly that row.
    """
    bound, _ = _bound(2.5, 2999, binning_z=4)
    measured = (30, 35, 40, 45, 50, 65, 70, 80, 100, 105, 110, 140)
    admitted = [z for z in measured if z <= bound]
    assert admitted == list(measured), f"only {admitted} admitted of {measured}"


def test_the_fraction_has_headroom_over_the_largest_repeatable_shift():
    """140 um, measured in two independent stitches of one 7497 um stack, is
    1.87% of depth. The fraction is set from that, with ~1.3x headroom, rather
    than picked."""
    largest_repeatable = 140.0 / 7497.0
    assert _Z_CLAMP_DEPTH_FRACTION >= 1.25 * largest_repeatable
    assert _Z_CLAMP_DEPTH_FRACTION <= 2.0 * largest_repeatable, (
        "headroom is for measurement scatter, not for switching the guard off"
    )


def test_it_still_rejects_a_peak_found_far_down_the_stack():
    """Widening is not the same as removing. The point of the bound is a
    correlation peak that landed somewhere unrelated."""
    bound, _ = _bound(2.5, 2999, binning_z=4)
    for absurd in (200, 400, 1000):
        assert absurd > bound


def test_no_earlier_run_gets_a_narrower_bound_than_it_had():
    """2026-09-18: 643 planes at 10 um had 80 um. Widening the fraction may
    only admit more of what was measured there, never less."""
    bound, _ = _bound(10.0, 643)
    assert bound >= 80.0
    assert bound == pytest.approx(_Z_CLAMP_DEPTH_FRACTION * 6430.0, rel=1e-2)


# --------------------------------------------------------------------------
# The rule itself
# --------------------------------------------------------------------------


def test_the_physical_floor_holds_however_finely_z_is_sampled():
    """The invariant that was violated.

    A first draft of this test asserted the bound never NARROWS as Z is sampled
    more finely, and it failed: at a 10 um step the quantization term (8 steps =
    80 um) exceeds 1% of a 6000 um stack, so going to a 5 um step drops the
    bound to 60. That is not the bug. Both terms are legitimate floors and the
    rule takes the wider; what matters is that the PHYSICAL one is always
    honoured, because drift does not care how finely the stack was sampled.
    """
    depth_um = 6000.0
    physical = _Z_CLAMP_DEPTH_FRACTION * depth_um
    for vz in (10.0, 5.0, 2.5, 1.25, 0.5):
        bound, _ = _bound(vz, depth_um / vz)
        assert bound >= physical - 1e-6, (
            f"at a {vz} um step the bound is {bound} um, below the {physical} um "
            f"a {depth_um} um stack can drift regardless of sampling"
        )


def test_quantization_may_make_the_bound_wider_but_never_narrower_than_physical():
    """A coarse step keeps its generous quantization allowance."""
    # 6000 um at 10 um: max(8 steps = 80, floor 25, depth term)
    coarse, _ = _bound(10.0, 600)
    fine, _ = _bound(2.5, 2400)      # same depth, 8 steps = 20
    physical = _Z_CLAMP_DEPTH_FRACTION * 6000.0
    assert coarse == pytest.approx(max(80.0, physical))
    assert fine == pytest.approx(max(25.0, physical))
    assert min(coarse, fine) >= physical


def test_a_deeper_stack_gets_a_wider_bound():
    shallow, _ = _bound(2.5, 200)
    deep, _ = _bound(2.5, 4000)
    assert deep > shallow


def test_a_shallow_stack_keeps_the_absolute_floor():
    """1% of a thin stack is a meaningless tolerance; the floor still rules."""
    bound, _ = _bound(2.5, 128)      # 320 um deep, 1% = 3.2 um
    assert bound == pytest.approx(25.0)


def test_the_bound_never_exceeds_a_quarter_of_the_stack():
    bound, _ = _bound(50.0, 20)      # 1000 um deep; 8 steps would be 400
    assert bound <= 0.25 * 1000 + 1e-6


def test_binned_quantization_still_sets_a_floor():
    """Pass 1 can only express multiples of the binned step; a bound below one
    would clamp pure quantization and report it as a measurement."""
    bound, _ = _bound(10.0, 60, binning_z=8)   # floor = 2*8*10 = 160
    assert bound >= 160.0


def test_an_explicit_setting_still_wins():
    bound, source = _bound(2.5, 2999, configured=12.0)
    assert source == "config"
    assert bound == pytest.approx(12.0)


def test_an_explicit_setting_cannot_go_below_quantization():
    bound, source = _bound(10.0, 600, binning_z=4, configured=1.0)
    assert source == "config"
    assert bound == pytest.approx(80.0)   # 2 * 4 * 10
