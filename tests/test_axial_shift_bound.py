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
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flamingo_stitcher.pipeline import StitchingPipeline


def _bound(vz, planes, *, binning_z=1, configured=0.0):
    p = StitchingPipeline.__new__(StitchingPipeline)
    p.config = SimpleNamespace(
        max_registration_shift_z_um=configured,
        registration_binning={"z": binning_z, "y": 1, "x": 1},
    )
    return p._axial_shift_bound(planes * vz, vz)


# --------------------------------------------------------------------------
# The regression
# --------------------------------------------------------------------------


def test_a_finely_sampled_deep_stack_is_no_longer_punished():
    """The 2026-10-07 run: 2999 planes at 2.5 um."""
    bound, source = _bound(2.5, 2999, binning_z=4)
    assert source == "auto"
    assert bound == pytest.approx(93.7, rel=1e-2)   # 1.25% of 7497 um
    assert bound > 25.0, "the old rule fell through to the 25 um floor here"


def test_it_admits_the_row_to_row_drift_the_rig_actually_measured():
    """The Y-seam Z shifts from the 2026-10-07 8x run, by row pair:

        006->005  45        002->001  65 65 65 70 80 80
        004->003  30        001->000 105 105 140
        003->002  35 40 50

    Six independent columns agreeing on 65-80 um at one row pair is a
    measurement, not a stray correlation peak.
    """
    bound, _ = _bound(2.5, 2999, binning_z=4)
    measured = (30, 35, 40, 45, 50, 65, 70, 80, 105, 140)
    admitted = [z for z in measured if z <= bound]
    assert len(admitted) == 8, f"only {admitted} admitted of {measured}"


def test_the_physical_fraction_matches_what_the_working_run_had():
    """The constant is carried from the 2026-09-18 run (80 um / 6430 um =
    1.24%), which registered these seams fine, rather than picked."""
    bound, _ = _bound(10.0, 643)
    assert bound / (643 * 10.0) == pytest.approx(0.0125, rel=0.02)


def test_it_still_rejects_a_peak_found_far_down_the_stack():
    """Widening is not the same as removing. The point of the bound is a
    correlation peak that landed somewhere unrelated."""
    bound, _ = _bound(2.5, 2999, binning_z=4)
    for absurd in (195, 400, 1000):
        assert absurd > bound


def test_the_coarse_step_run_is_essentially_unchanged():
    """2026-09-18: 643 planes at 10 um had 80.0 um and registered fine.

    It now gets 80.4 um. The two rules nearly coincide there BY CONSTRUCTION —
    the depth fraction was taken from this run's own tolerance — so agreement
    to half a micron is the property worth pinning, not exact equality. A first
    version of this test asserted 80.0 exactly and failed on the 0.4 um, which
    is 0.5% and changes no decision.
    """
    bound, _ = _bound(10.0, 643)
    assert bound == pytest.approx(80.4, abs=0.5)
    assert abs(bound - 80.0) < 1.0, "the run that worked must not move materially"


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
    physical = 0.0125 * depth_um
    for vz in (10.0, 5.0, 2.5, 1.25, 0.5):
        bound, _ = _bound(vz, depth_um / vz)
        assert bound >= physical - 1e-6, (
            f"at a {vz} um step the bound is {bound} um, below the {physical} um "
            f"a {depth_um} um stack can drift regardless of sampling"
        )


def test_quantization_may_make_the_bound_wider_but_never_narrower_than_physical():
    """A coarse step keeps its generous quantization allowance."""
    coarse, _ = _bound(10.0, 600)    # 6000 um: max(80, 25, 75) = 80
    fine, _ = _bound(2.5, 2400)      # 6000 um: max(20, 25, 75) = 75
    assert coarse == pytest.approx(80.0)
    assert fine == pytest.approx(75.0)
    assert min(coarse, fine) >= 0.0125 * 6000.0


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
