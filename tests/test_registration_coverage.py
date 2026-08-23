"""The guard that refuses a registration whose seams do not connect the mosaic.

The failure this exists for is not a crash: it is a run that reports
"Registration complete", writes a plausible-looking report, and produces output
visibly worse than not registering at all. It happened on a real 4x7
acquisition where the sample occupied the middle two columns — 8 of 45 seams
registered, all inside the sample, leaving 18 tiles connected to nothing. The
sample block slid ~190 px past the background tiles that stayed at their stage
position.

The mechanism is in multiview-stitcher, not in us: ``param_resolution`` loops
over ``nx.connected_components`` and resolves each independently, giving an
edgeless component the identity transform. Two components are each internally
consistent and mutually free-floating.
"""

from types import SimpleNamespace

from flamingo_stitcher import registration_report as rr


def seam(a, b, status, axis="x"):
    return rr.SeamResult(
        tile_a=f"t{a}", tile_b=f"t{b}", index_a=a, index_b=b, axis=axis, status=status
    )


def grid_seams(cols, rows, registered):
    """Every adjacent pair in a cols x rows grid, only `registered` succeeding."""
    registered = {tuple(sorted(e)) for e in registered}
    out = []
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        for j, axis in ((i + 1, "x") if c < cols - 1 else (None, None), (
            i + cols, "y"
        ) if r < rows - 1 else (None, None)):
            if j is None:
                continue
            status = (
                rr.STATUS_REGISTERED
                if tuple(sorted((i, j))) in registered
                else rr.STATUS_BELOW_QUALITY
            )
            out.append(seam(i, j, status, axis))
    return out


# The real failing run, reduced to its graph.
REAL_RUN_REGISTERED = [
    (9, 10), (13, 14), (14, 15), (17, 18), (18, 19), (21, 22), (9, 13), (14, 18)
]


class TestCoverage:
    def test_the_real_failing_run_is_refused(self):
        cov = rr.mosaic_coverage(28, grid_seams(4, 7, REAL_RUN_REGISTERED))
        assert not cov.is_safe_to_apply
        assert cov.n_registered_seams == 8
        assert cov.n_isolated == 18
        assert cov.largest_component == 8

    def test_a_fully_registered_grid_is_applied(self):
        every_pair = [(s.index_a, s.index_b) for s in grid_seams(3, 3, [])]
        cov = rr.mosaic_coverage(9, grid_seams(3, 3, every_pair))
        assert cov.is_safe_to_apply
        assert cov.spans_mosaic
        assert not cov.unconstrained

    def test_a_spanning_tree_is_applied_but_flagged_as_redundancy_free(self):
        # 4 tiles in a line: 3 seams span them with no second path anywhere.
        seams = [
            seam(0, 1, rr.STATUS_REGISTERED),
            seam(1, 2, rr.STATUS_REGISTERED),
            seam(2, 3, rr.STATUS_REGISTERED),
        ]
        cov = rr.mosaic_coverage(4, seams)
        assert cov.is_safe_to_apply
        assert cov.is_tree

    def test_islands_that_share_no_seam_are_safe(self):
        # Two 2-tile islands, no expected seam between them: sliding one past
        # the other tears nothing, so refusing would be wrong.
        seams = [seam(0, 1, rr.STATUS_REGISTERED), seam(2, 3, rr.STATUS_REGISTERED)]
        cov = rr.mosaic_coverage(4, seams)
        assert not cov.spans_mosaic
        assert cov.is_safe_to_apply

    def test_one_adjacent_pair_across_two_groups_is_enough_to_refuse(self):
        # Same two islands, but now they are neighbours. That single expected
        # seam is unmeasured and would tear.
        seams = [
            seam(0, 1, rr.STATUS_REGISTERED),
            seam(2, 3, rr.STATUS_REGISTERED),
            seam(1, 2, rr.STATUS_BELOW_QUALITY),
        ]
        cov = rr.mosaic_coverage(4, seams)
        assert not cov.is_safe_to_apply
        assert cov.unconstrained == [(1, 2)]

    def test_a_single_isolated_tile_with_neighbours_is_refused(self):
        seams = [
            seam(0, 1, rr.STATUS_REGISTERED),
            seam(1, 2, rr.STATUS_REGISTERED),
            seam(2, 3, rr.STATUS_BELOW_QUALITY),
        ]
        cov = rr.mosaic_coverage(4, seams)
        assert not cov.is_safe_to_apply
        assert cov.n_isolated == 1

    def test_pruned_and_below_quality_seams_do_not_connect(self):
        seams = [
            seam(0, 1, rr.STATUS_PRUNED),
            seam(1, 2, rr.STATUS_BELOW_QUALITY),
            seam(2, 3, rr.STATUS_DROPPED),
        ]
        cov = rr.mosaic_coverage(4, seams)
        assert cov.n_registered_seams == 0
        assert not cov.is_safe_to_apply

    def test_no_tiles_is_not_safe_and_does_not_raise(self):
        cov = rr.mosaic_coverage(0, [])
        assert not cov.is_safe_to_apply
        assert cov.describe() == "no tiles"

    def test_out_of_range_indices_are_ignored(self):
        cov = rr.mosaic_coverage(2, [seam(0, 99, rr.STATUS_REGISTERED)])
        assert cov.n_registered_seams == 0


class TestRefusedReport:
    """A refused run must still say what it measured, and must not claim the
    proposed shifts were applied."""

    def _report(self):
        tiles = [
            SimpleNamespace(
                x_mm=0.0, y_mm=0.0, z_min_mm=0.0, folder=None, tile_index=i
            )
            for i in range(2)
        ]
        seams = [seam(0, 1, rr.STATUS_BELOW_QUALITY)]
        return rr.build_report(
            tiles=tiles,
            params=[None, None],
            voxel_size_um={"z": 1.0, "y": 1.0, "x": 1.0},
            seams=seams,
            applied=False,
            reason="the registered seams do not connect the mosaic",
        )

    def test_ran_but_not_applied_is_a_distinct_state(self):
        report = self._report()
        assert report.ran is True
        assert report.applied is False

    def test_the_text_report_says_it_was_not_applied_and_why(self):
        text = rr.format_report_text(self._report())
        assert "NOT APPLIED" in text
        assert "do not connect the mosaic" in text

    def test_a_refused_tile_reports_zero_shift_not_the_proposal(self):
        import numpy as np

        tiles = [
            SimpleNamespace(
                x_mm=0.0, y_mm=0.0, z_min_mm=0.0, folder=None, tile_index=i
            )
            for i in range(1)
        ]
        param = np.eye(4)
        param[:3, 3] = [11.0, 22.0, 33.0]
        rows = rr.build_tile_shifts(
            tiles=tiles,
            params=[param],
            voxel_size_um={"z": 1.0, "y": 1.0, "x": 1.0},
            applied=False,
        )
        assert (rows[0].dz_um, rows[0].dy_um, rows[0].dx_um) == (0.0, 0.0, 0.0)
        assert rows[0].shift_before_clamp_um == (11.0, 22.0, 33.0)


# ---------------------------------------------------------------------------
# The pipeline's two-part decision: is it trustworthy, and is it safe to apply?
# ---------------------------------------------------------------------------

pytest = __import__("pytest")
pytest.importorskip("multiview_stitcher")
from multiview_stitcher import param_utils  # noqa: E402
import numpy as np  # noqa: E402

from flamingo_stitcher.pipeline import (  # noqa: E402
    StitchingConfig,
    StitchingPipeline,
)


def _params(translations):
    return [
        param_utils.affine_from_translation(np.asarray(t, float))
        for t in translations
    ]


def _trans(param):
    arr = np.asarray(param)
    if arr.ndim == 3:
        arr = arr[0]
    return arr[:3, 3]


class TestBindUnconstrainedTiles:
    """A tile with no registered seam must move WITH the mosaic, not be left
    behind it. Being left behind is what opened 90-194 px steps on the real
    run: the correction is only a tear if the neighbours do not share it."""

    def _coverage(self, n, registered):
        return rr.mosaic_coverage(n, grid_seams(2, n // 2, registered))

    def test_a_loose_tile_takes_the_mosaics_consensus_shift(self):
        pipe = StitchingPipeline(StitchingConfig())
        # 6 tiles, 2 wide. Tiles 0-3 register together; 4 and 5 register with
        # nothing, so MVS hands them identity while 0-3 move ~-100 µm in x.
        cov = self._coverage(6, [(0, 1), (2, 3), (0, 2), (1, 3)])
        params = _params(
            [(0, 0, -100), (0, 0, -102), (0, 0, -98), (0, 0, -100), (0, 0, 0), (0, 0, 0)]
        )
        out, bound = pipe._bind_unconstrained_tiles(params, cov)
        assert sorted(bound) == [4, 5]
        assert _trans(out[4])[2] == pytest.approx(-100.0)
        assert _trans(out[5])[2] == pytest.approx(-100.0)

    def test_the_tear_it_removes_is_the_whole_correction(self):
        pipe = StitchingPipeline(StitchingConfig())
        cov = self._coverage(6, [(0, 1), (2, 3), (0, 2), (1, 3)])
        params = _params(
            [(0, 0, -100), (0, 0, -100), (0, 0, -100), (0, 0, -100), (0, 0, 0), (0, 0, 0)]
        )
        tear_before = abs(_trans(params[2])[2] - _trans(params[4])[2])
        out, _ = pipe._bind_unconstrained_tiles(params, cov)
        tear_after = abs(_trans(out[2])[2] - _trans(out[4])[2])
        assert tear_before == pytest.approx(100.0)
        assert tear_after == pytest.approx(0.0)

    def test_registered_tiles_keep_their_own_measured_shift(self):
        pipe = StitchingPipeline(StitchingConfig())
        cov = self._coverage(6, [(0, 1), (2, 3), (0, 2), (1, 3)])
        params = _params(
            [(0, 0, -100), (0, 0, -102), (0, 0, -98), (0, 0, -104), (0, 0, 0), (0, 0, 0)]
        )
        out, _ = pipe._bind_unconstrained_tiles(params, cov)
        for i in range(4):
            assert _trans(out[i])[2] == pytest.approx(_trans(params[i])[2])

    def test_a_safe_mosaic_is_left_completely_alone(self):
        pipe = StitchingPipeline(StitchingConfig())
        every = [(s.index_a, s.index_b) for s in grid_seams(2, 2, [])]
        cov = rr.mosaic_coverage(4, grid_seams(2, 2, every))
        params = _params([(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)])
        out, bound = pipe._bind_unconstrained_tiles(params, cov)
        assert bound == []
        assert out is params

    def test_a_second_real_group_keeps_its_own_internal_alignment(self):
        """The bug this replaced: flattening every non-dominant tile onto one
        number destroys the alignment a second group actually measured.

        BigStitcher states the rule — align components relative to each other
        from metadata, "while keeping the results from the first round within a
        component". On the real run, tiles 21 and 22 formed a group with a
        measured dx of 68.8 um between them; flattening set both to the same
        value and threw that measurement away.
        """
        pipe = StitchingPipeline(StitchingConfig())
        # 0-3 are the dominant group; 4 and 5 form a second REAL group with a
        # measured 20 um between them.
        cov = rr.mosaic_coverage(
            6, grid_seams(2, 3, [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5)])
        )
        params = _params(
            [(0, 0, -100), (0, 0, -100), (0, 0, -100), (0, 0, -100),
             (0, 0, 10), (0, 0, 30)]
        )
        out, bound = pipe._bind_unconstrained_tiles(params, cov)
        assert sorted(bound) == [4, 5]
        # Their 20 um internal separation survives -- this is the point.
        internal = _trans(out[5])[2] - _trans(out[4])[2]
        assert internal == pytest.approx(20.0)
        # ...and the group as a whole has moved onto the mosaic. (_median_shift
        # takes the UPPER median on an even count, so the group's median tile
        # lands exactly on the anchor and the other sits 20 um off it.)
        assert _trans(out[4])[2] == pytest.approx(-120.0)
        assert _trans(out[5])[2] == pytest.approx(-100.0)

    def test_the_group_lands_centred_on_the_mosaics_placement(self):
        pipe = StitchingPipeline(StitchingConfig())
        cov = rr.mosaic_coverage(
            6, grid_seams(2, 3, [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5)])
        )
        params = _params(
            [(0, 0, -100)] * 4 + [(0, 0, 10), (0, 0, 30)]
        )
        out, _ = pipe._bind_unconstrained_tiles(params, cov)
        # Its median moves onto the dominant group's median; the dominant group
        # itself is untouched.
        assert sorted([_trans(out[4])[2], _trans(out[5])[2]])[1] == pytest.approx(-100.0)
        for i in range(4):
            assert _trans(out[i])[2] == pytest.approx(-100.0)

    def test_it_declines_when_the_dominant_group_is_too_small_to_have_a_consensus(self):
        # 2 tiles cannot vote on where the mosaic is; binding 4 others to them
        # would be guessing, not consensus.
        pipe = StitchingPipeline(StitchingConfig())
        cov = self._coverage(6, [(0, 1)])
        params = _params([(0, 0, -100), (0, 0, -100)] + [(0, 0, 0)] * 4)
        _out, bound = pipe._bind_unconstrained_tiles(params, cov)
        assert bound == []


class TestTrustThreshold:
    """Binding makes a partly-registered mosaic safe. It cannot make it right."""

    def _cov(self, registered):
        return rr.mosaic_coverage(28, grid_seams(4, 7, registered))

    def test_the_real_failing_run_is_not_trusted(self):
        pipe = StitchingPipeline(StitchingConfig())
        reason = pipe._untrustworthy_registration_reason(
            self._cov(REAL_RUN_REGISTERED)
        )
        assert reason is not None
        assert "8 of 45" in reason

    def test_a_fully_registered_mosaic_is_trusted(self):
        pipe = StitchingPipeline(StitchingConfig())
        every = [(s.index_a, s.index_b) for s in grid_seams(4, 7, [])]
        assert pipe._untrustworthy_registration_reason(self._cov(every)) is None

    def test_the_threshold_is_configurable_not_baked_in(self):
        # The same 8/45 run is accepted once the scope's threshold says a
        # sparse sample is expected here.
        cov = self._cov(REAL_RUN_REGISTERED)
        strict = StitchingPipeline(StitchingConfig())
        lenient = StitchingPipeline(
            StitchingConfig(min_registered_seam_frac=0.1)
        )
        assert strict._untrustworthy_registration_reason(cov) is not None
        assert lenient._untrustworthy_registration_reason(cov) is None

    def test_zero_disables_the_check(self):
        pipe = StitchingPipeline(StitchingConfig(min_registered_seam_frac=0.0))
        assert (
            pipe._untrustworthy_registration_reason(self._cov(REAL_RUN_REGISTERED))
            is None
        )

    def test_the_reason_tells_the_user_what_to_do_about_it(self):
        pipe = StitchingPipeline(StitchingConfig())
        reason = pipe._untrustworthy_registration_reason(
            self._cov(REAL_RUN_REGISTERED)
        )
        # It must name the knob, not just report a number. "Registration was
        # refused" with no next step is how a guard gets switched off wholesale.
        assert "Options tab" in reason
        assert "restrict the run to the tiles that contain it" in reason


class TestZeroMeansDisabled:
    """0.0 on a threshold means "do not check this". `x or DEFAULT` silently
    turns that back into the default, which is how a disabled gate keeps
    firing."""

    def test_zero_overlap_fraction_disables_the_overlap_gate(self):
        pipe = StitchingPipeline(
            StitchingConfig(min_registration_overlap_frac=0.0)
        )
        tiles = [
            SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=0.0),
            # pitch 99 µm against a 100 µm frame: 1% overlap, far below 5%.
            SimpleNamespace(x_mm=0.099, y_mm=0.0, z_min_mm=0.0),
        ]
        extent = {"x": 100.0, "y": 100.0, "z": 10.0}
        assert pipe._registration_overlap_gate(tiles, extent) is None

    def test_the_default_still_gates_that_same_mosaic(self):
        pipe = StitchingPipeline(StitchingConfig())
        tiles = [
            SimpleNamespace(x_mm=0.0, y_mm=0.0, z_min_mm=0.0),
            SimpleNamespace(x_mm=0.099, y_mm=0.0, z_min_mm=0.0),
        ]
        extent = {"x": 100.0, "y": 100.0, "z": 10.0}
        assert pipe._registration_overlap_gate(tiles, extent) is not None


class TestZSearchRangeMatchesTheBound:
    """Searching past the clamp's Z bound cannot produce anything that survives.

    On the real run the search was +/-40 um against a 25 um bound, and 14 of 28
    corrections came back "at the search limit" -- reported as failed
    measurements when they were really the settings asking for something the
    clamp would refuse to use.
    """

    def _summary(self, requested, bound):
        from types import SimpleNamespace

        pipe = StitchingPipeline(
            StitchingConfig(
                registration_z_refine=False,          # stop after the summary
                registration_z_refine_range_um=requested,
            )
        )
        clamp = SimpleNamespace(bound_z_um=bound)
        _params, summary = pipe._refine_z_shifts(
            [], [], [], {"z": 1.0, "y": 1.0, "x": 1.0}, None, clamp
        )
        return summary

    def test_a_range_beyond_the_bound_is_reduced_to_it(self):
        assert self._summary(40.0, 25.0).search_range_um == pytest.approx(25.0)

    def test_a_range_inside_the_bound_is_left_alone(self):
        assert self._summary(15.0, 25.0).search_range_um == pytest.approx(15.0)

    def test_no_bound_leaves_the_request_untouched(self):
        assert self._summary(40.0, None).search_range_um == pytest.approx(40.0)
