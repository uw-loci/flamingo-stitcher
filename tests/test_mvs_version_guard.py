"""Guards for the multiview-stitcher version floor and the tile-overlap check.

Background — the bug this exists to prevent from recurring silently:

multiview-stitcher fuses the output one dask chunk at a time. Through 0.1.56 it
did that with a **zero-pixel halo** unless content-based weighting was on
(``fusion.fuse`` set ``overlap_in_pixels = 0``, discarding even an explicitly
passed value). When a tile's translation is not a whole number of output voxels
— the normal case, since stage positions are arbitrary relative to the output
grid — the order-1 interpolation at each chunk boundary had no neighbouring
source sample and returned 0. The fused volume came out with a regular grid of
black one-voxel lines that looks like a tile-seam artifact but actually tracks
the chunk grid.

Reproduced against real releases while diagnosing a 104-tile brain stitch:
0.1.44, 0.1.48, 0.1.49, 0.1.52, 0.1.56 all produce the lines; 0.1.57, 0.1.58 and
0.1.59 do not. It affects ``blend`` and ``max`` fusion alike; content-based
weighting masks it only because it requests a ``2 * sigma_2`` halo.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from flamingo_stitcher.pipeline import (
    MIN_SAFE_MVS_VERSION,
    MIN_USEFUL_TILE_OVERLAP,
    _detect_tile_spacing_gaps,
    _parse_version_tuple,
    check_multiview_stitcher_version,
)


class _Tile:
    """Minimal duck-type for _detect_tile_spacing_gaps (reads x_mm/y_mm only)."""

    def __init__(self, x_mm, y_mm):
        self.x_mm = x_mm
        self.y_mm = y_mm


def _grid(x_step_mm, y_step_mm, n=4):
    return [
        _Tile(round(i * x_step_mm, 6), round(j * y_step_mm, 6))
        for i in range(n)
        for j in range(n)
    ]


class TestParseVersionTuple:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("0.1.57", (0, 1, 57)),
            ("0.1.44", (0, 1, 44)),
            ("1.0", (1, 0)),
            ("0.2.0rc1", (0, 2, 0)),
            # Parsing stops at the first non-numeric component, which is what
            # we want: a post/rc suffix must not change the ordering decision.
            ("0.1.57.post1", (0, 1, 57)),
        ],
    )
    def test_parses_numeric_prefix(self, text, expected):
        assert _parse_version_tuple(text) == expected

    @pytest.mark.parametrize("text", ["", "?", None, "abc", "v"])
    def test_unparseable_returns_none(self, text):
        assert _parse_version_tuple(text) is None


class TestVersionGuard:
    @pytest.mark.parametrize("version", ["0.1.44", "0.1.48", "0.1.52", "0.1.56"])
    def test_known_bad_versions_warn(self, version):
        """Every version verified to produce the black-line artifact."""
        lines = check_multiview_stitcher_version(version)
        assert lines, f"{version} must be flagged"
        assert "TOO OLD" in lines[0]
        assert "0.1.57" in lines[0]

    @pytest.mark.parametrize("version", ["0.1.57", "0.1.58", "0.1.59", "0.2.0", "1.0.0"])
    def test_known_good_versions_are_silent(self, version):
        assert check_multiview_stitcher_version(version) == []

    @pytest.mark.parametrize("version", ["?", "", "not-a-version"])
    def test_unparseable_version_stays_silent(self, version):
        """A false alarm on every run is worse than a missed one."""
        assert check_multiview_stitcher_version(version) == []

    def test_boundary_is_exclusive_below_inclusive_at(self):
        assert check_multiview_stitcher_version("0.1.56") != []
        assert check_multiview_stitcher_version("0.1.57") == []

    def test_installed_version_satisfies_the_floor(self):
        """The env running these tests must itself be safe.

        Without this, the suite could pass on a box that would silently produce
        corrupt stitches.
        """
        assert check_multiview_stitcher_version() == [], (
            "installed multiview-stitcher is below the safe floor; "
            "run: pip install -U 'multiview-stitcher>=0.1.57'"
        )


class TestPinStaysInStepWithCode:
    def test_pyproject_floor_matches_min_safe_constant(self):
        """pyproject and MIN_SAFE_MVS_VERSION must not drift apart.

        They are two halves of one decision: the pin keeps a fresh install
        correct, the constant catches a bundle that was built against something
        older anyway (which is exactly what reached the rig).
        """
        text = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        match = re.search(r'"multiview-stitcher>=([0-9.]+)', text)
        assert match, "could not find the multiview-stitcher pin in pyproject.toml"
        assert _parse_version_tuple(match.group(1)) == MIN_SAFE_MVS_VERSION


class TestNearZeroOverlapWarning:
    """The blind spot: a 0.01% overlap is not a gap, but is not usable either."""

    def test_zero_overlap_is_flagged_on_both_axes(self):
        # Frame 1024 px x 1.0475 um = 1.07264 mm; step it at ~exactly that.
        msgs = _detect_tile_spacing_gaps(_grid(1.07250, 1.07273), 1024, 1024, 1.0475)
        assert len(msgs) == 2, msgs
        assert all("barely overlap" in m for m in msgs)
        assert any(m.startswith("X") for m in msgs)
        assert any(m.startswith("Y") for m in msgs)

    def test_message_names_the_two_real_consequences(self):
        msgs = _detect_tile_spacing_gaps(_grid(1.07250, 1.07250), 1024, 1024, 1.0475)
        joined = " ".join(msgs)
        assert "registration" in joined
        assert "falloff" in joined or "brightness step" in joined

    def test_healthy_overlap_is_silent(self):
        # 10% overlap -> step = 0.9 * coverage.
        step = 1024 * 1.0475 / 1000.0 * 0.90
        assert _detect_tile_spacing_gaps(_grid(step, step), 1024, 1024, 1.0475) == []

    def test_threshold_boundary(self):
        cov = 1024 * 1.0475 / 1000.0
        just_under = cov * (1 - (MIN_USEFUL_TILE_OVERLAP * 0.5))
        just_over = cov * (1 - (MIN_USEFUL_TILE_OVERLAP * 2.0))
        assert _detect_tile_spacing_gaps(_grid(just_under, just_under), 1024, 1024, 1.0475)
        assert (
            _detect_tile_spacing_gaps(_grid(just_over, just_over), 1024, 1024, 1.0475)
            == []
        )

    def test_real_gap_still_reports_as_a_gap_not_as_overlap(self):
        """A genuine gap must keep its own (different, stronger) message."""
        cov = 1024 * 1.0475 / 1000.0
        msgs = _detect_tile_spacing_gaps(_grid(cov * 1.2, cov * 1.2), 1024, 1024, 1.0475)
        assert len(msgs) == 2
        assert all("do not overlap" in m and "blank gaps" in m for m in msgs)
        assert not any("barely overlap" in m for m in msgs)


class TestFusionHasNoBlockBoundaryZeros:
    """End-to-end guard against the artifact itself, not just the version string.

    Fuses tiles whose translations are deliberately NOT whole output voxels and
    asserts no all-zero row/column appears. On an affected multiview-stitcher
    this fails with black lines on the chunk grid; it is the test that would
    have caught the regression regardless of what the version metadata claimed.
    """

    @staticmethod
    def _fuse(pitch, chunk, fusion_mode="blend"):
        from multiview_stitcher import fusion
        from multiview_stitcher import spatial_image_utils as siu

        rng = np.random.default_rng(0)
        sims = [
            siu.get_sim_from_array(
                (rng.random((4, 64, 64)) * 400 + 800).astype(np.uint16),
                dims=["z", "y", "x"],
                scale={"z": 1, "y": 1, "x": 1},
                translation={"z": 0, "y": i * pitch, "x": i * pitch},
                transform_key="stage",
            )
            for i in range(3)
        ]
        kwargs = dict(
            transform_key="stage",
            output_spacing={"z": 1, "y": 1, "x": 1},
            output_chunksize={"z": 4, "y": chunk, "x": chunk},
            blending_widths={"z": 5, "y": 10, "x": 10},
        )
        if fusion_mode == "max":
            kwargs["fusion_func"] = fusion.max_fusion
        vol = np.asarray(fusion.fuse(sims, **kwargs).data.compute()).squeeze()
        plane = vol[vol.shape[0] // 2]
        # Drop the outer volume edge, which is legitimately background.
        return plane[:-1, :-1]

    @pytest.mark.parametrize("fusion_mode", ["blend", "max"])
    @pytest.mark.parametrize("chunk", [32, 64])
    def test_fractional_tile_pitch_leaves_no_black_lines(self, chunk, fusion_mode):
        plane = self._fuse(pitch=57.6, chunk=chunk, fusion_mode=fusion_mode)
        black_cols = np.flatnonzero((plane == 0).all(axis=0))
        black_rows = np.flatnonzero((plane == 0).all(axis=1))
        assert black_cols.size == 0, (
            f"black columns at {black_cols.tolist()} — fusion is writing zeros at "
            f"block boundaries; multiview-stitcher is too old"
        )
        assert black_rows.size == 0, f"black rows at {black_rows.tolist()}"

    def test_integer_pitch_is_clean_too(self):
        """Control: the artifact never occurred for grid-aligned translations."""
        plane = self._fuse(pitch=58.0, chunk=32)
        assert not (plane == 0).all(axis=0).any()
        assert not (plane == 0).all(axis=1).any()


class TestStaleMetadataCannotFakeTheVerdict:
    """The guard must judge the code that RUNS, not leftover packaging.

    On 2026-08-08 a rig running v0.10.0 — an installer verified in CI to bundle
    multiview-stitcher 0.1.59 — logged "multiview-stitcher 0.1.44" and tripped
    the too-old guard. Nothing was rebuilt wrong: Inno Setup's [Files] only
    overwrites, so every old release's versioned *.dist-info directory was
    still sitting in the install dir and importlib.metadata answered with the
    oldest one it found.

    Left alone this guard would cry wolf on healthy installs and, worse, go
    quiet on a genuinely old one whose stale metadata happened to read new.
    """

    def test_stale_old_metadata_does_not_condemn_a_healthy_module(self):
        lines = check_multiview_stitcher_version(
            "0.1.44", module_version="0.1.59", module_path="/app/_internal/mvs.py"
        )
        assert not any("TOO OLD" in ln for ln in lines), (
            "a healthy 0.1.59 module must not be failed for stale metadata"
        )

    def test_the_mismatch_is_still_reported_as_the_real_problem(self):
        lines = check_multiview_stitcher_version(
            "0.1.44", module_version="0.1.59", module_path="/app/_internal/mvs.py"
        )
        assert lines, "a version mismatch must not pass in silence"
        joined = "\n".join(lines)
        assert "0.1.44" in joined and "0.1.59" in joined
        assert "dist-info" in joined, "must name what is actually stale"
        assert "/app/_internal/mvs.py" in joined, "must say which module won"

    def test_stale_new_metadata_cannot_hide_an_old_module(self):
        """The dangerous direction: metadata reads safe, the code is not."""
        lines = check_multiview_stitcher_version(
            "0.1.59", module_version="0.1.44", module_path="/app/_internal/mvs.py"
        )
        assert any("TOO OLD" in ln for ln in lines), (
            "an actually-old module must be flagged however new metadata reads"
        )
        assert any("0.1.44" in ln for ln in lines if "TOO OLD" in ln)

    def test_agreement_stays_silent(self):
        assert (
            check_multiview_stitcher_version("0.1.59", module_version="0.1.59") == []
        )

    def test_agreeing_old_versions_warn_once_not_twice(self):
        lines = check_multiview_stitcher_version("0.1.44", module_version="0.1.44")
        assert lines
        assert not any("dist-info" in ln for ln in lines), (
            "no mismatch to report when both agree"
        )

    def test_an_unreadable_module_falls_back_to_metadata(self):
        """Better to judge on metadata than to skip the check entirely."""
        lines = check_multiview_stitcher_version("0.1.44", module_version="")
        assert any("TOO OLD" in ln for ln in lines)

    def test_explicit_version_is_not_overruled_by_the_live_env(self):
        """Callers naming a version get that version judged, full stop."""
        lines = check_multiview_stitcher_version("0.1.44")
        assert any("TOO OLD" in ln for ln in lines)


class TestInstallerClearsTheOldBundle:
    """The packaging half of the same bug — see the class above."""

    def _iss(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "installer" / "installer.iss"
        if not path.exists():
            pytest.skip("installer.iss not present")
        return path.read_text(encoding="utf-8", errors="replace")

    def test_install_delete_wipes_the_payload_directory(self):
        text = self._iss()
        assert "[InstallDelete]" in text, (
            "without [InstallDelete] every upgrade layers onto the last one "
            "and stale dist-info accumulates forever"
        )
        assert "{app}\\_internal" in text

    def test_the_delete_precedes_the_copy(self):
        """Ordering is the whole point: delete-after would erase the install."""
        text = self._iss()
        assert text.index("[InstallDelete]") < text.index("[Files]")
