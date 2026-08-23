"""Per-microscope, per-objective stitching options.

The values these carry are judgements about an instrument, not constants — how
much of a mosaic must register before the result is believed, how far the stage
can plausibly be wrong. The point of the store is that tuning one rig cannot
silently retune another, and that a corrupt or hand-edited file degrades to the
defaults instead of feeding a nonsense number to the pipeline.
"""

from __future__ import annotations

import json

import pytest

from flamingo_stitcher import scope_profiles as sp


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Never touch the developer's real ~/.flamingo_stitcher."""
    store = tmp_path / "scope_profiles.json"
    monkeypatch.setattr(sp, "profiles_path", lambda: store)
    return store


class TestKeys:
    def test_objective_normalises_to_one_decimal(self):
        assert sp.normalize_objective(17.0) == "17.0x"
        assert sp.normalize_objective(17.000) == "17.0x"
        assert sp.normalize_objective("17x") == "17.0x"

    def test_a_missing_or_nonsense_objective_becomes_the_wildcard(self):
        for value in (None, "", "unknown", 0, -4):
            assert sp.normalize_objective(value) == sp.ANY_OBJECTIVE

    def test_scope_names_are_case_and_space_insensitive(self):
        assert sp.profile_key("  Liara ", 17) == sp.profile_key("liara", 17.0)

    def test_no_microscope_name_means_no_key(self):
        assert sp.profile_key(None, 17) == ""
        assert sp.profile_key("   ", 17) == ""


class TestRoundTrip:
    def test_save_then_load(self):
        assert sp.save_profile("Liara", 17.0, {"min_registered_seam_frac": 0.2})
        values, source = sp.load_profile("Liara", 17.0)
        assert values == {"min_registered_seam_frac": 0.2}
        assert source == "liara|17.0x"

    def test_a_scope_wide_entry_serves_any_objective(self):
        sp.save_profile("Liara", None, {"quality_threshold": 0.3})
        values, source = sp.load_profile("Liara", 25.0)
        assert values == {"quality_threshold": 0.3}
        assert source == "liara|*"

    def test_an_objective_entry_wins_over_the_scope_wide_one(self):
        sp.save_profile("Liara", None, {"quality_threshold": 0.3})
        sp.save_profile("Liara", 17.0, {"quality_threshold": 0.5})
        assert sp.load_profile("Liara", 17.0)[0] == {"quality_threshold": 0.5}
        assert sp.load_profile("Liara", 25.0)[0] == {"quality_threshold": 0.3}

    def test_the_two_entries_are_not_merged(self):
        # A half-inherited profile is the hardest kind to explain from a report.
        sp.save_profile("Liara", None, {"quality_threshold": 0.3})
        sp.save_profile("Liara", 17.0, {"min_registered_seam_frac": 0.2})
        values, _ = sp.load_profile("Liara", 17.0)
        assert values == {"min_registered_seam_frac": 0.2}
        assert "quality_threshold" not in values

    def test_tuning_one_scope_does_not_touch_another(self):
        sp.save_profile("Liara", 17.0, {"quality_threshold": 0.9})
        assert sp.load_profile("Bruce", 17.0) == ({}, "")

    def test_saving_without_a_microscope_name_fails_rather_than_writing_junk(self):
        assert sp.save_profile(None, 17.0, {"quality_threshold": 0.5}) is False

    def test_delete(self):
        sp.save_profile("Liara", 17.0, {"quality_threshold": 0.5})
        assert sp.delete_profile("Liara", 17.0)
        assert sp.load_profile("Liara", 17.0) == ({}, "")
        assert sp.delete_profile("Liara", 17.0) is False


class TestWhitelist:
    def test_unknown_fields_are_dropped(self):
        sp.save_profile("Liara", 17.0, {"quality_threshold": 0.5, "output_format": "tiff"})
        values, _ = sp.load_profile("Liara", 17.0)
        assert values == {"quality_threshold": 0.5}

    def test_an_out_of_range_value_is_dropped_not_clamped(self, isolated_store):
        # Hand-edited to something impossible. Silently clamping would hide it;
        # using it would hand the pipeline a nonsense threshold.
        isolated_store.parent.mkdir(parents=True, exist_ok=True)
        isolated_store.write_text(
            json.dumps({"version": 1, "profiles": {"liara|17.0x": {
                "quality_threshold": 45.0, "min_registered_seam_frac": 0.2
            }}})
        )
        values, _ = sp.load_profile("Liara", 17.0)
        assert values == {"min_registered_seam_frac": 0.2}

    def test_a_corrupt_file_reads_as_no_profiles(self, isolated_store):
        isolated_store.parent.mkdir(parents=True, exist_ok=True)
        isolated_store.write_text("{not json")
        assert sp.load_profile("Liara", 17.0) == ({}, "")
        assert sp.list_profiles() == {}

    def test_a_missing_file_reads_as_no_profiles(self):
        assert sp.load_profile("Liara", 17.0) == ({}, "")


class TestApply:
    def test_only_the_fields_the_profile_carries_are_touched(self):
        from flamingo_stitcher.pipeline import StitchingConfig

        config = StitchingConfig()
        before_quality = config.quality_threshold
        changed = sp.apply_profile(config, {"min_registered_seam_frac": 0.2})
        assert config.min_registered_seam_frac == 0.2
        assert config.quality_threshold == before_quality
        assert len(changed) == 1

    def test_it_reports_what_it_changed_in_words(self):
        from flamingo_stitcher.pipeline import StitchingConfig

        changed = sp.apply_profile(
            StitchingConfig(), {"min_registered_seam_frac": 0.2}
        )
        assert "Minimum share of seams that must register" in changed[0]

    def test_a_value_already_in_place_is_not_reported_as_a_change(self):
        from flamingo_stitcher.pipeline import StitchingConfig

        config = StitchingConfig()
        assert sp.apply_profile(
            config, {"quality_threshold": config.quality_threshold}
        ) == []

    def test_every_tunable_is_a_real_config_field(self):
        # The store writes StitchingConfig attribute names directly, so a typo
        # here would be a setting that saves, loads, and does nothing.
        from flamingo_stitcher.pipeline import StitchingConfig

        config = StitchingConfig()
        for field in sp.TUNABLE_FIELDS:
            assert hasattr(config, field), field

    def test_every_tunable_is_shareable_so_it_lands_in_stitch_metadata(self):
        from flamingo_stitcher.pipeline import SHAREABLE_CONFIG_FIELDS

        for field in sp.TUNABLE_FIELDS:
            assert field in SHAREABLE_CONFIG_FIELDS, field


class TestDescribeResolution:
    def test_an_unknown_scope_says_so_and_says_what_to_do(self):
        applied, message = sp.describe_resolution("Liara", 17.0, {}, "")
        assert applied is False
        assert "Options tab" in message

    def test_an_unreadable_acquisition_names_the_missing_file(self):
        applied, message = sp.describe_resolution(None, None, {}, "")
        assert applied is False
        assert "ScopeSettings.txt" in message

    def test_a_loaded_profile_says_which_entry_it_came_from(self):
        _, message = sp.describe_resolution(
            "Liara", 17.0, {"quality_threshold": 0.5}, "liara|17.0x"
        )
        assert "objective 17.0x" in message
        _, message = sp.describe_resolution(
            "Liara", 17.0, {"quality_threshold": 0.5}, "liara|*"
        )
        assert "all objectives" in message


class TestPipelineApplication:
    """The layer that makes a headless/CLI run honour a rig's tuning at all."""

    def _acquisition(self, tmp_path, scope="Liara", mag=17.0):
        acq = tmp_path / "acq"
        acq.mkdir()
        (acq / "ScopeSettings.txt").write_text(
            f"Microscope name = {scope}\nObjective lens magnification = {mag}\n"
        )
        return acq

    def _pipeline(self, **overrides):
        from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline

        return StitchingPipeline(StitchingConfig(**overrides))

    def test_it_reads_the_scope_and_objective_off_the_acquisition(self, tmp_path):
        acq = self._acquisition(tmp_path)
        scope, objective, _values, _source = sp.resolve_for_acquisition(acq)
        assert scope == "Liara"
        assert objective == pytest.approx(17.0)

    def test_a_saved_profile_reaches_the_config(self, tmp_path):
        sp.save_profile("Liara", 17.0, {"min_registered_seam_frac": 0.2})
        pipe = self._pipeline()
        pipe._apply_scope_profile(self._acquisition(tmp_path))
        assert pipe.config.min_registered_seam_frac == pytest.approx(0.2)
        assert pipe.config.scope_profile_source == "liara|17.0x"

    def test_a_profile_for_another_scope_is_not_applied(self, tmp_path):
        sp.save_profile("Bruce", 17.0, {"min_registered_seam_frac": 0.2})
        pipe = self._pipeline()
        before = pipe.config.min_registered_seam_frac
        pipe._apply_scope_profile(self._acquisition(tmp_path, scope="Liara"))
        assert pipe.config.min_registered_seam_frac == before
        assert pipe.config.scope_profile_source == ""

    def test_an_already_resolved_profile_is_not_re_applied_over_an_override(
        self, tmp_path
    ):
        # The GUI resolves per queue item and lets the user override a control
        # afterwards. Re-applying here would silently undo that.
        sp.save_profile("Liara", 17.0, {"quality_threshold": 0.15})
        pipe = self._pipeline(
            quality_threshold=0.6, scope_profile_source="liara|17.0x"
        )
        pipe._apply_scope_profile(self._acquisition(tmp_path))
        assert pipe.config.quality_threshold == pytest.approx(0.6)

    def test_an_unreadable_acquisition_leaves_the_config_alone(self, tmp_path):
        pipe = self._pipeline()
        before = pipe.config.quality_threshold
        pipe._apply_scope_profile(tmp_path / "does-not-exist")
        assert pipe.config.quality_threshold == before

    def test_a_broken_profile_store_never_stops_a_stitch(self, tmp_path, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(sp, "resolve_for_acquisition", boom)
        pipe = self._pipeline()
        pipe._apply_scope_profile(self._acquisition(tmp_path))  # must not raise

    def test_the_profile_source_is_recorded_for_provenance(self, tmp_path):
        from flamingo_stitcher.pipeline import SHAREABLE_CONFIG_FIELDS

        # It lands in stitch_metadata.json, so a run's thresholds are traceable
        # to the profile that set them.
        assert "scope_profile_source" in SHAREABLE_CONFIG_FIELDS
