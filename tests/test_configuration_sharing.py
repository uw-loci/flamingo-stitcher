"""A configuration file has to reproduce a run, not half of one.

The use case is concrete: someone tunes a stitch on the microscope computer,
then wants the same treatment applied to five more datasets on their own laptop.
That only works if the file carries EVERYTHING that shaped the output.

It did not. `SHAREABLE_CONFIG_FIELDS` was a hand-maintained whitelist that had
to be edited whenever a setting was added, and nobody ever did — so 34 real
processing settings never travelled at all, including every destripe tuning
parameter (sigma, wavelet, level), every deconvolution parameter, and depth
attenuation. The gate is subtractive now: a new setting travels the day it is
added, and the only way to keep one out is to say so explicitly.

Run: python -m pytest tests/test_configuration_sharing.py -q
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from flamingo_stitcher.pipeline import (
    ACQUISITION_CONFIG_FIELDS,
    MACHINE_CONFIG_FIELDS,
    NON_APPLICABLE_CONFIG_FIELDS,
    PROVENANCE_CONFIG_FIELDS,
    SHAREABLE_CONFIG_FIELDS,
    StitchingConfig,
    serialize_stitching_config,
)

ALL_FIELDS = {f.name for f in dataclasses.fields(StitchingConfig)}


class TestEverythingThatShapesTheOutputTravels:
    @pytest.mark.parametrize(
        "field",
        [
            # The ones the user actually asked about, and the reason for this file.
            "destripe_params",
            "destripe_direction",
            "destripe_output_axis",
            "deconvolution_engine",
            "deconvolution_iterations",
            "deconvolution_na",
            "deconvolution_wavelength_nm",
            "deconvolution_n_immersion",
            "depth_attenuation",
            "depth_attenuation_mu",
            # Registration tuning that lives in the Options tab, not the dialog.
            "quality_threshold",
            "min_tile_structure",
            "registration_channel_auto",
            "registration_seam_content_gate",
            # Output shape.
            "zarr_chunks",
            "pyramid_levels",
            "blending_widths",
        ],
    )
    def test_field_is_in_a_shared_configuration(self, field):
        assert field in SHAREABLE_CONFIG_FIELDS

    def test_only_the_local_scratch_path_is_withheld(self):
        # The other private names (acquisition_dir, output_dir, channels) are
        # not config fields at all — they reach the worker separately — so
        # scratch_dir is the only one this can actually exclude.
        assert ALL_FIELDS - set(SHAREABLE_CONFIG_FIELDS) == {"scratch_dir"}

    def test_a_new_setting_travels_without_anyone_editing_a_list(self):
        # The regression this whole change exists to prevent: the old whitelist
        # carried 65 of 101 fields and had to be hand-edited to grow.
        assert len(SHAREABLE_CONFIG_FIELDS) == len(ALL_FIELDS) - 1

    def test_no_duplicates(self):
        assert len(SHAREABLE_CONFIG_FIELDS) == len(set(SHAREABLE_CONFIG_FIELDS))


class TestWhatIsRecordedButNotApplied:
    def test_acquisition_geometry_is_recorded_for_provenance(self):
        # Discover re-measures these, but the run's own numbers stay readable.
        for f in ACQUISITION_CONFIG_FIELDS:
            assert f in SHAREABLE_CONFIG_FIELDS
            assert f in NON_APPLICABLE_CONFIG_FIELDS

    def test_machine_limits_are_recorded_but_never_applied(self):
        # A 191 GB box's ceiling and core count must not follow a config onto a
        # laptop — but "how many workers did that run use" is exactly what a
        # slow run gets diagnosed from, so they are still written down.
        for f in ("max_memory_gb", "preprocess_workers", "fuse_workers"):
            assert f in SHAREABLE_CONFIG_FIELDS
            assert f in MACHINE_CONFIG_FIELDS
            assert f in NON_APPLICABLE_CONFIG_FIELDS

    def test_scope_profile_source_is_provenance_only(self):
        assert "scope_profile_source" in PROVENANCE_CONFIG_FIELDS
        assert "scope_profile_source" in NON_APPLICABLE_CONFIG_FIELDS


class TestItSurvivesJson:
    def _round_trip(self, config):
        return json.loads(json.dumps(serialize_stitching_config(config)))

    def test_a_default_config_serializes(self):
        assert self._round_trip(StitchingConfig())

    def test_destripe_tuning_survives(self):
        cfg = StitchingConfig()
        cfg.destripe_params = {
            "sigma_foreground": 64.0,
            "sigma_background": 512.0,
            "level": 5,
            "wavelet": "db4",
        }
        assert self._round_trip(cfg)["destripe_params"] == cfg.destripe_params

    def test_int_keyed_dicts_are_stringified_for_json(self):
        # json.dumps refuses nothing here, but int keys come BACK as strings —
        # the loader has to undo that or a per-channel threshold silently
        # matches no channel and nothing raises.
        cfg = StitchingConfig()
        cfg.background_zero_thresholds = {3: 120}
        assert self._round_trip(cfg)["background_zero_thresholds"] == {"3": 120}

    def test_every_field_is_json_safe(self):
        # One unserializable default would take the whole file down.
        blob = self._round_trip(StitchingConfig())
        assert set(blob) == set(SHAREABLE_CONFIG_FIELDS)
