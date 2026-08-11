"""Every registration knob must survive the trip from YAML/CLI/GUI to the config.

A setting that exists in one layer and is dropped by another is worse than one
that does not exist: the UI says it is on, the run behaves as though it is off,
and nothing reports the disagreement. Two specific traps are pinned here.

Run: python3 -m pytest tests/test_registration_config_wiring.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from flamingo_stitcher import config_loader
from flamingo_stitcher.config_loader import apply_stitching_yaml_to_config
from flamingo_stitcher.pipeline import SHAREABLE_CONFIG_FIELDS, StitchingConfig

REPO = Path(__file__).resolve().parents[1]
DIALOG = REPO / "src" / "flamingo_stitcher" / "gui" / "stitching_dialog.py"
YAML_PATH = REPO / "src" / "flamingo_stitcher" / "configs" / "stitching_config.yaml"

NEW_FIELDS = [
    "max_registration_shift_z_um",
    "min_registration_overlap_frac",
    "registration_upsample_factor",
    "registration_z_refine",
    "registration_z_refine_range_um",
    "registration_z_refine_binning",
    "registration_z_refine_upsample",
    "registration_report_enabled",
    "registration_report_json",
]


@pytest.fixture
def from_yaml(monkeypatch):
    """Apply an arbitrary YAML document to a fresh config.

    `apply_stitching_yaml_to_config` reads the shipped file itself rather than
    taking a document, so the loader is what gets substituted.
    """

    def _apply(document):
        monkeypatch.setattr(
            config_loader, "get_stitching_defaults", lambda: document
        )
        config = StitchingConfig()
        apply_stitching_yaml_to_config(config)
        return config

    return _apply


class TestYamlWiring:
    @pytest.mark.parametrize(
        "yaml_key,attr,value,expected",
        [
            ("max_shift_z_um", "max_registration_shift_z_um", 12.5, 12.5),
            ("min_overlap_fraction", "min_registration_overlap_frac", 0.2, 0.2),
            ("upsample_factor", "registration_upsample_factor", 8, 8),
            ("report", "registration_report_enabled", False, False),
            ("report_json", "registration_report_json", True, True),
        ],
    )
    def test_a_top_level_registration_key_reaches_the_config(
        self, from_yaml, yaml_key, attr, value, expected
    ):
        config = from_yaml({"registration": {yaml_key: value}})
        assert getattr(config, attr) == expected

    @pytest.mark.parametrize(
        "yaml_key,attr,value,expected",
        [
            ("enabled", "registration_z_refine", True, True),
            ("range_um", "registration_z_refine_range_um", 75.0, 75.0),
            ("upsample_factor", "registration_z_refine_upsample", 4, 4),
        ],
    )
    def test_a_z_refine_key_reaches_the_config(
        self, from_yaml, yaml_key, attr, value, expected
    ):
        config = from_yaml({"registration": {"z_refine": {yaml_key: value}}})
        assert getattr(config, attr) == expected

    def test_the_z_refine_binning_survives_as_a_dict(self, from_yaml):
        config = from_yaml(
            {"registration": {"z_refine": {"binning": {"z": 1, "y": 2, "x": 2}}}}
        )
        assert config.registration_z_refine_binning == {"z": 1, "y": 2, "x": 2}

    def test_the_shipped_yaml_documents_every_new_key(self):
        # A knob nobody can find in the config file may as well not exist.
        reg = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))["registration"]
        for key in (
            "max_shift_z_um",
            "min_overlap_fraction",
            "upsample_factor",
            "report",
            "report_json",
        ):
            assert key in reg, key
        for key in ("enabled", "range_um", "binning", "upsample_factor"):
            assert key in reg["z_refine"], key

    def test_the_shipped_yaml_agrees_with_the_dataclass_defaults(self):
        # If they drift, a fresh install behaves differently from a bare
        # StitchingConfig() and neither is obviously the intended one.
        shipped = StitchingConfig.with_yaml_defaults()
        defaults = StitchingConfig()
        for attr in NEW_FIELDS:
            assert getattr(shipped, attr) == getattr(defaults, attr), attr


class TestSharedConfigBlock:
    @pytest.mark.parametrize("field", NEW_FIELDS)
    def test_new_fields_are_recorded_in_stitch_metadata(self, field):
        # "Load Configuration" reads this block to reproduce a setup that
        # worked. A field missing from it silently reverts to the default.
        assert field in SHAREABLE_CONFIG_FIELDS


def _method_body(class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(DIALOG.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == method_name
                ):
                    return child
    raise AssertionError(f"{class_name}.{method_name} not found")


def _setting_keys(method: ast.AST) -> set:
    """The literal keys passed to s.setValue(...) / s.value(...) in a method."""
    keys = set()
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("setValue", "value"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str):
                keys.add(value)
    return keys


class TestTheNativeTabDoesNotForgetSettings:
    """NativeStitchingDialog overrides save/restore WITHOUT calling super().

    So a QSettings key added to the base dialog is silently dropped by the
    Native tab: the user sets it, closes the app, and it is gone with no error.
    Rather than hope the next person notices the override, assert the two stay
    in step.
    """

    def test_both_dialogs_save_the_same_keys(self):
        base = _setting_keys(_method_body("StitchingDialog", "_save_settings"))
        native = _setting_keys(_method_body("NativeStitchingDialog", "_save_settings"))
        missing = base - native
        assert not missing, (
            "NativeStitchingDialog._save_settings does not call super(), so "
            f"these keys are silently forgotten on that tab: {sorted(missing)}"
        )

    def test_both_dialogs_restore_the_same_keys(self):
        base = _setting_keys(_method_body("StitchingDialog", "_restore_settings"))
        native = _setting_keys(
            _method_body("NativeStitchingDialog", "_restore_settings")
        )
        missing = base - native
        assert not missing, (
            "NativeStitchingDialog._restore_settings does not call super(), so "
            f"these keys are never restored on that tab: {sorted(missing)}"
        )

    @pytest.mark.parametrize(
        "key",
        ["max_reg_shift_z", "z_refine", "z_refine_range_um", "registration_report"],
    )
    def test_the_new_registration_keys_are_persisted(self, key):
        assert key in _setting_keys(
            _method_body("StitchingDialog", "_save_settings")
        ), key
        assert key in _setting_keys(
            _method_body("StitchingDialog", "_restore_settings")
        ), key


class TestTheTimingCacheKey:
    """A cost swing the ETA cannot see makes the ETA wrong for both runs."""

    def test_z_refine_is_part_of_the_key(self):
        from flamingo_stitcher.timing_cache import StitchingTimingKey

        base = dict(
            n_tiles=10,
            n_channels=1,
            n_pyramid_levels=0,
            n_timepoints=1,
            output_format="ome-tiff",
            fusion_method="cosine",
            skip_registration=False,
            planes_per_tile=100,
        )
        plain = StitchingTimingKey(**base).serialize()
        refined = StitchingTimingKey(**base, z_refine=True).serialize()
        assert plain != refined

    def test_an_unrefined_key_is_byte_identical_to_the_old_format(self):
        # The token is APPENDED only when set, so every key string already in a
        # user's timing cache stays valid and their learned timings survive.
        from flamingo_stitcher.timing_cache import StitchingTimingKey

        plain = StitchingTimingKey(
            n_tiles=10,
            n_channels=1,
            n_pyramid_levels=0,
            n_timepoints=1,
            output_format="ome-tiff",
            fusion_method="cosine",
            skip_registration=False,
            planes_per_tile=100,
        ).serialize()
        assert plain.endswith("dst=")
        assert "zr=" not in plain

    def test_the_config_drives_the_key(self):
        from flamingo_stitcher.pipeline import build_timing_key

        cfg = StitchingConfig()
        cfg.registration_z_refine = True
        assert build_timing_key([], cfg, None).z_refine is True
