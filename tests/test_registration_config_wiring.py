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
    "registration_z_snap_to_plane",
    "stitching_approach",
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
            # Off is the interesting direction: the default is on, so a
            # config that means to keep sub-plane Z has to be able to say so.
            ("z_snap_to_plane", "registration_z_snap_to_plane", False, False),
            ("approach", "stitching_approach", "center_xy", "center_xy"),
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
            "z_snap_to_plane",
            "approach",
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
        [
            "max_reg_shift_z",
            "z_refine",
            "z_refine_range_um",
            "z_snap_to_plane",
            "stitching_approach",
            "registration_report",
        ],
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


class TestEveryWidgetReferenceIsAssigned:
    """No method may use a `self._widget` the constructor never creates.

    The GUI cannot be imported in CI (no PyQt5), so a typo or a widget added to
    a layout before it exists raises only on a real user's machine, at dialog
    construction, as an AttributeError with the tab already half-built. This
    walks the class instead: every `self._x` READ anywhere in a dialog class
    must be assigned somewhere in that class or a base class in this file.
    """

    _CLASSES = ("StitchingDialog", "NativeStitchingDialog", "MultiViewStitchingDialog")

    @staticmethod
    def _tree():
        return ast.parse(DIALOG.read_text(encoding="utf-8"))

    def _class_nodes(self):
        by_name = {
            n.name: n
            for n in self._tree().body
            if isinstance(n, ast.ClassDef)
        }
        return by_name

    def test_no_method_reads_an_attribute_nothing_assigns(self):
        by_name = self._class_nodes()
        assigned = set()
        tree = self._tree()
        # `self.x = ...` anywhere in the file (subclasses inherit, and
        # setattr-style dynamic creation is rare enough to ignore).
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    assigned.add(node.attr)
        # ...plus class-level constants (`_SETTINGS_GROUP = "..."`), which are
        # read through self but never assigned through it.
        for cls in by_name.values():
            for stmt in cls.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            assigned.add(target.id)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    assigned.add(stmt.target.id)

        missing = {}
        for cls_name in self._CLASSES:
            cls = by_name.get(cls_name)
            if cls is None:
                continue
            for node in ast.walk(cls):
                if not (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    continue
                name = node.attr
                if not name.startswith("_"):
                    continue  # public API / inherited Qt methods
                if name in assigned:
                    continue
                # Methods defined on the class are attributes too.
                if any(
                    isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and c.name == name
                    for other in by_name.values()
                    for c in other.body
                ):
                    continue
                missing.setdefault(cls_name, set()).add(name)

        assert not missing, (
            "these attributes are read but never assigned — the dialog will "
            f"raise AttributeError when built: {missing}"
        )
