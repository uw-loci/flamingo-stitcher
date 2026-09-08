"""The fusion region size has to be reachable from outside the source.

`fusion_superblock_target_gb` bounds how much output is fused at a time when
streaming. The 4 GB default was chosen against a machine that ran out of memory;
on a 191 GB box a 49-tile run fused in 81 regions, each re-fusing its boundary.
Whether a larger region is faster has NOT been measured — which is the point:
it cannot be measured while the only way to change it is editing Python.

Before this it was settable nowhere. No GUI control, no CLI flag, no YAML key,
absent from SHAREABLE_CONFIG_FIELDS — and the run log said "Set 'Fusion
super-block chunks' to override", naming a control that did not exist.

Run: python -m pytest tests/test_fusion_region_setting.py -q
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
SRC = REPO / "src" / "flamingo_stitcher"
DIALOG = (SRC / "gui" / "stitching_dialog.py").read_text(encoding="utf-8")
MAIN = (SRC / "__main__.py").read_text(encoding="utf-8")
PIPELINE = (SRC / "pipeline.py").read_text(encoding="utf-8")
YAML_PATH = SRC / "configs" / "stitching_config.yaml"


@pytest.fixture
def from_yaml(monkeypatch):
    def _apply(document):
        monkeypatch.setattr(config_loader, "get_stitching_defaults", lambda: document)
        config = StitchingConfig()
        apply_stitching_yaml_to_config(config)
        return config

    return _apply


class TestItIsReachable:
    def test_from_the_yaml(self, from_yaml):
        config = from_yaml({"memory": {"fusion_region_gb": 32.0}})
        assert config.fusion_superblock_target_gb == 32.0

    def test_zero_is_honoured_rather_than_treated_as_unset(self, from_yaml):
        """0 means 'whole output' — a real choice, not a missing value."""
        config = from_yaml({"memory": {"fusion_region_gb": 0.0}})
        assert config.fusion_superblock_target_gb == 0.0

    def test_the_explicit_chunk_override_is_reachable_too(self, from_yaml):
        config = from_yaml({"memory": {"fusion_superblock_chunks": 12}})
        assert config.fusion_superblock_chunks == 12

    def test_the_shipped_yaml_documents_both(self):
        mem = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))["memory"]
        assert "fusion_region_gb" in mem
        assert "fusion_superblock_chunks" in mem

    def test_the_shipped_yaml_matches_the_dataclass(self):
        shipped = StitchingConfig.with_yaml_defaults()
        default = StitchingConfig()
        assert (
            shipped.fusion_superblock_target_gb
            == default.fusion_superblock_target_gb
        )

    def test_from_the_cli(self):
        assert "--fusion-region-gb" in MAIN
        assert '("fusion_superblock_target_gb", args.fusion_region_gb)' in MAIN

    def test_from_the_gui(self):
        assert "_fusion_region_combo" in DIALOG
        assert "Fusion region size" in DIALOG

    def test_load_configuration_can_restore_it(self):
        for field in ("fusion_superblock_target_gb", "fusion_superblock_chunks"):
            assert field in SHAREABLE_CONFIG_FIELDS, field


class TestTheGuiControl:
    def _method(self, name):
        tree = ast.parse(DIALOG)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(DIALOG, node)
        raise AssertionError(name)

    def test_the_whole_output_choice_is_not_dropped_by_truthiness(self):
        """`if region_gb:` would silently discard 0.0, the one value that means
        something different rather than nothing."""
        source = self._method("_build_config")
        assert "region_gb is not None" in source

    def test_it_offers_a_larger_region_than_the_default(self):
        assert "32.0" in DIALOG and "Whole output" in DIALOG

    def test_it_is_persisted_and_restored(self):
        assert DIALOG.count('"fusion_region_gb"') >= 3


class TestTheLogNamesSomethingReal:
    def test_it_no_longer_points_at_a_control_that_does_not_exist(self):
        assert "Set 'Fusion super-block chunks' to override" not in PIPELINE

    def test_it_names_all_three_routes(self):
        assert "Fusion region size" in PIPELINE
        assert "--fusion-region-gb" in PIPELINE
        assert "memory.fusion_region_gb" in PIPELINE


class TestNoConfigFieldIsUnreachable:
    """Every StitchingConfig field must be settable without editing Python.

    fusion_superblock_target_gb was not, and the run log told operators to set a
    control that did not exist. An audit found seven more in the same state —
    including the three resource_guard fields, which decide whether a long run
    is allowed to START rather than dying on disk hours in.

    This is the guard against the next one.
    """

    def test_every_field_is_reachable_from_somewhere(self):
        import dataclasses

        loader = (SRC / "config_loader.py").read_text(encoding="utf-8")
        options = (SRC / "gui" / "options_panel.py").read_text(encoding="utf-8")
        reachable_text = DIALOG + options + MAIN + loader

        unreachable = []
        for field in dataclasses.fields(StitchingConfig):
            name = field.name
            if (
                name in reachable_text
                or name.replace("_", "-") in reachable_text
                or name in SHAREABLE_CONFIG_FIELDS
            ):
                continue
            unreachable.append(name)
        assert not unreachable, (
            "settable only by editing Python: " + ", ".join(unreachable)
        )

    def test_the_resource_guard_is_reachable(self):
        """It decides whether a run starts; at 4x-in-Z the disk fraction is the
        difference between a refusal up front and a failure hours in."""
        for field in (
            "resource_guard_enabled",
            "resource_guard_ram_fraction",
            "resource_guard_disk_fraction",
        ):
            assert field in SHAREABLE_CONFIG_FIELDS, field
        mem = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
        assert set(mem["resource_guard"]) == {
            "enabled", "ram_fraction", "disk_fraction"
        }

    def test_a_psf_file_can_finally_be_supplied(self):
        mem = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
        assert "psf_path" in mem["deconvolution"]
