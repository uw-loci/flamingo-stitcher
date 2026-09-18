"""The `illumination:` block must actually reach the run.

It did not. The block was read early in `apply_stitching_yaml_to_config`, then a
legacy top-level `illumination_fusion:` key was read LATER and overwrote it —
and the shipped YAML carried that legacy key set to "max". So editing
`illumination.fusion` to `content` in the very file that documents it resolved
to `max`, and nothing said so: the run log echoed the method actually in use,
which reads as correct unless you remember what you asked for.

Run: python -m pytest tests/test_illumination_config_precedence.py -q
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from flamingo_stitcher import config_loader
from flamingo_stitcher.pipeline import StitchingConfig

SHIPPED = Path(config_loader.__file__).parent / "configs" / "stitching_config.yaml"


def _apply(monkeypatch, text):
    data = yaml.safe_load(textwrap.dedent(text)) or {}
    monkeypatch.setattr(config_loader, "get_stitching_defaults", lambda *a, **k: data)
    cfg = StitchingConfig()
    config_loader.apply_stitching_yaml_to_config(cfg)
    return cfg


class TestTheBlockWins:
    def test_the_block_sets_the_fusion_method(self, monkeypatch):
        cfg = _apply(monkeypatch, "illumination:\n  fusion: content\n")
        assert cfg.illumination_fusion == "content"

    def test_the_block_beats_a_legacy_top_level_key(self, monkeypatch):
        # The exact shape that was broken: both present, the block must win.
        cfg = _apply(
            monkeypatch,
            """
            illumination_fusion: "max"
            illumination:
              fusion: blend
            """,
        )
        assert cfg.illumination_fusion == "blend"

    def test_a_legacy_key_alone_still_works(self, monkeypatch):
        # An existing user config must not stop working.
        cfg = _apply(monkeypatch, 'illumination_fusion: "mean"\n')
        assert cfg.illumination_fusion == "mean"

    @pytest.mark.parametrize(
        "key,value,field",
        [
            ("low_side", 1, "illumination_low_side"),
            ("pure_frac", 0.25, "illumination_pure_frac"),
            ("axis", "x", "illumination_axis"),
        ],
    )
    def test_the_geometry_settings_reach_the_config(
        self, monkeypatch, key, value, field
    ):
        cfg = _apply(monkeypatch, f"illumination:\n  {key}: {value}\n")
        assert getattr(cfg, field) == value


class TestTheShippedFileIsUsable:
    def test_editing_the_documented_block_has_an_effect(self, monkeypatch):
        # Reads the REAL shipped YAML, changes only the documented key, and
        # checks it survives. This is the test the original bug fails.
        data = yaml.safe_load(SHIPPED.read_text())
        data.setdefault("illumination", {})["fusion"] = "split"
        monkeypatch.setattr(
            config_loader, "get_stitching_defaults", lambda *a, **k: data
        )
        cfg = StitchingConfig()
        config_loader.apply_stitching_yaml_to_config(cfg)
        assert cfg.illumination_fusion == "split"

    def test_the_shipped_file_carries_no_competing_duplicate(self):
        data = yaml.safe_load(SHIPPED.read_text())
        assert "illumination_fusion" not in data, (
            "a top-level illumination_fusion key is back; it silently competes "
            "with the illumination: block that documents the same setting"
        )
        assert data["illumination"]["fusion"] == "max"
