"""Per-acquisition tile orientation (auto_tile_orientation).

When ``auto_tile_orientation`` is on, the pipeline must resolve each
acquisition's orientation from ITS OWN microscope name — so a batch mixing
systems (or a stale choice from another scope) can't mis-orient an item.
"""

from pathlib import Path

import flamingo_stitcher.orientation as orient
from flamingo_stitcher.pipeline import (
    StitchingConfig,
    StitchingPipeline,
    discover_tiles,
)

from _synth_acq import write_synth_acquisition


def _named_acq(tmp_path: Path, microscope: str) -> Path:
    acq = write_synth_acquisition(tmp_path, grid=(2, 2))
    # The synth generator writes ScopeSettings.txt without a microscope name;
    # append one so orientation resolution has something to key on.
    scope = acq / "ScopeSettings.txt"
    scope.write_text(scope.read_text() + f"\n  Microscope name = {microscope}\n")
    return acq


def _resolve_orientation(acq: Path, config: StitchingConfig):
    pipe = StitchingPipeline(config)
    tiles = discover_tiles(acq)
    assert tiles
    pipe._apply_and_log_geometry(tiles, acq)
    return (
        pipe.config.tile_orientation,
        pipe.config.reverse_x_tiles,
        pipe.config.reverse_y_tiles,
    )


def test_auto_orientation_uses_bundled_n7_preset(tmp_path, monkeypatch):
    monkeypatch.setattr(orient, "_user_presets_path", lambda: tmp_path / "none.json")
    acq = _named_acq(tmp_path / "n7acq", "n7")
    name, rx, ry = _resolve_orientation(
        acq, StitchingConfig(auto_tile_orientation=True)
    )
    assert name == "identity"
    assert rx is True  # per the bundled n7 preset (identity + reverse_x)
    assert ry is False


def test_auto_orientation_prefers_user_preset(tmp_path, monkeypatch):
    # A user-saved preset for this scope overrides the bundled YAML.
    preset_file = tmp_path / "presets.json"
    monkeypatch.setattr(orient, "_user_presets_path", lambda: preset_file)
    acq = _named_acq(tmp_path / "n7acq", "n7")
    orient.save_microscope_orientation("n7", "rot180", reverse_x=False, reverse_y=True)

    name, rx, ry = _resolve_orientation(
        acq, StitchingConfig(auto_tile_orientation=True)
    )
    assert name == "rot180"
    assert rx is False
    assert ry is True


def test_unknown_scope_keeps_legacy_default(tmp_path, monkeypatch):
    # No preset anywhere → leave the config default (empty tile_orientation,
    # which the pipeline treats as flip_h when camera_x_inverted).
    monkeypatch.setattr(orient, "_user_presets_path", lambda: tmp_path / "none.json")
    acq = _named_acq(tmp_path / "mystery", "totally-unknown-scope")
    name, rx, ry = _resolve_orientation(
        acq, StitchingConfig(auto_tile_orientation=True)
    )
    assert name == ""  # untouched; _preprocess derives flip_h from camera_x_inverted
    assert rx is False
    assert ry is False
