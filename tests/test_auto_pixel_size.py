"""Per-acquisition XY pixel size (auto_pixel_size).

When ``auto_pixel_size`` is on, each acquisition must resolve its OWN XY pixel
size from its OWN recorded objective (ScopeSettings.txt), so a batch that mixes
objectives stitches each item at the right scale instead of one value applied to
all. A manual value (auto off) must be left exactly as configured.
"""

from pathlib import Path

from flamingo_stitcher.pipeline import (
    StitchingConfig,
    StitchingPipeline,
    discover_tiles,
)

from _synth_acq import write_synth_acquisition

_SENSOR_UM = 6.5  # matches the synth generator + bundled hardware config default


def _resolve_pixel(acq: Path, config: StitchingConfig) -> float:
    """Run the geometry-resolution step and return the resulting pixel size."""
    pipe = StitchingPipeline(config)
    tiles = discover_tiles(acq)
    assert tiles, "synthetic acquisition produced no tiles"
    pipe._apply_and_log_geometry(tiles, acq)
    return pipe.config.pixel_size_um


def test_auto_pixel_size_derives_per_acquisition(tmp_path):
    # Two acquisitions, different objectives → different implied pixel sizes.
    acq_a = write_synth_acquisition(tmp_path / "a", grid=(2, 2), pixel_size_um=0.5)
    acq_b = write_synth_acquisition(tmp_path / "b", grid=(2, 2), pixel_size_um=1.0)

    # A deliberately-wrong fallback proves the value came from the objective.
    px_a = _resolve_pixel(acq_a, StitchingConfig(pixel_size_um=99.0, auto_pixel_size=True))
    px_b = _resolve_pixel(acq_b, StitchingConfig(pixel_size_um=99.0, auto_pixel_size=True))

    assert px_a == round(_SENSOR_UM / 13.0, 4)  # objective 13.0× → 0.5 µm
    assert px_b == round(_SENSOR_UM / 6.5, 4)  # objective 6.5×  → 1.0 µm
    assert px_a != px_b  # per-entry, not one value for the whole batch


def test_manual_pixel_size_is_left_untouched(tmp_path):
    # Objective implies ~0.5 µm, but a manual value must win unchanged.
    acq = write_synth_acquisition(tmp_path / "m", grid=(2, 2), pixel_size_um=0.5)
    px = _resolve_pixel(acq, StitchingConfig(pixel_size_um=0.406, auto_pixel_size=False))
    assert px == 0.406


def test_auto_falls_back_when_objective_missing(tmp_path):
    acq = write_synth_acquisition(tmp_path / "n", grid=(2, 2), pixel_size_um=0.5)
    (acq / "ScopeSettings.txt").unlink()  # no objective to read
    px = _resolve_pixel(acq, StitchingConfig(pixel_size_um=0.406, auto_pixel_size=True))
    assert px == 0.406  # keeps the fallback rather than guessing
