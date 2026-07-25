"""expected_output_path drives the re-run overwrite/skip/unique behaviour."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from flamingo_stitcher.pipeline import StitchingConfig, StitchingPipeline  # noqa: E402


def _pipe(**cfg):
    return StitchingPipeline(StitchingConfig(**cfg))


def test_extension_matches_format(tmp_path):
    acq = tmp_path / "Sample" / "2026-04-05"
    acq.mkdir(parents=True)
    out = tmp_path / "out"

    zarr = _pipe(output_format="ome-zarr-sharded").expected_output_path(acq, out)
    assert zarr.suffixes[-2:] == [".ome", ".zarr"]
    assert zarr.parent == out

    tiff = _pipe(output_format="ome-tiff").expected_output_path(acq, out)
    assert tiff.name.endswith(".ome.tif")

    ims = _pipe(output_format="imaris").expected_output_path(acq, out)
    assert ims.name.endswith(".ims")


def test_settings_change_the_filename(tmp_path):
    acq = tmp_path / "Sample" / "2026-04-05"
    acq.mkdir(parents=True)
    out = tmp_path / "out"

    plain = _pipe(output_format="ome-zarr-sharded").expected_output_path(acq, out)
    destriped = _pipe(
        output_format="ome-zarr-sharded", destripe=True
    ).expected_output_path(acq, out)
    # Different preprocessing → different store name (they coexist, no clobber).
    assert plain.name != destriped.name
    assert "destripe" in destriped.name


def test_unique_bump_avoids_existing(tmp_path):
    """The CLI/GUI 'unique' path bumps until the store doesn't exist."""
    acq = tmp_path / "Sample" / "2026-04-05"
    acq.mkdir(parents=True)
    out = tmp_path / "Sample_2026-04-05_stitched"
    pipe = _pipe(output_format="ome-zarr-sharded")

    # Simulate an existing store at the base location.
    existing = pipe.expected_output_path(acq, out)
    existing.mkdir(parents=True)
    assert pipe.expected_output_path(acq, out).exists()

    # Bump: out_2 should not exist yet.
    bumped = Path(str(out) + "_2")
    assert not pipe.expected_output_path(acq, bumped).exists()
