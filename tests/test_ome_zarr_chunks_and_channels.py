"""Read the written store back: chunk shape per axis, one channel per chunk,
channel names and colors.

The sharded writer passed ``chunks[0]`` to ngff-zarr, which applies one int to
every axis: the configured 32 x 256 x 256 became 32 x 32 x 32, and the channel
axis became min(32, C), so every chunk held every channel and a viewer showing
one channel read all of them. Found 2026-11-06 by checking a stitched store
against the ome-zarr-skills inspector's anomaly list (CHUNK_SPANS_CHANNEL,
OMERO_MISSING). These tests read the files, not the writer's arguments.
"""

import json

import numpy as np
import pytest

from flamingo_stitcher.writers.ome_zarr_writer import (
    _chunks_by_dim,
    wavelength_color_hex,
    write_ome_zarr_sharded,
    write_ome_zarr_v2,
)

VOXEL = {"z": 2.5, "y": 1.0475, "x": 1.0475}


def _volume():
    rng = np.random.default_rng(0)
    data = rng.integers(100, 4000, size=(2, 40, 96, 80), dtype=np.uint16)
    data[1] //= 4  # a dimmer second channel, so the windows differ
    return data


def test_the_mapping_puts_one_channel_in_a_chunk():
    assert _chunks_by_dim(("c", "z", "y", "x"), (32, 256, 256)) == {
        "c": 1,
        "z": 32,
        "y": 256,
        "x": 256,
    }
    assert _chunks_by_dim(("z", "y", "x"), (16, 64, 64)) == {"z": 16, "y": 64, "x": 64}


def test_colors_match_the_apps_channel_colors():
    """visualization_3d_config.yaml: 405 blue, 488 green, 561 red, 640 magenta."""
    assert [wavelength_color_hex(w) for w in (405, 488, 561, 640)] == [
        "0000FF",
        "00FF00",
        "FF0000",
        "FF00FF",
    ]
    assert wavelength_color_hex(None) == "FFFFFF"
    assert wavelength_color_hex(900) == "FFFFFF"


def test_sharded_v05_chunks_and_channels_as_configured(tmp_path):
    pytest.importorskip("ngff_zarr")
    out = write_ome_zarr_sharded(
        _volume(),
        tmp_path / "s.ome.zarr",
        VOXEL,
        chunks=(16, 32, 32),
        shard_chunks=(2, 2, 2),
        pyramid_levels=1,
        channel_names=["Channel_1", "Channel_3"],
        channel_wavelengths_nm=[488.0, 640.0],
    )
    root = json.loads((out / "zarr.json").read_text())["attributes"]["ome"]
    level0 = root["multiscales"][0]["datasets"][0]["path"]
    meta = json.loads((out / level0 / "zarr.json").read_text())
    sharding = next(c for c in meta["codecs"] if c["name"] == "sharding_indexed")
    inner = sharding["configuration"]["chunk_shape"]
    outer = meta["chunk_grid"]["configuration"]["chunk_shape"]
    assert inner == [1, 16, 32, 32], "chunk shape must be what was configured"
    assert outer[0] == 1, "a shard must hold one channel"

    channels = root["omero"]["channels"]
    assert [c["label"] for c in channels] == [
        "Channel_1 (488 nm)",
        "Channel_3 (640 nm)",
    ]
    assert channels[1]["color"] == "FF00FF"
    # Each window is that channel's own: the dimmer one ends lower.
    assert channels[1]["window"]["end"] < channels[0]["window"]["end"]
    assert channels[0]["window"]["max"] == 65535


def test_v04_writes_channels_at_the_root(tmp_path):
    out = write_ome_zarr_v2(
        _volume(),
        tmp_path / "v2.ome.zarr",
        VOXEL,
        chunks=(16, 32, 32),
        pyramid_levels=1,
        channel_names=["Channel_1", "Channel_3"],
        channel_wavelengths_nm=[488.0, 640.0],
    )
    attrs = json.loads((out / ".zattrs").read_text())
    assert [c["label"] for c in attrs["omero"]["channels"]] == [
        "Channel_1 (488 nm)",
        "Channel_3 (640 nm)",
    ]
    level0 = json.loads((out / "0" / ".zarray").read_text())
    assert level0["chunks"][0] == 1
