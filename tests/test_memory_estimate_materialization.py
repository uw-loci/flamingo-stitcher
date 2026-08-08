"""The in-memory projection must model dask's 2x materialization.

A 104-tile single-channel run projected 82.7 GB and ran at 128.1 GB. Part of the
gap was a real bug (an extra fused copy — see ``test_fused_uint16_conversion``);
the rest was the estimator's fuse-phase term, which was written as::

    output_gb + max(pyramid_overhead_gb, per_channel_gb)

That happens to equal 2 x output when there is exactly ONE channel, which is why
it looked right for years. With several channels ``per_channel_gb`` shrinks to
``output_gb / n`` and the term collapses to the pyramid floor, silently
under-counting every multi-channel run.

What is actually live during the fuse of one channel:

* the pre-allocated stacked (C,Z,Y,X) array — only once C > 1; at C == 1 the
  computed channel IS the stacked array (``stacked = vol``, no copy)
* the per-chunk results dask holds, plus the concatenated array it builds from
  them — ``.compute()`` cannot avoid having both

so the term is ``2 x per_channel`` (+ ``output`` when C > 1). These tests assert
that contract against the exposed ``materialize_fused_gb``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from flamingo_stitcher.pipeline import (  # noqa: E402
    RawTileInfo,
    StitchingConfig,
    estimate_memory_usage,
)


def _tiles(channels, grid=(6, 6), n_planes=512, frame=1024, pitch_mm=0.9):
    """A mosaic big enough that rounding to 2 decimal GB is not a factor."""
    out = []
    for iy in range(grid[1]):
        for ix in range(grid[0]):
            out.append(
                RawTileInfo(
                    folder=Path(f"X{ix}_Y{iy}"),
                    x_mm=2.0 + ix * pitch_mm,
                    y_mm=2.0 + iy * pitch_mm,
                    z_min_mm=10.0,
                    z_max_mm=12.0,
                    n_planes=n_planes,
                    raw_files={c: {0: Path("a")} for c in channels},
                    channels=list(channels),
                    illumination_sides=[0],
                    frame_width=frame,
                    frame_height=frame,
                )
            )
    return out


def _config():
    cfg = StitchingConfig.with_yaml_defaults()
    cfg.pixel_size_um = 1.0
    cfg.downsample_xy = 2
    cfg.downsample_z = 1
    cfg.skip_registration = True
    return cfg


def _est(channels):
    tiles = _tiles(channels)
    return estimate_memory_usage(tiles, list(channels), _config())


class TestMaterializationTerm:
    def test_single_channel_models_two_copies(self):
        est = _est([1])
        # At one channel the computed array becomes the stacked array, so the
        # only duplication is dask's chunks + concatenated result.
        assert est["materialize_fused_gb"] == pytest.approx(
            2 * est["output_gb"], rel=0.02
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_multi_channel_covers_the_stacked_array_too(self, n):
        """The regression: the old term dropped below one whole output here."""
        channels = list(range(1, n + 1))
        est = _est(channels)
        output_gb = est["output_gb"]
        per_channel = output_gb / n

        assert est["materialize_fused_gb"] >= output_gb, (
            f"{n}-channel projection budgets only "
            f"{est['materialize_fused_gb']:.2f} GB for the fuse phase, less than "
            f"the {output_gb:.2f} GB stacked array that is allocated up front "
            f"and stays live for the whole loop."
        )
        expected = output_gb + 2 * per_channel
        assert est["materialize_fused_gb"] >= 0.98 * expected, (
            "must also cover dask holding the chunk results AND the "
            f"concatenated channel: {est['materialize_fused_gb']:.2f} GB "
            f"budgeted vs {expected:.2f} GB live"
        )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_peak_includes_the_term(self, n):
        est = _est(list(range(1, n + 1)))
        assert est["in_memory_gb"] >= est["materialize_fused_gb"]
        assert est["in_memory_gb"] >= (
            est["held_tiles_gb"] + est["materialize_fused_gb"] - 0.05
        ), "held tiles and the fuse-phase working set are live at the same time"

    def test_more_channels_costs_more(self):
        """Sanity: the projection must not be flat in channel count."""
        peaks = [_est(list(range(1, n + 1)))["in_memory_gb"] for n in (1, 2, 3)]
        assert peaks == sorted(peaks) and peaks[0] < peaks[-1], peaks


class TestFusionWorkingSet:
    def test_content_based_blending_is_modelled_as_more_expensive(self):
        """Content-based keeps a longer per-block buffer chain alive.

        Counted from ``multiview_stitcher.weights.content_based``: float64
        stacked views + float64 normalized weights + a float32 astype copy + the
        accumulated/stacked/normalized float32 weights + the nan-gaussian
        scratch. Plain cosine blending has none of the last four.
        """
        tiles = _tiles([1])
        plain = _config()
        plain.tile_overlap_fusion = "blend"
        plain.content_based_fusion = False

        content = _config()
        content.tile_overlap_fusion = "blend"
        content.content_based_fusion = True

        plain_est = estimate_memory_usage(tiles, [1], plain)
        content_est = estimate_memory_usage(tiles, [1], content)

        ratio = content_est["fusion_gb"] / plain_est["fusion_gb"]
        # The 2 x sigma_2 halo alone inflates the block by ~1.8x for this
        # geometry. The extra buffers roughly double it again (coexist 2.5 ->
        # 5.5), so a ratio near the halo figure means the buffer chain is not
        # being counted — which is what shipped before, and what made the
        # content-based projection optimistic on every run that enabled it.
        assert ratio > 2.5, (
            f"content-based blending is projected at only {ratio:.2f}x plain "
            f"blending ({content_est['fusion_gb']} vs {plain_est['fusion_gb']} GB). "
            "That is about the halo's contribution on its own — is the "
            "content-based coexistence factor still being applied?"
        )

    def test_max_fusion_ignores_the_content_based_flag(self):
        """`max` short-circuits content-based, so it must not pay for the halo."""
        tiles = _tiles([1])
        cfg = _config()
        cfg.tile_overlap_fusion = "max"
        cfg.content_based_fusion = True

        ref = _config()
        ref.tile_overlap_fusion = "max"
        ref.content_based_fusion = False

        assert (
            estimate_memory_usage(tiles, [1], cfg)["fusion_gb"]
            == estimate_memory_usage(tiles, [1], ref)["fusion_gb"]
        )
