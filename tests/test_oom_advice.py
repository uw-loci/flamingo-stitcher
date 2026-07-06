"""Tests for step-aware out-of-memory advice (pure, no Qt/numpy)."""

from flamingo_stitcher.oom_advice import (
    format_oom_advice,
    is_memory_error,
    oom_advice,
)


class TestIsMemoryError:
    def test_numpy_unable_to_allocate(self):
        assert is_memory_error(
            "Unable to allocate 471. GiB for an array with shape "
            "(1662, 7424, 5120) and data type float64"
        )

    def test_python_memoryerror(self):
        assert is_memory_error("MemoryError")

    def test_cpp_bad_alloc(self):
        assert is_memory_error("std::bad_alloc")

    def test_os_enomem(self):
        assert is_memory_error("Cannot allocate memory")

    def test_non_memory_error(self):
        assert not is_memory_error("FileNotFoundError: no such file")

    def test_empty(self):
        assert not is_memory_error("")


class TestWriteStepAdvice:
    def test_ome_tiff_recommends_zarr_and_keeps_resolution(self):
        lines = oom_advice("write", {"output_format": "ome-tiff"})
        joined = " ".join(lines).lower()
        assert "ome-zarr" in joined
        # The v0.5.0 pyramid fix is mentioned for TIFF/Imaris outputs.
        assert "v0.5.0" in " ".join(lines)

    def test_imaris_also_flagged_as_heavy(self):
        lines = oom_advice("write", {"output_format": "imaris"})
        assert any("ome-zarr" in l.lower() for l in lines)

    def test_zarr_does_not_recommend_switching_format(self):
        lines = oom_advice("write", {"output_format": "ome-zarr-sharded"})
        # Already zarr: no "switch to OME-Zarr" nudge, but still pyramid advice.
        assert not any("switch output format" in l.lower() for l in lines)
        assert any("pyramid" in l.lower() for l in lines)


class TestFuseStepAdvice:
    def test_workers_and_content_based_and_heavy_buffers(self):
        lines = oom_advice(
            "fuse",
            {
                "preprocess_workers": 4,
                "fuse_workers": 4,
                "content_based_fusion": True,
                "deconvolution_enabled": True,
                "depth_attenuation": True,
            },
        )
        joined = " ".join(lines).lower()
        assert "worker" in joined
        assert "content-based" in joined
        assert "deconvolution" in joined
        assert "depth attenuation" in joined

    def test_no_heavy_buffers_when_all_off(self):
        lines = oom_advice(
            "fuse",
            {
                "content_based_fusion": False,
                "deconvolution_enabled": False,
                "depth_attenuation": False,
            },
        )
        joined = " ".join(lines).lower()
        assert "content-based" not in joined
        assert "extra whole-tile float buffer" not in joined

    def test_leonardo_illumination_flagged(self):
        lines = oom_advice("fuse", {"illumination_fusion": "leonardo"})
        assert any("leonardo" in l.lower() for l in lines)

    def test_auto_workers_not_named_as_too_many(self):
        # 0/0 => auto (~4); still worth suggesting a cap.
        lines = oom_advice("fuse", {"preprocess_workers": 0, "fuse_workers": 0})
        assert any("worker" in l.lower() for l in lines)


class TestRegisterStepAdvice:
    def test_offers_skip_registration_when_not_already_skipping(self):
        lines = oom_advice("register", {"skip_registration": False})
        assert any("skip registration" in l.lower() for l in lines)

    def test_no_skip_suggestion_when_already_skipping(self):
        lines = oom_advice("register", {"skip_registration": True})
        assert not any(
            "tick 'skip registration" in l.lower() for l in lines
        )

    def test_mentions_pandas(self):
        lines = oom_advice("register", {})
        assert any("pandas" in l.lower() for l in lines)


class TestStreamingLever:
    def test_forced_in_memory_offers_streaming_first(self):
        lines = oom_advice("fuse", {}, use_streaming=False)
        assert "streaming mode" in lines[0].lower()

    def test_streaming_on_does_not_offer_streaming(self):
        lines = oom_advice("fuse", {}, use_streaming=True)
        assert not any("enable streaming mode" in l.lower() for l in lines)

    def test_streaming_never_re_suggested_for_any_step(self):
        # The watchdog fires with mode already resolved; if it's streaming we
        # must never tell the user to "enable streaming".
        for step in ("discover", "register", "preprocess", "fuse", "write",
                     "metadata", None):
            lines = oom_advice(step, {}, use_streaming=True)
            assert not any("enable streaming mode" in l.lower() for l in lines)

    def test_watchdog_fuse_streaming_gives_actionable_levers(self):
        # Exact screenshot scenario: fuse phase, already streaming, nothing
        # heavy enabled. Should still offer worker reduction, not streaming.
        lines = oom_advice(
            "fuse",
            {"illumination_fusion": "max", "content_based_fusion": False},
            use_streaming=True,
        )
        assert not any("enable streaming mode" in l.lower() for l in lines)
        assert any("worker" in l.lower() for l in lines)


class TestUniversalClosers:
    def test_downsample_is_last_and_flagged_as_resolution_loss(self):
        lines = oom_advice("write", {"output_format": "ome-tiff"})
        assert "downsample" in lines[-1].lower()
        assert "lowers resolution" in lines[-1].lower()

    def test_unknown_step_gives_broad_advice(self):
        lines = oom_advice(None, {"output_format": "ome-tiff"})
        joined = " ".join(lines).lower()
        assert "ome-zarr" in joined  # write-step lever included
        assert "worker" in joined  # fuse-step lever included


class TestFormatting:
    def test_format_names_the_step_and_numbers_lines(self):
        text = format_oom_advice("write", {"output_format": "ome-tiff"})
        assert "writing the output" in text.lower()
        assert "same" in text.lower() and "resolution" in text.lower()
        assert "\n  1. " in text

    def test_format_unknown_step_has_no_where_clause(self):
        text = format_oom_advice(None, {})
        assert "while" not in text.split("\n")[0].lower()
