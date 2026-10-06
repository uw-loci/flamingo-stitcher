"""The Leonardo gate: it must assert a GPU, not a successful import.

`import leonardo_toolset` succeeding was the whole check. On Windows there is
no CUDA jax to import -- conda-forge ships no jaxlib for win-64, and PyPI's
jax-cuda12-plugin/pjrt are Linux-only -- so a native install gives CPU jax,
which imports, runs, produces output, and is slow for no visible reason. Same
silent-wrong-answer shape as an inverted illumination_low_side.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flamingo_stitcher import leonardo_capability as lc


def _runner(table):
    """Fake subprocess.run. `table` maps a substring of the command to
    (returncode, output)."""

    def run(cmd, **kw):
        joined = " ".join(str(c) for c in cmd)
        for needle, (code, out) in table.items():
            if needle in joined:
                return SimpleNamespace(returncode=code, stdout=out, stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="no match")

    return run


GPU_OK = {"nvidia-smi": (0, "NVIDIA RTX A4000\n")}
NO_GPU = {"nvidia-smi": (1, "command not found")}
WSL_OK = {"wsl.exe -l -q": (0, "Ubuntu\n")}
NO_WSL = {"wsl.exe -l -q": (1, "")}
IMPORT_OK = {"import leonardo_toolset": (0, "")}
JAX_GPU = {"jax.devices": (0, "cuda,gpu\n")}
JAX_CPU = {"jax.devices": (0, "cpu\n")}


def test_everything_present_is_available():
    cap = lc.detect(
        runner=_runner({**WSL_OK, **GPU_OK, **IMPORT_OK, **JAX_GPU}),
        system="Windows",
        env_python="/leo/python",
    )
    assert cap.available
    assert "RTX A4000" in cap.reason
    assert all(cap.checks.values())


def test_cpu_only_jax_is_refused_even_though_everything_imports():
    """The case this gate exists for: nothing errors, and it would be slow."""
    cap = lc.detect(
        runner=_runner({**WSL_OK, **GPU_OK, **IMPORT_OK, **JAX_CPU}),
        system="Windows",
        env_python="/leo/python",
    )
    assert not cap.available
    assert cap.first_failure == "jax_gpu"
    assert "only see the CPU" in cap.reason
    assert "slower" in cap.reason


def test_a_laptop_with_no_wsl_is_told_to_install_wsl_not_leonardo():
    cap = lc.detect(runner=_runner(NO_WSL), system="Windows")
    assert not cap.available
    assert cap.first_failure == "wsl2"
    assert "wsl --install" in cap.reason
    # It must NOT send them to the Leonardo setup button — wrong problem.
    assert "Set up Leonardo" not in cap.reason


def test_a_windows_driver_without_wsl_passthrough_is_caught():
    """nvidia-smi is probed INSIDE WSL on purpose: a working desktop driver
    does not imply the distro can see the GPU."""
    cap = lc.detect(
        runner=_runner({**WSL_OK, **NO_GPU}), system="Windows", env_python="/leo/python"
    )
    assert not cap.available
    assert cap.first_failure == "gpu"
    assert "inside WSL2" in cap.reason


def test_the_gpu_probe_runs_through_wsl_on_windows_and_directly_on_linux():
    seen = []

    def spy(cmd, **kw):
        seen.append(" ".join(str(c) for c in cmd))
        return SimpleNamespace(returncode=0, stdout="Ubuntu\nNVIDIA A4000\n", stderr="")

    lc.detect(runner=spy, system="Windows")
    assert any(c.startswith("wsl.exe -e nvidia-smi") for c in seen), seen
    seen.clear()
    lc.detect(runner=spy, system="Linux")
    assert any(c.startswith("nvidia-smi") for c in seen), seen


def test_linux_needs_no_wsl():
    cap = lc.detect(
        runner=_runner({**GPU_OK, **IMPORT_OK, **JAX_GPU}),
        system="Linux",
        env_python="/leo/python",
    )
    assert cap.available
    assert cap.checks["wsl2"] is True  # not applicable, not a blocker


def test_a_gpu_but_no_environment_points_at_the_setup_button():
    cap = lc.detect(
        runner=_runner({**WSL_OK, **GPU_OK}), system="Windows", env_python=None
    )
    assert not cap.available
    assert cap.first_failure == "env"
    assert "Set up Leonardo" in cap.reason
    assert "several-GB" in cap.reason


def test_an_environment_that_will_not_import_says_rebuild():
    cap = lc.detect(
        runner=_runner({**WSL_OK, **GPU_OK, "import leonardo_toolset": (1, "ImportError")}),
        system="Windows",
        env_python="/leo/python",
    )
    assert not cap.available
    assert cap.first_failure == "env"
    assert "ImportError" in cap.detail


@pytest.mark.parametrize("system", ["Darwin", "FreeBSD"])
def test_an_unsupported_platform_says_so_plainly(system):
    cap = lc.detect(runner=_runner({}), system=system)
    assert not cap.available
    assert cap.first_failure == "platform"
    assert system in cap.reason


def test_a_probe_that_raises_reads_as_unavailable_not_a_crash():
    """The Options tab must survive a broken WSL install."""

    def boom(cmd, **kw):
        raise OSError("wsl.exe is corrupt")

    cap = lc.detect(runner=boom, system="Windows")
    assert not cap.available
    assert cap.first_failure == "wsl2"


def test_a_hanging_probe_cannot_hang_the_tab():
    import inspect

    src = inspect.getsource(lc._run)
    assert "timeout=_PROBE_TIMEOUT" in src


def test_wsl_utf16_nul_padding_is_stripped():
    """`wsl.exe -l -q` emits UTF-16; unstripped NULs made every distro name
    non-empty, so WSL always looked present."""
    cap = lc.detect(
        runner=_runner({"wsl.exe -l -q": (0, "\x00\x00\n\x00 \x00\n")}),
        system="Windows",
    )
    assert cap.first_failure == "wsl2", "NUL-only lines are not distributions"


def test_every_refusal_names_a_next_action():
    """A disabled control with no reason is the thing this replaces."""
    cases = [
        lc.detect(runner=_runner(NO_WSL), system="Windows"),
        lc.detect(runner=_runner({**WSL_OK, **NO_GPU}), system="Windows",
                  env_python="/p"),
        lc.detect(runner=_runner({**WSL_OK, **GPU_OK}), system="Windows"),
        lc.detect(runner=_runner({**WSL_OK, **GPU_OK, **IMPORT_OK, **JAX_CPU}),
                  system="Windows", env_python="/p"),
    ]
    for cap in cases:
        assert not cap.available
        assert cap.tooltip() == cap.reason
        assert len(cap.reason) > 60, cap.reason
        assert any(w in cap.reason for w in
                   ("Install", "install", "Check", "Re-run", "Rebuild", "Use ")), cap.reason
