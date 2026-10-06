"""Can Leonardo FUSE actually run here, on a GPU?

Leonardo needs **jax and torch both on GPU**. Until now the only check was
``import leonardo_toolset`` succeeding in the isolated env, and an import says
nothing about a GPU: on Windows there is no CUDA jax to import in the first
place. conda-forge ships no ``jaxlib`` for win-64 at all, and PyPI's
``jax-cuda12-plugin`` / ``jax-cuda12-pjrt`` are Linux-only -- so a native
Windows install resolves to **CPU jax**, which imports fine, runs, produces
output, and is slow for no reason the operator can see.

That is the same silent-wrong-answer shape as an inverted
``illumination_low_side``: plausible output, no error, and nothing on screen
says the run did not do what was asked. So the gate asserts a GPU DEVICE rather
than a successful import, and every refusal names the next action rather than
greying a control with no reason.

The target deployment is broad on purpose -- the stitcher is a standalone app
that may be on a laptop with no GPU, a workstation with one, or the microscope
PC. Anything short of a working GPU path disables the option; nothing else
about the run changes.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Each probe shells out; keep them short so a broken WSL install cannot hang
# the Options tab.
_PROBE_TIMEOUT = 20.0

# Ordered, so the FIRST failure is the one reported — telling someone their
# Leonardo env is missing when WSL2 itself is absent sends them to the wrong
# problem.
CHECK_ORDER = ("platform", "wsl2", "gpu", "env", "jax_gpu")


@dataclass(frozen=True)
class Capability:
    """Whether Leonardo FUSE can run, and if not, what to do about it."""

    available: bool
    reason: str
    """One line, user-facing. Names the next action, not just the fault."""
    checks: Dict[str, bool] = field(default_factory=dict)
    detail: str = ""

    @property
    def first_failure(self) -> Optional[str]:
        for name in CHECK_ORDER:
            if name in self.checks and not self.checks[name]:
                return name
        return None

    def tooltip(self) -> str:
        if self.available:
            return "Leonardo FUSE is available: GPU detected and the environment imports."
        return self.reason


def _run(cmd: Sequence[str], runner: Callable) -> tuple[int, str]:
    try:
        r = runner(cmd, capture_output=True, text=True, timeout=_PROBE_TIMEOUT)
        return int(r.returncode), f"{r.stdout or ''}{r.stderr or ''}"
    except Exception as exc:  # noqa: BLE001 - a probe must never raise
        logger.debug("Leonardo probe %s failed: %s", cmd, exc)
        return 1, str(exc)


def _wsl_distros(runner: Callable) -> List[str]:
    """Registered WSL distributions, or [] if WSL is absent or broken."""
    code, out = _run(["wsl.exe", "-l", "-q"], runner)
    if code != 0:
        return []
    # wsl.exe -l -q emits UTF-16 with NULs when not redirected cleanly.
    return [d for d in (ln.replace("\x00", "").strip() for ln in out.splitlines()) if d]


def detect(
    runner: Callable = subprocess.run,
    system: Optional[str] = None,
    env_python: Optional[str] = None,
) -> Capability:
    """Probe the machine. Never raises; a broken probe reads as unavailable.

    ``runner`` and ``system`` are injected so the decision table can be tested
    without WSL, a GPU, or a built environment.
    """
    system = system or platform.system()
    checks: Dict[str, bool] = {}

    # --- 1. platform -----------------------------------------------------
    if system == "Linux":
        checks["platform"] = True
        checks["wsl2"] = True  # not applicable; nothing to install
    elif system == "Windows":
        checks["platform"] = True
        distros = _wsl_distros(runner)
        checks["wsl2"] = bool(distros)
        if not distros:
            return Capability(
                False,
                "Leonardo FUSE needs a GPU, and on Windows that means WSL2: "
                "jax has no CUDA build for Windows, so a native install would "
                "run on the CPU without saying so. Install WSL2 with a Linux "
                "distribution (`wsl --install`), then set up the Leonardo "
                "environment from this tab.",
                checks,
            )
    else:
        checks["platform"] = False
        return Capability(
            False,
            f"Leonardo FUSE is not supported on {system}. It needs jax and "
            f"torch on a CUDA GPU, which is available on Linux, or on Windows "
            f"through WSL2.",
            checks,
        )

    prefix = ["wsl.exe", "-e"] if system == "Windows" else []

    # --- 2. a CUDA GPU the Linux side can actually see --------------------
    # Deliberately probed INSIDE WSL: the Windows driver can be present while
    # the WSL CUDA passthrough is not, and only the inside view decides.
    code, out = _run(
        [*prefix, "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], runner
    )
    gpus = [g.strip() for g in out.splitlines() if g.strip()] if code == 0 else []
    checks["gpu"] = bool(gpus)
    if not gpus:
        where = "inside WSL2" if system == "Windows" else "on this machine"
        return Capability(
            False,
            f"No CUDA GPU is visible {where}, so Leonardo FUSE would fall back "
            f"to the CPU. Check `nvidia-smi` {where}; on Windows a working "
            f"desktop driver does not by itself give WSL2 GPU access.",
            checks,
            detail=out.strip()[:400],
        )

    # --- 3. the environment exists and imports ---------------------------
    if not env_python:
        checks["env"] = False
        return Capability(
            False,
            f"GPU found ({gpus[0]}), but the Leonardo environment is not set "
            f"up yet. Use 'Set up Leonardo (GPU)…' in this tab — it is a "
            f"several-GB download.",
            checks,
        )
    code, out = _run([str(env_python), "-c", "import leonardo_toolset"], runner)
    checks["env"] = code == 0
    if code != 0:
        return Capability(
            False,
            "The Leonardo environment exists but `leonardo_toolset` will not "
            "import. Re-run 'Set up Leonardo (GPU)…' to rebuild it.",
            checks,
            detail=out.strip()[:400],
        )

    # --- 4. jax really has the GPU ---------------------------------------
    # The whole point. `import jax` succeeds on a CPU-only build, so asking
    # for devices is the only question worth asking.
    code, out = _run(
        [
            str(env_python),
            "-c",
            "import jax;print(','.join(sorted({d.platform for d in jax.devices()})))",
        ],
        runner,
    )
    platforms = {p.strip().lower() for p in out.replace("\n", ",").split(",") if p.strip()}
    checks["jax_gpu"] = bool(platforms & {"gpu", "cuda", "rocm"})
    if not checks["jax_gpu"]:
        return Capability(
            False,
            "jax in the Leonardo environment can only see the CPU, so FUSE "
            "would run far slower than expected without failing. Rebuild the "
            "environment with 'Set up Leonardo (GPU)…'.",
            checks,
            detail=f"jax.devices() platforms: {sorted(platforms) or 'none'}",
        )

    return Capability(
        True,
        f"Leonardo FUSE is available ({gpus[0]}).",
        checks,
        detail=f"jax platforms: {sorted(platforms)}",
    )
