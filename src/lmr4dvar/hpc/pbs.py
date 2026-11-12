"""Portable PBS script rendering and explicit submission."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from lmr4dvar.config import ProjectConfig
from lmr4dvar.workflow import IMPLEMENTED_STAGES, Stage


def render_pbs_script(
    config: ProjectConfig,
    config_path: Path,
    *,
    through: Stage = Stage.TRAIN_LIM,
    profile_path: Path | None = None,
    force: bool = False,
) -> str:
    """Render a PBS job that invokes the same local workflow CLI."""
    hpc = config.hpc
    if hpc is None:
        raise ValueError("hpc configuration is required to render PBS")
    if through not in IMPLEMENTED_STAGES:
        raise ValueError(f"PBS execution through {through.value} is not implemented")
    job_name = re.sub(r"[^A-Za-z0-9_-]", "-", config.project.name)[:15]
    log_path = (
        config.paths.output_root
        / config.project.name
        / "logs"
        / f"through-{through.value}.pbs.log"
    ).resolve()
    resources = f"select=1:ncpus={hpc.cpus}:mem={hpc.memory_gb}GB"
    if hpc.gpus:
        resources += f":ngpus={hpc.gpus}"
    command = [
        hpc.python_executable,
        "-m",
        "lmr4dvar.cli",
        "run",
        str(config_path.expanduser().resolve()),
        "--through",
        through.value,
    ]
    if profile_path is not None:
        command.extend(["--profile", str(profile_path.expanduser().resolve())])
    if force:
        command.append("--force")
    quoted_command = " ".join(shlex.quote(part) for part in command)
    return "\n".join(
        [
            "#!/bin/bash",
            f"#PBS -N {job_name}",
            f"#PBS -A {hpc.account}",
            f"#PBS -q {hpc.queue}",
            f"#PBS -l {resources}",
            f"#PBS -l walltime={hpc.walltime}",
            "#PBS -j oe",
            f"#PBS -o {log_path}",
            "#PBS -V",
            "",
            "set -euo pipefail",
            f'export OMP_NUM_THREADS="${{PBS_NCPUS:-{hpc.cpus}}}"',
            f'export MKL_NUM_THREADS="${{PBS_NCPUS:-{hpc.cpus}}}"',
            f'export OPENBLAS_NUM_THREADS="${{PBS_NCPUS:-{hpc.cpus}}}"',
            f'export NUMEXPR_NUM_THREADS="${{PBS_NCPUS:-{hpc.cpus}}}"',
            'cd "${PBS_O_WORKDIR}"',
            quoted_command,
            "",
        ]
    )


def write_pbs_script(script: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script, encoding="utf-8")
    output.chmod(0o750)
    return output


def submit_pbs(script_path: Path) -> str:
    """Submit an existing script and return the scheduler job identifier."""
    if not script_path.is_file():
        raise FileNotFoundError(script_path)
    result = subprocess.run(
        ["qsub", str(script_path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip()
    if not job_id:
        raise RuntimeError("qsub returned an empty job identifier")
    return job_id
