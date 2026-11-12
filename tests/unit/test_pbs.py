from pathlib import Path
from types import SimpleNamespace

import pytest

from lmr4dvar.config.models import ProjectConfig
from lmr4dvar.hpc import render_pbs_script, submit_pbs, write_pbs_script
from lmr4dvar.workflow import Stage


def _config(tmp_path: Path, *, gpus: int = 0) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "project": {"name": "pbs.test"},
            "paths": {"simulation_root": tmp_path, "output_root": tmp_path / "output"},
            "state": {"variables": {"tas": {"domain": "global", "n_modes": 1}}},
            "observations": {"pages2k": {"enabled": True}},
            "assimilation": {"start": "0850-01", "end": "2000-12"},
            "reconstruction": {"variables": ["tas"]},
            "verification": {
                "anomaly_period": [850, 1849],
                "comparison_period": [850, 2000],
            },
            "hpc": {
                "account": "P12345678",
                "queue": "casper",
                "python_executable": "/envs/lmr4dvar/bin/python",
                "cpus": 4,
                "memory_gb": 32,
                "gpus": gpus,
                "walltime": "01:30:00",
            },
        }
    )


def test_render_pbs_uses_profile_resources_and_same_cli(tmp_path) -> None:
    config_path = tmp_path / "project config.yml"
    profile_path = tmp_path / "profile.yml"
    script = render_pbs_script(
        _config(tmp_path, gpus=1),
        config_path,
        through=Stage.TRAIN_LIM,
        profile_path=profile_path,
        force=True,
    )
    assert "#PBS -N pbs-test" in script
    assert "#PBS -A P12345678" in script
    assert "select=1:ncpus=4:mem=32GB:ngpus=1" in script
    assert 'export OMP_NUM_THREADS="${PBS_NCPUS:-4}"' in script
    assert 'export MKL_NUM_THREADS="${PBS_NCPUS:-4}"' in script
    assert 'export OPENBLAS_NUM_THREADS="${PBS_NCPUS:-4}"' in script
    assert f"#PBS -o {tmp_path}/output/pbs.test/logs/through-train-lim.pbs.log" in script
    assert "#PBS -l walltime=01:30:00" in script
    assert "/envs/lmr4dvar/bin/python -m lmr4dvar.cli run" in script
    assert "'" + str(config_path.resolve()) + "'" in script
    assert "--through train-lim" in script
    assert "--force" in script


def test_write_and_submit_are_separate_operations(tmp_path, monkeypatch) -> None:
    path = write_pbs_script("#!/bin/bash\ntrue\n", tmp_path / "job.sh")
    assert path.stat().st_mode & 0o100
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="12345.desched\n")

    monkeypatch.setattr("lmr4dvar.hpc.pbs.subprocess.run", fake_run)
    assert submit_pbs(path) == "12345.desched"
    assert calls[0][0] == ["qsub", str(path.resolve())]
    assert calls[0][1]["check"] is True


def test_render_requires_hpc_configuration(tmp_path) -> None:
    data = _config(tmp_path).model_dump()
    data["hpc"] = None
    with pytest.raises(ValueError, match="hpc configuration"):
        render_pbs_script(ProjectConfig.model_validate(data), tmp_path / "config.yml")
