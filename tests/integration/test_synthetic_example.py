import importlib.util
import json
from pathlib import Path

import numpy as np
import xarray as xr

from lmr4dvar.config import load_project_config
from lmr4dvar.workflow import StageStatus, build_plan


def test_public_example_uses_real_observations_and_completes_all_stages(tmp_path):
    path = Path(__file__).resolve().parents[2] / "examples/synthetic_workflow.py"
    spec = importlib.util.spec_from_file_location("synthetic_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "example"
    root = module.run_example(output)
    config = load_project_config(output / "project.yml")
    assert all(item.status is StageStatus.COMPLETE for item in build_plan(config))
    diagnostics = json.loads((root / "stages/assimilate/diagnostics.json").read_text())
    assert diagnostics["observation_used_count_history"][-1] == 20
    assert diagnostics["cost_history"][-1] < diagnostics["cost_history"][0]
    for variable in ("tas", "pr"):
        with xr.open_dataset(root / f"stages/reconstruct/fields/{variable}.nc") as dataset:
            assert dataset[variable].shape == (80, 2, 2)
            assert np.isfinite(dataset[variable]).all()
    assert (root / "stages/verify/summary.json").is_file()
