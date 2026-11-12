import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lmr4dvar.config.models import ProjectConfig
from lmr4dvar.workflow.executor import _execute_eof, _execute_preprocess, _execute_train_lim


def test_excluded_member_forecast_pipeline_with_training_only_transforms(tmp_path):
    path = Path(__file__).resolve().parents[2] / "analyze/verify_cesm_precipitation_heldout.py"
    spec = importlib.util.spec_from_file_location("heldout", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    rng = np.random.default_rng(3)
    dates = pd.date_range("2000-01-01", periods=97, freq="MS")
    for variable in ("tas", "pr"):
        for member in ("001", "002", "011"):
            values = rng.normal(size=(96, 2, 2))
            if variable == "pr":
                values = 1e-7 + values * 1e-8
            dataset = xr.Dataset(
                {
                    variable: (
                        ("time", "lat", "lon"),
                        values,
                        {"units": "K" if variable == "tas" else "m/s"},
                    ),
                    "time_bnds": (("time", "bnds"), np.stack([dates[:-1], dates[1:]], axis=1)),
                },
                coords={
                    "time": ("time", dates[1:], {"bounds": "time_bnds"}),
                    "lat": [-1.0, 1.0],
                    "lon": [0.0, 2.0],
                },
            )
            for name in ("time", "time_bnds"):
                dataset[name].encoding["units"] = "days since 2000-01-01"
            dataset.to_netcdf(tmp_path / f"{variable}.{member}.nc", engine="h5netcdf")
    config = ProjectConfig.model_validate(
        {
            "project": {"name": "tiny"},
            "paths": {"simulation_root": tmp_path, "output_root": tmp_path / "runs"},
            "state": {
                "variables": {
                    name: {
                        "domain": "global",
                        "n_modes": 1,
                        "units": "K" if name == "tas" else "mm/day",
                        "input_pattern": f"{name}.*.nc",
                        "exclude_input_patterns": ["*.011.*"],
                        "input_combine": "concatenate_members",
                        "monthly_time_policy": "cf_bounds",
                    }
                    for name in ("tas", "pr")
                }
            },
            "preprocess": {
                "regrid": "none",
                "training_period": [2000, 2007],
                "require_complete_years": True,
            },
            "lim": {"fit_noise": False, "require_stable": False, "segment_length_steps": 28},
            "observations": {"pages2k": {"enabled": True, "path": tmp_path / "proxy.pkl"}},
            "assimilation": {"start": "2000-01", "end": "2007-12"},
            "reconstruction": {"variables": ["tas", "pr"]},
            "verification": {"anomaly_period": [2000, 2007], "comparison_period": [2000, 2007]},
        }
    )
    _execute_preprocess(config)
    _execute_eof(config)
    _execute_train_lim(config)
    output = tmp_path / "evaluation"
    output.mkdir()
    state, truth, labels, model, lim, inputs = script.assemble(
        config, "011", [(2001, 2003), (2005, 2007)], output
    )
    assert state.shape == (24, 2)
    assert len(inputs["pr"]["training"]) == 2
    scores = script.evaluate(state, truth, labels, model, lim, slice(1, 2))
    assert scores.sizes["lead_months"] == 4
    assert (
        scores.sample_count.sel({"method": "LIM", "lead_months": 12, "season": "all"}) == 16
    ).all()
    assert np.isfinite(scores.rmse).all()
    script.plot_scores(scores, output)
    assert (output / "pr_correlation.png").is_file()
    cached, *_ = script.assemble(config, "011", [(2001, 2003), (2005, 2007)], output)
    np.testing.assert_allclose(cached, state)
