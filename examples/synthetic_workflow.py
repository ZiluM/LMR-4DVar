"""Run a deterministic, data-free CPU demonstration through verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

from lmr4dvar.config import load_project_config
from lmr4dvar.observations.calibrated_artifact import save_calibrated_observations
from lmr4dvar.observations.records import ObservationCollection, ObservationRecord
from lmr4dvar.workflow import Stage, execute_through


def run_example(output: Path) -> Path:
    """Generate tiny known inputs and run the unmodified scientific handlers."""
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"choose an empty output directory: {output}")
    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    dates = pd.date_range("1950-01-01", "1990-01-01", freq="MS")
    samples = len(dates) - 1
    coefficients = np.zeros((samples, 2))
    for i in range(1, samples):
        coefficients[i] = 0.8 * coefficients[i - 1] + rng.normal(scale=0.3, size=2)
    pattern = np.array([[1.0, -0.5], [0.7, -0.2]])
    for i, (name, units) in enumerate((("tas", "K"), ("pr", "mm/day"))):
        dataset = xr.Dataset(
            {
                name: (
                    ("time", "lat", "lon"),
                    coefficients[:, i, None, None] * pattern,
                    {"units": units, "description": "synthetic monthly anomalies"},
                ),
                "time_bnds": (
                    ("time", "bnds"), np.stack([dates[:-1], dates[1:]], axis=1)
                ),
            },
            coords={
                "time": ("time", dates[1:], {"bounds": "time_bnds"}),
                "lat": [-45.0, 45.0],
                "lon": [0.0, 180.0],
            },
            attrs={"Conventions": "CF-1.8", "source": "synthetic example, seed 42"},
        )
        for variable in ("time", "time_bnds"):
            dataset[variable].encoding["units"] = "days since 1950-01-01"
        dataset.to_netcdf(inputs / f"{name}.nc", engine="h5netcdf")

    years = np.arange(1960, 1980)
    # JJA at the first grid cell: known slope 1, intercept 0, noise variance 0.04.
    truth = np.array(
        [coefficients[(dates[:-1].year == year) & dates[:-1].month.isin([6, 7, 8]), 0].mean()
         for year in years]
    )
    proxy_path = save_calibrated_observations(
        inputs / "synthetic_proxy.nc",
        ObservationCollection([
            ObservationRecord(
                record_id="synthetic-temperature-proxy",
                family="pages2k",
                variable="tas",
                latitude=-45.0,
                longitude=0.0,
                time=years.astype(float),
                value=truth + rng.normal(scale=0.2, size=len(years)),
                variance=np.full(len(years), 0.04),
                metadata={
                    "synthetic": True,
                    "psm_kind": "additive_linear_v1",
                    "intercept": 0.0,
                    "predictors": [{
                        "variable": "tas", "units": "K", "slope": 1.0,
                        "seasonality": [6, 7, 8],
                    }],
                },
            )
        ]),
    )
    root = output / "results"
    prior = root / "synthetic" / "stages" / "preprocess" / "fields"
    data = {
        "project": {"name": "synthetic", "description": "seed-42 CPU demonstration"},
        "paths": {"simulation_root": str(inputs), "output_root": str(root)},
        "state": {"variables": {
            name: {
                "domain": "global", "n_modes": 1, "units": units,
                "input_pattern": f"{name}.nc", "monthly_time_policy": "cf_bounds",
            }
            for name, units in (("tas", "K"), ("pr", "mm/day"))
        }},
        "preprocess": {
            "regrid": "none", "quarterly_mode": "complete", "require_complete_years": True,
            "remove_monthly_climatology": False, "detrend": "none",
        },
        "lim": {"fit_noise": False},
        "observations": {"pages2k": {
            "enabled": True, "artifact_format": "calibrated_netcdf", "path": str(proxy_path),
        }},
        "assimilation": {
            "start": "1960-01", "end": "1979-12", "device": "cpu", "precision": "float64",
            "max_iterations": 30, "inner_iterations": 10, "minimum_outer_iterations": 2,
        },
        "reconstruction": {"variables": ["tas", "pr"], "indices": ["GMT"],
                           "plot_figures": False},
        "verification": {
            "anomaly_period": [1960, 1969], "comparison_period": [1960, 1979],
            "running_mean_years": 1, "plot_figures": False,
            "reference_fields": {name: str(prior / f"{name}.nc") for name in ("tas", "pr")},
        },
    }
    config_path = output / "project.yml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    config = load_project_config(config_path)
    execute_through(config, end=Stage.VERIFY)
    return root / config.project.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/synthetic"))
    args = parser.parse_args()
    root = run_example(args.output)
    print(f"Completed synthetic example: {root}")
    print(f"Reusable configuration: {args.output.resolve() / 'project.yml'}")
    print(f"Verification: {root / 'stages/verify/summary.json'}")


if __name__ == "__main__":
    main()
