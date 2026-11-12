from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lmr4dvar.config.models import ProjectConfig
from lmr4dvar.data import inspect_simulation_inputs


def _write_member(path: Path, variable: str, periods: int = 8, start: str = "2000-01-01") -> None:
    field = xr.DataArray(
        np.zeros((periods, 2, 2), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range(start, periods=periods, freq="QS-JAN"),
            "lat": [-1.0, 1.0],
            "lon": [0.0, 2.0],
        },
        name=variable,
        attrs={"units": "K"},
    )
    field.to_dataset().to_netcdf(path, engine="h5netcdf")


def _config(root: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "project": {"name": "preflight"},
            "paths": {"simulation_root": root, "output_root": root / "output"},
            "state": {
                "variables": {
                    "tas": {
                        "domain": "global",
                        "n_modes": 1,
                        "input_pattern": "tas_*.nc",
                        "input_combine": "concatenate_members",
                    },
                    "sic": {
                        "domain": "northern_hemisphere",
                        "n_modes": 1,
                        "input_pattern": "sic_*.nc",
                        "input_combine": "concatenate_members",
                    },
                }
            },
            "preprocess": {
                "input_kind": "prepared_quarterly",
                "training_period": [2000, 2001],
            },
            "observations": {"pages2k": {"enabled": True, "path": root / "proxy.pkl"}},
            "assimilation": {"start": "2000-01", "end": "2001-12"},
            "reconstruction": {"variables": ["tas", "sic"]},
            "verification": {
                "anomaly_period": [2000, 2001],
                "comparison_period": [2000, 2001],
            },
        }
    )


def test_inspect_simulation_inputs_reports_members_domain_and_common_total(tmp_path) -> None:
    _write_member(tmp_path / "tas_01.nc", "tas")
    _write_member(tmp_path / "tas_02.nc", "tas")
    _write_member(tmp_path / "sic_01.nc", "sic")
    _write_member(tmp_path / "sic_02.nc", "sic")

    report = inspect_simulation_inputs(_config(tmp_path))

    assert not report["errors"]
    assert report["training_samples"] == 16
    assert report["variables"]["tas"]["samples_per_file"] == [8]
    assert report["variables"]["sic"]["grid"] == {"lat": 1, "lon": 2}


def test_inspect_simulation_inputs_rejects_cross_variable_sample_mismatch(tmp_path) -> None:
    _write_member(tmp_path / "tas_01.nc", "tas")
    _write_member(tmp_path / "tas_02.nc", "tas")
    _write_member(tmp_path / "sic_01.nc", "sic")

    report = inspect_simulation_inputs(_config(tmp_path))

    assert any("different training sample totals" in error for error in report["errors"])


def test_preflight_rejects_equal_counts_with_different_physical_times(tmp_path):
    _write_member(tmp_path / "tas_01.nc", "tas", periods=4)
    _write_member(tmp_path / "sic_01.nc", "sic", periods=4, start="2000-04-01")
    report = inspect_simulation_inputs(_config(tmp_path))
    assert report["training_samples"] == 4
    assert any("physical member times differ" in error for error in report["errors"])
