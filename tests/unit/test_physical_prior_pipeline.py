import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lmr4dvar.config.models import ProjectConfig
from lmr4dvar.data.preprocess import complete_quarterly_years
from lmr4dvar.lim import LIMModel
from lmr4dvar.workflow.executor import _execute_eof, _execute_preprocess, _execute_train_lim


def test_complete_year_selection_drops_boundary_not_internal_quarters():
    time = pd.date_range("2000-04-01", "2003-10-01", freq="QS-JAN")
    field = xr.DataArray(np.arange(len(time)), dims="time", coords={"time": time})
    complete = complete_quarterly_years(field)
    assert complete.sizes["time"] == 12
    assert complete.time.dt.year.values[0] == 2001
    with pytest.raises(ValueError, match="missing quarters"):
        complete_quarterly_years(field.isel(time=np.delete(np.arange(len(time)), 5)))


def test_monthly_to_joint_lim_keeps_physical_labels_units_and_member_boundaries(tmp_path):
    rng = np.random.default_rng(4)
    dates = pd.date_range("2000-01-01", periods=49, freq="MS")
    for variable in ("tas", "pr"):
        for member in (1, 2):
            values = rng.normal(size=(48, 2, 2))
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
            dataset.time.encoding["units"] = "days since 2000-01-01"
            dataset.time_bnds.encoding["units"] = "days since 2000-01-01"
            dataset.to_netcdf(tmp_path / f"{variable}_{member}.nc", engine="h5netcdf")
    config = ProjectConfig.model_validate(
        {
            "project": {"name": "tiny-physical-prior"},
            "paths": {"simulation_root": tmp_path, "output_root": tmp_path / "output"},
            "state": {
                "variables": {
                    name: {
                        "domain": "global",
                        "n_modes": 1,
                        "input_pattern": f"{name}_*.nc",
                        "input_combine": "concatenate_members",
                        "monthly_time_policy": "cf_bounds",
                        "units": "K" if name == "tas" else "mm/day",
                    }
                    for name in ("tas", "pr")
                }
            },
            "preprocess": {
                "training_period": [2000, 2003],
                "regrid": "none",
                "detrend": "none",
                "require_complete_years": True,
            },
            "lim": {"fit_noise": False, "require_stable": False, "segment_length_steps": 12},
            "observations": {"pages2k": {"enabled": True, "path": tmp_path / "proxy.pkl"}},
            "assimilation": {"start": "2000-01", "end": "2003-12"},
            "reconstruction": {"variables": ["tas", "pr"]},
            "verification": {"anomaly_period": [2000, 2003], "comparison_period": [2000, 2003]},
        }
    )
    assert all(path.exists() for path in _execute_preprocess(config))
    root = tmp_path / "output" / config.project.name / "stages"
    with xr.open_dataset(root / "preprocess/fields/pr.nc") as dataset:
        assert dataset.pr.attrs["units"] == "mm/day"
        assert dataset.sizes["time"] == 24
        assert dataset.physical_year.values.tolist() == list(np.repeat([2001, 2002, 2003], 4)) * 2
    _execute_eof(config)
    with xr.open_dataset(root / "eof/state_pcs.nc") as dataset:
        assert dataset.physical_month.values.tolist() == [1, 4, 7, 10] * 6
        assert dataset.member.values.tolist() == [0] * 12 + [1] * 12
    _execute_train_lim(config)
    model = LIMModel.load(root / "train-lim")
    assert model.segment_length_steps == 12
    with xr.open_dataset(root / "train-lim/training_metrics.nc") as dataset:
        assert dataset.pair_count.values[0] == 22
    config.lim.segment_length_steps = 8
    with pytest.raises(ValueError, match="physical member blocks"):
        _execute_train_lim(config)
