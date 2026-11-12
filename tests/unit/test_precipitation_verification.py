import json

import numpy as np
import xarray as xr

from lmr4dvar.config.models import ProjectConfig
from lmr4dvar.verification.precipitation import (
    plot_precipitation_comparison,
    precipitation_comparison,
    reference_on_model_grid,
)
from lmr4dvar.workflow.reconstruct import _calculate_indices
from lmr4dvar.workflow.verify import execute_verification


def test_gpcc_coverage_is_fixed_and_never_filled_over_ocean(tmp_path):
    time = 1950 + np.arange(40) / 4
    rng = np.random.default_rng(5)
    field = xr.DataArray(
        rng.normal(size=(40, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-20, 20], "lon": [0, 180]},
        attrs={"units": "mm/day"},
    )
    reference = field.copy(deep=True)
    reference.values[:, 0, 0] = np.nan
    reference.values[3, 0, 1] = np.nan
    dataset, summary = precipitation_comparison(field, reference, period=(1950, 1959), window=3)
    assert summary["coverage_gridpoints"] == 2
    assert summary["raw"]["rmse"] == 0
    np.testing.assert_allclose(dataset.annual_spatial_pattern_correlation, 1)
    assert not bool(dataset.common_coverage_mask[0, 0])
    assert not bool(dataset.common_coverage_mask[0, 1])
    paths = plot_precipitation_comparison(dataset, summary, tmp_path)
    assert len(paths) == 3
    assert all(path.stat().st_size > 0 for path in paths)


def test_reference_regridding_wraps_longitude_and_preserves_missing_cells():
    reference = xr.DataArray(
        np.ones((2, 4)),
        dims=("lat", "lon"),
        coords={"lat": [-45, 45], "lon": [0.5, 120, 240, 359.5]},
    )
    model = xr.DataArray(
        np.zeros((2, 3)), dims=("lat", "lon"), coords={"lat": [-20, 20], "lon": [0, 180, 358]}
    )
    result = reference_on_model_grid(reference, model)
    np.testing.assert_allclose(result, 1)
    reference.values[:, 1:3] = np.nan
    assert bool(reference_on_model_grid(reference, model).sel(lon=180).isnull().all())


def test_precipitation_outputs_are_part_of_verification_workflow(tmp_path):
    time = 1950 + np.arange(40) / 4
    rng = np.random.default_rng(8)
    field = xr.DataArray(
        rng.normal(size=(40, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-20.0, 20.0], "lon": [0.0, 180.0]},
        attrs={"units": "mm/day"},
        name="pr",
    )
    root = tmp_path / "runs/test/stages/reconstruct/fields"
    root.mkdir(parents=True)
    field.to_dataset().to_netcdf(root / "pr.nc", engine="h5netcdf")
    reference = tmp_path / "gpcc.nc"
    field.to_dataset().to_netcdf(reference, engine="h5netcdf")
    config = ProjectConfig.model_validate(
        {
            "project": {"name": "test"},
            "paths": {"simulation_root": tmp_path, "output_root": tmp_path / "runs"},
            "state": {"variables": {"pr": {"domain": "global", "n_modes": 1, "units": "mm/day"}}},
            "observations": {"pages2k": {"enabled": True}},
            "assimilation": {"start": "1950-01", "end": "1959-12"},
            "reconstruction": {"variables": ["pr"], "indices": ["GMP"]},
            "verification": {
                "anomaly_period": [1950, 1954],
                "comparison_period": [1950, 1959],
                "running_mean_years": 3,
                "plot_figures": False,
                "reference_fields": {"pr": {"path": reference, "variable": "pr"}},
            },
        }
    )
    outputs = execute_verification(config)
    indices = _calculate_indices(config, {"pr": field})
    np.testing.assert_allclose(indices["GMP"], field.mean(("lat", "lon")))
    assert indices["GMP"].attrs["units"] == "mm/day"
    assert {path.name for path in outputs} == {
        "precipitation_gpcc.nc",
        "pattern_metrics.nc",
        "summary.json",
    }
    summary = json.loads(next(path for path in outputs if path.name == "summary.json").read_text())
    assert summary["precipitation_gpcc"]["raw"]["rmse"] == 0
    assert summary["spatial_correlation"]["pr"]["annual"]["area_weighted_mean_correlation"] > 0.999
