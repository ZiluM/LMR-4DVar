import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lmr4dvar.verification.precipitation_products import verify_product


def script_module():
    path = Path(__file__).parents[1] / "analyze" / "verify_precipitation_products.py"
    spec = importlib.util.spec_from_file_location("precipitation_longest_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_longest_keeps_edge_quarters_but_not_partial_annual_means():
    times = np.arange(1980.25, 1984, 0.25)
    field = xr.DataArray(
        np.arange(60.0).reshape(15, 2, 2),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [-45, 45], "lon": [90, 270]},
        attrs={"units": "mm/day"},
    )
    dataset, summary = verify_product(
        field,
        field,
        xr.ones_like(field.isel(time=0), dtype=bool),
        baseline=[1981, 1983],
        period=[1806, 2000],
    )
    assert summary["annual_period"] == [1981, 1983]
    assert summary["quarterly"]["time_samples"] == 15
    assert summary["annual"]["time_samples"] == 3
    assert summary["DJF"]["time_range"] == [1981, 1983]
    assert summary["MAM"]["time_range"] == [1980.25, 1983.25]
    assert dataset.attrs["comparison_period"] == "1980-1983"


def test_era5_native_hourly_monthly_reduction_handles_cross_month_forecasts(tmp_path):
    script = script_module()
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    cache.mkdir()
    for month in range(1, 13):
        start = pd.Timestamp(year=1940, month=month, day=1) + pd.Timedelta(hours=6)
        end = (start + pd.offsets.MonthBegin(1)) - pd.Timedelta(hours=12)
        initial = pd.date_range(start, end, freq="12h")
        path = raw / f"1940{month:02d}"
        path.mkdir(parents=True)
        field = xr.DataArray(
            np.full((len(initial), 12, 2, 4), 1e-5),
            dims=("forecast_initial_time", "forecast_hour", "latitude", "longitude"),
            coords={
                "forecast_initial_time": initial,
                "forecast_hour": np.arange(1, 13),
                "latitude": [-45, 45],
                "longitude": [0, 90, 180, 270],
            },
            attrs={"units": "kg m**-2 s**-1"},
            name="MTPR",
        )
        field.to_dataset().to_netcdf(path / "toy_mtpr.nc", engine="h5netcdf")
    target = xr.Dataset(coords={"lat": [-45, 45], "lon": [45, 135, 225, 315]})
    result = script.prepare_era5_hourly({"hourly_root": str(raw)}, target, cache, 1940, 1940)
    assert result.sizes["time"] == 11
    assert result.time.dt.month.values.tolist() == list(range(2, 13))
    np.testing.assert_allclose(result, 0.864)
    assert result.attrs["units"] == "mm/day"


def test_independent_product_summary_preserves_pending_products(tmp_path):
    script = script_module()
    (tmp_path / "summary_ERA5.json").write_text(json.dumps({"status": "failed", "error": "toy"}))
    script.aggregate_summaries(
        tmp_path,
        {
            "baseline": [1981, 2000],
            "expected_products": ["ERA5", "20CRv3"],
            "products": {},
        },
    )
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["products"]["ERA5"]["error"] == "toy"
    assert result["products"]["20CRv3"]["status"] == "pending"
