import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lmr4dvar.verification.precipitation_products import (
    conservative_grid,
    era5_interval_midpoints,
    monthly_to_quarters,
    temporal_map_title,
    verify_product,
)


def test_conservative_remapping_preserves_constant_and_global_integral():
    source = xr.DataArray(
        np.arange(32).reshape(4, 8),
        dims=("lat", "lon"),
        coords={"lat": [-67.5, -22.5, 22.5, 67.5], "lon": np.arange(0, 360, 45)},
    )
    target = xr.Dataset(coords={"lat": [-45, 45], "lon": [45, 135, 225, 315]})
    constant = conservative_grid(xr.ones_like(source) * 3, target)
    np.testing.assert_allclose(constant, 3)
    remapped = conservative_grid(source, target)
    weights = np.diff(np.sin(np.deg2rad([-90, -45, 0, 45, 90])))
    source_mean = (source * xr.DataArray(weights, dims="lat")).sum() / (weights.sum() * 8)
    np.testing.assert_allclose(remapped.mean(), source_mean)


def test_conservative_missing_cells_are_not_filled():
    source = xr.DataArray(
        [[1.0, np.nan], [1.0, 1.0]], dims=("lat", "lon"), coords={"lat": [-45, 45], "lon": [0, 180]}
    )
    result = conservative_grid(source, xr.Dataset(coords={"lat": [-45, 45], "lon": [90, 270]}))
    assert np.isnan(result.sel(lat=-45)).all()
    np.testing.assert_allclose(result.sel(lat=45), 1)


def test_rounded_grib_longitudes_still_close_periodically():
    source = xr.DataArray(
        np.ones((2, 512)),
        dims=("lat", "lon"),
        coords={"lat": [-45, 45], "lon": np.linspace(0, 359.297, 512)},
    )
    target = xr.Dataset(coords={"lat": [-45, 45], "lon": np.arange(1.25, 360, 2.5)})
    np.testing.assert_allclose(conservative_grid(source, target), 1)


def test_era5_hourly_midpoints_resolve_month_boundary():
    initial = np.array(["1940-01-31T06:00", "1940-01-31T18:00"], dtype="datetime64[ns]")
    actual = era5_interval_midpoints(initial, np.arange(1, 13))
    assert actual[1, 5] == np.datetime64("1940-01-31T23:30")
    assert actual[1, 6] == np.datetime64("1940-02-01T00:30")
    with pytest.raises(ValueError, match="steps"):
        era5_interval_midpoints(initial, [1, 2])
    with pytest.raises(ValueError, match="consecutive"):
        era5_interval_midpoints(initial[[0, 0]], np.arange(1, 13))


def test_map_title_labels_gridpoint_area_mean_not_mean_series_r():
    assert temporal_map_title("global", 0.12345) == (
        "global: Temporal correlation\nArea-weighted R=0.123"
    )


def test_monthly_complete_quarters_and_missing_month_rejection():
    field = xr.DataArray(
        np.arange(12.0),
        dims="time",
        coords={"time": pd.date_range("1979-01-01", periods=12, freq="MS")},
    )
    result = monthly_to_quarters(field)
    np.testing.assert_allclose(result.time, [1979.25, 1979.5, 1979.75])
    np.testing.assert_allclose(result, [3, 6, 9])
    with pytest.raises(ValueError, match="missing months"):
        monthly_to_quarters(field.isel(time=[0, 1, 3, 4]))


def test_product_metrics_fixed_mask_and_complete_baseline():
    times = np.arange(1980, 1984, 0.25)
    field = xr.DataArray(
        np.arange(64).reshape(16, 2, 2).astype(float),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [-45, 45], "lon": [90, 270]},
        attrs={"units": "mm/day"},
    )
    reference = field.copy()
    reference.values[0, 0, 0] = np.nan
    mask = xr.ones_like(field.isel(time=0), dtype=bool)
    dataset, summary = verify_product(
        field, reference, mask, baseline=[1981, 1983], period=[1980, 1983]
    )
    assert summary["coverage_gridpoints"] == 3
    assert summary["annual_mean"]["correlation"] == pytest.approx(1)
    assert summary["annual_mean"]["rmse"] == pytest.approx(0)
    np.testing.assert_allclose(
        dataset.ce.sel(aggregation="annual").values[
            np.isfinite(dataset.ce.sel(aggregation="annual"))
        ],
        1,
    )
    with pytest.raises(ValueError, match="every baseline quarter"):
        verify_product(
            field,
            reference.isel(time=slice(None, -1)),
            mask,
            baseline=[1981, 1983],
            period=[1980, 1983],
        )


def test_era20c_requires_complete_nonnegative_daily_accumulation():
    path = Path(__file__).parents[1] / "analyze" / "verify_precipitation_products.py"
    spec = importlib.util.spec_from_file_location("precipitation_products_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    np.testing.assert_allclose(script.daily_accumulation(np.array([0.003]), 24), 3)
    with pytest.raises(ValueError, match="24-hour"):
        script.daily_accumulation(np.array([0.003]), 6)
    with pytest.raises(ValueError, match="negative"):
        script.daily_accumulation(np.array([-0.003]), 24)
