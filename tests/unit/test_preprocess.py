import json

import numpy as np
import pandas as pd
import xarray as xr

from lmr4dvar.data import (
    linear_detrend_by_month,
    monthly_climatology_anomaly,
    prepare_quarterly_field,
    quarterly_mean,
    regular_lat_lon_grid,
    write_preprocessed_artifact,
)


def monthly_field(periods: int = 24) -> xr.DataArray:
    return xr.DataArray(
        np.arange(periods, dtype=float)[:, None, None],
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=periods, freq="MS"),
            "lat": [45.0],
            "lon": [0.0],
        },
        name="tas",
        attrs={"units": "K"},
    )


def test_monthly_anomaly_has_zero_month_means() -> None:
    anomaly, climatology = monthly_climatology_anomaly(monthly_field())

    np.testing.assert_allclose(anomaly.groupby("time.month").mean("time"), 0.0)
    assert climatology.sizes["month"] == 12
    assert anomaly.attrs["units"] == "K"


def test_legacy_quarterly_mode_preserves_incomplete_edges() -> None:
    result = quarterly_mean(monthly_field(12), "legacy_qs_dec")

    np.testing.assert_allclose(result[:, 0, 0], [0.5, 3.0, 6.0, 9.0, 11.0])
    assert result.time.dt.month.values.tolist() == [1, 4, 7, 10, 12]


def test_complete_quarterly_mode_drops_incomplete_edges() -> None:
    result = quarterly_mean(monthly_field(12), "complete")

    np.testing.assert_allclose(result[:, 0, 0], [3.0, 6.0, 9.0])
    assert result.time.dt.month.values.tolist() == [4, 7, 10]


def test_linear_detrend_by_month_removes_separate_trends() -> None:
    time = pd.date_range("2000-01-01", periods=60, freq="MS")
    month = time.month.to_numpy()
    year_index = np.arange(60) // 12
    values = month + year_index * month * 0.2
    field = xr.DataArray(values[:, None], dims=("time", "site"), coords={"time": time})

    result = linear_detrend_by_month(field)

    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_detrend_marks_entire_month_location_missing_after_one_nan() -> None:
    field = monthly_field(24)
    field[0, 0, 0] = np.nan

    result = linear_detrend_by_month(field)

    assert result.sel(time=result.time.dt.month == 1).isnull().all()
    assert result.sel(time=result.time.dt.month == 2).notnull().all()


def test_two_degree_grid_matches_legacy_cell_centers() -> None:
    lat, lon = regular_lat_lon_grid(2.0)

    assert lat.size == 90
    assert lon.size == 180
    np.testing.assert_array_equal(lat[[0, -1]], [-89.0, 89.0])
    np.testing.assert_array_equal(lon[[0, -1]], [0.0, 358.0])


def test_prepared_quarterly_field_selects_period_months_and_domain() -> None:
    field = xr.DataArray(
        np.arange(12, dtype=float).reshape(6, 2, 1),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.to_datetime(
                ["1999-10-01", "1999-12-01", "2000-01-01", "2000-04-01", "2000-07-01", "2001-01-01"]
            ),
            "lat": [-1.0, 1.0],
            "lon": [0.0],
        },
    )

    result = prepare_quarterly_field(
        field,
        domain="northern_hemisphere",
        seasonal_months=[1, 4, 7, 10],
        training_period=(2000, 2000),
    )

    assert result.sizes == {"time": 3, "lat": 1, "lon": 1}
    assert result.time.dt.month.values.tolist() == [1, 4, 7]


def test_preprocessed_artifact_roundtrip(tmp_path) -> None:
    field = monthly_field(3)
    climatology = field.groupby("time.month").mean("time")

    write_preprocessed_artifact(
        tmp_path, {"tas": field}, {"tas": climatology}, {"dataset": "synthetic"}
    )

    with xr.open_dataarray(
        tmp_path / "fields" / "tas.nc", engine="h5netcdf"
    ) as loaded:
        xr.testing.assert_allclose(loaded, field.astype("float32"))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset"] == "synthetic"
    assert manifest["variables"] == ["tas"]
