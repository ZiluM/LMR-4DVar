import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from lmr4dvar.config.models import ReconstructionClimatologyConfig
from lmr4dvar.eof import EOFModel
from lmr4dvar.reconstruction import (
    decode_trajectory,
    global_mean_ocean_heat_content,
    global_mean_temperature,
    nino34,
    regular_grid_cell_area,
    sea_ice_area,
    sea_ice_volume,
)
from lmr4dvar.state import StateLayout
from lmr4dvar.verification import (
    SeriesMetrics,
    anomaly,
    centered_running_mean,
    gridpoint_temporal_correlation,
    plot_correlation_maps,
    plot_index_comparison,
    seasonal_anomaly,
    series_metrics,
    spatial_pattern_metrics,
)
from lmr4dvar.workflow.reconstruct import _restore_climatology


def _field() -> xr.DataArray:
    time = np.arange(12)
    score = np.sin(time * 2 * np.pi / 12)
    pattern = np.array([[1.0, -1.0], [0.5, -0.5]])
    return xr.DataArray(
        2.0 + score[:, None, None] * pattern,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-45.0, 45.0], "lon": [0.0, 90.0]},
        name="tas",
    )


def test_decode_trajectory_matches_eof_decode() -> None:
    model, pcs = EOFModel.fit(_field(), 1, center=True)
    layout = StateLayout.from_mode_counts({"tas": 1})
    decoded = decode_trajectory(
        torch.tensor(pcs.values, dtype=torch.float64), pcs.time, {"tas": model}, layout
    )
    xr.testing.assert_allclose(decoded.tas, model.decode(pcs))


def test_area_indices_have_expected_geometry_and_units() -> None:
    grid = xr.DataArray(
        np.ones((2, 4)),
        dims=("lat", "lon"),
        coords={"lat": [-45.0, 45.0], "lon": [45.0, 135.0, 225.0, 315.0]},
    )
    area = regular_grid_cell_area(grid)
    expected_earth_area = 4 * np.pi * 6_371_000.0**2
    np.testing.assert_allclose(float(area.sum()), expected_earth_area, rtol=1e-12)
    sia = sea_ice_area(grid, cell_area=area)
    siv = sea_ice_volume(grid, 2 * grid, cell_area=area)
    np.testing.assert_allclose(float(sia), expected_earth_area / 2)
    np.testing.assert_allclose(float(siv), expected_earth_area)
    assert sia.attrs["units"] == "m2"
    assert siv.attrs["units"] == "m3"


def test_temperature_indices_use_latitude_weights_and_region() -> None:
    field = xr.DataArray(
        np.arange(18.0).reshape(1, 3, 6),
        dims=("time", "lat", "lon"),
        coords={"time": [2000], "lat": [-10.0, 0.0, 10.0], "lon": np.arange(0, 360, 60)},
    )
    gmt = global_mean_temperature(field)
    assert gmt.name == "GMT"
    nino = nino34(field)
    np.testing.assert_allclose(nino, field.sel(lat=0, lon=240))


def test_global_mean_ocean_heat_content_uses_latitude_weights() -> None:
    field = xr.DataArray(
        np.array([[[1.0, 1.0], [3.0, 3.0]]]),
        dims=("time", "lat", "lon"),
        coords={"time": [2000], "lat": [0.0, 60.0], "lon": [0.0, 180.0]},
    )

    result = global_mean_ocean_heat_content(field)

    np.testing.assert_allclose(result, [5 / 3])
    assert result.name == "GMOHC"
    assert result.attrs["units"] == "J m-2"


@pytest.mark.parametrize("global_climatology", [False, True])
def test_restore_monthly_sea_ice_climatology(tmp_path, global_climatology) -> None:
    latitudes = [-45.0, 45.0] if global_climatology else [45.0]
    climatology = xr.DataArray(
        np.broadcast_to(np.arange(1.0, 13.0)[:, None, None], (12, len(latitudes), 1)),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2001-01-01", periods=12, freq="MS"),
            "lat": latitudes,
            "lon": [0.0],
        },
        name="sic_climatology",
    )
    path = tmp_path / "sic_climatology.nc"
    climatology.to_dataset().to_netcdf(path, engine="h5netcdf")
    anomaly = xr.DataArray(
        np.ones((2, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": [1000.0, 1000.25], "lat": [45.0], "lon": [0.0]},
    )

    restored = _restore_climatology(
        anomaly,
        ReconstructionClimatologyConfig(path=path, variable="sic_climatology"),
        "missing",
    )

    np.testing.assert_allclose(restored[:, 0, 0], [2.0, 5.0])
    np.testing.assert_array_equal(restored.lat, anomaly.lat)


@pytest.mark.parametrize("coordinate", ["lat", "lon"])
def test_restore_climatology_rejects_mismatched_grid(tmp_path, coordinate) -> None:
    climatology = _field().assign_coords(time=pd.date_range("2001-01-01", periods=12, freq="MS"))
    path = tmp_path / "climatology.nc"
    climatology.to_dataset().to_netcdf(path, engine="h5netcdf")
    field = _field().assign_coords({coordinate: _field()[coordinate] + 0.5})
    with pytest.raises(ValueError, match=f"different {coordinate} grid"):
        _restore_climatology(
            field,
            ReconstructionClimatologyConfig(path=path, variable="tas"),
            "missing",
        )


def test_restore_month_dimension_climatology_uses_complete_physical_quarters(tmp_path):
    climatology = xr.DataArray(
        np.arange(1.0, 13.0)[:, None, None],
        dims=("month", "lat", "lon"),
        coords={"month": np.arange(1, 13), "lat": [45.0], "lon": [0.0]},
        name="sic",
    )
    path = tmp_path / "climatology.nc"
    climatology.to_dataset().to_netcdf(path, engine="h5netcdf")
    field = xr.DataArray(
        np.ones((4, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": [1000.0, 1000.25, 1000.5, 1000.75], "lat": [45.0], "lon": [0.0]},
    )
    config = ReconstructionClimatologyConfig(path=path, variable="sic")
    restored = _restore_climatology(field, config, "missing")
    np.testing.assert_allclose(restored[:, 0, 0], [6.0, 5.0, 8.0, 11.0])
    climatology.isel(month=slice(1, None)).to_dataset().to_netcdf(path, engine="h5netcdf")
    with pytest.raises(ValueError, match="twelve unique"):
        _restore_climatology(field, config, "missing")


def test_series_metrics_align_common_time_and_running_mean() -> None:
    left = xr.DataArray([1.0, 2.0, 4.0, 8.0], dims="time", coords={"time": [0, 1, 2, 3]})
    right = xr.DataArray([2.0, 4.0, 8.0, 16.0], dims="time", coords={"time": [1, 2, 3, 4]})
    metrics = series_metrics(left, right)
    assert metrics.count == 3
    assert metrics.correlation == 1.0
    np.testing.assert_allclose(metrics.rmse, 0.0)
    anomalies = anomaly(left, (0, 1))
    np.testing.assert_allclose(anomalies, [-0.5, 0.5, 2.5, 6.5])
    rolling = centered_running_mean(left, 3)
    assert np.isnan(rolling.isel(time=0)) and np.isnan(rolling.isel(time=-1))
    np.testing.assert_allclose(rolling.isel(time=1), 7 / 3)


def test_seasonal_anomaly_removes_quarter_specific_climatology() -> None:
    field = xr.DataArray(
        [1.0, 10.0, 3.0, 14.0],
        dims="time",
        coords={"time": [2000.0, 2000.25, 2001.0, 2001.25]},
    )

    actual = seasonal_anomaly(field, (2000, 2001))

    np.testing.assert_allclose(actual, [-1.0, -2.0, 1.0, 2.0])


def test_spatial_metrics_are_weighted_and_time_resolved() -> None:
    time = [1000, 1001]
    lat = [-60.0, 0.0, 60.0]
    lon = [0.0, 90.0]
    values = np.arange(12.0).reshape(2, 3, 2)
    left = xr.DataArray(
        values, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}
    )
    right = 2 * left + 3
    metrics = spatial_pattern_metrics(left, right)
    np.testing.assert_allclose(metrics.pattern_correlation, [1.0, 1.0])
    assert metrics.pattern_rmse.sizes["time"] == 2
    np.testing.assert_array_equal(metrics.valid_grid_points, [6, 6])


def test_spatial_metrics_accept_exact_spatial_subset_but_reject_shifted_grid() -> None:
    left = _field()
    subset = left.sel(lat=[45.0])

    metrics = spatial_pattern_metrics(subset, left)

    np.testing.assert_array_equal(metrics.valid_grid_points, np.full(12, 2))
    shifted = left.assign_coords(lat=left.lat + 0.5)
    with pytest.raises(ValueError, match="spatial coordinate lat"):
        spatial_pattern_metrics(left, shifted)


def test_gridpoint_temporal_correlation_is_mapped_and_area_weighted() -> None:
    left = _field()
    right = left.copy()
    right.loc[{"lat": 45.0}] = -right.sel(lat=45.0)

    metrics = gridpoint_temporal_correlation(left, right)

    np.testing.assert_allclose(metrics.gridpoint_temporal_correlation.sel(lat=-45.0), 1.0)
    np.testing.assert_allclose(metrics.gridpoint_temporal_correlation.sel(lat=45.0), -1.0)
    np.testing.assert_allclose(metrics.area_weighted_mean_correlation, 0.0, atol=1e-15)
    np.testing.assert_array_equal(metrics.valid_time_samples, np.full((2, 2), 12))


def test_verification_plots_write_legacy_named_outputs(tmp_path) -> None:
    correlation = xr.DataArray(
        np.array([[[[1.0, 0.5], [-0.5, -1.0]]]]),
        dims=("variable", "aggregation", "lat", "lon"),
        coords={
            "variable": ["tas"],
            "aggregation": ["annual"],
            "lat": [-45.0, 45.0],
            "lon": [0.0, 180.0],
        },
    )
    metrics = xr.Dataset(
        {
            "gridpoint_temporal_correlation": correlation,
            "area_weighted_mean_correlation": correlation.mean(("lat", "lon")),
        }
    )

    map_paths = plot_correlation_maps(metrics, tmp_path / "pattern", dpi=50)

    assert map_paths == (tmp_path / "pattern" / "tas" / "ANN_annual.png",)
    assert map_paths[0].stat().st_size > 0

    years = np.arange(1900, 1910)
    reference = xr.DataArray(np.arange(10.0), dims="time", coords={"time": years})
    reconstruction = reference + 0.1
    index_path = plot_index_comparison(
        "GMT",
        reconstruction,
        reference,
        SeriesMetrics(correlation=1.0, rmse=0.1, count=10),
        (1900, 1909),
        tmp_path / "indices" / "GMT" / "compare_annual.png",
        dpi=50,
    )

    assert index_path.stat().st_size > 0
