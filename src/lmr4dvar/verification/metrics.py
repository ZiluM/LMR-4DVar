"""Explicit common-time series and spatial verification metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class SeriesMetrics:
    correlation: float
    rmse: float
    count: int


def anomaly(field: xr.DataArray, period: tuple[int, int]) -> xr.DataArray:
    """Subtract a baseline selected from numeric or datetime time coordinates."""
    if "time" not in field.dims:
        raise ValueError("field must have a time dimension")
    start, end = sorted(period)
    years = _years(field.time)
    baseline = field.where((years >= start) & (years <= end), drop=True)
    if baseline.sizes.get("time", 0) == 0:
        raise ValueError(f"no samples in anomaly period {period}")
    return field - baseline.mean("time")


def seasonal_anomaly(field: xr.DataArray, period: tuple[int, int]) -> xr.DataArray:
    """Subtract a separate climatology for each numeric quarterly time slot."""
    if "time" not in field.dims:
        raise ValueError("field must have a time dimension")
    years = _years(field.time)
    quarter = np.rint((field.time.astype(float) - years) * 4).astype(int) % 4
    field = field.assign_coords(quarter=("time", np.asarray(quarter)))
    start, end = sorted(period)
    baseline = field.where((years >= start) & (years <= end), drop=True)
    if baseline.sizes.get("time", 0) == 0:
        raise ValueError(f"no samples in anomaly period {period}")
    climatology = baseline.groupby("quarter").mean("time")
    return field.groupby("quarter") - climatology


def centered_running_mean(series: xr.DataArray, window: int, *, min_periods=None) -> xr.DataArray:
    if window < 1:
        raise ValueError("running-mean window must be positive")
    if min_periods is None:
        min_periods = window
    if not 1 <= min_periods <= window:
        raise ValueError("min_periods must be between one and window")
    return series.rolling(time=window, center=True, min_periods=min_periods).mean()


def series_metrics(
    reconstruction: xr.DataArray,
    reference: xr.DataArray,
    *,
    period: tuple[int, int] | None = None,
) -> SeriesMetrics:
    left, right = _common_time(reconstruction, reference, period)
    if left.dims != ("time",) or right.dims != ("time",):
        raise ValueError("series metrics require one-dimensional time series")
    left_values = np.asarray(left, dtype=float).reshape(-1)
    right_values = np.asarray(right, dtype=float).reshape(-1)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if np.count_nonzero(finite) < 2:
        raise ValueError("series metrics require at least two common finite values")
    left_values = left_values[finite]
    right_values = right_values[finite]
    if np.std(left_values) == 0 or np.std(right_values) == 0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(left_values, right_values)[0, 1])
    rmse = float(np.sqrt(np.mean(np.square(left_values - right_values))))
    return SeriesMetrics(correlation, rmse, int(finite.sum()))


def spatial_pattern_metrics(
    reconstruction: xr.DataArray,
    reference: xr.DataArray,
    *,
    period: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Return latitude-weighted pattern correlation and RMSE at common times."""
    left, right = _common_time(reconstruction, reference, period)
    if "lat" not in left.dims or "lon" not in left.dims:
        raise ValueError("spatial metrics require lat and lon dimensions")
    if left.dims != right.dims:
        right = right.transpose(*left.dims)
    weights = xr.broadcast(
        xr.DataArray(np.cos(np.deg2rad(left.lat)), dims="lat", coords={"lat": left.lat}),
        left.isel(time=0, drop=True),
    )[0]
    valid = np.isfinite(left) & np.isfinite(right)
    weights = weights.where(valid)
    weight_sum = weights.sum(("lat", "lon"))
    if bool((weight_sum <= 0).any()):
        raise ValueError("at least one common time has no finite spatial overlap")
    left_mean = (left * weights).sum(("lat", "lon")) / weight_sum
    right_mean = (right * weights).sum(("lat", "lon")) / weight_sum
    left_anomaly = left - left_mean
    right_anomaly = right - right_mean
    covariance = (left_anomaly * right_anomaly * weights).sum(("lat", "lon"))
    left_variance = (left_anomaly**2 * weights).sum(("lat", "lon"))
    right_variance = (right_anomaly**2 * weights).sum(("lat", "lon"))
    denominator = np.sqrt(left_variance * right_variance)
    correlation = xr.where(denominator > 0, covariance / denominator, np.nan)
    rmse = np.sqrt((((left - right) ** 2) * weights).sum(("lat", "lon")) / weight_sum)
    return xr.Dataset(
        {
            "pattern_correlation": correlation,
            "pattern_rmse": rmse,
            "valid_grid_points": valid.sum(("lat", "lon")),
        }
    )


def gridpoint_temporal_correlation(
    reconstruction: xr.DataArray,
    reference: xr.DataArray,
    *,
    period: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Return gridpoint temporal correlation and its area-weighted spatial mean."""
    left, right = _common_time(reconstruction, reference, period)
    if "lat" not in left.dims or "lon" not in left.dims:
        raise ValueError("gridpoint temporal correlation requires lat and lon dimensions")
    if left.dims != right.dims:
        right = right.transpose(*left.dims)
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum("time")
    left_valid = left.where(valid)
    right_valid = right.where(valid)
    left_mean = left_valid.sum("time") / count
    right_mean = right_valid.sum("time") / count
    left_anomaly = left_valid - left_mean
    right_anomaly = right_valid - right_mean
    covariance = (left_anomaly * right_anomaly).sum("time")
    left_variance = (left_anomaly**2).sum("time")
    right_variance = (right_anomaly**2).sum("time")
    denominator = np.sqrt(left_variance * right_variance)
    correlation = xr.where((count >= 2) & (denominator > 0), covariance / denominator, np.nan)
    weights = xr.DataArray(
        np.cos(np.deg2rad(correlation.lat)),
        dims="lat",
        coords={"lat": correlation.lat},
    )
    mean_correlation = correlation.weighted(weights).mean(("lat", "lon"))
    correlation.attrs.update(
        {
            "long_name": "gridpoint Pearson correlation through time",
            "correlation_dimension": "time",
        }
    )
    count.attrs["long_name"] = "paired finite time samples used for correlation"
    mean_correlation.attrs.update(
        {
            "long_name": "area-weighted mean gridpoint temporal correlation",
            "spatial_weighting": "cosine(latitude)",
        }
    )
    return xr.Dataset(
        {
            "gridpoint_temporal_correlation": correlation,
            "valid_time_samples": count,
            "area_weighted_mean_correlation": mean_correlation,
        }
    )


def _common_time(left, right, period):
    left_spatial = {dim for dim in left.dims if dim != "time"}
    right_spatial = {dim for dim in right.dims if dim != "time"}
    if left_spatial != right_spatial:
        raise ValueError("inputs have different non-time dimensions")
    for dimension in left_spatial:
        left_values = np.asarray(left[dimension])
        right_values = np.asarray(right[dimension])
        common = np.intersect1d(left_values, right_values)
        if common.size == 0 or common.size != min(left_values.size, right_values.size):
            raise ValueError(f"inputs differ on spatial coordinate {dimension}")
    left, right = xr.align(left, right, join="inner")
    if left.sizes.get("time", 0) == 0:
        raise ValueError("inputs have no common times")
    if period is not None:
        start, end = sorted(period)
        years = _years(left.time)
        selection = (years >= start) & (years <= end)
        left = left.where(selection, drop=True)
        right = right.where(selection, drop=True)
        if left.sizes.get("time", 0) == 0:
            raise ValueError(f"no common samples in comparison period {period}")
    return left, right


def _years(time: xr.DataArray) -> xr.DataArray:
    try:
        return time.dt.year
    except AttributeError:
        return np.floor(time.astype(float)).astype(int)
