"""Common-grid precipitation diagnostics and conservative rectilinear remapping."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix

from lmr4dvar.data.preprocess import quarterly_mean
from lmr4dvar.reconstruction.indices import latitude_weighted_mean
from lmr4dvar.verification.metrics import (
    gridpoint_temporal_correlation,
    seasonal_anomaly,
    series_metrics,
    spatial_pattern_metrics,
)


def conservative_grid(field: xr.DataArray, target: xr.Dataset) -> xr.DataArray:
    """First-order spherical cell overlaps; full longitude and latitude coverage.

    Latitude cell edges are inferred from centers, with polar edges at +/-90.
    Longitude centers must be uniformly spaced around a complete periodic grid.
    Missing contributing source cells invalidate the destination cell.
    """
    field = field.assign_coords(lon=field.lon % 360).sortby("lon").sortby("lat")
    lat, lon = np.asarray(field.lat), np.asarray(field.lon)
    tlat, tlon = np.asarray(target.lat), np.asarray(target.lon)
    if np.any(np.diff(lat) <= 0) or np.any(np.diff(tlat) <= 0):
        raise ValueError("latitudes must be unique and increasing")
    if len(lat) < 2 or len(tlat) < 2:
        raise ValueError("at least two latitude cells required")
    slat = np.sin(np.deg2rad(np.r_[-90, (lat[:-1] + lat[1:]) / 2, 90]))
    dlat = np.sin(np.deg2rad(np.r_[-90, (tlat[:-1] + tlat[1:]) / 2, 90]))
    wy = _overlap(slat, dlat)
    sx, dx = _longitude_edges(lon), _longitude_edges(tlon)
    wx = np.zeros((len(tlon), len(lon)))
    for shift in (-360, 0, 360):
        wx += _overlap(sx + shift, dx)
    if not np.allclose(wy.sum(1), 1) or not np.allclose(wx.sum(1), 1):
        raise ValueError("source grid does not completely cover target")
    wy, wx = csr_matrix(wy), csr_matrix(wx)
    prefix = tuple(dim for dim in field.dims if dim not in ("lat", "lon"))
    data = np.asarray(field.transpose(*prefix, "lat", "lon"), dtype=float)
    flat = data.reshape((-1, len(lat), len(lon)))
    result = []
    for plane in flat:
        finite = np.isfinite(plane)
        values = wx.dot(wy.dot(np.where(finite, plane, 0)).T).T
        coverage = wx.dot(wy.dot(finite.astype(float)).T).T
        result.append(np.where(coverage >= 1 - 1e-10, values, np.nan))
    values = np.asarray(result).reshape((*data.shape[:-2], len(tlat), len(tlon)))
    coordinates = {dim: field[dim] for dim in prefix}
    coordinates.update(lat=tlat, lon=tlon)
    output = xr.DataArray(values, dims=(*prefix, "lat", "lon"), coords=coordinates)
    output.attrs = field.attrs.copy()
    output.attrs["regridding"] = "first-order conservative spherical overlaps; inferred cell edges"
    return output


def _overlap(source, destination):
    overlap = np.maximum(
        0,
        np.minimum(destination[1:, None], source[None, 1:])
        - np.maximum(destination[:-1, None], source[None, :-1]),
    )
    return overlap / np.diff(destination)[:, None]


def _longitude_edges(centers):
    if len(centers) < 2:
        raise ValueError("at least two longitude cells required")
    spacing = 360 / len(centers)
    if not np.allclose(np.diff(centers), spacing, atol=1e-4):
        raise ValueError("longitude grid must be uniform and global")
    # Some GRIB-derived coordinates have a rounded final longitude. Use the
    # cyclic neighbor, rather than a nominal spacing, to close exactly at 360.
    first = (centers[-1] - 360 + centers[0]) / 2
    return np.r_[first, (centers[:-1] + centers[1:]) / 2, first + 360]


def monthly_to_quarters(field: xr.DataArray) -> xr.DataArray:
    """Require exactly one consecutive sample per physical month before averaging."""
    months = np.asarray(field.time.dt.year) * 12 + np.asarray(field.time.dt.month)
    if len(months) < 3 or np.any(np.diff(months) != 1):
        raise ValueError("monthly reference has duplicate or missing months")
    result = quarterly_mean(field)
    result = result.assign_coords(
        time=np.asarray(result.time.dt.year) + (np.asarray(result.time.dt.month) - 1) / 12
    )
    return result


def era5_interval_midpoints(initial_times, forecast_hours):
    """ERA5 MTPR is the mean rate in the hour preceding each validity time."""
    initial = np.asarray(initial_times, dtype="datetime64[ns]")
    hours = np.asarray(forecast_hours)
    if not np.array_equal(hours, np.arange(1, 13)):
        raise ValueError("ERA5 hourly MTPR requires forecast steps 1 through 12")
    if np.any(np.diff(initial) != np.timedelta64(12, "h")):
        raise ValueError("ERA5 forecast initializations must be consecutive 12-hour cycles")
    return initial[:, None] + hours[None, :].astype("timedelta64[h]") - np.timedelta64(30, "m")


def temporal_map_title(region, correlation):
    return f"{region}: Temporal correlation\nArea-weighted R={correlation:.3f}"


def verify_product(reconstructed, reference, region_mask, *, baseline, period):
    """Quarter/annual/seasonal anomaly diagnostics on fixed paired coverage."""
    for field in (reconstructed, reference):
        if field.attrs.get("units") != "mm/day":
            raise ValueError("both precipitation fields must have units mm/day")
        times = np.asarray(field.time, dtype=float)
        if np.any(np.diff(times) <= 0) or not np.allclose(times * 4, np.rint(times * 4)):
            raise ValueError("quarter coordinates must be unique, ordered year+quarter/4")
        baseline_times = times[(times >= baseline[0]) & (times < baseline[1] + 1)]
        expected = np.arange(baseline[0], baseline[1] + 1, 0.25)
        if not np.array_equal(baseline_times, expected):
            raise ValueError("reference and reconstruction require every baseline quarter")
    left = seasonal_anomaly(reconstructed, baseline)
    right = seasonal_anomaly(reference, baseline)
    left, right = xr.align(left, right, join="inner")
    years = np.floor(left.time).astype(int)
    selected = (years >= period[0]) & (years <= period[1])
    left, right = left.isel(time=selected), right.isel(time=selected)
    if not left.sizes["time"]:
        raise ValueError("no common comparison quarters")
    mask = region_mask & (np.isfinite(left) & np.isfinite(right)).all("time")
    if int(mask.sum()) < 2:
        raise ValueError("insufficient fixed spatial overlap")
    left, right = left.where(mask), right.where(mask)
    year = xr.DataArray(np.floor(left.time).astype(int).values, dims="time", name="year")
    counts = xr.DataArray(np.ones(len(year)), dims="time").groupby(year).sum()
    complete = counts.year.values[counts.values == 4]
    annual_left = left.groupby(year).mean("time").sel(year=complete).rename(year="time")
    annual_right = right.groupby(year).mean("time").sel(year=complete).rename(year="time")
    annual_pattern = spatial_pattern_metrics(annual_left, annual_right)
    quarterly_pattern = spatial_pattern_metrics(left, right)
    means = (latitude_weighted_mean(annual_left), latitude_weighted_mean(annual_right))
    metrics = series_metrics(*means)
    seasons = np.rint((np.asarray(left.time) % 1) * 4).astype(int)
    aggregations = [("annual", annual_left, annual_right), ("quarterly", left, right)]
    for index, name in enumerate(("DJF", "MAM", "JJA", "SON")):
        aggregations.append(
            (name, left.isel(time=seasons == index), right.isel(time=seasons == index))
        )
    maps, summary = [], {}
    for name, a, b in aggregations:
        diagnostics = gridpoint_temporal_correlation(a, b)
        variance = ((b - b.mean("time")) ** 2).sum("time")
        left_variance = ((a - a.mean("time")) ** 2).sum("time")
        threshold = 1e-12 * a.sizes["time"]
        stable = (variance > threshold) & (left_variance > threshold)
        diagnostics["gridpoint_temporal_correlation"] = (
            diagnostics.gridpoint_temporal_correlation.where(stable)
        )
        diagnostics["area_weighted_mean_correlation"] = latitude_weighted_mean(
            diagnostics.gridpoint_temporal_correlation
        )
        diagnostics["ce"] = xr.where(
            variance > threshold, 1 - ((a - b) ** 2).sum("time") / variance, np.nan
        )
        diagnostics["rmse"] = np.sqrt(((a - b) ** 2).mean("time"))
        summary[name] = {
            "gridpoint_temporal_r": float(diagnostics.area_weighted_mean_correlation),
            "gridpoint_ce": float(latitude_weighted_mean(diagnostics.ce)),
            "gridpoint_rmse_mm_day": float(latitude_weighted_mean(diagnostics.rmse)),
            "time_samples": a.sizes["time"],
            "time_range": [float(a.time[0]), float(a.time[-1])],
            "ce_valid_gridpoints": int(diagnostics.ce.count()),
            "temporal_r_valid_gridpoints": int(diagnostics.gridpoint_temporal_correlation.count()),
        }
        maps.append(diagnostics.expand_dims(aggregation=[name]))
    dataset = xr.concat(maps, dim="aggregation")
    dataset["common_coverage_mask"] = mask
    dataset["reconstruction_annual_mean"] = means[0].rename(time="year")
    dataset["reference_annual_mean"] = means[1].rename(time="year")
    dataset["annual_pattern_r"] = annual_pattern.pattern_correlation.rename(time="year")
    dataset["quarterly_pattern_r"] = quarterly_pattern.pattern_correlation.rename(
        time="quarter_time"
    )
    dataset.attrs.update(
        units="mm/day",
        anomaly_period=f"{baseline[0]}-{baseline[1]}",
        requested_comparison_period=f"{period[0]}-{period[1]}",
        comparison_period=f"{int(np.floor(left.time[0]))}-{int(np.floor(left.time[-1]))}",
        annual_period=f"{int(complete[0])}-{int(complete[-1])}",
        quarterly_time_start=float(left.time[0]),
        quarterly_time_end=float(left.time[-1]),
        mean_definition="cosine-latitude weights; fixed paired coverage",
        annual_definition="equal average of four complete DJF/MAM/JJA/SON quarters",
        minimum_temporal_standard_deviation_mm_day=1e-6,
    )
    summary.update(
        annual_mean=metrics.__dict__,
        coverage_gridpoints=int(mask.sum()),
        annual_spatial_pattern_r=float(annual_pattern.pattern_correlation.mean()),
        quarterly_spatial_pattern_r=float(quarterly_pattern.pattern_correlation.mean()),
        annual_period=[int(complete[0]), int(complete[-1])],
        quarterly_time_range=[float(left.time[0]), float(left.time[-1])],
    )
    return dataset, summary
