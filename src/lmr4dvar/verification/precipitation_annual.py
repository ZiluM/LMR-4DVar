"""Annual-only precipitation diagnostics, without synthetic seasonal replication."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .metrics import gridpoint_temporal_correlation, series_metrics, spatial_pattern_metrics


def annual_years(field: xr.DataArray) -> xr.DataArray:
    """Label annual fields by integer year, preserving rather than averaging samples."""
    try:
        years = field.time.dt.year.values
    except AttributeError:
        values = np.asarray(field.time, dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, np.rint(values)):
            raise ValueError("annual time must consist of integer years or decoded dates") from None
        years = values.astype(int)
    if np.any(np.diff(years) <= 0):
        raise ValueError("annual years must be unique and increasing")
    return field.assign_coords(time=years)


def complete_quarter_years(field: xr.DataArray) -> xr.DataArray:
    times = np.asarray(field.time, dtype=float)
    if np.any(np.diff(times) <= 0) or not np.allclose(times * 4, np.rint(times * 4)):
        raise ValueError("quarters must be unique, increasing, and year+quarter/4")
    years = np.floor(times).astype(int)
    labels, counts = np.unique(years, return_counts=True)
    annual = field.assign_coords(year=("time", years)).groupby("year").mean("time", skipna=False)
    return annual.sel(year=labels[counts == 4]).rename(year="time")


def verify_annual_pair(reconstruction, reference, mask, *, baseline):
    """Evaluate already matched years with an externally shared fixed coverage mask."""
    reconstruction, reference = (annual_years(f) for f in (reconstruction, reference))
    for field in (reconstruction, reference):
        if field.attrs.get("units") != "mm/day":
            raise ValueError("annual precipitation requires mm/day units")
        expected = np.arange(baseline[0], baseline[1] + 1)
        actual = field.time.sel(time=slice(*baseline)).values
        if not np.array_equal(actual, expected):
            raise ValueError("every annual baseline year is required")
    left, right = xr.align(reconstruction, reference, join="exact")
    if left.sizes["time"] < 2:
        raise ValueError("annual verification needs at least two matched years")
    if int(mask.sum()) < 2:
        raise ValueError("annual verification needs at least two common cells")
    left = (left - left.sel(time=slice(*baseline)).mean("time")).where(mask)
    right = (right - right.sel(time=slice(*baseline)).mean("time")).where(mask)
    if bool(((~np.isfinite(left) | ~np.isfinite(right)) & mask).any()):
        raise ValueError("shared coverage includes nonfinite annual values")
    weights = xr.DataArray(np.cos(np.deg2rad(left.lat)), dims="lat", coords={"lat": left.lat})

    def mean(field):
        return field.weighted(weights).mean(("lat", "lon"))

    maps = gridpoint_temporal_correlation(left, right)
    n = left.sizes["time"]
    left_var = ((left - left.mean("time")) ** 2).sum("time")
    right_var = ((right - right.mean("time")) ** 2).sum("time")
    maps["gridpoint_temporal_correlation"] = maps.gridpoint_temporal_correlation.where(
        (left_var > 1e-12 * n) & (right_var > 1e-12 * n)
    )
    maps["ce"] = (1 - ((left - right) ** 2).sum("time") / right_var).where(right_var > 1e-12 * n)
    maps["rmse"] = np.sqrt(((left - right) ** 2).mean("time"))
    maps["reconstruction_mean"] = mean(left)
    maps["reference_mean"] = mean(right)
    maps["pattern_r"] = spatial_pattern_metrics(left, right).pattern_correlation
    maps["coverage_mask"] = mask
    maps.attrs.update(units="mm/day", baseline=f"{baseline[0]}-{baseline[1]}")
    return maps, {
        "period": [int(left.time[0]), int(left.time[-1])],
        "count": n,
        "coverage_gridpoints": int(mask.sum()),
        "annual_mean": series_metrics(maps.reconstruction_mean, maps.reference_mean).__dict__,
        "gridpoint_temporal_r": float(mean(maps.gridpoint_temporal_correlation)),
        "gridpoint_ce": float(mean(maps.ce)),
        "gridpoint_rmse_mm_day": float(mean(maps.rmse)),
        "spatial_pattern_r": float(maps.pattern_r.mean()),
    }
