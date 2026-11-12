"""Reusable climate-field preprocessing operations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import xarray as xr
from scipy import signal

from lmr4dvar.config.models import Domain, QuarterlyMode


def regular_lat_lon_grid(resolution: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-center coordinates matching CDO's global rNxM grids."""
    nlon = round(360.0 / resolution)
    nlat = round(180.0 / resolution)
    if not np.isclose(nlon * resolution, 360.0) or not np.isclose(nlat * resolution, 180.0):
        raise ValueError("resolution must divide both 360 and 180 degrees")
    lon = np.arange(nlon, dtype=float) * resolution
    lat = -90.0 + resolution / 2.0 + np.arange(nlat, dtype=float) * resolution
    return lat, lon


def normalize_longitude(field: xr.DataArray) -> xr.DataArray:
    """Normalize longitude to [0, 360) and sort it."""
    if "lon" not in field.coords:
        raise ValueError("field must have a lon coordinate")
    normalized = field.assign_coords(lon=np.mod(field.lon, 360.0)).sortby("lon")
    if np.unique(normalized.lon.values).size != normalized.sizes["lon"]:
        raise ValueError("longitude normalization produced duplicate coordinates")
    return normalized


def regrid_regular(field: xr.DataArray, resolution: float = 2.0) -> xr.DataArray:
    """Linearly interpolate a rectilinear field to a regular cell-center grid."""
    if "lat" not in field.coords or "lon" not in field.coords:
        raise ValueError("field must have lat and lon coordinates")
    source = normalize_longitude(field).sortby("lat")
    lat, lon = regular_lat_lon_grid(resolution)
    result = source.interp(lat=lat, lon=lon, method="linear")
    result.attrs = field.attrs.copy()
    return result


def monthly_climatology_anomaly(
    field: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Remove the full-record monthly climatology."""
    _require_time(field)
    climatology = field.groupby("time.month").mean("time", keep_attrs=True)
    anomaly = field.groupby("time.month") - climatology
    anomaly.attrs = field.attrs.copy()
    anomaly.attrs["anomaly_reference"] = "full-record monthly climatology"
    return anomaly, climatology


def quarterly_mean(
    field: xr.DataArray,
    mode: QuarterlyMode | str = QuarterlyMode.COMPLETE,
) -> xr.DataArray:
    """Average DJF-aligned quarters with explicit incomplete-edge handling."""
    _require_time(field)
    mode = QuarterlyMode(mode)
    quarterly = field.resample(time="QS-DEC").mean(keep_attrs=True)
    counts = (
        xr.DataArray(
            np.ones(field.sizes["time"], dtype=np.int8),
            coords={"time": field.time},
            dims="time",
        )
        .resample(time="QS-DEC")
        .sum()
    )

    if mode is QuarterlyMode.COMPLETE:
        quarterly = quarterly.where(counts == 3, drop=True)
        quarterly = quarterly.assign_coords(time=quarterly.indexes["time"].shift(1, "MS"))
    else:
        legacy_time = list(field.time.values[::3])
        if not np.array_equal(field.time.values[-1:], field.time.values[::3][-1:]):
            legacy_time.append(field.time.values[-1])
        if len(legacy_time) != quarterly.sizes["time"]:
            raise ValueError(
                "legacy_qs_dec timestamps require a complete monthly series starting in January"
            )
        quarterly = quarterly.assign_coords(time=legacy_time)

    quarterly.attrs = field.attrs.copy()
    quarterly.attrs["quarterly_mode"] = mode.value
    return quarterly


def linear_detrend_by_month(field: xr.DataArray) -> xr.DataArray:
    """Linearly detrend each calendar-month group, matching legacy sacpy behavior."""
    _require_time(field)
    result = field.groupby("time.month").map(_detrend_time)
    result.attrs = field.attrs.copy()
    result.attrs["detrend"] = "linear separately within each calendar month"
    return result


def select_domain(field: xr.DataArray, domain: Domain | str) -> xr.DataArray:
    """Select a configured latitude domain."""
    domain = Domain(domain)
    if domain is Domain.GLOBAL:
        return field
    if "lat" not in field.coords:
        raise ValueError("field must have a lat coordinate for domain selection")
    return field.where(field.lat >= 0, drop=True)


def select_year_period(field: xr.DataArray, period: tuple[int, int] | None) -> xr.DataArray:
    """Select an inclusive year range without assuming a specific calendar."""
    if period is None:
        return field
    _require_time(field)
    selected = field.where(
        (field.time.dt.year >= period[0]) & (field.time.dt.year <= period[1]),
        drop=True,
    )
    if selected.sizes["time"] == 0:
        raise ValueError(f"training period {period[0]}-{period[1]} selects no samples")
    return selected


def preprocess_field(
    field: xr.DataArray,
    *,
    domain: Domain | str,
    resolution: float = 2.0,
    regrid: bool = True,
    remove_climatology: bool = True,
    quarterly_mode: QuarterlyMode | str = QuarterlyMode.COMPLETE,
    seasonal_months: Iterable[int] = (1, 4, 7, 10),
    detrend: bool = True,
    training_period: tuple[int, int] | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Apply the standard simulation-to-quarterly-field preprocessing chain."""
    result = select_year_period(field, training_period)
    result = regrid_regular(result, resolution) if regrid else result.copy()
    climatology = None
    if remove_climatology:
        result, climatology = monthly_climatology_anomaly(result)
    result = quarterly_mean(result, quarterly_mode)
    months = tuple(seasonal_months)
    result = result.where(result.time.dt.month.isin(months), drop=True)
    if detrend:
        result = linear_detrend_by_month(result)
    result = select_domain(result, domain)
    result.attrs["preprocessing"] = (
        f"regrid={regrid}; resolution={resolution}; climatology={remove_climatology}; "
        f"quarterly={QuarterlyMode(quarterly_mode).value}; detrend={detrend}"
    )
    return result, climatology


def prepare_quarterly_field(
    field: xr.DataArray,
    *,
    domain: Domain | str,
    seasonal_months: Iterable[int] = (1, 4, 7, 10),
    training_period: tuple[int, int] | None = None,
) -> xr.DataArray:
    """Validate and select an already anomalized, quarterly simulation field."""
    result = select_year_period(field, training_period)
    months = tuple(seasonal_months)
    result = result.where(result.time.dt.month.isin(months), drop=True)
    if result.sizes["time"] == 0:
        raise ValueError("prepared quarterly input has no configured seasonal months")
    result = select_domain(result, domain)
    result.attrs = field.attrs.copy()
    result.attrs["preprocessing"] = "prepared_quarterly; no transformations applied"
    return result


def _detrend_time(group: xr.DataArray) -> xr.DataArray:
    def detrend(values: np.ndarray) -> np.ndarray:
        missing = np.any(~np.isfinite(values), axis=-1)
        safe = np.where(np.isfinite(values), values, 0.0)
        output = signal.detrend(safe, axis=-1, type="linear")
        return np.where(missing[..., np.newaxis], np.nan, output)

    return xr.apply_ufunc(
        detrend,
        group,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[float],
    ).transpose(*group.dims)


def complete_quarterly_years(field: xr.DataArray) -> xr.DataArray:
    """Trim incomplete boundary years; reject missing/reordered internal quarters."""
    years = np.asarray(field.time.dt.year)
    months = np.asarray(field.time.dt.month)
    if not years.size or not np.isin(months, [1, 4, 7, 10]).all():
        raise ValueError("complete quarterly years require Jan/Apr/Jul/Oct samples")
    labels = years * 12 + months
    if np.any(np.diff(labels) <= 0):
        raise ValueError("quarterly time must be ordered and unique")
    if np.any(np.diff(labels) != 3):
        raise ValueError("missing quarters within the training period")
    complete = [
        year for year in np.unique(years) if np.array_equal(months[years == year], [1, 4, 7, 10])
    ]
    if not complete:
        raise ValueError("no complete quarterly training years")
    expected = np.arange(complete[0], complete[-1] + 1)
    if not np.array_equal(complete, expected):
        raise ValueError("missing quarters within the training period")
    interior = (years > years.min()) & (years < years.max())
    if not np.isin(years[interior], complete).all():
        raise ValueError("incomplete internal training year")
    result = field.where(field.time.dt.year.isin(complete), drop=True)
    result.attrs = field.attrs.copy()
    result.attrs["complete_quarterly_years"] = f"{complete[0]}-{complete[-1]}"
    return result


def _require_time(field: xr.DataArray) -> None:
    if "time" not in field.dims:
        raise ValueError("field must have a time dimension")
