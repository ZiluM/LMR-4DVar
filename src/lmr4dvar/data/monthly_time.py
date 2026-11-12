"""Explicit physical month labels for interval-averaged simulation inputs."""

from __future__ import annotations

import numpy as np
import xarray as xr


def label_monthly_intervals(
    dataset: xr.Dataset,
    *,
    bounds_source: xr.Dataset | None = None,
    drop_partial_boundaries: bool = False,
    restrict_to_bounds_source: bool = False,
) -> xr.Dataset:
    """Use CF bounds to label complete calendar-month averages at month start.

    CESM CAM h0 files can label January means at February 1. Never infer
    physical months from such coordinates alone. This opt-in transformation
    leaves values and interval bounds unchanged and rejects ambiguous inputs.
    """
    if "time" not in dataset.coords or dataset.time.dims != ("time",):
        raise ValueError("monthly inputs require a 1D time coordinate")
    if bounds_source is not None:
        if restrict_to_bounds_source:
            indices = dataset.indexes["time"].get_indexer(bounds_source.time.values)
            if np.any(indices < 0) or np.any(np.diff(indices) != 1):
                raise ValueError(
                    "external bounds coverage is not a contiguous exact source-time subset"
                )
            dataset = dataset.isel(time=indices)
            dataset.attrs = dataset.attrs.copy()
            dataset.attrs["time_bounds_coverage_selection"] = (
                "explicit contiguous bounds-source subset"
            )
        if not np.array_equal(dataset.time.values, bounds_source.time.values):
            raise ValueError("external monthly bounds source has different time coordinates")
        original_bounds = dataset.time.attrs.get("bounds")
        if original_bounds and original_bounds in dataset:
            raise ValueError("external bounds must not replace existing input time bounds")
        source_name = bounds_source.time.attrs.get("bounds")
        if not source_name or source_name not in bounds_source:
            raise ValueError("external bounds source lacks explicit CF time bounds")
        dataset = dataset.assign(lmr_monthly_bounds=bounds_source[source_name])
        dataset.time.attrs = {**dataset.time.attrs, "bounds": "lmr_monthly_bounds"}
    bounds_name = dataset.time.attrs.get("bounds")
    if not bounds_name or bounds_name not in dataset:
        raise ValueError("monthly time correction requires explicit CF time bounds")
    bounds = dataset[bounds_name]
    other = [dim for dim in bounds.dims if dim != "time"]
    if bounds.ndim != 2 or "time" not in bounds.dims or len(other) != 1:
        raise ValueError("monthly time bounds must have dimensions (time, bounds)")
    if bounds.sizes[other[0]] != 2 or not dataset.sizes["time"]:
        raise ValueError("monthly time bounds require two endpoints and nonempty time")
    bounds = bounds.transpose("time", other[0])
    lower = bounds.isel({other[0]: 0}, drop=True)
    upper = bounds.isel({other[0]: 1}, drop=True)
    start_masks = []
    for endpoint in (lower, upper):
        try:
            valid = (
                (endpoint.dt.day == 1)
                & (endpoint.dt.hour == 0)
                & (endpoint.dt.minute == 0)
                & (endpoint.dt.second == 0)
                & (endpoint.dt.microsecond == 0)
            )
        except (AttributeError, TypeError) as exc:
            raise ValueError("monthly bounds must be decoded calendar dates") from exc
        start_masks.append(valid.values)
    lower_month = lower.dt.year.values * 12 + lower.dt.month.values
    upper_month = upper.dt.year.values * 12 + upper.dt.month.values
    complete = start_masks[0] & start_masks[1] & (upper_month - lower_month == 1)
    if drop_partial_boundaries and not complete.all():
        valid = np.flatnonzero(complete)
        if not valid.size or not complete[valid[0] : valid[-1] + 1].all():
            raise ValueError("incomplete internal monthly interval or no complete months")
        result = label_monthly_intervals(dataset.isel(time=slice(valid[0], valid[-1] + 1)))
        result.attrs["dropped_partial_boundary_months"] = int((~complete).sum())
        return result
    if not (start_masks[0] & start_masks[1]).all():
        raise ValueError("monthly bounds must be midnight calendar-month starts")
    if not np.all(upper_month - lower_month == 1):
        raise ValueError("each monthly interval must span exactly one calendar month")
    if not np.all(np.diff(lower_month) == 1):
        raise ValueError("monthly intervals must be consecutive, ordered and unique")
    try:
        inside = (dataset.time.values >= lower.values) & (dataset.time.values <= upper.values)
    except TypeError as exc:
        raise ValueError("time coordinates and bounds must use compatible calendars") from exc
    if not inside.all():
        raise ValueError("time coordinate must lie inside its declared monthly interval")
    time = xr.DataArray(lower.values, dims="time", attrs=dataset.time.attrs.copy())
    time.encoding = dataset.time.encoding.copy()
    result = dataset.assign_coords(time=time)
    result.attrs = dataset.attrs.copy()
    result.attrs["physical_month_labels"] = f"calendar-month starts from CF bounds {bounds_name}"
    return result
