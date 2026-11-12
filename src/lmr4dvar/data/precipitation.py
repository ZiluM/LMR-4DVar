"""Explicit precipitation-rate conversion, including calendar-month totals."""

from __future__ import annotations

import xarray as xr


def precipitation_mm_per_day(
    field: xr.DataArray, *, source_units: str | None = None
) -> xr.DataArray:
    units = source_units or field.attrs.get("units")
    aliases = {
        "mm/day": "mm/day",
        "mm d-1": "mm/day",
        "mm/month": "mm/month",
        "mm month-1": "mm/month",
        "m/s": "m/s",
        "m s-1": "m/s",
        "kg m-2 s-1": "kg m-2 s-1",
        "kg/m2/s": "kg m-2 s-1",
    }
    if units not in aliases:
        raise ValueError(f"unsupported or missing precipitation units: {units!r}")
    if source_units is not None and field.attrs.get("units") is not None:
        declared = aliases.get(field.attrs["units"])
        if declared != aliases[units]:
            raise ValueError("configured precipitation units contradict declared source units")
    kind = aliases[units]
    if kind == "mm/month":
        if "time" not in field.dims:
            raise ValueError("monthly precipitation totals require calendar time")
        result = field / field.time.dt.days_in_month
    elif kind == "m/s":
        result = field * 86400000.0
    elif kind == "kg m-2 s-1":
        result = field * 86400.0
    else:
        result = field.copy()
    result.attrs = field.attrs.copy()
    result.attrs.update(units="mm/day", precipitation_source_units=units)
    return result
