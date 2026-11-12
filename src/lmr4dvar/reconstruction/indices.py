"""Area-aware climate indices used by LMR4DVar verification."""

from __future__ import annotations

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0


def latitude_weighted_mean(field: xr.DataArray, *, lat_bounds=None) -> xr.DataArray:
    """Mean over latitude/longitude using cosine latitude or explicit bounds."""
    _require_lat_lon(field)
    if lat_bounds is not None:
        south, north = sorted(lat_bounds)
        field = field.sel(lat=slice(south, north))
    weights = xr.DataArray(
        np.cos(np.deg2rad(field.lat)), dims="lat", coords={"lat": field.lat}
    )
    return field.weighted(weights).mean(("lat", "lon"))


def global_mean_temperature(tas: xr.DataArray) -> xr.DataArray:
    result = latitude_weighted_mean(tas)
    result.name = "GMT"
    result.attrs.update({"long_name": "global mean surface air temperature"})
    return result


def global_mean_ocean_heat_content(ohc: xr.DataArray, *, name: str = "GMOHC") -> xr.DataArray:
    result = latitude_weighted_mean(ohc)
    result.name = name
    result.attrs.update(
        {"long_name": "global mean ocean heat content", "units": "J m-2"}
    )
    return result


def nino34(tos: xr.DataArray) -> xr.DataArray:
    """Area-weighted SST over 5S--5N, 170W--120W."""
    _require_lat_lon(tos)
    longitude = tos.lon % 360
    region = tos.where(
        (tos.lat >= -5) & (tos.lat <= 5) & (longitude >= 190) & (longitude <= 240),
        drop=True,
    )
    if region.sizes.get("lat", 0) == 0 or region.sizes.get("lon", 0) == 0:
        raise ValueError("tos grid has no points inside the Nino3.4 region")
    result = latitude_weighted_mean(region)
    result.name = "Nino34"
    return result


def regular_grid_cell_area(field: xr.DataArray) -> xr.DataArray:
    """Calculate spherical cell areas from monotonic 1D center coordinates."""
    _require_lat_lon(field)
    lat = np.asarray(field.lat, dtype=float)
    lon = np.asarray(field.lon, dtype=float)
    if lat.size < 2 or lon.size < 2 or np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        raise ValueError("area calculation requires increasing 1D lat/lon coordinates")
    lat_edges = _center_edges(lat, lower=-90, upper=90)
    lon_edges = _center_edges(lon, lower=None, upper=None)
    lat_factor = np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1]))
    lon_width = np.deg2rad(np.diff(lon_edges))
    area = EARTH_RADIUS_M**2 * lat_factor[:, None] * lon_width[None, :]
    return xr.DataArray(area, dims=("lat", "lon"), coords={"lat": field.lat, "lon": field.lon})


def sea_ice_area(
    concentration: xr.DataArray,
    *,
    cell_area: xr.DataArray | None = None,
    concentration_scale: float = 1.0,
    northern_hemisphere: bool = True,
) -> xr.DataArray:
    if concentration_scale <= 0:
        raise ValueError("concentration_scale must be positive")
    concentration = _hemisphere(concentration, northern_hemisphere)
    area = regular_grid_cell_area(concentration) if cell_area is None else cell_area
    result = ((concentration / concentration_scale) * area).sum(("lat", "lon"), skipna=True)
    result.name = "SIA"
    result.attrs["units"] = "m2"
    return result


def sea_ice_volume(
    concentration: xr.DataArray,
    thickness: xr.DataArray,
    *,
    cell_area: xr.DataArray | None = None,
    concentration_scale: float = 1.0,
    northern_hemisphere: bool = True,
) -> xr.DataArray:
    concentration, thickness = xr.align(concentration, thickness, join="exact")
    concentration = _hemisphere(concentration, northern_hemisphere)
    thickness = _hemisphere(thickness, northern_hemisphere)
    area = regular_grid_cell_area(concentration) if cell_area is None else cell_area
    result = (
        (concentration / concentration_scale) * thickness * area
    ).sum(("lat", "lon"), skipna=True)
    result.name = "SIV"
    result.attrs["units"] = "m3"
    return result


def _hemisphere(field: xr.DataArray, northern: bool) -> xr.DataArray:
    _require_lat_lon(field)
    return field.where(field.lat >= 0, drop=True) if northern else field


def _require_lat_lon(field: xr.DataArray) -> None:
    if "lat" not in field.dims or "lon" not in field.dims:
        raise ValueError("field must have 1D lat and lon dimensions")


def _center_edges(values: np.ndarray, lower: float | None, upper: float | None) -> np.ndarray:
    edges = np.empty(values.size + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    if lower is not None:
        edges[0] = max(edges[0], lower)
    if upper is not None:
        edges[-1] = min(edges[-1], upper)
    return edges
