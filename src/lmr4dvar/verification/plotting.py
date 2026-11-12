"""Headless verification figures compatible with the legacy output set."""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .metrics import SeriesMetrics

plt.switch_backend("Agg")


_AGGREGATION_FILENAMES = {
    "quarterly": "all.png",
    "annual": "ANN_annual.png",
    "DJF": "DJF_annual.png",
    "MAM": "MAM_annual.png",
    "JJA": "JJA_annual.png",
    "SON": "SON_annual.png",
}


def plot_correlation_maps(
    metrics: xr.Dataset,
    output_directory: Path,
    *,
    dpi: int = 150,
) -> tuple[Path, ...]:
    """Plot every variable/aggregation gridpoint temporal-correlation map."""
    required = {"gridpoint_temporal_correlation", "area_weighted_mean_correlation"}
    missing = required - set(metrics.data_vars)
    if missing:
        raise KeyError(f"correlation map dataset is missing {sorted(missing)}")
    paths = []
    for variable in map(str, metrics.variable.values):
        variable_directory = output_directory / variable
        variable_directory.mkdir(parents=True, exist_ok=True)
        for aggregation in map(str, metrics.aggregation.values):
            correlation = metrics.gridpoint_temporal_correlation.sel(
                variable=variable, aggregation=aggregation
            )
            correlation = _drop_empty_edges(correlation)
            mean = float(
                metrics.area_weighted_mean_correlation.sel(
                    variable=variable, aggregation=aggregation
                )
            )
            path = variable_directory / _AGGREGATION_FILENAMES[aggregation]
            _plot_correlation_map(correlation, variable, aggregation, mean, path, dpi)
            paths.append(path)
    return tuple(paths)


def plot_index_comparison(
    name: str,
    reconstruction: xr.DataArray,
    reference: xr.DataArray,
    metrics: SeriesMetrics,
    period: tuple[int, int],
    output_path: Path,
    *,
    dpi: int = 150,
    label: str = "Annual",
) -> Path:
    """Plot an aligned reconstruction/reference index comparison."""
    left, right = xr.align(reconstruction, reference, join="inner")
    years = _years(left.time)
    start, end = sorted(period)
    selected = (years >= start) & (years <= end)
    left = left.isel(time=np.flatnonzero(selected))
    right = right.isel(time=np.flatnonzero(selected))
    finite = np.isfinite(left) & np.isfinite(right)
    left = left.where(finite)
    right = right.where(finite)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 4.2))
    axis.plot(left.time, right, color="#222222", linewidth=1.4, label="Reference")
    axis.plot(left.time, left, color="#087e8b", linewidth=1.25, label="Reconstruction")
    axis.axhline(0, color="#7a7a7a", linewidth=0.7)
    axis.set_xlim(start, end)
    axis.set_xlabel("Year")
    axis.set_ylabel("Anomaly")
    axis.set_title(
        f"{name} {label}: r={metrics.correlation:.3f}, "
        f"RMSE={metrics.rmse:.3f}, n={metrics.count}"
    )
    axis.grid(color="#d9dde1", linewidth=0.6, alpha=0.8)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_correlation_map(
    correlation: xr.DataArray,
    variable: str,
    aggregation: str,
    mean: float,
    output_path: Path,
    dpi: int,
) -> None:
    northern_hemisphere = correlation.lat.size > 0 and float(correlation.lat.min()) >= 0
    projection = ccrs.NorthPolarStereo() if northern_hemisphere else ccrs.Robinson()
    figure = plt.figure(figsize=(8.6, 5.2 if northern_hemisphere else 4.8))
    axis = figure.add_subplot(1, 1, 1, projection=projection)
    latitude = np.asarray(correlation.lat, dtype=float)
    longitude = np.asarray(correlation.lon, dtype=float)
    image = axis.imshow(
        np.asarray(correlation, dtype=float),
        origin="lower",
        extent=_cell_edge_extent(longitude, latitude),
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axis.coastlines(resolution="110m", linewidth=0.65, color="#30343b")
    if northern_hemisphere:
        axis.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
    else:
        axis.set_global()
    mean_text = "nan" if not np.isfinite(mean) else f"{mean:.2f}"
    axis.set_title(
        f"{variable.upper()} {aggregation} temporal correlation "
        f"(area mean = {mean_text})",
        fontsize=11,
    )
    colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.06, shrink=0.78)
    colorbar.set_label("Pearson correlation")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _drop_empty_edges(field: xr.DataArray) -> xr.DataArray:
    finite = np.isfinite(field)
    valid_latitude = finite.any("lon")
    valid_longitude = finite.any("lat")
    if bool(valid_latitude.any()) and bool(valid_longitude.any()):
        return field.sel(
            lat=field.lat.where(valid_latitude, drop=True),
            lon=field.lon.where(valid_longitude, drop=True),
        )
    return field


def _cell_edge_extent(longitude: np.ndarray, latitude: np.ndarray) -> list[float]:
    if longitude.size < 2 or latitude.size < 2:
        raise ValueError("correlation maps require at least two latitudes and longitudes")
    half_lon = float(np.median(np.diff(longitude))) / 2
    half_lat = float(np.median(np.diff(latitude))) / 2
    return [
        float(longitude[0] - half_lon),
        float(longitude[-1] + half_lon),
        float(latitude[0] - half_lat),
        float(latitude[-1] + half_lat),
    ]


def _years(time: xr.DataArray) -> np.ndarray:
    try:
        return np.asarray(time.dt.year, dtype=int)
    except AttributeError:
        return np.floor(np.asarray(time, dtype=float)).astype(int)
