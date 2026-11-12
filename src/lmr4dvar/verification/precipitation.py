"""Precipitation verification on fixed GPCC/reconstruction common coverage."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from lmr4dvar.reconstruction.indices import latitude_weighted_mean

from .metrics import centered_running_mean, series_metrics, spatial_pattern_metrics
from .plotting import plot_index_comparison


def reference_on_model_grid(reference: xr.DataArray, model: xr.DataArray) -> xr.DataArray:
    """Explicit bilinear reference interpolation with periodic longitude.

    Missing land/ocean reference cells remain missing; no nearest-valid fallback.
    """
    reference = reference.assign_coords(lon=reference.lon % 360).sortby("lon").sortby("lat")
    if reference.lon.to_index().has_duplicates:
        raise ValueError("reference longitude normalization produced duplicate points")
    if float(reference.lon[-1] - reference.lon[0]) > 350:
        west = reference.isel(lon=[-1]).assign_coords(lon=reference.lon.values[-1:] - 360)
        east = reference.isel(lon=[0]).assign_coords(lon=reference.lon.values[:1] + 360)
        reference = xr.concat([west, reference, east], dim="lon")
    result = reference.interp(lat=model.lat, lon=model.lon, method="linear")
    result.attrs = reference.attrs.copy()
    result.attrs["verification_regridding"] = "linear reference-to-model; periodic longitude"
    return result


def precipitation_comparison(reconstruction, reference, *, period, window):
    """Inputs are already quarterly, on one grid and seasonal-baseline anomalies."""
    left, right = xr.align(reconstruction, reference, join="inner")
    years = np.floor(np.asarray(left.time)).astype(int)
    selected = (years >= period[0]) & (years <= period[1])
    left, right = left.isel(time=selected), right.isel(time=selected)
    if not left.sizes["time"]:
        raise ValueError("precipitation verification has no common comparison samples")
    mask = (np.isfinite(left) & np.isfinite(right)).all("time")
    if int(mask.sum()) < 2:
        raise ValueError("precipitation verification needs at least two fixed common gridpoints")
    left, right = left.where(mask), right.where(mask)
    quarterly_pattern = spatial_pattern_metrics(left, right)
    years = np.floor(np.asarray(left.time)).astype(int)
    group = xr.DataArray(years, dims="time", name="year")
    counts = xr.DataArray(np.ones(len(years)), dims="time").groupby(group).sum()
    complete_years = counts.year.values[counts.values == 4]
    annual_left = left.groupby(group).mean("time").sel(year=complete_years).rename(year="time")
    annual_right = right.groupby(group).mean("time").sel(year=complete_years).rename(year="time")
    annual_pattern = spatial_pattern_metrics(annual_left, annual_right)
    mean_left, mean_right = (
        latitude_weighted_mean(annual_left),
        latitude_weighted_mean(annual_right),
    )
    smooth_left = centered_running_mean(mean_left, window)
    smooth_right = centered_running_mean(mean_right, window)
    raw = series_metrics(mean_left, mean_right)
    finite_smooth = np.isfinite(smooth_left) & np.isfinite(smooth_right)
    smooth = series_metrics(smooth_left, smooth_right) if int(finite_smooth.sum()) >= 2 else None
    dataset = xr.Dataset(
        {
            "common_coverage_mask": mask,
            "reconstruction_mean": mean_left,
            "gpcc_mean": mean_right,
            "reconstruction_running_mean": smooth_left,
            "gpcc_running_mean": smooth_right,
            "annual_spatial_pattern_correlation": annual_pattern.pattern_correlation,
            "annual_spatial_pattern_rmse": annual_pattern.pattern_rmse,
            "quarterly_spatial_pattern_correlation": quarterly_pattern.pattern_correlation.rename(
                time="quarter_time"
            ),
            "quarterly_spatial_pattern_rmse": quarterly_pattern.pattern_rmse.rename(
                time="quarter_time"
            ),
        },
        attrs={
            "coverage": "fixed paired finite GPCC/reconstruction coverage; not global ocean+land",
            "spatial_pattern_definition": (
                "cosine-latitude weighted Pearson correlation across gridpoints"
            ),
            "mean_definition": "cosine-latitude weighted mean on fixed common coverage",
            "units": "mm/day",
            "running_mean_years": window,
            "comparison_period": f"{period[0]}-{period[1]}",
            "evaluation_scope": "GPCC calibration input, not independent validation",
        },
    )
    for name in (
        "reconstruction_mean",
        "gpcc_mean",
        "reconstruction_running_mean",
        "gpcc_running_mean",
        "annual_spatial_pattern_rmse",
        "quarterly_spatial_pattern_rmse",
    ):
        dataset[name].attrs["units"] = "mm/day"
    summary = {
        "coverage_gridpoints": int(mask.sum()),
        "raw": raw.__dict__,
        f"running_mean_{window}_years": smooth.__dict__ if smooth else None,
        "annual_mean_spatial_pattern_correlation": float(annual_pattern.pattern_correlation.mean()),
        "quarterly_mean_spatial_pattern_correlation": float(
            quarterly_pattern.pattern_correlation.mean()
        ),
        "coverage": dataset.attrs["coverage"],
        "evaluation_scope": dataset.attrs["evaluation_scope"],
    }
    return dataset, summary


def plot_precipitation_comparison(dataset, summary, directory: Path, *, dpi=150):
    from .metrics import SeriesMetrics

    directory.mkdir(parents=True, exist_ok=True)
    period = tuple(int(value) for value in dataset.attrs["comparison_period"].split("-"))
    paths = [
        plot_index_comparison(
            "Precipitation (GPCC common coverage, mm/day)",
            dataset.reconstruction_mean,
            dataset.gpcc_mean,
            SeriesMetrics(**summary["raw"]),
            period,
            directory / "mean_annual.png",
            dpi=dpi,
        )
    ]
    window = dataset.attrs["running_mean_years"]
    smooth = summary[f"running_mean_{window}_years"]
    if smooth is not None:
        paths.append(
            plot_index_comparison(
                "Precipitation (GPCC common coverage, mm/day)",
                dataset.reconstruction_running_mean,
                dataset.gpcc_running_mean,
                SeriesMetrics(**smooth),
                period,
                directory / f"mean_running_mean_{window}_years.png",
                dpi=dpi,
                label=f"{window}-year running mean",
            )
        )
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(dataset.time, dataset.annual_spatial_pattern_correlation, color="#087e8b")
    axis.axhline(0, color="#777777", linewidth=0.7)
    axis.set_ylim(-1, 1)
    axis.set_xlabel("Year")
    axis.set_ylabel("Spatial pattern correlation")
    axis.set_title("Annual precipitation anomaly pattern vs GPCC (common coverage)")
    figure.tight_layout()
    path = directory / "annual_spatial_pattern_correlation.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return tuple([*paths, path])
