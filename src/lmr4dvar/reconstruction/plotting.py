"""Headless figures for reconstructed climate indices."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

plt.switch_backend("Agg")

_PLOT_SPECS = {
    "GMT": (1.0, "Global mean temperature"),
    "GMP": (1.0, "Global mean precipitation anomaly (mm/day)"),
    "Nino34": (1.0, "Nino3.4 temperature"),
    "GMOHC": (1.0e8, "Global mean OHC (10^8 J m$^{-2}$)"),
    "GMOHC2000": (1.0e8, "Global mean OHC (10^8 J m$^{-2}$)"),
    "SIA": (1.0e12, "Northern Hemisphere SIA (10^6 km$^2$)"),
    "SIV": (1.0e12, "Northern Hemisphere SIV (10^3 km$^3$)"),
}


def plot_reconstructed_gmt(
    gmt: xr.DataArray,
    output_directory: Path,
    *,
    dpi: int = 150,
) -> tuple[Path, Path]:
    """Plot quarterly and annual reconstructed global mean temperature."""
    return plot_reconstructed_index("GMT", gmt, output_directory, dpi=dpi)


def plot_reconstructed_index(
    name: str,
    series: xr.DataArray,
    output_directory: Path,
    *,
    dpi: int = 150,
) -> tuple[Path, Path]:
    """Plot quarterly and annual values for one reconstructed index."""
    if series.dims != ("time",):
        raise ValueError(f"reconstructed {name} must have only a time dimension")
    output_directory.mkdir(parents=True, exist_ok=True)
    annual = _annual_mean(series)
    scale, ylabel = _PLOT_SPECS.get(name, (1.0, name))
    outputs = (
        output_directory / "seasonal.png",
        output_directory / "annual.png",
    )
    _plot_series(
        series / scale,
        outputs[0],
        f"Reconstructed {name} (quarterly)",
        ylabel,
        dpi=dpi,
    )
    _plot_series(
        annual / scale,
        outputs[1],
        f"Reconstructed {name} (annual mean)",
        ylabel,
        dpi=dpi,
    )
    return outputs


def _plot_series(
    series: xr.DataArray,
    output_path: Path,
    title: str,
    ylabel: str,
    *,
    dpi: int,
) -> None:
    values = np.asarray(series, dtype=float)
    years = _year_values(series.time)
    finite = np.isfinite(years) & np.isfinite(values)
    figure, axis = plt.subplots(figsize=(11, 4.2))
    axis.plot(years[finite], values[finite], color="#087e8b", linewidth=0.9)
    finite_values = values[finite]
    if finite_values.size and np.nanmin(finite_values) <= 0 <= np.nanmax(finite_values):
        axis.axhline(0, color="#777777", linewidth=0.7)
    axis.set_xlim(float(np.nanmin(years)), float(np.nanmax(years)))
    axis.set_xlabel("Year")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(color="#d9dde1", linewidth=0.6, alpha=0.8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _annual_mean(series: xr.DataArray) -> xr.DataArray:
    years = np.floor(_year_values(series.time)).astype(int)
    annual = series.groupby(xr.DataArray(years, dims="time", name="year")).mean("time")
    return annual.rename(year="time")


def _year_values(time: xr.DataArray) -> np.ndarray:
    try:
        years = np.asarray(time.dt.year, dtype=float)
        months = np.asarray(time.dt.month, dtype=float)
        return years + (months - 1) / 12
    except AttributeError:
        return np.asarray(time, dtype=float)
