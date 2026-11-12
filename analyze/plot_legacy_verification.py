"""Plot standalone verification artifacts using the legacy FDVar layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lmr4dvar.config import load_project_config
from lmr4dvar.verification import anomaly, series_metrics

QUARTERS = ((1, "DJF"), (4, "MAM"), (7, "JJA"), (10, "SON"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--show-period", nargs=2, type=int, default=(1800, 2001))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config, args.profile)
    run_root = config.paths.output_root / config.project.name
    plot_index_verification(config, run_root, tuple(args.show_period))
    plot_pattern_verification(config, run_root)
    print(f"Wrote legacy-style figures below {run_root}")


def plot_index_verification(config, run_root: Path, show_period: tuple[int, int]) -> None:
    with xr.open_dataset(run_root / "stages/reconstruct/indices.nc", engine="h5netcdf") as ds:
        reconstructed = ds.load()
    references = config.verification.reference_indices
    if not isinstance(references, dict):
        return
    for name, reference_config in references.items():
        if name not in reconstructed:
            continue
        output = run_root / "pic1" / name
        output.mkdir(parents=True, exist_ok=True)
        reference_name = reference_config.variable or name
        reference = _load_field(reference_config.path, reference_name)
        left = anomaly(_quarter_coordinate(reconstructed[name]), config.verification.anomaly_period)
        right = anomaly(_quarter_coordinate(reference), config.verification.anomaly_period)
        comparison = reference_config.comparison_period or config.verification.comparison_period

        _plot_single_series(left, name, show_period, output / "seasonal.png")
        _plot_single_series(
            _annual_mean(left), f"{name} Annual", show_period, output / "annual.png"
        )
        _plot_comparison(
            left,
            right,
            name,
            comparison,
            show_period,
            output / "compare_seasonal.png",
        )
        for month, season in QUARTERS:
            _plot_comparison(
                _season(left, month),
                _season(right, month),
                f"{name} {season}",
                comparison,
                show_period,
                output / f"{name} compare_{month}.png",
            )
        _plot_comparison(
            _annual_mean(left),
            _annual_mean(right),
            f"{name} Annual",
            comparison,
            show_period,
            output / "compare_annual.png",
        )


def plot_pattern_verification(config, run_root: Path) -> None:
    for variable, reference_config in config.verification.reference_fields.items():
        if isinstance(reference_config, Path):
            reference_path = reference_config
            reference_name = variable
            comparison = config.verification.comparison_period
        else:
            reference_path = reference_config.path
            reference_name = reference_config.variable or variable
            comparison = reference_config.comparison_period or config.verification.comparison_period
        periods = (config.verification.anomaly_period, comparison)
        reconstruction = _load_field(
            run_root / f"stages/reconstruct/fields/{variable}.nc",
            variable,
            periods=periods,
        )
        reference = _load_field(reference_path, reference_name, periods=periods)
        left = anomaly(_quarter_coordinate(reconstruction), config.verification.anomaly_period)
        right = anomaly(_quarter_coordinate(reference), config.verification.anomaly_period)
        left, right = _common_grid_and_period(left, right, comparison)
        output = run_root / "pattern1" / variable
        output.mkdir(parents=True, exist_ok=True)
        _plot_correlation_map(left, right, variable, "All quarterly", output / "all.png")
        _plot_correlation_map(
            _annual_mean(left),
            _annual_mean(right),
            variable,
            "ANN",
            output / "ANN_annual.png",
        )
        for month, season in QUARTERS:
            _plot_correlation_map(
                _season(left, month),
                _season(right, month),
                variable,
                season,
                output / f"{season}_annual.png",
            )


def _plot_single_series(
    series: xr.DataArray, title: str, show_period: tuple[int, int], output: Path
) -> None:
    selected = _select_period(series, show_period)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(selected.time, selected, color="#1769aa", linewidth=1.1, label="Reconstruction")
    _finish_series_axis(ax, title, show_period)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(
    reconstruction: xr.DataArray,
    reference: xr.DataArray,
    title: str,
    comparison_period: tuple[int, int],
    show_period: tuple[int, int],
    output: Path,
) -> None:
    left, right = xr.align(reconstruction, reference, join="inner")
    metrics = series_metrics(left, right, period=comparison_period)
    visible_left = _select_period(left, show_period)
    visible_right = _select_period(right, show_period)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(visible_left.time, visible_left, color="#1769aa", linewidth=1.1, label="Reconstruction")
    ax.plot(visible_right.time, visible_right, color="#222222", linewidth=1.0, label="Observation")
    _finish_series_axis(
        ax,
        f"{title} (r={metrics.correlation:.2f}, RMSE={metrics.rmse:.2f})",
        show_period,
    )
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finish_series_axis(ax, title: str, show_period: tuple[int, int]) -> None:
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_xlim(*show_period)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly")
    ax.set_title(title)
    ax.grid(color="#dddddd", linewidth=0.6)


def _plot_correlation_map(
    reconstruction: xr.DataArray,
    reference: xr.DataArray,
    variable: str,
    label: str,
    output: Path,
) -> None:
    left, right = xr.align(reconstruction, reference, join="inner")
    correlation = xr.corr(left, right, dim="time")
    weights = np.cos(np.deg2rad(correlation.lat))
    mean = float(correlation.weighted(weights).mean(("lat", "lon"), skipna=True))
    polar = variable == "sic"
    projection = ccrs.NorthPolarStereo() if polar else ccrs.Robinson()
    fig = plt.figure(figsize=(8.5, 4.8 if not polar else 6.2))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    cyclic, cyclic_lon = add_cyclic_point(
        np.asarray(correlation), coord=np.asarray(correlation.lon)
    )
    mesh = ax.contourf(
        cyclic_lon,
        correlation.lat,
        cyclic,
        levels=np.linspace(-1, 1, 21),
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        extend="neither",
    )
    ax.coastlines(linewidth=0.6)
    if polar:
        ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    else:
        ax.set_global()
    ax.set_title(f"{variable} {label} correlation (area mean={mean:.2f})")
    fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, shrink=0.82, label="r")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_field(
    path: Path,
    variable: str,
    *,
    periods: tuple[tuple[int, int], ...] | None = None,
) -> xr.DataArray:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        field = dataset[variable]
        if periods is not None:
            years = _years(field.time)
            selected = np.zeros(field.sizes["time"], dtype=bool)
            for start, end in periods:
                selected |= (years >= min(start, end)) & (years <= max(start, end))
            field = field.isel(time=np.flatnonzero(selected))
        return field.load()


def _common_grid_and_period(
    left: xr.DataArray, right: xr.DataArray, period: tuple[int, int]
) -> tuple[xr.DataArray, xr.DataArray]:
    left, right = xr.align(left, right, join="inner")
    left = _select_period(left, period)
    right = _select_period(right, period)
    if not left.sizes.get("time", 0):
        raise ValueError(f"no common values in {period}")
    return left, right


def _quarter_coordinate(field: xr.DataArray) -> xr.DataArray:
    try:
        values = field.time.dt.year + (field.time.dt.month - 1) / 12
    except AttributeError:
        values = field.time.astype(float)
    return field.assign_coords(time=np.asarray(values, dtype=float))


def _annual_mean(series: xr.DataArray) -> xr.DataArray:
    years = np.floor(np.asarray(series.time, dtype=float)).astype(int)
    return series.groupby(xr.DataArray(years, dims="time", name="year")).mean("time").rename(
        year="time"
    )


def _season(series: xr.DataArray, month: int) -> xr.DataArray:
    target = (month - 1) / 12
    quarter = np.asarray(series.time, dtype=float) - np.floor(np.asarray(series.time, dtype=float))
    return series.isel(time=np.flatnonzero(np.isclose(quarter, target))).assign_coords(
        time=np.floor(np.asarray(series.time, dtype=float)[np.isclose(quarter, target)]).astype(int)
    )


def _select_period(series: xr.DataArray, period: tuple[int, int]) -> xr.DataArray:
    start, end = sorted(period)
    years = np.floor(np.asarray(series.time, dtype=float)).astype(int)
    return series.isel(time=np.flatnonzero((years >= start) & (years <= end)))


def _years(time: xr.DataArray) -> np.ndarray:
    try:
        return np.asarray(time.dt.year, dtype=int)
    except AttributeError:
        return np.floor(np.asarray(time, dtype=float)).astype(int)


if __name__ == "__main__":
    main()
