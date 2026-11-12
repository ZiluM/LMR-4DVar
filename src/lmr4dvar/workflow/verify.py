"""Verification workflow handler."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from lmr4dvar.config import ProjectConfig
from lmr4dvar.verification import (
    anomaly,
    centered_running_mean,
    gridpoint_temporal_correlation,
    plot_correlation_maps,
    plot_index_comparison,
    seasonal_anomaly,
    series_metrics,
    spatial_pattern_metrics,
)
from lmr4dvar.verification.precipitation import (
    plot_precipitation_comparison,
    precipitation_comparison,
    reference_on_model_grid,
)

from .planner import stage_directory
from .stages import Stage


def execute_verification(config: ProjectConfig) -> tuple[Path, ...]:
    if not config.verification.reference_fields and config.verification.reference_indices is None:
        raise ValueError("verification requires reference_fields or reference_indices")
    reconstruction = stage_directory(config, Stage.RECONSTRUCT)
    directory = stage_directory(config, Stage.VERIFY)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    figure_paths = []
    pattern_results = []
    precipitation_summary = None
    spatial_correlation_summary = {
        "_metadata": {
            "definition": "gridpoint Pearson correlation through time",
            "spatial_summary": "cosine-latitude weighted mean",
            "aggregations": ["quarterly", "annual", "DJF", "MAM", "JJA", "SON"],
        }
    }
    for variable, reference_config in config.verification.reference_fields.items():
        if isinstance(reference_config, Path):
            reference_path = reference_config
            reference_variable = variable
            comparison_period = config.verification.comparison_period
        else:
            reference_path = reference_config.path
            reference_variable = reference_config.variable or variable
            comparison_period = (
                reference_config.comparison_period or config.verification.comparison_period
            )
        reconstruction_path = reconstruction / "fields" / f"{variable}.nc"
        periods = (config.verification.anomaly_period, comparison_period)
        reconstructed = _load_variable(reconstruction_path, variable, periods=periods)
        reference = _load_variable(reference_path, reference_variable, periods=periods)
        if variable == "pr" and (
            config.state.variables["pr"].units != "mm/day"
            or reference.attrs.get("units") != "mm/day"
        ):
            raise ValueError("precipitation verification requires model/reference units mm/day")
        if not isinstance(reference_config, Path) and reference_config.regrid_to_reconstruction:
            reference = reference_on_model_grid(reference, reconstructed)
        reconstructed = seasonal_anomaly(
            _quarter_coordinate(reconstructed), config.verification.anomaly_period
        )
        reference = seasonal_anomaly(
            _quarter_coordinate(reference), config.verification.anomaly_period
        )
        if variable == "pr":
            comparison, precipitation_summary = precipitation_comparison(
                reconstructed,
                reference,
                period=comparison_period,
                window=config.verification.running_mean_years,
            )
            comparison.attrs["anomaly_period"] = "-".join(
                map(str, config.verification.anomaly_period)
            )
            if not isinstance(reference_config, Path) and reference_config.evaluation_scope:
                comparison.attrs["evaluation_scope"] = reference_config.evaluation_scope
                precipitation_summary["evaluation_scope"] = reference_config.evaluation_scope
            precipitation_path = directory / "precipitation_gpcc.nc"
            comparison.to_netcdf(precipitation_path, engine="h5netcdf")
            outputs.append(precipitation_path)
            reconstructed = reconstructed.where(comparison.common_coverage_mask)
            reference = reference.where(comparison.common_coverage_mask)
            if config.verification.plot_figures:
                figure_paths.extend(
                    plot_precipitation_comparison(
                        comparison,
                        precipitation_summary,
                        directory / "figures" / "precipitation",
                        dpi=config.verification.figure_dpi,
                    )
                )
        pattern_metrics = spatial_pattern_metrics(
            reconstructed,
            reference,
            period=comparison_period,
        ).drop_vars("pattern_correlation")
        correlation_results = []
        variable_summary = {}
        reference_aggregations = dict(_temporal_aggregations(reference))
        for aggregation, reconstructed_aggregate in _temporal_aggregations(reconstructed):
            reference_aggregate = reference_aggregations[aggregation]
            correlations = gridpoint_temporal_correlation(
                reconstructed_aggregate,
                reference_aggregate,
                period=comparison_period,
            )
            correlations = correlations.expand_dims(aggregation=[aggregation])
            correlation_results.append(correlations)
            variable_summary[aggregation] = {
                "area_weighted_mean_correlation": float(
                    correlations.area_weighted_mean_correlation.squeeze()
                ),
                "valid_grid_points": int(correlations.gridpoint_temporal_correlation.count()),
            }
        temporal_metrics = xr.concat(correlation_results, dim="aggregation")
        metrics = xr.merge([pattern_metrics, temporal_metrics]).expand_dims(variable=[variable])
        pattern_results.append(metrics)
        spatial_correlation_summary[variable] = variable_summary
    if pattern_results:
        pattern_path = directory / "pattern_metrics.nc"
        pattern_dataset = xr.concat(pattern_results, dim="variable")
        pattern_dataset.attrs.update(
            {
                "correlation_definition": "gridpoint Pearson correlation through time",
                "correlation_spatial_summary": "cosine-latitude weighted mean",
                "field_anomaly_method": "quarter-specific climatology",
                "anomaly_period": "-".join(map(str, config.verification.anomaly_period)),
            }
        )
        pattern_dataset.to_netcdf(pattern_path, engine="h5netcdf")
        outputs.append(pattern_path)
        if config.verification.plot_figures:
            figure_paths.extend(
                plot_correlation_maps(
                    pattern_dataset,
                    directory / "figures" / "pattern",
                    dpi=config.verification.figure_dpi,
                )
            )

    series_results = {}
    if precipitation_summary is not None:
        series_results["precipitation_gpcc"] = precipitation_summary
    reference_indices = config.verification.reference_indices
    if reference_indices is not None:
        with xr.open_dataset(reconstruction / "indices.nc", engine="h5netcdf") as dataset:
            reconstructed_indices = dataset.load()
        if isinstance(reference_indices, Path):
            with xr.open_dataset(reference_indices, engine="h5netcdf") as dataset:
                loaded_indices = dataset.load()
            common = sorted(set(reconstructed_indices.data_vars) & set(loaded_indices.data_vars))
            if not common:
                raise ValueError("reconstruction and reference contain no common indices")
            configured = {
                name: (
                    loaded_indices[name],
                    config.verification.comparison_period,
                )
                for name in common
            }
        else:
            configured = {}
            for name, reference_config in reference_indices.items():
                if name not in reconstructed_indices:
                    raise KeyError(f"reconstructed index {name!r} is absent")
                reference_name = reference_config.variable or name
                configured[name] = (
                    _load_variable(reference_config.path, reference_name),
                    reference_config.comparison_period or config.verification.comparison_period,
                )
        for name, (reference_series, comparison_period) in configured.items():
            reconstructed = anomaly(
                _annual_mean(reconstructed_indices[name]), config.verification.anomaly_period
            )
            reference = anomaly(_annual_mean(reference_series), config.verification.anomaly_period)
            raw = series_metrics(reconstructed, reference, period=comparison_period)
            window = config.verification.running_mean_years
            reconstructed_smooth = centered_running_mean(reconstructed, window)
            reference_smooth = centered_running_mean(reference, window)
            smooth = series_metrics(
                reconstructed_smooth,
                reference_smooth,
                period=comparison_period,
            )
            series_results[name] = {
                "raw": raw.__dict__,
                f"running_mean_{window}_years": smooth.__dict__,
            }
            if config.verification.plot_figures:
                figure_directory = directory / "figures" / "indices" / name
                figure_paths.extend(
                    (
                        plot_index_comparison(
                            name,
                            reconstructed,
                            reference,
                            raw,
                            comparison_period,
                            figure_directory / "compare_annual.png",
                            dpi=config.verification.figure_dpi,
                        ),
                        plot_index_comparison(
                            name,
                            reconstructed_smooth,
                            reference_smooth,
                            smooth,
                            comparison_period,
                            figure_directory / f"compare_running_mean_{window}_years.png",
                            dpi=config.verification.figure_dpi,
                            label=f"{window}-year running mean",
                        ),
                    )
                )
    if pattern_results:
        series_results["spatial_correlation"] = spatial_correlation_summary
    if figure_paths:
        series_results["figures"] = [str(path.relative_to(directory)) for path in figure_paths]
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(series_results, indent=2) + "\n", encoding="utf-8")
    outputs.append(summary_path)
    outputs.extend(figure_paths)
    return tuple(outputs)


def _load_variable(
    path: Path,
    variable: str,
    *,
    periods: tuple[tuple[int, int], ...] | None = None,
) -> xr.DataArray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        if variable not in dataset:
            raise KeyError(f"variable {variable!r} is absent from {path}")
        field = dataset[variable]
        if periods is not None:
            years = _coordinate_years(field.time)
            selected = np.zeros(field.sizes["time"], dtype=bool)
            for period in periods:
                start, end = sorted(period)
                selected |= (years >= start) & (years <= end)
            field = field.isel(time=np.flatnonzero(selected))
        return field.load()


def _quarter_coordinate(field: xr.DataArray) -> xr.DataArray:
    try:
        years = field.time.dt.year.astype(float)
        months = field.time.dt.month.astype(float)
        values = years + (months - 1) / 12
    except AttributeError:
        values = field.time.astype(float)
    return field.assign_coords(time=np.asarray(values, dtype=float))


def _annual_mean(series: xr.DataArray) -> xr.DataArray:
    series = _quarter_coordinate(series)
    years = np.floor(np.asarray(series.time, dtype=float)).astype(int)
    annual = series.groupby(xr.DataArray(years, dims="time", name="year")).mean("time")
    return annual.rename(year="time")


def _temporal_aggregations(field: xr.DataArray) -> tuple[tuple[str, xr.DataArray], ...]:
    field = _quarter_coordinate(field)
    years = np.floor(np.asarray(field.time, dtype=float)).astype(int)
    quarters = np.rint((np.asarray(field.time, dtype=float) - years) * 4).astype(int) % 4
    results = [("quarterly", field), ("annual", _annual_mean(field))]
    for quarter, name in enumerate(("DJF", "MAM", "JJA", "SON")):
        indices = np.flatnonzero(quarters == quarter)
        seasonal = field.isel(time=indices).assign_coords(time=years[indices])
        results.append((name, seasonal))
    return tuple(results)


def _coordinate_years(time: xr.DataArray) -> np.ndarray:
    try:
        return np.asarray(time.dt.year, dtype=int)
    except AttributeError:
        return np.floor(np.asarray(time, dtype=float)).astype(int)
