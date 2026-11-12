"""Diagnose spatial verification skill and the EOF-subspace ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from lmr4dvar.config import load_project_config
from lmr4dvar.eof import EOFModel
from lmr4dvar.verification import spatial_pattern_metrics
from lmr4dvar.workflow.planner import artifact_stage_directory, stage_directory
from lmr4dvar.workflow.stages import Stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def numeric_quarter_time(field: xr.DataArray) -> xr.DataArray:
    try:
        years = np.asarray(field.time.dt.year, dtype=float)
        months = np.asarray(field.time.dt.month, dtype=float)
        values = years + (months - 1) / 12
    except AttributeError:
        values = np.asarray(field.time, dtype=float)
    return field.assign_coords(time=values)


def years(field: xr.DataArray) -> np.ndarray:
    return np.floor(np.asarray(field.time, dtype=float)).astype(int)


def select_periods(
    field: xr.DataArray, periods: tuple[tuple[int, int], ...]
) -> xr.DataArray:
    field_years = years(field)
    selected = np.zeros(field.sizes["time"], dtype=bool)
    for start, end in periods:
        lower, upper = sorted((start, end))
        selected |= (field_years >= lower) & (field_years <= upper)
    return field.isel(time=np.flatnonzero(selected))


def load_field(
    path: Path, variable: str, periods: tuple[tuple[int, int], ...]
) -> xr.DataArray:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        field = select_periods(numeric_quarter_time(dataset[variable]), periods).load()
    return field


def overall_anomaly(field: xr.DataArray, period: tuple[int, int]) -> xr.DataArray:
    lower, upper = sorted(period)
    field_years = years(field)
    baseline = field.isel(
        time=np.flatnonzero((field_years >= lower) & (field_years <= upper))
    )
    return field - baseline.mean("time")


def seasonal_anomaly(field: xr.DataArray, period: tuple[int, int]) -> xr.DataArray:
    quarter = np.rint((np.asarray(field.time) - years(field)) * 4).astype(int) % 4
    field = field.assign_coords(quarter=("time", quarter))
    lower, upper = sorted(period)
    field_years = years(field)
    baseline = field.isel(
        time=np.flatnonzero((field_years >= lower) & (field_years <= upper))
    )
    climatology = baseline.groupby("quarter").mean("time")
    return field.groupby("quarter") - climatology


def annual_mean(field: xr.DataArray) -> xr.DataArray:
    year = xr.DataArray(years(field), dims="time", name="year")
    return field.groupby(year).mean("time").rename(year="time")


def period(field: xr.DataArray, bounds: tuple[int, int]) -> xr.DataArray:
    lower, upper = sorted(bounds)
    field_years = years(field)
    return field.isel(
        time=np.flatnonzero((field_years >= lower) & (field_years <= upper))
    )


def metric_summary(left: xr.DataArray, right: xr.DataArray) -> dict[str, float | int]:
    metrics = spatial_pattern_metrics(left, right)
    correlation = np.asarray(metrics.pattern_correlation, dtype=float)
    rmse = np.asarray(metrics.pattern_rmse, dtype=float)
    valid = np.isfinite(correlation) & np.isfinite(rmse)
    return {
        "count": int(valid.sum()),
        "correlation_mean": float(correlation[valid].mean()),
        "correlation_median": float(np.median(correlation[valid])),
        "rmse_mean": float(rmse[valid].mean()),
    }


def spatial_std_ratio(left: xr.DataArray, right: xr.DataArray) -> float:
    left, right = xr.align(left, right, join="inner")
    latitude_weights = xr.DataArray(
        np.cos(np.deg2rad(left.lat)), dims="lat", coords={"lat": left.lat}
    )
    weights = xr.broadcast(latitude_weights, left.isel(time=0, drop=True))[0]
    valid = np.isfinite(left) & np.isfinite(right)
    weights = weights.where(valid)
    weight_sum = weights.sum(("lat", "lon"))
    left_mean = (left * weights).sum(("lat", "lon")) / weight_sum
    right_mean = (right * weights).sum(("lat", "lon")) / weight_sum
    left_std = np.sqrt((((left - left_mean) ** 2) * weights).sum(("lat", "lon")) / weight_sum)
    right_std = np.sqrt(
        (((right - right_mean) ** 2) * weights).sum(("lat", "lon")) / weight_sum
    )
    return float((left_std / right_std).mean(skipna=True))


def eof_oracle(reference: xr.DataArray, model: EOFModel) -> xr.DataArray:
    reference = reference.sel(
        {dimension: model.valid_mask[dimension] for dimension in model.valid_mask.dims}
    )
    n_modes = model.components.sizes["mode"]
    unit_pcs = xr.DataArray(
        np.eye(n_modes),
        dims=("time", "mode"),
        coords={"time": np.arange(n_modes), "mode": model.components.mode},
    )
    zero_pcs = xr.zeros_like(unit_pcs.isel(time=slice(0, 1)))
    basis = np.asarray(model.decode(unit_pcs) - model.decode(zero_pcs).isel(time=0))
    reference = reference.transpose("time", "lat", "lon")
    latitude_weight = np.broadcast_to(
        np.sqrt(np.cos(np.deg2rad(reference.lat.values)))[:, None],
        reference.shape[1:],
    )
    full_basis_valid = np.isfinite(basis).all(axis=0)
    result = np.full(reference.shape, np.nan, dtype=float)
    for index, target in enumerate(np.asarray(reference, dtype=float)):
        valid = full_basis_valid & np.isfinite(target)
        design = (basis[:, valid] * latitude_weight[valid]).T
        values = target[valid] * latitude_weight[valid]
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        result[index, full_basis_valid] = coefficients @ basis[:, full_basis_valid]
    return xr.DataArray(result, dims=reference.dims, coords=reference.coords)


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    reconstruction = stage_directory(config, Stage.RECONSTRUCT) / "fields"
    eof_directory = artifact_stage_directory(config, Stage.EOF)
    baseline = config.verification.anomaly_period
    summary = {}
    for variable, reference_config in config.verification.reference_fields.items():
        if isinstance(reference_config, Path):
            reference_path = reference_config
            reference_variable = variable
            comparison = config.verification.comparison_period
        else:
            reference_path = reference_config.path
            reference_variable = reference_config.variable or variable
            comparison = reference_config.comparison_period or config.verification.comparison_period
        periods = (baseline, comparison)
        reconstructed = load_field(reconstruction / f"{variable}.nc", variable, periods)
        reference = load_field(reference_path, reference_variable, periods)

        reconstructed_overall = period(overall_anomaly(reconstructed, baseline), comparison)
        reference_overall = period(overall_anomaly(reference, baseline), comparison)
        reconstructed_seasonal = period(seasonal_anomaly(reconstructed, baseline), comparison)
        reference_seasonal = period(seasonal_anomaly(reference, baseline), comparison)
        reconstructed_annual = annual_mean(reconstructed_seasonal)
        reference_annual = annual_mean(reference_seasonal)
        model = EOFModel.load(eof_directory / variable)
        oracle = eof_oracle(reference_seasonal, model)
        summary[variable] = {
            "retained_eof_variance_fraction": float(model.explained_variance_ratio.sum()),
            "quarterly_overall_baseline": metric_summary(
                reconstructed_overall, reference_overall
            ),
            "quarterly_seasonal_baseline": metric_summary(
                reconstructed_seasonal, reference_seasonal
            ),
            "annual_seasonal_baseline": metric_summary(
                reconstructed_annual, reference_annual
            ),
            "quarterly_spatial_std_ratio": spatial_std_ratio(
                reconstructed_seasonal, reference_seasonal
            ),
            "eof_oracle_quarterly": metric_summary(oracle, reference_seasonal),
        }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
