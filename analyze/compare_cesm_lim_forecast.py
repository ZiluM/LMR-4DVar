"""Compare legacy and member-aligned CESM LIM forecasts against CCSM4."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr

from lmr4dvar.eof.model import EOFModel
from lmr4dvar.reconstruction.indices import nino34

VARIABLES = {
    "tas": (15, "tas", "anom_remap2x2_tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc"),
    "ohc300": (
        30,
        "ohc300",
        "anom_remap2x2_ohc300_Omon_CCSM4_past1000_r1i1p1_085001-185012.nc",
    ),
    "tos": (15, "tos", "anom_remap2x2_tos_Omon_CCSM4_past1000_r1i1p1_085001-185012.nc"),
    "sic": (15, "sic", "anom_remap2x2_sic_OImon_CCSM4_past1000_r1i1p1_085001-185012.nc"),
    "sit": (15, "sit", "anom_remap2x2_sit_OImon_CCSM4_past1000_r1i1p1_085001-185012.nc"),
    "ohc2000": (30, "ohc2000", "anom_remap2x2_ohc2000.nc"),
}

# os.listdir() order used independently by each variable in the legacy EOF build.
LEGACY_MEMBER_ORDERS = {
    "tas": (8, 3, 9, 13, 6, 7, 11, 12, 1, 10, 2, 4, 5),
    "ohc300": (7, 9, 6, 5, 2, 1, 3, 12, 10, 13, 11, 8, 4),
    "tos": (2, 3, 12, 10, 8, 13, 6, 7, 1, 5, 9, 11, 4),
    "sic": (11, 6, 4, 1, 8, 13, 3, 7, 9, 2, 12, 10, 5),
    "sit": (8, 7, 12, 3, 11, 10, 4, 1, 6, 2, 13, 5, 9),
    "ohc2000": (7, 9, 6, 5, 2, 1, 3, 12, 10, 13, 11, 8, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eof-dir", type=Path, required=True)
    parser.add_argument("--training-pcs", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    parser.add_argument("--legacy-correlations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--member-length", type=int, default=4000)
    parser.add_argument("--max-lead", type=int, default=4)
    return parser.parse_args()


def fit_propagator(values: np.ndarray, member_length: int | None) -> np.ndarray:
    x0 = values[:-1]
    x1 = values[1:]
    if member_length is not None:
        valid = (np.arange(values.shape[0] - 1) + 1) % member_length != 0
        x0 = x0[valid]
        x1 = x1[valid]
    return (x1.T @ x0) @ np.linalg.pinv(x0.T @ x0)


def legacy_misaligned_training(values: np.ndarray, member_length: int) -> np.ndarray:
    expected = member_length * len(next(iter(LEGACY_MEMBER_ORDERS.values())))
    if values.shape[0] != expected:
        raise ValueError(f"legacy member ordering expects {expected} training samples")
    reordered = np.empty_like(values)
    offset = 0
    for name, (n_modes, _, _) in VARIABLES.items():
        for destination, member in enumerate(LEGACY_MEMBER_ORDERS[name]):
            source = slice((member - 1) * member_length, member * member_length)
            target = slice(destination * member_length, (destination + 1) * member_length)
            reordered[target, offset : offset + n_modes] = values[source, offset : offset + n_modes]
        offset += n_modes
    return reordered


def load_verification_field(path: Path, source_name: str, northern: bool) -> xr.DataArray:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        field = dataset[source_name].isel(time=slice(0, 4000)).load()
    if northern:
        field = field.sel(lat=slice(0, 90))
    return field


def align_spatial_coordinates(field: xr.DataArray, model: EOFModel) -> xr.DataArray:
    field = field.reset_coords(drop=True)
    coordinates = {}
    for dimension in model.valid_mask.dims:
        actual = np.asarray(field[dimension])
        expected = np.asarray(model.valid_mask[dimension])
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-10):
            raise ValueError(f"verification {dimension} grid differs from the EOF grid")
        coordinates[dimension] = model.valid_mask[dimension]
    return field.assign_coords(coordinates)


def legacy_field_correlation(reference: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    reference_anomaly = reference - reference.mean(axis=0)
    forecast_anomaly = forecast - forecast.mean(axis=0)
    covariance = np.einsum("t...,t...->...", reference_anomaly, forecast_anomaly) / (
        reference.shape[0] - 1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            covariance / np.std(reference_anomaly, axis=0) / np.std(forecast_anomaly, axis=0)
        ).real


def area_weighted_mean(correlation: np.ndarray, latitude: np.ndarray) -> float:
    latitude_weight = np.sqrt(np.clip(np.cos(np.deg2rad(latitude)), 0.0, None))
    weights = np.broadcast_to(latitude_weight[:, None], correlation.shape)
    valid = np.isfinite(correlation)
    return float(np.sum(correlation[valid] * weights[valid]) / np.sum(weights[valid]))


def evaluate_model(
    propagator: np.ndarray,
    encoded: np.ndarray,
    fields: dict[str, xr.DataArray],
    models: dict[str, EOFModel],
    max_lead: int,
) -> dict[str, np.ndarray]:
    correlations: dict[str, list[np.ndarray]] = {name: [] for name in VARIABLES}
    transition = np.eye(propagator.shape[0])
    for lead in range(1, max_lead + 1):
        transition = transition @ propagator
        predicted = encoded @ transition.T
        offset = 0
        for name, (n_modes, _, _) in VARIABLES.items():
            pcs = xr.DataArray(
                predicted[:, offset : offset + n_modes],
                dims=("time", "mode"),
                coords={"time": fields[name].time, "mode": np.arange(n_modes)},
            )
            decoded = models[name].decode(pcs)
            correlations[name].append(
                legacy_field_correlation(fields[name].values[lead:], decoded.values[:-lead])
            )
            offset += n_modes
    return {name: np.stack(values) for name, values in correlations.items()}


def series_metrics(forecast: xr.DataArray, reference: xr.DataArray) -> dict[str, float | int]:
    forecast_values = np.asarray(forecast, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    valid = np.isfinite(forecast_values) & np.isfinite(reference_values)
    forecast_values = forecast_values[valid]
    reference_values = reference_values[valid]
    return {
        "correlation": float(np.corrcoef(forecast_values, reference_values)[0, 1]),
        "rmse": float(np.sqrt(np.mean(np.square(forecast_values - reference_values)))),
        "count": int(valid.sum()),
    }


def evaluate_nino34(
    propagators: dict[str, np.ndarray],
    encoded: np.ndarray,
    tos: xr.DataArray,
    model: EOFModel,
    max_lead: int,
) -> tuple[dict[str, dict[str, list[dict[str, float | int]]]], xr.Dataset]:
    truth = nino34(tos)
    forecasts = np.full((len(propagators), max_lead, tos.sizes["time"]), np.nan)
    summaries: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    tos_offset = sum(VARIABLES[name][0] for name in ("tas", "ohc300"))

    for model_index, (model_name, propagator) in enumerate(propagators.items()):
        model_metrics = []
        transition = np.eye(propagator.shape[0])
        for lead in range(1, max_lead + 1):
            transition = transition @ propagator
            predicted = encoded @ transition.T
            pcs = xr.DataArray(
                predicted[:, tos_offset : tos_offset + VARIABLES["tos"][0]],
                dims=("time", "mode"),
                coords={"time": tos.time, "mode": model.components.mode},
            )
            forecast = nino34(model.decode(pcs))
            forecasts[model_index, lead - 1, lead:] = forecast.values[:-lead]
            model_metrics.append(series_metrics(forecast[:-lead], truth[lead:]))
        summaries[model_name] = {"lead_metrics": model_metrics}

    persistence_metrics = []
    persistence = np.full((max_lead, tos.sizes["time"]), np.nan)
    for lead in range(1, max_lead + 1):
        persistence[lead - 1, lead:] = truth.values[:-lead]
        persistence_metrics.append(series_metrics(truth[:-lead], truth[lead:]))
    summaries["persistence"] = {"lead_metrics": persistence_metrics}

    dataset = xr.Dataset(
        {
            "truth": truth,
            "forecast": (("model", "lead", "time"), forecasts),
            "persistence": (("lead", "time"), persistence),
        },
        coords={
            "model": list(propagators),
            "lead": np.arange(1, max_lead + 1),
            "time": tos.time,
        },
        attrs={
            "index": "area-weighted SST over 5S-5N, 170W-120W",
            "forecast_alignment": "forecast initialized at time-lead and verified at time",
        },
    )
    return summaries, dataset


def summarize(
    correlations: dict[str, np.ndarray], fields: dict[str, xr.DataArray]
) -> dict[str, list[float]]:
    return {
        name: [area_weighted_mean(item, fields[name].lat.values) for item in values]
        for name, values in correlations.items()
    }


def write_correlations(
    path: Path, correlations: dict[str, np.ndarray], fields: dict[str, xr.DataArray]
) -> None:
    arrays = {}
    for name, values in correlations.items():
        arrays[name] = xr.DataArray(
            values,
            dims=("lead", "lat", "lon"),
            coords={
                "lead": np.arange(1, values.shape[0] + 1),
                "lat": fields[name].lat,
                "lon": fields[name].lon,
            },
        )
    xr.Dataset(arrays).to_netcdf(path, engine="h5netcdf")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(args.training_pcs, engine="h5netcdf") as dataset:
        training = np.asarray(dataset.pcs.values)

    models = {name: EOFModel.load(args.eof_dir / name) for name in VARIABLES}
    fields = {}
    encoded_parts = []
    for name, (_, source_name, filename) in VARIABLES.items():
        field = load_verification_field(
            args.verification_dir / filename, source_name, name in {"sic", "sit"}
        )
        field = align_spatial_coordinates(field, models[name])
        fields[name] = field
        encoded_parts.append(models[name].encode(field).values)
    encoded = np.concatenate(encoded_parts, axis=1)

    propagators = {
        "legacy_misaligned": fit_propagator(
            legacy_misaligned_training(training, args.member_length), None
        ),
        "aligned_with_boundaries": fit_propagator(training, None),
        "aligned_without_boundaries": fit_propagator(training, args.member_length),
    }
    all_correlations = {
        name: evaluate_model(matrix, encoded, fields, models, args.max_lead)
        for name, matrix in propagators.items()
    }
    with args.legacy_correlations.open("rb") as stream:
        legacy = {name: np.asarray(value) for name, value in pickle.load(stream).items()}

    legacy_difference = max(
        float(np.nanmax(np.abs(legacy[name] - all_correlations["legacy_misaligned"][name])))
        for name in VARIABLES
    )
    if legacy_difference > 1e-10:
        raise ValueError(
            "reconstructed legacy forecast differs from saved correlations: "
            f"{legacy_difference:.3e}"
        )

    summaries = {"legacy_misaligned": summarize(legacy, fields)}
    summaries.update({name: summarize(values, fields) for name, values in all_correlations.items()})
    for values in summaries.values():
        flat = [metric for variable in values.values() for metric in variable]
        values["mean_across_variables_and_leads"] = float(np.mean(flat))

    write_correlations(args.output_dir / "legacy_correlations.nc", legacy, fields)
    for name, values in all_correlations.items():
        write_correlations(args.output_dir / f"{name}_correlations.nc", values, fields)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    nino_summary, nino_dataset = evaluate_nino34(
        propagators, encoded, fields["tos"], models["tos"], args.max_lead
    )
    nino_summary["legacy_regression_max_correlation_difference"] = legacy_difference
    (args.output_dir / "nino34_summary.json").write_text(
        json.dumps(nino_summary, indent=2) + "\n", encoding="utf-8"
    )
    nino_dataset.to_netcdf(args.output_dir / "nino34_forecasts.nc", engine="h5netcdf")
    print(json.dumps(summaries, indent=2))
    print(json.dumps(nino_summary, indent=2))


if __name__ == "__main__":
    main()
