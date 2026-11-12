"""Compare the standalone MRI LIM forecast skill with the legacy result."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from compare_cesm_lim_forecast import (
    VARIABLES,
    align_spatial_coordinates,
    evaluate_model,
    load_verification_field,
    summarize,
    write_correlations,
)

from lmr4dvar.eof.model import EOFModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eof-dir", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    parser.add_argument("--legacy-correlations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-lead", type=int, default=4)
    return parser.parse_args()


def map_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    difference = candidate[valid] - reference[valid]
    return {
        "spatial_correlation": float(np.corrcoef(reference[valid], candidate[valid])[0, 1]),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "valid_grid_points": int(valid.sum()),
    }


def legacy_compatible_encode(field: xr.DataArray, model: EOFModel) -> np.ndarray:
    """Project after replacing validation-field NaNs with zero, as legacy EOF did."""
    valid = model.valid_mask.stack(space=model.valid_mask.dims).values
    stacked = field.stack(space=model.valid_mask.dims).transpose("time", "space")
    values = stacked.values[:, valid]
    weights = model.weights.stack(space=model.valid_mask.dims).values[valid]
    mean = model.weighted_mean.stack(space=model.valid_mask.dims).values[valid]
    if not np.allclose(mean, 0.0):
        raise ValueError("legacy-compatible projection requires an uncentered EOF")
    components = model.components.stack(space=model.valid_mask.dims).values[:, valid]
    weighted = np.nan_to_num(values * weights, nan=0.0)
    return (weighted @ components.T) / model.pc_scale.values


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {name: EOFModel.load(args.eof_dir / name) for name in VARIABLES}

    fields = {}
    encoded_parts = []
    for name, (_, source_name, filename) in VARIABLES.items():
        field = load_verification_field(
            args.verification_dir / filename, source_name, name in {"sic", "sit"}
        )
        field = align_spatial_coordinates(field, models[name])
        fields[name] = field
        encoded_parts.append(legacy_compatible_encode(field, models[name]))
    encoded = np.concatenate(encoded_parts, axis=1)

    with xr.open_dataset(args.dynamics, engine="h5netcdf") as dataset:
        propagator = np.asarray(dataset.G.values)
    standalone = evaluate_model(propagator, encoded, fields, models, args.max_lead)
    with args.legacy_correlations.open("rb") as stream:
        legacy = {name: np.asarray(value) for name, value in pickle.load(stream).items()}

    legacy_skill = summarize(legacy, fields)
    standalone_skill = summarize(standalone, fields)
    comparison = {
        name: [
            map_metrics(legacy[name][lead], standalone[name][lead])
            for lead in range(args.max_lead)
        ]
        for name in VARIABLES
    }
    skill_differences = {
        name: [
            float(standalone_skill[name][lead] - legacy_skill[name][lead])
            for lead in range(args.max_lead)
        ]
        for name in VARIABLES
    }
    flat_legacy = [value for values in legacy_skill.values() for value in values]
    flat_standalone = [value for values in standalone_skill.values() for value in values]
    flat_maps = [metric for values in comparison.values() for metric in values]
    summary = {
        "legacy_area_weighted_skill": legacy_skill,
        "standalone_area_weighted_skill": standalone_skill,
        "standalone_minus_legacy_skill": skill_differences,
        "map_comparison": comparison,
        "aggregate": {
            "legacy_mean_skill": float(np.mean(flat_legacy)),
            "standalone_mean_skill": float(np.mean(flat_standalone)),
            "mean_skill_difference": float(np.mean(flat_standalone) - np.mean(flat_legacy)),
            "mean_map_spatial_correlation": float(
                np.mean([metric["spatial_correlation"] for metric in flat_maps])
            ),
            "minimum_map_spatial_correlation": float(
                np.min([metric["spatial_correlation"] for metric in flat_maps])
            ),
            "mean_map_rmse": float(np.mean([metric["rmse"] for metric in flat_maps])),
            "maximum_map_absolute_difference": float(
                np.max([metric["maximum_absolute_difference"] for metric in flat_maps])
            ),
        },
    }
    write_correlations(args.output_dir / "standalone_correlations.nc", standalone, fields)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
