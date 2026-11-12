"""Compare standalone MRI EOF artifacts with their legacy pickle counterparts."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment

VARIABLES = {
    "tas": (15, "tas.pkl", "finite_complete_record"),
    "ohc300": (30, "ohc300.pkl", "finite_complete_record"),
    "tos": (15, "tos.pkl", "finite_complete_record"),
    "sic": (15, "sic_NH.pkl", "fillna_zero"),
    "sit": (15, "sit_NH.pkl", "fillna_zero"),
    "ohc2000": (30, "ohc2000.pkl", "finite_complete_record"),
}


class LegacyEOFArtifact:
    """Attribute container used to deserialize the legacy EOF artifact."""


class LegacyEOFUnpickler(pickle.Unpickler):
    """Load EOF state without importing the legacy implementation."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "EOF" and name == "EOF":
            return LegacyEOFArtifact
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-eof-dir", type=Path, required=True)
    parser.add_argument("--standalone-eof-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_anomaly = left - left.mean(axis=1, keepdims=True)
    right_anomaly = right - right.mean(axis=1, keepdims=True)
    left_norm = np.linalg.norm(left_anomaly, axis=1)
    right_norm = np.linalg.norm(right_anomaly, axis=1)
    return (left_anomaly @ right_anomaly.T) / np.outer(left_norm, right_norm)


def matched_correlation(matrix: np.ndarray) -> dict[str, Any]:
    old_indices, new_indices = linear_sum_assignment(-np.abs(matrix))
    values = matrix[old_indices, new_indices]
    return {
        "old_mode": old_indices.tolist(),
        "standalone_mode": new_indices.tolist(),
        "correlation": values.tolist(),
        "mean_absolute_correlation": float(np.mean(np.abs(values))),
        "minimum_absolute_correlation": float(np.min(np.abs(values))),
        "same_index_mean_absolute_correlation": float(np.mean(np.abs(np.diag(matrix)))),
    }


def reconstruction_metrics(
    old_pcs: np.ndarray,
    old_scales: np.ndarray,
    old_components: np.ndarray,
    new_pcs: np.ndarray,
    new_scales: np.ndarray,
    new_components: np.ndarray,
    *,
    chunk_size: int = 100,
) -> dict[str, float]:
    count = 0
    sum_old = sum_new = sum_old2 = sum_new2 = sum_cross = sum_diff2 = 0.0
    n_time = min(old_pcs.shape[1], new_pcs.shape[0])
    for start in range(0, n_time, chunk_size):
        stop = min(start + chunk_size, n_time)
        old_values = (old_pcs[:, start:stop].T * old_scales) @ old_components
        new_values = (new_pcs[start:stop] * new_scales) @ new_components
        difference = new_values - old_values
        count += difference.size
        sum_old += float(old_values.sum())
        sum_new += float(new_values.sum())
        sum_old2 += float(np.square(old_values).sum())
        sum_new2 += float(np.square(new_values).sum())
        sum_cross += float((old_values * new_values).sum())
        sum_diff2 += float(np.square(difference).sum())
    mean_old = sum_old / count
    mean_new = sum_new / count
    covariance = sum_cross / count - mean_old * mean_new
    variance_old = sum_old2 / count - mean_old**2
    variance_new = sum_new2 / count - mean_new**2
    rmse = np.sqrt(sum_diff2 / count)
    return {
        "weighted_field_correlation": float(covariance / np.sqrt(variance_old * variance_new)),
        "weighted_field_rmse": float(rmse),
        "relative_rmse_to_legacy_rms": float(rmse / np.sqrt(sum_old2 / count)),
    }


def compare_variable(
    legacy_path: Path,
    standalone_directory: Path,
    n_modes: int,
    legacy_missing_value_treatment: str,
) -> dict[str, Any]:
    with legacy_path.open("rb") as stream:
        legacy = LegacyEOFUnpickler(stream).load()
    with xr.open_dataset(standalone_directory / "eofs.nc", engine="h5netcdf") as dataset:
        standalone_eofs = dataset.load()
    with xr.open_dataset(standalone_directory / "pcs.nc", engine="h5netcdf") as dataset:
        standalone_pcs = np.asarray(dataset.pcs.values)

    old_pcs = np.asarray(legacy.pc[:n_modes]).real
    old_scales = np.asarray(legacy.pc_std[:n_modes]).real.reshape(-1)
    old_components_valid = np.asarray(legacy.e_vector[:n_modes]).real
    old_mask = np.asarray(legacy.flag, dtype=bool).reshape(-1)
    new_mask = np.asarray(standalone_eofs.valid_mask.values, dtype=bool).reshape(-1)
    new_components_full = np.asarray(standalone_eofs.components.values).reshape(n_modes, -1)
    if old_mask.shape != new_mask.shape:
        raise ValueError(f"spatial shape differs: {old_mask.shape} versus {new_mask.shape}")

    old_components_full = np.zeros((n_modes, old_mask.size))
    old_components_full[:, old_mask] = old_components_valid
    common = old_mask & new_mask
    old_common = old_components_full[:, common]
    new_common = new_components_full[:, common]
    if not np.isfinite(new_common).all():
        raise ValueError("standalone components are non-finite inside the common mask")

    old_basis = np.linalg.qr(old_common.T, mode="reduced")[0]
    new_basis = np.linalg.qr(new_common.T, mode="reduced")[0]
    basis_overlap = old_basis.T @ new_basis
    subspace_cosines = np.linalg.svd(basis_overlap, compute_uv=False)
    projector_difference_squared = max(
        0.0, float(2 * n_modes - 2 * np.square(basis_overlap).sum())
    )
    component_correlations = row_correlation(old_common, new_common)
    n_time = min(old_pcs.shape[1], standalone_pcs.shape[0])
    pc_correlations = row_correlation(old_pcs[:, :n_time], standalone_pcs[:n_time].T)

    old_variance = np.asarray(legacy.varperc[:n_modes]).real
    new_variance = np.asarray(standalone_eofs.explained_variance_ratio.values)
    result = {
        "legacy_path": str(legacy_path),
        "standalone_directory": str(standalone_directory),
        "n_modes": n_modes,
        "legacy_training_samples": int(old_pcs.shape[1]),
        "standalone_training_samples": int(standalone_pcs.shape[0]),
        "common_training_samples": int(n_time),
        "legacy_missing_value_treatment": legacy_missing_value_treatment,
        "legacy_valid_grid_points": int(old_mask.sum()),
        "standalone_valid_grid_points": int(new_mask.sum()),
        "common_valid_grid_points": int(common.sum()),
        "retained_subspace": {
            "mean_cosine": float(np.mean(subspace_cosines)),
            "minimum_cosine": float(np.min(subspace_cosines)),
            "maximum_principal_angle_degrees": float(
                np.rad2deg(np.arccos(np.clip(np.min(subspace_cosines), -1.0, 1.0)))
            ),
            "projection_relative_frobenius_difference": float(
                np.sqrt(projector_difference_squared) / np.sqrt(n_modes)
            ),
        },
        "matched_components": matched_correlation(component_correlations),
        "matched_pcs": matched_correlation(pc_correlations),
        "retained_explained_variance": {
            "legacy": float(old_variance.sum()),
            "standalone": float(new_variance.sum()),
            "difference": float(new_variance.sum() - old_variance.sum()),
        },
        "retained_reconstruction": reconstruction_metrics(
            old_pcs,
            old_scales,
            old_common,
            standalone_pcs,
            np.asarray(standalone_eofs.pc_scale.values),
            new_common,
        ),
    }
    del legacy, standalone_eofs, standalone_pcs
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    results = {}
    for name, (n_modes, legacy_filename, missing_treatment) in VARIABLES.items():
        print(f"Comparing {name}", flush=True)
        results[name] = compare_variable(
            args.legacy_eof_dir / legacy_filename,
            args.standalone_eof_dir / name,
            n_modes,
            missing_treatment,
        )
        brief = {
            "subspace_min_cosine": results[name]["retained_subspace"]["minimum_cosine"],
            "matched_pc_mean_abs_r": results[name]["matched_pcs"][
                "mean_absolute_correlation"
            ],
            "reconstruction_r": results[name]["retained_reconstruction"][
                "weighted_field_correlation"
            ],
        }
        print(json.dumps(brief, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
