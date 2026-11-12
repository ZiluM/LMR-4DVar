"""SedPSM PPE adapters using a separately installed local SedPSM checkout."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .linear import EOFPointDecoder, annualize_quarterly
from .matching import match_nearest_times
from .records import ObservationCollection, ObservationFamily, ObservationRecord

ClimatologySource = float | Mapping[str, float] | Callable[[ObservationRecord], float]


def load_sedpsm_ppe(
    path: str | Path, *, experiment: int, variance_floor: float = 1e-8
) -> ObservationCollection:
    """Load a SedPSM PPE pickle and reduce ensemble members to mean/variance."""
    if experiment not in range(5):
        raise ValueError("SedPSM PPE experiment must be in 0..4")
    try:
        with Path(path).open("rb") as stream:
            database = pickle.load(stream)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SedPSM must be installed into the lmr4dvar environment from the configured "
            "local checkout before loading PPE pickle files"
        ) from exc
    records = [
        sedpsm_record_to_observation(
            record, index=index, experiment=experiment, variance_floor=variance_floor
        )
        for index, record in enumerate(database)
    ]
    return ObservationCollection(records)


def sedpsm_record_to_observation(
    record: Any, *, index: int, experiment: int, variance_floor: float = 1e-8
) -> ObservationRecord:
    """Convert one SedPSM record without importing SedPSM implementation types."""
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    metadata = dict(getattr(record, "meta", {}))
    dataset_name = str(
        metadata.get("dataSetName") or metadata.get("dataset_name") or f"record_{index:03d}"
    )
    values = np.asarray(record.values, dtype=float)
    time = np.asarray(record.time, dtype=float)
    if values.ndim == 1:
        mean = values
        variance = np.full(time.shape, variance_floor)
        ensemble_size = 1
    elif values.ndim == 2 and values.shape[0] == time.size:
        mean = np.nanmean(values, axis=1)
        variance = np.nanvar(values, axis=1)
        variance = np.where(
            np.isfinite(variance), np.maximum(variance, variance_floor), variance_floor
        )
        ensemble_size = values.shape[1]
    else:
        raise ValueError("SedPSM values must have shape (time,) or (time, ensemble)")
    return ObservationRecord(
        record_id=f"SedPSM_PPE{experiment}_{dataset_name}",
        family=ObservationFamily.SEDPSM,
        variable="tas",
        latitude=float(metadata["lat"]),
        longitude=float(metadata["lon"]),
        time=time,
        value=mean,
        variance=variance,
        metadata={
            **metadata,
            "experiment": experiment,
            "ensemble_size": ensemble_size,
            "proxy_type": metadata.get("proxy_type", getattr(record, "name", "")),
        },
    )


def sedpsm_experiment_parameters(
    experiment: int,
    proxy_type: str,
    dataset_name: str,
    ensemble_size: int,
    *,
    n_age_realizations: int = 1,
) -> dict[str, Any]:
    """Return the documented SedPSM PPE forward parameters for experiments 0--4."""
    if experiment not in range(5):
        raise ValueError("SedPSM PPE experiment must be in 0..4")
    if ensemble_size < 1 or n_age_realizations < 1:
        raise ValueError("ensemble_size and n_age_realizations must be positive")
    normalized = str(proxy_type).lower()
    if normalized == "uk37":
        sensor = {
            "proxy_type": "Uk37",
            "calibration": "bayspline",
            "output_units": "proxy",
            "nens": int(ensemble_size),
            "sample_params": True,
            "sample_noise": True,
            "propagate_ensemble": True,
        }
    elif normalized == "mgca":
        sensor = {
            "proxy_type": "MgCa",
            "calibration": "baymag_lite",
            "output_units": "proxy",
            "species": (
                "bulloides" if dataset_name == "Ocn-CariacoBasin.Black.2007" else "ruber"
            ),
            "nens": int(ensemble_size),
            "sample_params": True,
            "sample_noise": True,
            "propagate_ensemble": True,
        }
    else:
        raise ValueError(f"unsupported SedPSM proxy type: {proxy_type}")

    archive = (
        {"bio_depth": 0.5, "sed_acc_rate": 200, "layer_width": 1.0, "n_samples": np.inf}
        if experiment in (0, 1)
        else {"bio_depth": 5, "sed_acc_rate": 100, "layer_width": 1.0, "n_samples": np.inf}
    )
    low_noise = experiment in (0, 1, 2)
    observation = {
        "sigma_meas": 1.0e-5 if low_noise else (0.23 if normalized == "uk37" else 0.26),
        "sigma_ind": (0.0 if normalized == "uk37" else 0.02)
        if low_noise
        else (0.0 if normalized == "uk37" else 2.0),
        "scale_noise": True,
        "n_replicates": 1,
        "seed": 42,
    }
    age = {}
    if experiment == 4:
        age = {
            "age_model": "baconish",
            "bacon_kwargs": {"n_tie_points": 4, "age_err": 40.0},
            "n_age_realizations": int(n_age_realizations),
        }
    return {**sensor, **archive, **observation, **age}


class SedPSMOperator:
    """Run the external differentiable SedPSM pipeline on decoded local ``tas``."""

    def __init__(
        self,
        decoder: EOFPointDecoder,
        *,
        climatology: ClimatologySource,
        n_age_realizations: int = 1,
        time_tolerance: float = 1.0e-6,
        edge_policy: str = "legacy_repeat",
        run_pipeline: Callable[..., Any] | None = None,
        tensor_climate_type: type | None = None,
    ) -> None:
        if n_age_realizations < 1:
            raise ValueError("n_age_realizations must be positive")
        if (run_pipeline is None) != (tensor_climate_type is None):
            raise ValueError("run_pipeline and tensor_climate_type must be supplied together")
        if run_pipeline is None:
            try:
                from sedPSM_pytorch import run_pipeline as imported_pipeline
                from sedPSM_pytorch.climate import TensorClimate
            except ImportError as exc:
                raise RuntimeError(
                    "SedPSM must be installed in the lmr4dvar environment"
                ) from exc
            run_pipeline = imported_pipeline
            tensor_climate_type = TensorClimate
        self.decoder = decoder
        self.climatology = climatology
        self.n_age_realizations = n_age_realizations
        self.time_tolerance = time_tolerance
        self.edge_policy = edge_policy
        self.run_pipeline = run_pipeline
        self.tensor_climate_type = tensor_climate_type

    def predict(
        self,
        trajectory: torch.Tensor,
        model_time: torch.Tensor,
        observations: ObservationCollection,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        if trajectory.ndim != 2 or trajectory.shape[1] != self.decoder.layout.size:
            raise ValueError("trajectory must have shape (time, state_layout.size)")
        annual_time, annual_state = annualize_quarterly(
            trajectory, model_time, edge_policy=self.edge_policy
        )
        predictions = []
        observed = []
        variances = []
        families = []
        for record in observations:
            if (
                not isinstance(record, ObservationRecord)
                or record.family is not ObservationFamily.SEDPSM
            ):
                raise TypeError("SedPSMOperator accepts only SedPSM ObservationRecord values")
            transform = self.decoder.transform("tas", record.latitude, record.longitude)
            weights = torch.as_tensor(
                transform.weights, dtype=trajectory.dtype, device=trajectory.device
            )
            local_tas = annual_state @ weights + transform.bias + self._climatology(record)
            climate = self.tensor_climate_type(
                values=local_tas.to(torch.float64),
                years=annual_time.detach().cpu().numpy(),
                name="tas",
                units="degC",
                lat=record.latitude,
                lon=record.longitude,
            )
            metadata = record.metadata
            dataset_name = str(
                metadata.get("dataSetName")
                or metadata.get("dataset_name")
                or record.record_id
            )
            parameters = sedpsm_experiment_parameters(
                int(metadata["experiment"]),
                str(metadata["proxy_type"]),
                dataset_name,
                int(metadata["ensemble_size"]),
                n_age_realizations=self.n_age_realizations,
            )
            _, final = self.run_pipeline(
                climate, sample_timepoints=record.time.copy(), **parameters
            )
            mean_record = final.select_member("mean")
            model_values = mean_record.values
            if not isinstance(model_values, torch.Tensor):
                raise TypeError("SedPSM tensor pipeline returned non-tensor values")
            match = match_nearest_times(
                np.asarray(mean_record.time, dtype=float),
                record.time,
                tolerance=self.time_tolerance,
            )
            if match.observation_indices.size == 0:
                continue
            model_indices = torch.as_tensor(
                match.model_indices, dtype=torch.long, device=model_values.device
            )
            predictions.append(model_values[model_indices].to(trajectory))
            observed.append(
                torch.as_tensor(
                    record.value[match.observation_indices],
                    dtype=trajectory.dtype,
                    device=trajectory.device,
                )
            )
            variances.append(
                torch.as_tensor(
                    record.variance[match.observation_indices],
                    dtype=trajectory.dtype,
                    device=trajectory.device,
                )
            )
            families.extend([record.family.value] * match.observation_indices.size)
        if not predictions:
            empty = trajectory.new_empty((0,))
            return empty, empty, empty, []
        return torch.cat(predictions), torch.cat(observed), torch.cat(variances), families

    def _climatology(self, record: ObservationRecord) -> float:
        source = self.climatology
        if callable(source):
            value = source(record)
        elif isinstance(source, Mapping):
            value = source.get(record.record_id)
        else:
            value = source
        if value is None or not np.isfinite(value):
            raise ValueError(f"missing finite tas climatology for {record.record_id}")
        return float(value)
