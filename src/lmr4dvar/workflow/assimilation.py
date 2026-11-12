"""Artifact and observation assembly for the assimilation workflow stage."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import xarray as xr

from lmr4dvar.assimilation import OperatorBinding, WeakConstraintObjective, assimilate_lbfgs
from lmr4dvar.config import ProjectConfig
from lmr4dvar.eof import EOFModel
from lmr4dvar.lim import LIMModel
from lmr4dvar.observations import (
    BoreholeOperator,
    EOFPointDecoder,
    LinearProxyOperator,
    ObservationCollection,
    SedPSMOperator,
    load_borehole_observations,
    load_cfr_observations,
    load_sedpsm_ppe,
)
from lmr4dvar.observations.additive import AdditiveProxyOperator, predictor_terms
from lmr4dvar.observations.calibrated_artifact import load_calibrated_observations
from lmr4dvar.state import StateLayout

from .planner import artifact_stage_directory, stage_directory
from .stages import Stage


def execute_assimilation(config: ProjectConfig) -> tuple[Path, ...]:
    """Load independent artifacts, optimize the state trajectory, and save analysis."""
    layout = StateLayout.from_mode_counts(
        {name: item.n_modes for name, item in config.state.variables.items()}
    )
    models = _load_eof_models(config)
    pages2k = config.observations.pages2k
    additive = pages2k.enabled and (
        pages2k.artifact_format == "calibrated_netcdf"
        or (pages2k.calibration.enabled and pages2k.calibration.method == "tardif2019_quarterly")
    )
    decoder = EOFPointDecoder(
        models, layout, spatial_policy="great_circle" if additive else "coordinate"
    )
    lim = LIMModel.load(artifact_stage_directory(config, Stage.TRAIN_LIM))
    if tuple(str(value) for value in lim.training_mean.state.values) != layout.labels:
        raise ValueError("LIM state labels differ from configured StateLayout")
    device = _device(config)
    dtype = getattr(torch, config.assimilation.precision.value)
    objective = WeakConstraintObjective(
        lim,
        dtype=dtype,
        device=device,
        initial_weight=config.assimilation.initial_weight,
        model_error_weight=config.assimilation.model_error_weight,
    )
    model_time = assimilation_time(config).to(device=device, dtype=dtype)
    bindings = build_observation_bindings(config, decoder)
    result = assimilate_lbfgs(
        objective,
        model_time,
        bindings,
        max_iterations=config.assimilation.max_iterations,
        inner_iterations=config.assimilation.inner_iterations,
        minimum_outer_iterations=config.assimilation.minimum_outer_iterations,
        tolerance_gradient=config.assimilation.tolerance_gradient,
        tolerance_change=config.assimilation.tolerance_change,
        outer_tolerance_change=config.assimilation.outer_tolerance_change,
        history_size=config.assimilation.history_size,
    )
    directory = stage_directory(config, Stage.ASSIMILATE)
    directory.mkdir(parents=True, exist_ok=True)
    analysis_path = directory / "analysis.nc"
    analysis = xr.Dataset(
        {
            "control": (("time", "state"), result.control.cpu().numpy()),
            "trajectory": (("time", "state"), result.trajectory.cpu().numpy()),
        },
        coords={"time": model_time.cpu().numpy(), "state": list(layout.labels)},
        attrs={
            "control_parameterization": "state_trajectory",
            "numeric_precision": config.assimilation.precision.value,
        },
    )
    analysis.control.attrs["description"] = "directly optimized LIM state trajectory"
    analysis.trajectory.attrs["description"] = "analysis state trajectory (equal to control)"
    analysis.to_netcdf(analysis_path, engine="h5netcdf")
    diagnostics_path = directory / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "control_parameterization": "state_trajectory",
                "numeric_precision": config.assimilation.precision.value,
                "optimizer_settings": {
                    "name": config.assimilation.optimizer,
                    "max_outer_iterations": config.assimilation.max_iterations,
                    "inner_iterations": config.assimilation.inner_iterations,
                    "minimum_outer_iterations": (config.assimilation.minimum_outer_iterations),
                    "history_size": config.assimilation.history_size,
                    "tolerance_gradient": config.assimilation.tolerance_gradient,
                    "tolerance_change": config.assimilation.tolerance_change,
                    "outer_tolerance_change": (config.assimilation.outer_tolerance_change),
                },
                "function_evaluations": result.function_evaluations,
                "outer_iterations": result.outer_iterations,
                "outer_cost_history": result.outer_cost_history,
                "final_gradient_norm": result.final_gradient_norm,
                "converged": result.converged,
                "convergence_reason": result.convergence_reason,
                "cost_history": result.cost_history,
                "initial_history": result.initial_history,
                "model_error_history": result.model_error_history,
                "observation_history": result.observation_history,
                "observation_by_family_history": result.observation_by_family_history,
                "observation_used_count_history": result.observation_used_count_history,
                "observation_dropped_count_history": result.observation_dropped_count_history,
                "gradient_norm_history": result.gradient_norm_history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return analysis_path, diagnostics_path


def assimilation_time(config: ProjectConfig) -> torch.Tensor:
    if config.state.timestep_months != 3:
        raise ValueError("assimilation currently requires state.timestep_months=3")
    start_year, start_month = _parse_year_month(config.assimilation.start)
    end_year, end_month = _parse_year_month(config.assimilation.end)
    if start_month != 1 or end_month < 10:
        raise ValueError("assimilation window must contain complete Jan/Apr/Jul/Oct years")
    if end_year < start_year:
        raise ValueError("assimilation end must not precede start")
    values = [
        year + (month - 1) / 12
        for year in range(start_year, end_year + 1)
        for month in (1, 4, 7, 10)
    ]
    return torch.tensor(values, dtype=torch.float64)


def build_observation_bindings(
    config: ProjectConfig, decoder: EOFPointDecoder
) -> list[OperatorBinding]:
    bindings = []
    linear_records = []
    additive_records = []
    for family in ("pages2k", "temp12k"):
        source = getattr(config.observations, family)
        if source.enabled:
            path = source.path
            calibrated_netcdf = (
                family == "pages2k" and source.artifact_format == "calibrated_netcdf"
            )
            if family == "pages2k" and source.calibration.enabled:
                calibrated_netcdf = source.calibration.method == "tardif2019_quarterly"
                path = stage_directory(config, Stage.CALIBRATE_PAGES2K) / (
                    "calibrated_proxies.nc" if calibrated_netcdf else "cfr_job"
                )
            if family == "temp12k" and source.preparation.enabled:
                path = stage_directory(config, Stage.PREPARE_TEMP12K) / "temp12k.pkl"
            if path is None:
                raise ValueError(f"observations.{family}.path is required")
            if calibrated_netcdf:
                _validate_additive_prior(config)
                observations = load_calibrated_observations(path)
                for record in observations:
                    if record.family.value != "pages2k":
                        raise ValueError("PAGES2k calibrated artifact contains another family")
                    for term in predictor_terms(record):
                        variable = config.state.variables.get(term.variable)
                        if variable is None or variable.units != term.units:
                            raise ValueError(
                                f"{record.record_id}: state units differ from "
                                f"{term.variable} PSM units"
                            )
                additive_records.extend(observations)
                continue
            loaded = load_cfr_observations(
                path,
                family=family,
                calibrated_only=family == "pages2k",
                strict=source.strict,
            )
            linear_records.extend(loaded.observations)
    if linear_records:
        bindings.append(
            OperatorBinding(LinearProxyOperator(decoder), ObservationCollection(linear_records))
        )
    if additive_records:
        bindings.append(
            OperatorBinding(AdditiveProxyOperator(decoder), ObservationCollection(additive_records))
        )

    sedpsm = config.observations.sedpsm
    if sedpsm.enabled:
        if config.paths.sedpsm_root is None:
            raise ValueError("SedPSM root is required")
        observations = load_sedpsm_ppe(
            sedpsm.resolve_data_path(config.paths.sedpsm_root),
            experiment=sedpsm.experiment,
            variance_floor=sedpsm.variance_floor,
        )
        climatology = _tas_climatology(config)
        bindings.append(
            OperatorBinding(
                SedPSMOperator(
                    decoder,
                    climatology=lambda record: _site_climatology(climatology, record),
                    n_age_realizations=sedpsm.n_age_realizations,
                ),
                observations,
            )
        )

    borehole = config.observations.borehole
    if borehole.enabled:
        borehole_path = borehole.path
        if borehole.error_estimation.enabled:
            borehole_path = stage_directory(config, Stage.ESTIMATE_BOREHOLE_ERROR) / "boreholes.nc"
        if borehole_path is None:
            raise ValueError("observations.borehole.path is required")
        loaded = load_borehole_observations(borehole_path, strict=borehole.strict)
        bindings.append(
            OperatorBinding(
                BoreholeOperator(
                    decoder,
                    reference_period=borehole.reference_period,
                    history_policy=borehole.history_policy,
                ),
                loaded.observations,
            )
        )
    if not bindings:
        raise ValueError("no enabled observation records were loaded")
    return bindings


def _validate_additive_prior(config: ProjectConfig) -> None:
    root = config.paths.prior_run_root
    preprocess = (
        (root / "stages" / "preprocess") if root else stage_directory(config, Stage.PREPROCESS)
    )
    manifest = preprocess / "manifest.json"
    if not manifest.is_file():
        raise ValueError("additive PSM requires a physical-time preprocessing manifest")
    metadata = json.loads(manifest.read_text(encoding="ascii"))
    policies = metadata.get("monthly_time_policies", {})
    if any(policies.get(variable) != "cf_bounds" for variable in ("tas", "pr")):
        raise ValueError(
            "additive PSM requires tas/pr physical CF-month bounds; "
            "legacy quarterly prior is unsafe"
        )


def _load_eof_models(config: ProjectConfig) -> dict[str, EOFModel]:
    directory = artifact_stage_directory(config, Stage.EOF)
    return {variable: EOFModel.load(directory / variable) for variable in config.state.variables}


def _tas_climatology(config: ProjectConfig) -> xr.DataArray:
    sedpsm = config.observations.sedpsm
    path = sedpsm.climatology_path
    variable = sedpsm.climatology_variable
    if path is None:
        path = stage_directory(config, Stage.PREPROCESS) / "climatology" / "tas.nc"
        variable = "tas"
    if not path.is_file():
        raise ValueError(f"SedPSM tas climatology does not exist: {path}")
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        if variable not in dataset:
            raise ValueError(f"SedPSM climatology variable {variable!r} is absent from {path}")
        return dataset[variable].load()


def _site_climatology(field: xr.DataArray, record) -> float:
    longitude = record.longitude % 360
    if float(field.lon.max()) <= 180:
        longitude = (longitude + 180) % 360 - 180
    value = float(field.sel(lat=record.latitude, lon=longitude, method="nearest").mean().item())
    units = str(field.attrs.get("units", "")).strip().lower().replace("_", " ")
    if units in {"k", "kelvin"}:
        return value - 273.15
    if units in {
        "c",
        "degc",
        "degree c",
        "degrees c",
        "degree celsius",
        "degrees celsius",
    }:
        return value
    raise ValueError(f"unsupported tas climatology units: {field.attrs.get('units')!r}")


def _device(config: ProjectConfig) -> torch.device:
    requested = config.assimilation.device.value
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _parse_year_month(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(-?\d+)-(\d{2})", value)
    if match is None:
        raise ValueError(f"invalid year-month value: {value!r}")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"invalid month in {value!r}")
    return year, month
