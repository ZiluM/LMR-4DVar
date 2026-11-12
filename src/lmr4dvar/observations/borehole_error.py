"""Depth-dependent borehole error estimation from inversion ensembles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.special import erfc

from lmr4dvar.config.models import BoreholeErrorEstimationConfig

from .borehole import SECONDS_PER_YEAR

PROFILE_VARIABLES = {
    "mean": "perturbation",
    "plus_2sigma": "perturbation_plus_2sigma",
    "minus_2sigma": "perturbation_minus_2sigma",
}


@dataclass(frozen=True)
class BoreholeErrorOptions:
    svd_modes_by_step: dict[int, tuple[int, ...]]
    gradient_threshold_k_per_year: float
    deduplicate_diffusivities: bool
    variance_ddof: int


@dataclass(frozen=True)
class BoreholeErrorResult:
    variance: np.ndarray
    candidate_count: int
    retained_count: int


def effective_error_options(config: BoreholeErrorEstimationConfig) -> BoreholeErrorOptions:
    """Resolve SI or legacy-notebook defaults without hiding overrides."""
    if config.method_profile == "si_2026":
        default_modes = {25: (3, 4, 5), 40: (2, 3, 4), 50: (2, 3, 4)}
        default_gradient = 4.0 / 10.0
        default_deduplicate = False
        default_ddof = 0
    else:
        default_modes = {step: (2, 3, 4) for step in config.time_steps}
        default_gradient = 2.57 / 50.0
        default_deduplicate = True
        default_ddof = 1
    configured_modes = config.svd_modes_by_step
    modes = {
        step: tuple(
            int(value)
            for value in (
                configured_modes[step]
                if configured_modes is not None
                else default_modes.get(step, (2, 3, 4))
            )
        )
        for step in config.time_steps
    }
    return BoreholeErrorOptions(
        svd_modes_by_step=modes,
        gradient_threshold_k_per_year=(
            config.gradient_threshold_k_per_year
            if config.gradient_threshold_k_per_year is not None
            else default_gradient
        ),
        deduplicate_diffusivities=(
            config.deduplicate_diffusivities
            if config.deduplicate_diffusivities is not None
            else default_deduplicate
        ),
        variance_ddof=config.variance_ddof if config.variance_ddof is not None else default_ddof,
    )


def heat_conduction_matrix(
    depths: np.ndarray,
    years: np.ndarray,
    *,
    logging_year: float,
    diffusivity: float,
    time_step: float,
) -> np.ndarray:
    """Construct the step-response matrix used by the legacy SVD inversion."""
    depths = np.asarray(depths, dtype=float)
    years = np.asarray(years, dtype=float)
    if depths.ndim != 1 or years.ndim != 1 or depths.size == 0 or years.size == 0:
        raise ValueError("depths and years must be nonempty 1D arrays")
    ages = float(logging_year) - years[::-1]
    current = _erfc_kernel(depths, ages, diffusivity)
    previous = _erfc_kernel(depths, ages - float(time_step), diffusivity)
    return current - previous


def estimate_profile_error(
    profiles: dict[str, np.ndarray],
    depths: np.ndarray,
    *,
    logging_year: float,
    history_start_year: int,
    time_steps: list[int],
    thermal_conductivities: list[float],
    volumetric_heat_capacities: list[float],
    options: BoreholeErrorOptions,
    variance_floor: float,
) -> BoreholeErrorResult:
    """Estimate depthwise residual variance across retained inverse solutions."""
    depths = np.asarray(depths, dtype=float)
    mean_profile = np.asarray(profiles["mean"], dtype=float)
    if any(np.asarray(profile).shape != mean_profile.shape for profile in profiles.values()):
        raise ValueError("all borehole perturbation profiles must share one depth grid")
    if mean_profile.shape != depths.shape or not np.isfinite(mean_profile).all():
        raise ValueError("mean borehole profile must be finite and match depths")
    diffusivities = [
        conductivity / capacity
        for conductivity, capacity in product(
            thermal_conductivities, volumetric_heat_capacities
        )
    ]
    if options.deduplicate_diffusivities:
        diffusivities = list(dict.fromkeys(diffusivities))

    residuals = []
    candidate_count = 0
    for profile in profiles.values():
        profile = np.asarray(profile, dtype=float)
        if not np.isfinite(profile).all():
            continue
        for diffusivity in diffusivities:
            for time_step in time_steps:
                years = np.arange(
                    float(logging_year) - time_step,
                    float(history_start_year),
                    -time_step,
                    dtype=float,
                )[::-1]
                if years.size < 2:
                    continue
                matrix = heat_conduction_matrix(
                    depths,
                    years,
                    logging_year=logging_year,
                    diffusivity=diffusivity,
                    time_step=time_step,
                )
                u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
                for n_modes in options.svd_modes_by_step[time_step]:
                    candidate_count += 1
                    usable = min(n_modes, singular_values.size)
                    retained_values = singular_values[:usable]
                    if usable == 0 or np.any(retained_values <= np.finfo(float).eps):
                        continue
                    inverse = (
                        vt[:usable].T
                        @ np.diag(1.0 / retained_values)
                        @ u[:, :usable].T
                    )
                    history = (inverse @ profile)[::-1]
                    gradient = np.diff(history) / np.diff(years)
                    if np.any(np.abs(gradient) > options.gradient_threshold_k_per_year):
                        continue
                    reconstructed = matrix @ history[::-1]
                    residuals.append(reconstructed - mean_profile)
    retained_count = len(residuals)
    if retained_count <= options.variance_ddof:
        raise ValueError("no admissible borehole inversion ensemble remains")
    squared_sum = np.square(np.asarray(residuals)).sum(axis=0)
    variance = squared_sum / (retained_count - options.variance_ddof)
    variance = np.maximum(variance, variance_floor)
    return BoreholeErrorResult(variance, candidate_count, retained_count)


def estimate_borehole_error_artifact(
    config: BoreholeErrorEstimationConfig,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Estimate all station errors and write a compact NetCDF observation artifact."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target_depths = np.arange(
        config.depth_start,
        config.depth_end + config.depth_step / 2,
        config.depth_step,
    )
    options = effective_error_options(config)
    with xr.open_dataset(config.perturbation_path) as source:
        required = set(PROFILE_VARIABLES.values())
        missing = required - set(source.data_vars)
        if missing:
            raise ValueError(
                "borehole perturbation dataset is missing variables: "
                + ", ".join(sorted(missing))
            )
        if "station" not in source.dims or "depth" not in source.dims:
            raise ValueError("borehole perturbation dataset requires station and depth")
        station_ids = [str(value) for value in source.station.values]
        n_station = len(station_ids)
        shape = (n_station, target_depths.size)
        output_profiles = {
            variable: np.full(shape, np.nan) for variable in PROFILE_VARIABLES.values()
        }
        variance = np.full(shape, np.nan)
        candidates = np.zeros(n_station, dtype=np.int32)
        retained = np.zeros(n_station, dtype=np.int32)
        status = np.full(n_station, "not_processed", dtype=object)
        lat = np.asarray(source.lat.values, dtype=float)
        lon = np.asarray(source.lon.values, dtype=float)
        logging_year = np.asarray(source.logging_year.values, dtype=float)
        region = np.asarray(source.region.values, dtype=str)
        source_depth = np.asarray(source.depth.values, dtype=float)
        for index, _station_id in enumerate(station_ids):
            interpolated = {}
            for key, variable in PROFILE_VARIABLES.items():
                values = np.asarray(source[variable].isel(station=index).values, dtype=float)
                interpolated[key] = _interpolate_profile(
                    source_depth, values, target_depths
                )
                output_profiles[variable][index] = interpolated[key]
            valid = np.logical_and.reduce(
                [np.isfinite(values) for values in interpolated.values()]
            )
            if np.count_nonzero(valid) < config.minimum_depth_points:
                status[index] = "insufficient_depth_points"
                continue
            try:
                result = estimate_profile_error(
                    {key: values[valid] for key, values in interpolated.items()},
                    target_depths[valid],
                    logging_year=logging_year[index],
                    history_start_year=config.history_start_year,
                    time_steps=config.time_steps,
                    thermal_conductivities=config.thermal_conductivities,
                    volumetric_heat_capacities=config.volumetric_heat_capacities,
                    options=options,
                    variance_floor=config.variance_floor,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                status[index] = str(exc)
                continue
            variance[index, valid] = result.variance
            candidates[index] = result.candidate_count
            retained[index] = result.retained_count
            status[index] = "ok"

    dataset = xr.Dataset(
        {
            **{
                variable: (("station", "depth"), values)
                for variable, values in output_profiles.items()
            },
            "error": (("station", "depth"), variance),
            "candidate_count": ("station", candidates),
            "retained_count": ("station", retained),
            "status": ("station", status.astype(str)),
        },
        coords={
            "station": station_ids,
            "depth": target_depths,
            "lat": ("station", lat),
            "lon": ("station", lon),
            "logging_year": ("station", logging_year),
            "region": ("station", region),
        },
        attrs={
            "schema_version": 1,
            "method_profile": config.method_profile,
            "error_definition": "mean squared inverse-forward residual by depth",
            "error_units": "K2",
            "history_start_year": config.history_start_year,
            "gradient_threshold_k_per_year": options.gradient_threshold_k_per_year,
            "variance_ddof": options.variance_ddof,
        },
    )
    dataset.error.attrs.update(
        {"long_name": "effective borehole observation error variance", "units": "K2"}
    )
    dataset.perturbation.attrs.update(
        {"long_name": "geothermal-gradient-removed temperature", "units": "K"}
    )
    artifact = directory / "boreholes.nc"
    encoding = {
        name: {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        for name in (*PROFILE_VARIABLES.values(), "error")
    }
    dataset.to_netcdf(artifact, engine="h5netcdf", encoding=encoding)
    manifest = {
        "schema_version": 1,
        "source": str(config.perturbation_path),
        "method_profile": config.method_profile,
        "target_depths": {
            "start": config.depth_start,
            "end": config.depth_end,
            "step": config.depth_step,
        },
        "history_start_year": config.history_start_year,
        "time_steps": config.time_steps,
        "thermal_conductivities": config.thermal_conductivities,
        "volumetric_heat_capacities": config.volumetric_heat_capacities,
        "effective_options": {
            "svd_modes_by_step": options.svd_modes_by_step,
            "gradient_threshold_k_per_year": options.gradient_threshold_k_per_year,
            "deduplicate_diffusivities": options.deduplicate_diffusivities,
            "variance_ddof": options.variance_ddof,
        },
        "variance_floor": config.variance_floor,
        "counts": {
            "stations": int(dataset.sizes["station"]),
            "successful": int(np.count_nonzero(status == "ok")),
            "skipped": int(np.count_nonzero(status != "ok")),
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return artifact, manifest_path


def _erfc_kernel(depths: np.ndarray, ages: np.ndarray, diffusivity: float) -> np.ndarray:
    seconds = ages[None, :] * SECONDS_PER_YEAR
    valid = seconds > 0
    eta = np.full((depths.size, ages.size), np.inf)
    denominator = 2 * np.sqrt(np.where(valid, diffusivity * seconds, 1.0))
    np.divide(depths[:, None], denominator, out=eta, where=np.broadcast_to(valid, eta.shape))
    return erfc(eta)


def _interpolate_profile(
    source_depth: np.ndarray, source_value: np.ndarray, target_depth: np.ndarray
) -> np.ndarray:
    finite = np.isfinite(source_depth) & np.isfinite(source_value)
    if np.count_nonzero(finite) < 2:
        return np.full(target_depth.shape, np.nan)
    depth = source_depth[finite]
    value = source_value[finite]
    order = np.argsort(depth)
    depth = depth[order]
    value = value[order]
    unique_depth, unique_index = np.unique(depth, return_index=True)
    value = value[unique_index]
    return np.interp(
        target_depth,
        unique_depth,
        value,
        left=np.nan,
        right=np.nan,
    )
