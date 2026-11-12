"""Borehole profile loading and differentiable heat-conduction operators."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import xarray as xr

from .cfr import ObservationLoadResult, SkippedObservation
from .linear import EOFPointDecoder, annualize_quarterly
from .records import BoreholeRecord, ObservationCollection

SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
HistoryPolicy = Literal["full", "legacy_reference_start"]


@dataclass(frozen=True)
class _BoreholeBatchPlan:
    annual_time: torch.Tensor
    year_indices: torch.Tensor
    state_slice: slice
    site_weights: torch.Tensor
    site_bias: torch.Tensor
    coefficients: torch.Tensor
    depth_mask: torch.Tensor
    observed: torch.Tensor
    variance: torch.Tensor
    families: list[str]


def conduction_forward(
    surface_temperature: torch.Tensor,
    years: torch.Tensor | np.ndarray,
    depths: torch.Tensor | np.ndarray,
    *,
    logging_year: float,
    diffusivity: float = 1.0e-6,
    reference_period: tuple[float, float] | None = None,
    history_policy: HistoryPolicy = "full",
) -> torch.Tensor:
    """Map a piecewise-constant surface history to a subsurface profile.

    ``full`` subtracts the reference mean but retains earlier thermal history.
    ``legacy_reference_start`` also drops history before the reference start,
    reproducing the previous FDVar observation operator.
    """
    if surface_temperature.ndim != 1:
        raise ValueError("surface_temperature must be a 1D tensor")
    years_tensor = torch.as_tensor(
        years, dtype=surface_temperature.dtype, device=surface_temperature.device
    )
    depths_tensor = torch.as_tensor(
        depths, dtype=surface_temperature.dtype, device=surface_temperature.device
    )
    if years_tensor.ndim != 1 or years_tensor.shape != surface_temperature.shape:
        raise ValueError("years must be 1D and match surface_temperature")
    if depths_tensor.ndim != 1 or depths_tensor.numel() == 0:
        raise ValueError("depths must be a nonempty 1D array")
    if not torch.isfinite(surface_temperature).all() or not torch.isfinite(years_tensor).all():
        raise ValueError("surface_temperature and years must be finite")
    if not torch.isfinite(depths_tensor).all() or torch.any(depths_tensor < 0):
        raise ValueError("depths must be finite and nonnegative")
    if years_tensor.numel() > 1 and torch.any(torch.diff(years_tensor) <= 0):
        raise ValueError("years must be strictly increasing")
    if not np.isfinite(logging_year) or not np.isfinite(diffusivity) or diffusivity <= 0:
        raise ValueError("logging_year must be finite and diffusivity must be positive")
    if history_policy not in ("full", "legacy_reference_start"):
        raise ValueError("history_policy must be 'full' or 'legacy_reference_start'")

    anomalies = surface_temperature
    history_start = years_tensor[0]
    if reference_period is not None:
        start, end = sorted(float(value) for value in reference_period)
        reference_mask = (years_tensor >= start) & (years_tensor <= end)
        if not torch.any(reference_mask):
            raise ValueError(f"no surface temperatures in reference period {reference_period}")
        anomalies = anomalies - anomalies[reference_mask].mean()
        if history_policy == "legacy_reference_start":
            history_start = surface_temperature.new_tensor(start)

    use = (years_tensor >= history_start) & (years_tensor <= float(logging_year))
    if not torch.any(use):
        raise ValueError("surface history has no samples at or before the logging year")
    history = torch.flip(anomalies[use], dims=(0,))
    age = float(logging_year) - torch.flip(years_tensor[use], dims=(0,))
    kernel = _heat_kernel(depths_tensor, age, diffusivity)

    profile = history[0] * kernel[:, 0]
    if history.numel() > 1:
        increments = kernel[:, 1:] - kernel[:, :-1]
        profile = profile + torch.sum(history[1:].unsqueeze(0) * increments, dim=1)
    return profile


def _heat_kernel(depths: torch.Tensor, ages: torch.Tensor, diffusivity: float) -> torch.Tensor:
    positive_age = ages.unsqueeze(0) > 0
    denominator = 2 * torch.sqrt(
        torch.clamp(ages.unsqueeze(0), min=0) * diffusivity * SECONDS_PER_YEAR
    )
    eta = torch.where(
        positive_age,
        depths.unsqueeze(1) / torch.clamp(denominator, min=torch.finfo(depths.dtype).tiny),
        torch.full((), torch.inf, dtype=depths.dtype, device=depths.device),
    )
    kernel = torch.erfc(eta)
    return torch.where(depths.unsqueeze(1) == 0, torch.ones_like(kernel), kernel)


class BoreholeOperator:
    """Decode annual local temperature and predict borehole profiles."""

    def __init__(
        self,
        decoder: EOFPointDecoder,
        *,
        reference_period: tuple[float, float] | None = (1300, 1700),
        diffusivity: float = 1.0e-6,
        history_policy: HistoryPolicy = "full",
        edge_policy: str = "legacy_repeat",
    ) -> None:
        self.decoder = decoder
        self.reference_period = reference_period
        self.diffusivity = diffusivity
        self.history_policy = history_policy
        self.edge_policy = edge_policy
        self._plan_key: tuple[int, torch.dtype, torch.device] | None = None
        self._plan_collection: ObservationCollection | None = None
        self._plan: _BoreholeBatchPlan | None = None

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
        if not len(observations):
            empty = trajectory.new_empty((0,))
            return empty, empty, empty, []
        plan = self._prepare_plan(observations, annual_time, trajectory)
        local_temperature = (
            annual_state[plan.year_indices, plan.state_slice] @ plan.site_weights.T
            + plan.site_bias
        )
        padded_profiles = torch.einsum(
            "rdt,tr->rd", plan.coefficients, local_temperature
        )
        return (
            padded_profiles[plan.depth_mask],
            plan.observed,
            plan.variance,
            plan.families,
        )

    def _prepare_plan(
        self,
        observations: ObservationCollection,
        annual_time: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> _BoreholeBatchPlan:
        key = (id(observations), trajectory.dtype, trajectory.device)
        if (
            observations is self._plan_collection
            and key == self._plan_key
            and self._plan is not None
            and torch.equal(self._plan.annual_time, annual_time)
        ):
            return self._plan
        records = tuple(observations)
        if any(not isinstance(record, BoreholeRecord) for record in records):
            raise TypeError("BoreholeOperator accepts only BoreholeRecord values")
        state_slice = self.decoder.layout.slice_for("tas")
        transforms = [
            self.decoder.transform("tas", record.latitude, record.longitude)
            for record in records
        ]
        site_weights = torch.as_tensor(
            np.stack([transform.weights[state_slice] for transform in transforms]),
            dtype=trajectory.dtype,
            device=trajectory.device,
        )
        site_bias = torch.as_tensor(
            [transform.bias for transform in transforms],
            dtype=trajectory.dtype,
            device=trajectory.device,
        )
        start = float(annual_time[0])
        if self.reference_period is not None and self.history_policy == "legacy_reference_start":
            start = min(float(value) for value in self.reference_period)
        end = max(record.logging_year for record in records)
        if self.reference_period is not None:
            end = max(end, max(float(value) for value in self.reference_period))
        year_indices = torch.nonzero(
            (annual_time >= start) & (annual_time <= end), as_tuple=True
        )[0]
        operator_years = annual_time[year_indices].to(dtype=trajectory.dtype)
        maximum_depths = max(record.size for record in records)
        coefficients = trajectory.new_zeros(
            (len(records), maximum_depths, operator_years.numel())
        )
        depth_mask = torch.zeros(
            (len(records), maximum_depths), dtype=torch.bool, device=trajectory.device
        )
        observed = []
        variance = []
        families = []
        for record_index, record in enumerate(records):
            coefficients[record_index, : record.size] = _conduction_coefficients(
                operator_years,
                torch.as_tensor(
                    record.depth, dtype=trajectory.dtype, device=trajectory.device
                ),
                logging_year=record.logging_year,
                diffusivity=self.diffusivity,
                reference_period=self.reference_period,
                history_policy=self.history_policy,
            )
            depth_mask[record_index, : record.size] = True
            observed.append(
                torch.as_tensor(record.value, dtype=trajectory.dtype, device=trajectory.device)
            )
            variance.append(
                torch.as_tensor(
                    record.variance, dtype=trajectory.dtype, device=trajectory.device
                )
            )
            families.extend([record.family.value] * record.size)
        self._plan_key = key
        self._plan_collection = observations
        self._plan = _BoreholeBatchPlan(
            annual_time=annual_time.detach().clone(),
            year_indices=year_indices,
            state_slice=state_slice,
            site_weights=site_weights,
            site_bias=site_bias,
            coefficients=coefficients,
            depth_mask=depth_mask,
            observed=torch.cat(observed),
            variance=torch.cat(variance),
            families=families,
        )
        return self._plan


def _conduction_coefficients(
    years: torch.Tensor,
    depths: torch.Tensor,
    *,
    logging_year: float,
    diffusivity: float,
    reference_period: tuple[float, float] | None,
    history_policy: HistoryPolicy,
) -> torch.Tensor:
    """Build the fixed linear map from annual surface temperature to a profile."""
    history_start = years[0]
    reference = None
    if reference_period is not None:
        start, end = sorted(float(value) for value in reference_period)
        reference = (years >= start) & (years <= end)
        if not torch.any(reference):
            raise ValueError(f"no surface temperatures in reference period {reference_period}")
        if history_policy == "legacy_reference_start":
            history_start = years.new_tensor(start)
    use = (years >= history_start) & (years <= logging_year)
    if not torch.any(use):
        raise ValueError("surface history has no samples at or before the logging year")
    use_indices = torch.nonzero(use, as_tuple=True)[0].flip(0)
    ages = logging_year - years[use_indices]
    kernel = _heat_kernel(depths, ages, diffusivity)
    history_weights = kernel.clone()
    if use_indices.numel() > 1:
        history_weights[:, 1:] -= kernel[:, :-1]
    coefficients = years.new_zeros((depths.numel(), years.numel()))
    coefficients[:, use_indices] = history_weights
    if reference is not None:
        coefficients[:, reference] -= history_weights.sum(dim=1, keepdim=True) / int(
            reference.sum()
        )
    return coefficients


def load_borehole_observations(
    path: str | Path,
    *,
    target_depths: np.ndarray | None = None,
    value_name: str = "perturbation",
    error_name: str = "error",
    strict: bool = True,
) -> ObservationLoadResult:
    """Load compact NetCDF or legacy xarray pickles into depth-indexed records."""
    if target_depths is None:
        target_depths = np.arange(100.0, 305.0, 5.0)
    target_depths = np.asarray(target_depths, dtype=float)
    if target_depths.ndim != 1 or target_depths.size == 0:
        raise ValueError("target_depths must be a nonempty 1D array")
    path = Path(path)
    if path.suffix.lower() in {".nc", ".nc4"}:
        return _load_borehole_netcdf(
            path,
            target_depths=target_depths,
            value_name=value_name,
            error_name=error_name,
            strict=strict,
        )
    with path.open("rb") as stream:
        stations = pickle.load(stream)

    records = []
    skipped = []
    for index, station in enumerate(stations):
        record_id = _scalar_string(station, "station_id", default=f"borehole_{index:04d}")
        try:
            records.append(
                _station_to_record(
                    station,
                    record_id,
                    target_depths=target_depths,
                    value_name=value_name,
                    error_name=error_name,
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            skipped.append(SkippedObservation(record_id, str(exc)))
    if strict and skipped:
        examples = "; ".join(f"{item.record_id}: {item.reason}" for item in skipped[:3])
        raise ValueError(
            f"failed to convert {len(skipped)} of {len(records) + len(skipped)} boreholes; "
            f"examples: {examples}"
        )
    return ObservationLoadResult(ObservationCollection(records), tuple(skipped))


def _load_borehole_netcdf(
    path: Path,
    *,
    target_depths: np.ndarray,
    value_name: str,
    error_name: str,
    strict: bool,
) -> ObservationLoadResult:
    with xr.open_dataset(path, engine="h5netcdf") as source:
        dataset = source.load()
    if "station" not in dataset.dims:
        raise ValueError("borehole NetCDF requires a station dimension")
    records = []
    skipped = []
    for index, station_id in enumerate(dataset.station.values):
        record_id = str(station_id)
        station = dataset.isel(station=index)
        try:
            if "status" in station and str(station.status.item()) != "ok":
                raise ValueError(f"error estimation status is {station.status.item()}")
            records.append(
                _station_to_record(
                    station,
                    record_id,
                    target_depths=target_depths,
                    value_name=value_name,
                    error_name=error_name,
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            skipped.append(SkippedObservation(record_id, str(exc)))
    if strict and skipped:
        examples = "; ".join(f"{item.record_id}: {item.reason}" for item in skipped[:3])
        raise ValueError(
            f"failed to convert {len(skipped)} of {len(records) + len(skipped)} boreholes; "
            f"examples: {examples}"
        )
    return ObservationLoadResult(ObservationCollection(records), tuple(skipped))


def _station_to_record(
    station: xr.Dataset,
    record_id: str,
    *,
    target_depths: np.ndarray,
    value_name: str,
    error_name: str,
) -> BoreholeRecord:
    value = _interpolate_data_array(station[value_name], target_depths)
    variance = _interpolate_data_array(station[error_name], target_depths)
    keep = np.isfinite(value) & np.isfinite(variance) & (variance > 0)
    if not np.any(keep):
        raise ValueError("no finite profile values with positive variance at target depths")
    return BoreholeRecord(
        record_id=record_id,
        latitude=float(station["lat"].item()),
        longitude=float(station["lon"].item()),
        logging_year=float(station["logging_year"].item()),
        depth=target_depths[keep],
        value=value[keep],
        variance=variance[keep],
        metadata={"region": _scalar_string(station, "region", default="")},
    )


def _interpolate_data_array(array: xr.DataArray, target_depths: np.ndarray) -> np.ndarray:
    # Old xarray pickles can retain readable values/dimensions while their
    # serialized coordinate indexes are incompatible with newer xarray.
    depth_name = next((name for name in ("depth", "depths") if name in array.dims), None)
    if depth_name is None:
        raise ValueError(f"{array.name or 'array'} has no depth coordinate")
    source_depth = np.asarray(array.coords.variables[depth_name].values, dtype=float)
    source_value = np.asarray(array.values, dtype=float)
    finite = np.isfinite(source_depth) & np.isfinite(source_value)
    if np.count_nonzero(finite) < 2:
        raise ValueError(f"{array.name or 'array'} needs at least two finite depth values")
    order = np.argsort(source_depth[finite])
    return np.interp(
        target_depths,
        source_depth[finite][order],
        source_value[finite][order],
        left=np.nan,
        right=np.nan,
    )


def _scalar_string(station: xr.Dataset, name: str, *, default: str) -> str:
    if name not in station:
        return default
    return str(station[name].item())
