"""Linear EOF point decoding and calibrated proxy operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lmr4dvar.eof import EOFModel
from lmr4dvar.state import StateLayout

from .matching import (
    WindowIndex,
    average_indexed_windows,
    build_window_index,
    match_nearest_times,
)
from .records import BoreholeRecord, ObservationCollection, ObservationRecord

ANNUAL_MONTHS = (-12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
_MONTH_TO_QUARTER = {
    -12: (0, 0),
    1: (0, 0),
    2: (0, 0),
    3: (0, 1),
    4: (0, 1),
    5: (0, 1),
    6: (0, 2),
    7: (0, 2),
    8: (0, 2),
    9: (0, 3),
    10: (0, 3),
    11: (0, 3),
    12: (0, 3),
    -9: (-1, 3),
    -10: (-1, 3),
    -11: (-1, 3),
    -6: (-1, 2),
    -7: (-1, 2),
    -8: (-1, 2),
}


@dataclass(frozen=True)
class PointTransform:
    weights: np.ndarray
    bias: float
    grid_latitude: float
    grid_longitude: float


@dataclass(frozen=True)
class _LinearGroup:
    months: tuple[int, ...]
    variable: str
    records: tuple[ObservationRecord, ...]
    state_slice: slice
    weights: torch.Tensor
    bias: torch.Tensor


class EOFPointDecoder:
    """Construct full-state affine transforms for local climate values."""

    def __init__(
        self,
        models: dict[str, EOFModel],
        layout: StateLayout,
        *,
        spatial_policy: str = "coordinate",
    ) -> None:
        missing = set(layout.variable_slices) - set(models)
        if missing:
            raise ValueError(f"EOF models missing state variables: {sorted(missing)}")
        self.models = models
        self.layout = layout
        if spatial_policy not in {"coordinate", "great_circle"}:
            raise ValueError("unsupported EOF point selection policy")
        self.spatial_policy = spatial_policy
        self._transform_cache: dict[tuple[str, float, float, str], PointTransform] = {}

    def transform(self, variable: str, latitude: float, longitude: float) -> PointTransform:
        key = (variable, float(latitude), float(longitude) % 360.0, self.spatial_policy)
        cached = self._transform_cache.get(key)
        if cached is not None:
            return cached
        model = self.models[variable]
        variable_slice = self.layout.slice_for(variable)
        count = variable_slice.stop - variable_slice.start
        if count > model.components.sizes["mode"]:
            raise ValueError(f"EOF model for {variable} has fewer modes than the state layout")
        longitude = longitude % 360.0
        if self.spatial_policy == "great_circle":
            lat = np.deg2rad(model.valid_mask.lat.values)[:, None]
            lon = np.deg2rad(model.valid_mask.lon.values)[None, :]
            site_lat, site_lon = np.deg2rad([latitude, longitude])
            distance = (
                np.sin((lat - site_lat) / 2) ** 2
                + np.cos(lat) * np.cos(site_lat) * np.sin((lon - site_lon) / 2) ** 2
            )
            i, j = np.unravel_index(np.argmin(distance), distance.shape)
            selected_mask = model.valid_mask.isel(lat=i, lon=j)
        else:
            selected_mask = model.valid_mask.sel(lat=latitude, lon=longitude, method="nearest")
        if not bool(selected_mask.item()):
            raise ValueError(f"nearest {variable} EOF grid point is outside the valid mask")
        point = {"lat": selected_mask.lat.item(), "lon": selected_mask.lon.item()}
        component = model.components.isel(mode=slice(0, count)).sel(point)
        weight = model.weights.sel(point)
        mean = model.weighted_mean.sel(point)
        local = component.values * model.pc_scale.values[:count] / float(weight.item())
        full = np.zeros(self.layout.size, dtype=float)
        full[variable_slice] = local
        transform = PointTransform(
            weights=full,
            bias=float(mean.item() / weight.item()),
            grid_latitude=float(component.lat.item()),
            grid_longitude=float(component.lon.item()),
        )
        self._transform_cache[key] = transform
        return transform


def annualize_quarterly(
    values: torch.Tensor,
    time: torch.Tensor,
    months: list[int] | tuple[int, ...] = ANNUAL_MONTHS,
    *,
    edge_policy: str = "legacy_repeat",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average selected monthly seasons from an ordered quarterly trajectory."""
    if values.ndim not in (1, 2) or time.ndim != 1 or values.shape[0] != time.numel():
        raise ValueError("quarterly values must be 1D/2D and align with 1D time")
    if values.shape[0] < 4 or values.shape[0] % 4:
        raise ValueError("quarterly trajectory must contain complete four-step years")
    try:
        selected = sorted({_MONTH_TO_QUARTER[month] for month in months})
    except KeyError as exc:
        raise ValueError(f"unsupported seasonality month: {exc.args[0]}") from exc
    if not selected:
        raise ValueError("seasonality must select at least one quarter")
    n_years = values.shape[0] // 4
    if edge_policy not in ("drop", "legacy_repeat"):
        raise ValueError("edge_policy must be 'drop' or 'legacy_repeat'")
    quarterly = values.reshape(n_years, 4, *values.shape[1:])
    first_target = 1 if edge_policy == "drop" and any(offset < 0 for offset, _ in selected) else 0
    target_year = torch.arange(first_target, n_years, device=values.device)
    annual = torch.stack(
        [quarterly[(target_year + offset).clamp_min(0), quarter] for offset, quarter in selected],
        dim=0,
    ).mean(dim=0)
    years = torch.floor(time[::4]).to(torch.int64)
    return years[target_year], annual


class LinearProxyOperator:
    """Apply EOF point transforms and linear proxy calibrations."""

    def __init__(
        self,
        decoder: EOFPointDecoder,
        *,
        time_tolerance: float = 1e-6,
        edge_policy: str = "legacy_repeat",
    ) -> None:
        self.decoder = decoder
        self.time_tolerance = time_tolerance
        self.edge_policy = edge_policy
        self._window_model_time: dict[tuple[int, ...], torch.Tensor] = {}
        self._window_indices: dict[tuple[tuple[int, ...], str, str], WindowIndex] = {}
        self._group_cache_key: tuple[int, torch.dtype, torch.device] | None = None
        self._group_collection: ObservationCollection | None = None
        self._groups: tuple[_LinearGroup, ...] = ()
        self._record_group: dict[str, tuple[int, int]] = {}
        self._observation_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._exact_matches: dict[tuple[tuple[int, ...], str], tuple[torch.Tensor, np.ndarray]] = {}

    def predict(
        self,
        trajectory: torch.Tensor,
        model_time: torch.Tensor,
        observations: ObservationCollection,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        if trajectory.ndim != 2 or trajectory.shape[1] != self.decoder.layout.size:
            raise ValueError("trajectory must have shape (time, state_layout.size)")
        groups = self._prepare_groups(observations, trajectory)
        predictions = []
        observed = []
        variances = []
        families = []
        annualized: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        local_by_group: list[torch.Tensor] = []
        for group in groups:
            if group.months not in annualized:
                annualized[group.months] = annualize_quarterly(
                    trajectory, model_time, group.months, edge_policy=self.edge_policy
                )
                comparison_time = annualized[group.months][0].to(dtype=trajectory.dtype)
                self._prepare_window_time(group.months, comparison_time)
            annual_state = annualized[group.months][1][:, group.state_slice]
            local_by_group.append(annual_state @ group.weights.T + group.bias)
        for record in observations:
            if isinstance(record, BoreholeRecord):
                raise TypeError("BoreholeRecord requires BoreholeOperator")
            months = tuple(record.metadata.get("seasonality", ANNUAL_MONTHS))
            group_index, record_index = self._record_group[record.record_id]
            prediction, indices = self._predict_record(
                annualized[months][0],
                local_by_group[group_index][:, record_index],
                record,
                months=months,
            )
            if indices.size == 0:
                continue
            predictions.append(prediction)
            values, variance = self._observation_tensors[record.record_id]
            observed.append(values[indices])
            variances.append(variance[indices])
            families.extend([record.family.value] * indices.size)
        if not predictions:
            empty = trajectory.new_empty((0,))
            return empty, empty, empty, []
        return (
            torch.cat(predictions),
            torch.cat(observed),
            torch.cat(variances),
            families,
        )

    def _predict_record(
        self,
        annual_time: torch.Tensor,
        local: torch.Tensor,
        record: ObservationRecord,
        *,
        months: tuple[int, ...],
    ) -> tuple[torch.Tensor, np.ndarray]:
        reference_period = record.metadata.get("anomaly_reference_period")
        if reference_period is not None:
            start, end = sorted(float(value) for value in reference_period)
            reference = (annual_time >= start) & (annual_time <= end)
            if not torch.any(reference):
                raise ValueError(
                    f"{record.record_id} has no model values in anomaly reference period "
                    f"{reference_period}"
                )
            local = local - local[reference].mean().detach()
        local = local * float(record.metadata.get("slope", 1.0)) + float(
            record.metadata.get("intercept", 0.0)
        )
        comparison_time = annual_time.to(dtype=local.dtype)
        observation_time = torch.as_tensor(
            record.time, dtype=comparison_time.dtype, device=comparison_time.device
        )
        if record.time_bounds is not None:
            bounds = torch.as_tensor(
                record.time_bounds, dtype=comparison_time.dtype, device=comparison_time.device
            )
            index = self._window_index(
                months,
                record,
                comparison_time,
                observation_time,
                bounds=bounds,
            )
            prediction, indices, _ = average_indexed_windows(local, index)
            return prediction, indices.detach().cpu().numpy()
        if "resolution_years" in record.metadata:
            index = self._window_index(
                months,
                record,
                comparison_time,
                observation_time,
                widths=float(record.metadata["resolution_years"]),
            )
            prediction, indices, _ = average_indexed_windows(local, index)
            return prediction, indices.detach().cpu().numpy()
        key = (months, record.record_id)
        cached = self._exact_matches.get(key)
        if cached is None:
            match = match_nearest_times(
                annual_time.detach().cpu().numpy(), record.time, tolerance=self.time_tolerance
            )
            model_indices = torch.as_tensor(
                match.model_indices, dtype=torch.long, device=local.device
            )
            cached = (model_indices, match.observation_indices)
            self._exact_matches[key] = cached
        return local[cached[0]], cached[1]

    def _prepare_groups(
        self, observations: ObservationCollection, trajectory: torch.Tensor
    ) -> tuple[_LinearGroup, ...]:
        cache_key = (id(observations), trajectory.dtype, trajectory.device)
        if observations is self._group_collection and cache_key == self._group_cache_key:
            return self._groups
        grouped: dict[tuple[tuple[int, ...], str], list[ObservationRecord]] = {}
        for record in observations:
            if isinstance(record, BoreholeRecord):
                raise TypeError("BoreholeRecord requires BoreholeOperator")
            months = tuple(record.metadata.get("seasonality", ANNUAL_MONTHS))
            grouped.setdefault((months, record.variable), []).append(record)
        groups = []
        record_group = {}
        observation_tensors = {}
        for group_index, ((months, variable), records) in enumerate(grouped.items()):
            state_slice = self.decoder.layout.slice_for(variable)
            transforms = [
                self.decoder.transform(variable, record.latitude, record.longitude)
                for record in records
            ]
            weights = torch.as_tensor(
                np.stack([transform.weights[state_slice] for transform in transforms]),
                dtype=trajectory.dtype,
                device=trajectory.device,
            )
            bias = torch.as_tensor(
                [transform.bias for transform in transforms],
                dtype=trajectory.dtype,
                device=trajectory.device,
            )
            groups.append(
                _LinearGroup(months, variable, tuple(records), state_slice, weights, bias)
            )
            for record_index, record in enumerate(records):
                record_group[record.record_id] = (group_index, record_index)
                observation_tensors[record.record_id] = (
                    torch.as_tensor(record.value, dtype=trajectory.dtype, device=trajectory.device),
                    torch.as_tensor(
                        record.variance, dtype=trajectory.dtype, device=trajectory.device
                    ),
                )
        self._group_cache_key = cache_key
        self._group_collection = observations
        self._groups = tuple(groups)
        self._record_group = record_group
        self._observation_tensors = observation_tensors
        self._exact_matches.clear()
        self._window_indices.clear()
        return self._groups

    def _prepare_window_time(self, months: tuple[int, ...], model_time: torch.Tensor) -> None:
        cached = self._window_model_time.get(months)
        unchanged = (
            cached is not None
            and cached.dtype == model_time.dtype
            and cached.device == model_time.device
            and torch.equal(cached, model_time)
        )
        if unchanged:
            return
        self._window_model_time[months] = model_time.detach().clone()
        self._window_indices = {
            key: value for key, value in self._window_indices.items() if key[0] != months
        }
        self._exact_matches = {
            key: value for key, value in self._exact_matches.items() if key[0] != months
        }

    def _window_index(
        self,
        months: tuple[int, ...],
        record: ObservationRecord,
        model_time: torch.Tensor,
        observation_time: torch.Tensor,
        *,
        widths: float | None = None,
        bounds: torch.Tensor | None = None,
    ) -> WindowIndex:
        kind = "bounds" if bounds is not None else f"width:{widths}"
        key = (months, record.record_id, kind)
        index = self._window_indices.get(key)
        if index is None:
            index = build_window_index(
                model_time,
                observation_time,
                widths=widths,
                bounds=bounds,
            )
            self._window_indices[key] = index
        return index
