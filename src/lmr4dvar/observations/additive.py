"""Differentiable additive PSMs with independent complete-quarter seasons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .linear import EOFPointDecoder
from .matching import WindowIndex, average_indexed_windows, build_window_index
from .quarterly_seasons import complete_quarter_offsets, quarterly_season_mean
from .records import ObservationRecord


@dataclass(frozen=True)
class PredictorTerm:
    variable: str
    slope: float
    seasonality: tuple[int, ...]
    units: str
    reference_offset: float = 0.0

    def __post_init__(self):
        if (
            not self.variable
            or not self.units
            or not np.isfinite([self.slope, self.reference_offset]).all()
        ):
            raise ValueError("predictor requires variable, units and a finite slope")
        complete_quarter_offsets(self.seasonality)


def predictor_terms(record: ObservationRecord) -> tuple[PredictorTerm, ...]:
    if record.metadata.get("psm_kind") != "additive_linear_v1":
        raise ValueError(f"{record.record_id}: unsupported additive PSM contract")
    terms = tuple(
        PredictorTerm(
            variable=item["variable"],
            slope=float(item["slope"]),
            seasonality=tuple(item["seasonality"]),
            units=item["units"],
            reference_offset=float(item.get("anomaly_reference_offset", 0.0)),
        )
        for item in record.metadata["predictors"]
    )
    if not terms or len({term.variable for term in terms}) != len(terms):
        raise ValueError("additive predictors must be nonempty and have unique variables")
    if not np.isfinite(float(record.metadata["intercept"])):
        raise ValueError("additive intercept must be finite")
    return terms


class AdditiveProxyOperator:
    """Sum independently seasonalized predictors; preserve all state gradients.

    A reference-period normalization removes each predictor's model mean
    differentiably. It is explicit in metadata and is not a detached offset.
    Static seasonal gathers, point transforms and time-window indices are cached.
    """

    def __init__(self, decoder: EOFPointDecoder) -> None:
        if getattr(decoder, "spatial_policy", "great_circle") != "great_circle":
            raise ValueError("additive PSM requires great-circle EOF point selection")
        self.decoder = decoder
        self._cache_key = None
        self._prepared = ()
        self._seasons = {}

    def _prepare(self, trajectory, model_time, observations):
        key = (
            id(observations),
            id(model_time),
            model_time._version,
            tuple(model_time.shape),
            trajectory.dtype,
            trajectory.device,
        )
        if key == self._cache_key:
            return
        prepared = []
        seasons = {}
        for record in observations:
            if not isinstance(record, ObservationRecord):
                raise TypeError("additive PSM requires time-indexed proxy records")
            terms = predictor_terms(record)
            term_data = []
            for term in terms:
                if term.seasonality not in seasons:
                    years, _ = quarterly_season_mean(model_time, model_time, term.seasonality)
                    offsets = complete_quarter_offsets(term.seasonality)
                    target = years - torch.floor(model_time[0]).to(torch.int64)
                    indices = torch.stack(
                        [4 * (target + offset) + quarter for offset, quarter in offsets], dim=1
                    )
                    seasons[term.seasonality] = (years, indices)
                years, _ = seasons[term.seasonality]
                if not years.numel():
                    raise ValueError(f"{record.record_id}: no complete seasonal model years")
                transform = self.decoder.transform(term.variable, record.latitude, record.longitude)
                state_slice = self.decoder.layout.slice_for(term.variable)
                weights = torch.as_tensor(
                    transform.weights[state_slice], dtype=trajectory.dtype, device=trajectory.device
                )
                reference = record.metadata.get("anomaly_reference_period")
                reference_mask = None
                if reference is not None:
                    start, end = reference
                    if start > end or start != int(start) or end != int(end):
                        raise ValueError(
                            "additive anomaly reference period must be increasing integer years"
                        )
                    reference_mask = (years >= start) & (years <= end)
                    if int(reference_mask.sum()) != end - start + 1:
                        raise ValueError(f"{record.record_id}: incomplete anomaly reference period")
                term_data.append((term, state_slice, weights, transform.bias, reference_mask))
            first = max(int(seasons[t.seasonality][0][0]) for t in terms)
            last = min(int(seasons[t.seasonality][0][-1]) for t in terms)
            if first > last:
                raise ValueError(f"{record.record_id}: no common complete predictor years")
            common_years = torch.arange(
                first, last + 1, device=trajectory.device, dtype=trajectory.dtype
            )
            obs_time = torch.as_tensor(
                record.time, device=trajectory.device, dtype=trajectory.dtype
            )
            bounds = (
                None
                if record.time_bounds is None
                else torch.as_tensor(
                    record.time_bounds, device=trajectory.device, dtype=trajectory.dtype
                )
            )
            index = build_window_index(
                common_years,
                obs_time,
                bounds=bounds,
                widths=None if bounds is not None else record.metadata.get("resolution_years", 0.0),
            )
            if bounds is not None or "resolution_years" in record.metadata:
                if bounds is not None:
                    lower, upper = bounds[:, 0], bounds[:, 1]
                else:
                    width = float(record.metadata["resolution_years"])
                    lower, upper = obs_time - width / 2, obs_time + width / 2
                complete = (torch.ceil(lower[index.observation_indices]) >= first) & (
                    torch.floor(upper[index.observation_indices]) <= last
                )
                index = WindowIndex(
                    index.left[complete],
                    index.right[complete],
                    index.observation_indices[complete],
                    index.counts[complete],
                )
            values = torch.as_tensor(record.value, device=trajectory.device, dtype=trajectory.dtype)
            variance = torch.as_tensor(
                record.variance, device=trajectory.device, dtype=trajectory.dtype
            )
            slices = tuple(
                slice(
                    first - int(seasons[t.seasonality][0][0]),
                    last - int(seasons[t.seasonality][0][0]) + 1,
                )
                for t in terms
            )
            prepared.append(
                (record, tuple(term_data), slices, common_years.numel(), index, values, variance)
            )
        self._prepared = tuple(prepared)
        self._seasons = seasons
        self._cache_key = key
        self._observations = observations
        self._model_time = model_time

    def predict(self, trajectory, model_time, observations):
        if trajectory.ndim != 2 or trajectory.shape[1] != self.decoder.layout.size:
            raise ValueError("trajectory must have shape (time, state_layout.size)")
        if model_time.ndim != 1 or model_time.numel() != trajectory.shape[0]:
            raise ValueError("trajectory and model time must align")
        if model_time.device != trajectory.device:
            raise ValueError("model time and trajectory must be on the same device")
        self._prepare(trajectory, model_time, observations)
        seasonal = {
            months: trajectory[indices].mean(dim=1)
            for months, (_, indices) in self._seasons.items()
        }
        predictions, observed, variances, families = [], [], [], []
        for record, terms, slices, count, index, values, variance in self._prepared:
            prediction = trajectory.new_full((count,), float(record.metadata["intercept"]))
            for (term, state_slice, weights, bias, reference), selected in zip(
                terms, slices, strict=True
            ):
                local = seasonal[term.seasonality][:, state_slice] @ weights + bias
                if reference is not None:
                    local = local - local[reference].mean() + term.reference_offset
                prediction = prediction + term.slope * local[selected]
            if record.time_bounds is None and "resolution_years" not in record.metadata:
                prediction, indices = prediction[index.left], index.observation_indices
            else:
                prediction, indices, _ = average_indexed_windows(prediction, index)
            if not indices.numel():
                continue
            predictions.append(prediction)
            observed.append(values[indices])
            variances.append(variance[indices])
            families.extend([record.family.value] * indices.numel())
        if not predictions:
            raise ValueError("additive PSM has no observations matched to complete seasonal years")
        return torch.cat(predictions), torch.cat(observed), torch.cat(variances), families
