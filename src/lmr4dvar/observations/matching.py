"""Explicit time matching and differentiable temporal averaging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TimeMatch:
    model_indices: np.ndarray
    observation_indices: np.ndarray
    distance: np.ndarray
    unmatched_observation_indices: np.ndarray

    @property
    def count(self) -> int:
        return self.observation_indices.size


@dataclass(frozen=True)
class WindowIndex:
    """Precomputed half-open model index ranges for observation windows."""

    left: torch.Tensor
    right: torch.Tensor
    observation_indices: torch.Tensor
    counts: torch.Tensor

    @property
    def count(self) -> int:
        return self.observation_indices.numel()


def match_nearest_times(
    model_time: np.ndarray, observation_time: np.ndarray, *, tolerance: float = 1e-6
) -> TimeMatch:
    """Match each observation to its nearest model time within a tolerance."""
    model = _finite_1d(model_time, "model_time")
    observations = _finite_1d(observation_time, "observation_time")
    if tolerance < 0:
        raise ValueError("time matching tolerance must be non-negative")
    if model.size == 0:
        return TimeMatch(
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.arange(observations.size),
        )
    distances = np.abs(observations[:, None] - model[None, :])
    model_indices = np.argmin(distances, axis=1)
    nearest = distances[np.arange(observations.size), model_indices]
    matched = nearest <= tolerance
    return TimeMatch(
        model_indices[matched],
        np.flatnonzero(matched),
        nearest[matched],
        np.flatnonzero(~matched),
    )


def average_to_windows(
    values: torch.Tensor,
    model_time: torch.Tensor,
    observation_time: torch.Tensor,
    *,
    widths: torch.Tensor | float | None = None,
    bounds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average model values using indexed prefix sums with differentiable output."""
    if values.ndim not in (1, 2) or model_time.ndim != 1 or observation_time.ndim != 1:
        raise ValueError("values must be 1D/2D and time arrays must be 1D")
    if values.shape[0] != model_time.numel():
        raise ValueError("values and model_time lengths differ")
    index = build_window_index(
        model_time,
        observation_time,
        widths=widths,
        bounds=bounds,
    )
    return average_indexed_windows(values, index)


def build_window_index(
    model_time: torch.Tensor,
    observation_time: torch.Tensor,
    *,
    widths: torch.Tensor | float | None = None,
    bounds: torch.Tensor | None = None,
) -> WindowIndex:
    """Locate inclusive observation windows on a sorted model time axis."""
    if model_time.ndim != 1 or observation_time.ndim != 1:
        raise ValueError("model_time and observation_time must be 1D")
    if not torch.isfinite(model_time).all():
        raise ValueError("model_time must be finite")
    if model_time.numel() > 1 and torch.any(model_time[1:] < model_time[:-1]):
        raise ValueError("model_time must be sorted in increasing order")
    if bounds is not None and widths is not None:
        raise ValueError("specify either bounds or widths, not both")
    if bounds is not None:
        bounds = torch.as_tensor(
            bounds, dtype=model_time.dtype, device=model_time.device
        )
        if bounds.shape != (observation_time.numel(), 2):
            raise ValueError("bounds must have shape (observation, 2)")
        lower = torch.minimum(bounds[:, 0], bounds[:, 1])
        upper = torch.maximum(bounds[:, 0], bounds[:, 1])
    else:
        observation_time = torch.as_tensor(
            observation_time, dtype=model_time.dtype, device=model_time.device
        )
        widths = torch.as_tensor(
            0.0 if widths is None else widths,
            dtype=model_time.dtype,
            device=model_time.device,
        )
        if torch.any(widths < 0):
            raise ValueError("window widths must be non-negative")
        if widths.ndim == 0:
            widths = widths.expand(observation_time.numel())
        if widths.shape != observation_time.shape:
            raise ValueError("widths must be scalar or match observation_time")
        lower = observation_time - widths / 2
        upper = observation_time + widths / 2
    finite = torch.isfinite(lower) & torch.isfinite(upper)
    search_time = model_time.contiguous()
    safe_lower = torch.where(finite, lower, search_time.new_zeros(()))
    safe_upper = torch.where(finite, upper, search_time.new_full((), -torch.inf))
    left = torch.searchsorted(search_time, safe_lower.contiguous(), right=False)
    right = torch.searchsorted(search_time, safe_upper.contiguous(), right=True)
    counts = right - left
    valid = finite & (counts > 0)
    return WindowIndex(
        left=left[valid],
        right=right[valid],
        observation_indices=torch.nonzero(valid, as_tuple=True)[0],
        counts=counts[valid],
    )


def average_indexed_windows(
    values: torch.Tensor,
    index: WindowIndex,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average dynamic values over precomputed half-open index ranges."""
    if values.ndim not in (1, 2):
        raise ValueError("values must be 1D or 2D")
    if index.left.device != values.device:
        raise ValueError("window index and values must be on the same device")
    if index.count and int(index.right.max()) > values.shape[0]:
        raise ValueError("window index exceeds values time dimension")
    zero = torch.zeros_like(values[:1])
    prefix = torch.cat([zero, torch.cumsum(values, dim=0)], dim=0)
    totals = prefix[index.right] - prefix[index.left]
    divisor = index.counts.to(dtype=values.dtype)
    if values.ndim == 2:
        divisor = divisor[:, None]
    return totals / divisor, index.observation_indices, index.counts


def _finite_1d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 1D array")
    return array
