"""Validated, dependency-free observation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ObservationFamily(str, Enum):
    PAGES2K = "pages2k"
    SEDPSM = "sedpsm"
    BOREHOLE = "borehole"
    TEMP12K = "temp12k"


@dataclass(frozen=True)
class ObservationRecord:
    """One time-indexed proxy series with variance-valued uncertainty."""

    record_id: str
    family: ObservationFamily
    variable: str
    latitude: float
    longitude: float
    time: np.ndarray
    value: np.ndarray
    variance: np.ndarray | float
    time_bounds: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id or not self.variable:
            raise ValueError("record_id and variable must not be empty")
        if not np.isfinite(self.latitude) or not -90 <= self.latitude <= 90:
            raise ValueError(f"invalid observation latitude: {self.latitude}")
        if not np.isfinite(self.longitude):
            raise ValueError(f"invalid observation longitude: {self.longitude}")
        time = np.asarray(self.time, dtype=float)
        value = np.asarray(self.value, dtype=float)
        if time.ndim != 1 or value.ndim != 1 or time.shape != value.shape:
            raise ValueError("observation time and value must be equal-length 1D arrays")
        if time.size == 0 or not np.isfinite(time).all():
            raise ValueError("observation times must be nonempty and finite")
        variance = np.asarray(self.variance, dtype=float)
        if variance.ndim == 0:
            variance = np.full(time.shape, float(variance))
        if variance.shape != time.shape:
            raise ValueError("observation variance must be scalar or match time")
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            raise ValueError("observation variance must be finite and positive")
        bounds = None
        if self.time_bounds is not None:
            bounds = np.asarray(self.time_bounds, dtype=float)
            if bounds.shape != (time.size, 2) or not np.isfinite(bounds).all():
                raise ValueError("time_bounds must be a finite (observation, 2) array")
            bounds = np.sort(bounds, axis=1)
        object.__setattr__(self, "family", ObservationFamily(self.family))
        object.__setattr__(self, "longitude", float(self.longitude) % 360.0)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "time_bounds", bounds)

    @property
    def size(self) -> int:
        return self.time.size

    @property
    def valid_value_mask(self) -> np.ndarray:
        return np.isfinite(self.value)


@dataclass(frozen=True)
class BoreholeRecord:
    """One depth-indexed borehole temperature-anomaly profile."""

    record_id: str
    latitude: float
    longitude: float
    logging_year: float
    depth: np.ndarray
    value: np.ndarray
    variance: np.ndarray | float
    variable: str = "tas"
    family: ObservationFamily = ObservationFamily.BOREHOLE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id or self.variable != "tas":
            raise ValueError("borehole record_id must be nonempty and variable must be tas")
        if ObservationFamily(self.family) is not ObservationFamily.BOREHOLE:
            raise ValueError("BoreholeRecord family must be borehole")
        if not np.isfinite(self.latitude) or not -90 <= self.latitude <= 90:
            raise ValueError(f"invalid observation latitude: {self.latitude}")
        if not np.isfinite(self.longitude) or not np.isfinite(self.logging_year):
            raise ValueError("borehole longitude and logging_year must be finite")
        depth = np.asarray(self.depth, dtype=float)
        value = np.asarray(self.value, dtype=float)
        if depth.ndim != 1 or value.ndim != 1 or depth.shape != value.shape:
            raise ValueError("borehole depth and value must be equal-length 1D arrays")
        if depth.size == 0 or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError("borehole depths must be nonempty, finite, and nonnegative")
        if np.any(np.diff(depth) <= 0):
            raise ValueError("borehole depths must be strictly increasing")
        variance = np.asarray(self.variance, dtype=float)
        if variance.ndim == 0:
            variance = np.full(depth.shape, float(variance))
        if variance.shape != depth.shape:
            raise ValueError("borehole variance must be scalar or match depth")
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            raise ValueError("borehole variance must be finite and positive")
        object.__setattr__(self, "family", ObservationFamily.BOREHOLE)
        object.__setattr__(self, "longitude", float(self.longitude) % 360.0)
        object.__setattr__(self, "logging_year", float(self.logging_year))
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "variance", variance)

    @property
    def size(self) -> int:
        return self.depth.size

    @property
    def valid_value_mask(self) -> np.ndarray:
        return np.isfinite(self.value)


Observation = ObservationRecord | BoreholeRecord


class ObservationCollection:
    """Collection enforcing globally unique observation identifiers."""

    def __init__(self, records: list[Observation]) -> None:
        ids = [record.record_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("observation record_id values must be unique")
        self._records = tuple(records)

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def observation_count(self) -> int:
        return sum(record.size for record in self._records)

    def by_family(self, family: ObservationFamily | str) -> ObservationCollection:
        family = ObservationFamily(family)
        return ObservationCollection([record for record in self if record.family is family])

    def counts_by_family(self) -> dict[str, int]:
        return {
            family.value: sum(record.size for record in self if record.family is family)
            for family in ObservationFamily
        }
