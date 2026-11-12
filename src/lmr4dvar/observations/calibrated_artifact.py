"""Portable ragged NetCDF artifact for calibrated proxy observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from .records import ObservationCollection, ObservationRecord


def save_calibrated_observations(path: str | Path, observations: ObservationCollection) -> Path:
    records = list(observations)
    if not records or any(not isinstance(record, ObservationRecord) for record in records):
        raise ValueError("calibrated artifact requires nonempty time-indexed proxy records")
    dataset = xr.Dataset(
        {
            "proxy_time": ("sample", np.concatenate([r.time for r in records])),
            "proxy_value": ("sample", np.concatenate([r.value for r in records])),
            "proxy_variance": ("sample", np.concatenate([r.variance for r in records])),
            "record_index": (
                "sample",
                np.repeat(np.arange(len(records)), [r.size for r in records]),
            ),
            "time_bounds": (
                ("sample", "bound"),
                np.concatenate(
                    [
                        r.time_bounds if r.time_bounds is not None else np.full((r.size, 2), np.nan)
                        for r in records
                    ]
                ),
            ),
            "has_time_bounds": ("record", [int(r.time_bounds is not None) for r in records]),
            "record_id": ("record", [r.record_id for r in records]),
            "family": ("record", [r.family.value for r in records]),
            "variable": ("record", [r.variable for r in records]),
            "latitude": ("record", [r.latitude for r in records]),
            "longitude": ("record", [r.longitude for r in records]),
            "metadata_json": ("record", [json.dumps(r.metadata, allow_nan=False) for r in records]),
        },
        attrs={"artifact_kind": "calibrated_proxy_observations", "schema_version": 1},
    )
    dataset.proxy_variance.attrs["description"] = "variance, not standard deviation"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path, engine="h5netcdf")
    return path


def load_calibrated_observations(path: str | Path) -> ObservationCollection:
    with xr.open_dataset(path, engine="h5netcdf") as source:
        if source.attrs.get("artifact_kind") != "calibrated_proxy_observations":
            raise ValueError("not a calibrated proxy observation artifact")
        if source.attrs.get("schema_version") != 1:
            raise ValueError("unsupported calibrated proxy artifact schema")
        dataset = source.load()
    record_indices = dataset.record_index.values
    count = dataset.sizes["record"]
    if not np.isin(record_indices, np.arange(count)).all():
        raise ValueError("artifact sample references an invalid record index")
    records = []
    for i in range(count):
        selected = record_indices == i
        records.append(
            ObservationRecord(
                record_id=str(dataset.record_id.values[i]),
                family=str(dataset.family.values[i]),
                variable=str(dataset.variable.values[i]),
                latitude=float(dataset.latitude.values[i]),
                longitude=float(dataset.longitude.values[i]),
                time=dataset.proxy_time.values[selected],
                value=dataset.proxy_value.values[selected],
                variance=dataset.proxy_variance.values[selected],
                time_bounds=dataset.time_bounds.values[selected]
                if dataset.has_time_bounds.values[i]
                else None,
                metadata=json.loads(str(dataset.metadata_json.values[i])),
            )
        )
    return ObservationCollection(records)
