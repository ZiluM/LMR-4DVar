"""Adapters from CFR proxy databases to standalone observation records."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .records import ObservationCollection, ObservationFamily, ObservationRecord


@dataclass(frozen=True)
class SkippedObservation:
    record_id: str
    reason: str


@dataclass(frozen=True)
class ObservationLoadResult:
    observations: ObservationCollection
    skipped: tuple[SkippedObservation, ...]


def load_cfr_observations(
    path: str | Path,
    *,
    family: ObservationFamily | str,
    calibrated_only: bool = True,
    strict: bool = True,
) -> ObservationLoadResult:
    """Load a CFR ReconJob directory or pickled ProxyDatabase."""
    family = ObservationFamily(family)
    if family not in (ObservationFamily.PAGES2K, ObservationFamily.TEMP12K):
        raise ValueError("CFR adapter supports pages2k and temp12k families")
    database = _load_database(Path(path))
    if calibrated_only and hasattr(database, "filter"):
        database = database.filter(by="tag", keys=["calibrated"])
    records = []
    skipped = []
    for record_id, proxy in database.records.items():
        try:
            records.append(cfr_proxy_to_observation(str(record_id), proxy, family=family))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            skipped.append(SkippedObservation(str(record_id), str(exc)))
    if strict and skipped:
        examples = "; ".join(f"{item.record_id}: {item.reason}" for item in skipped[:3])
        raise ValueError(
            f"failed to convert {len(skipped)} of {len(records) + len(skipped)} CFR records; "
            f"examples: {examples}"
        )
    return ObservationLoadResult(ObservationCollection(records), tuple(skipped))


def cfr_proxy_to_observation(
    record_id: str,
    proxy: Any,
    *,
    family: ObservationFamily | str,
) -> ObservationRecord:
    """Convert one calibrated CFR ProxyRecord by public/serialized attributes."""
    family = ObservationFamily(family)
    metadata = {
        "proxy_type": str(proxy.ptype),
        "tags": sorted(str(tag) for tag in getattr(proxy, "tags", set())),
    }
    if family is ObservationFamily.TEMP12K:
        variable = "tas"
        metadata.update(
            {
                "slope": 1.0,
                "intercept": 0.0,
                "anomaly_reference_period": [0.0, 2000.0],
            }
        )
        seasonality = getattr(proxy, "seasonality", None)
    elif family is ObservationFamily.PAGES2K:
        details = proxy.psm.calib_details
        columns = list(details["df"].columns)
        if len(columns) < 2:
            raise ValueError("PSM calibration dataframe has no climate predictor")
        if len(columns) > 2:
            raise ValueError(
                "multi-predictor CFR PSM requires an explicit additive observation artifact"
            )
        variable = str(columns[1])
        parameters = proxy.psm.model.params
        metadata.update(
            {
                "slope": float(parameters[variable]),
                "intercept": float(parameters["Intercept"]),
            }
        )
        seasonality = details.get("seasonality")
    else:
        raise ValueError("CFR proxy conversion supports pages2k and temp12k")
    if seasonality is not None:
        metadata["seasonality"] = [int(month) for month in seasonality]
    if "low_res" in getattr(proxy, "tags", set()) or family is ObservationFamily.TEMP12K:
        resolution = float(proxy.dt)
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("low-resolution proxy dt must be finite and positive")
        metadata["resolution_years"] = resolution
    return ObservationRecord(
        record_id=record_id,
        family=family,
        variable=variable,
        latitude=float(proxy.lat),
        longitude=float(proxy.lon),
        time=np.asarray(proxy.time, dtype=float),
        value=np.asarray(proxy.value, dtype=float),
        variance=np.asarray(proxy.R, dtype=float),
        time_bounds=getattr(proxy, "time_bnds_ce", None),
        metadata=metadata,
    )


def _load_database(path: Path):
    try:
        import cfr
    except ImportError as exc:
        raise RuntimeError(
            "cfr==2024.4.4 and its runtime dependencies are required to load CFR artifacts"
        ) from exc
    if path.is_dir():
        job = cfr.ReconJob()
        job.load(str(path))
        return job.proxydb
    with path.open("rb") as stream:
        return pickle.load(stream)
