"""Build and deduplicate direct-temperature Temp12k observations."""

from __future__ import annotations

import copy
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lmr4dvar.config.models import Temp12kPreparationConfig

ANNUAL_SEASON = [-12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
SEASON_REMAP = {
    tuple(range(1, 13)): ANNUAL_SEASON,
    (7,): [6, 7, 8],
    (1,): [-12, 1, 2],
}
PAIR_COLUMNS = [
    "temp12k_id",
    "pages2k_id",
    "distance_km",
    "n_overlap_points",
    "corr",
    "abs_corr",
    "temp12k_ptype",
    "pages2k_ptype",
    "temp12k_unit",
    "pages2k_unit",
    "temp12k_lat",
    "temp12k_lon",
    "pages2k_lat",
    "pages2k_lon",
    "is_duplicate_by_rule",
    "overlap_year_min",
    "overlap_year_max",
]
REMOVAL_COLUMNS = [
    "temp12k_id",
    "points_before",
    "points_removed",
    "points_after",
    "years_removed_min",
    "years_removed_max",
    "n_years_removed",
    "dropped_record_for_short_length",
]


def prepare_temp12k(
    config: Temp12kPreparationConfig,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Run the legacy four-script Temp12k chain as one reproducible operation."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfr = _import_cfr()
    summary: Counter[str] = Counter()
    if config.source_format == "raw_temp12k":
        database = build_temp12k_database(config.source_path, cfr, summary)
    else:
        database = copy.deepcopy(_load_database(config.source_path, cfr))
        summary["source_cfr_records"] = len(database.records)

    source_count = len(database.records)
    dropped_baseline = apply_anomaly_reference(
        database,
        config.anomaly_period,
        require_baseline=config.require_anomaly_baseline,
    )
    seasonality_changed = normalize_seasonality(database)
    pages2k = _load_pages2k(config, cfr)
    pages_metadata = _database_metadata(
        pages2k, config.comparison_period, anomaly_period=config.anomaly_period
    )
    temp_metadata = _database_metadata(database, config.comparison_period)
    before, years_to_remove = find_candidate_pairs(
        temp_metadata,
        pages_metadata,
        maximum_distance_km=config.maximum_distance_km,
        duplicate_correlation_minimum=config.duplicate_correlation_minimum,
        duplicate_minimum_overlap=config.duplicate_minimum_overlap,
    )
    removals = remove_duplicate_years(
        database, years_to_remove, minimum_points=config.minimum_record_points
    )
    after, remaining = find_candidate_pairs(
        _database_metadata(database, config.comparison_period),
        pages_metadata,
        maximum_distance_km=config.maximum_distance_km,
        duplicate_correlation_minimum=config.duplicate_correlation_minimum,
        duplicate_minimum_overlap=config.duplicate_minimum_overlap,
    )

    database_path = directory / "temp12k.pkl"
    with database_path.open("wb") as stream:
        pickle.dump(database, stream, protocol=pickle.HIGHEST_PROTOCOL)
    build_diagnostics_path = directory / "build_diagnostics.csv"
    pd.DataFrame(
        sorted(summary.items()), columns=["outcome", "record_count"]
    ).to_csv(build_diagnostics_path, index=False)
    before_path = directory / "candidate_pairs_before.csv"
    after_path = directory / "candidate_pairs_after.csv"
    removals_path = directory / "removed_overlap_segments.csv"
    likely_path = directory / "likely_duplicate_candidates.csv"
    before.to_csv(before_path, index=False)
    after.to_csv(after_path, index=False)
    removals.to_csv(removals_path, index=False)
    likely = before[
        (before["n_overlap_points"] >= config.diagnostic_minimum_overlap)
        & (
            before["abs_corr"]
            >= config.diagnostic_absolute_correlation_minimum
        )
    ]
    likely.to_csv(likely_path, index=False)

    manifest = {
        "schema_version": 1,
        "method": "Temp12k direct-temperature preparation and PAGES2k deduplication",
        "source": str(config.source_path),
        "source_format": config.source_format,
        "pages2k_source": str(
            config.pages2k_source_path or config.pages2k_source_name
        ),
        "anomaly_period": list(config.anomaly_period),
        "require_anomaly_baseline": config.require_anomaly_baseline,
        "comparison_period": list(config.comparison_period),
        "duplicate_rule": {
            "maximum_distance_km": config.maximum_distance_km,
            "correlation": f"> {config.duplicate_correlation_minimum}",
            "overlap_points": f"> {config.duplicate_minimum_overlap}",
            "year_assignment": "numpy.rint",
        },
        "diagnostic_rule": {
            "absolute_correlation_minimum": (
                config.diagnostic_absolute_correlation_minimum
            ),
            "overlap_points_minimum": config.diagnostic_minimum_overlap,
        },
        "counts": {
            "source_temp12k": source_count,
            "dropped_without_anomaly_baseline": dropped_baseline,
            "seasonality_records_changed": seasonality_changed,
            "pages2k": len(pages2k.records),
            "location_candidates_before": len(before),
            "diagnostic_likely_pairs": len(likely),
            "duplicate_pairs_before": int(before["is_duplicate_by_rule"].sum()),
            "records_with_removed_segments": len(removals),
            "output_temp12k": len(database.records),
            "duplicate_pairs_after": int(after["is_duplicate_by_rule"].sum()),
        },
        "compatibility_notes": [
            "Only Temp12k entries in degC with reported temperature uncertainty are built.",
            "R is the square of paleoData_temperature12kUncertainty.",
            "Native sampling is preserved; duplicate timestamps keep the first value.",
            "Only overlapping rounded years are removed from Temp12k duplicate records.",
            "Records without anomaly-period data remain unchanged unless explicitly required.",
        ],
    }
    if remaining:
        raise ValueError(
            f"Temp12k deduplication left {sum(len(x) for x in remaining.values())} "
            "duplicate year assignments"
        )
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return (
        database_path,
        build_diagnostics_path,
        before_path,
        after_path,
        removals_path,
        likely_path,
        manifest_path,
    )


def build_temp12k_database(path: Path, cfr: Any, summary: Counter[str]):
    """Convert the Temp12k LiPD pickle representation into CFR records."""
    with Path(path).open("rb") as stream:
        source = pickle.load(stream)
    if not isinstance(source, dict) or not isinstance(source.get("TS"), list):
        raise ValueError("raw Temp12k pickle must contain a list at key 'TS'")
    records = {}
    for index, raw in enumerate(source["TS"]):
        if not _is_temperature_record(raw):
            summary["dropped_not_temp12k_degC"] += 1
            continue
        time, value, time_source = _extract_time_and_value(raw)
        if time is None or value is None:
            summary["dropped_no_usable_time_or_value"] += 1
            continue
        lat = _as_float(raw.get("geo_meanLat"))
        lon = _as_float(raw.get("geo_meanLon"))
        elev = _as_float(raw.get("geo_meanElev"))
        if lat is None or lon is None:
            summary["dropped_missing_lat_lon"] += 1
            continue
        time, value = _clean_time_value(time, value)
        if time.size < 2:
            summary["dropped_too_short_after_clean"] += 1
            continue
        sigma = _as_float(raw.get("paleoData_temperature12kUncertainty"))
        if sigma is None or sigma <= 0:
            summary["dropped_missing_temperature12k_uncertainty"] += 1
            continue
        archive = str(raw.get("archiveType", ""))
        proxy = str(raw.get("paleoData_proxy", ""))
        ptype = (
            cfr.proxy.get_ptype(archive, proxy)
            if archive and proxy
            else f"{archive}.{proxy}".lower()
        )
        record_id = _record_id(raw, index)
        record = cfr.ProxyRecord(
            pid=record_id,
            time=time,
            value=value,
            lat=lat,
            lon=lon,
            elev=elev,
            ptype=ptype,
            value_name=str(raw.get("paleoData_variableName", "temperature")),
            value_unit="degC",
            time_name="Year",
            time_unit="CE",
            seasonality=_parse_seasonality(raw, lat),
        )
        record.R = sigma**2
        record.R_sigma = sigma
        record.R_source = "reported:paleoData_temperature12kUncertainty"
        record.time_source = time_source
        record.archive_type = archive
        record.proxy_type = proxy
        records[record_id] = record
        summary["kept"] += 1
    return cfr.ProxyDatabase(
        records=records, source="Temp12k raw pickle -> CE direct temperature"
    )


def apply_anomaly_reference(
    database: Any,
    period: tuple[int, int],
    *,
    require_baseline: bool,
) -> int:
    """Reference Temp12k values to a rounded-year climatology in place."""
    dropped = 0
    for record_id, record in list(database.records.items()):
        mean = _baseline_mean(record.time, record.value, *period)
        if np.isfinite(mean):
            record.value = np.asarray(record.value, dtype=float) - mean
        elif require_baseline:
            del database.records[record_id]
            dropped += 1
    database.refresh()
    return dropped


def normalize_seasonality(database: Any) -> int:
    """Apply the final season convention without mutating the source artifact."""
    changed = 0
    for record in database.records.values():
        try:
            season = [int(value) for value in record.seasonality]
        except (TypeError, ValueError):
            season = []
        normalized = _normalize_season(season)
        changed += normalized != season
        record.seasonality = normalized
    return changed


def _normalize_season(season: list[int]) -> list[int]:
    remapped = SEASON_REMAP.get(tuple(season))
    if remapped is not None:
        return list(remapped)
    # A transition from December to an earlier month marks a season that
    # crosses New Year. Months before the transition belong to the preceding
    # model year. Other descending sequences are retained as reported.
    transition = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(season, season[1:], strict=False)
            )
            if left == 12 and 0 < right < 12
        ),
        None,
    )
    if transition is None:
        return list(season)
    return [
        -month if index <= transition and month > 0 else month
        for index, month in enumerate(season)
    ]


def find_candidate_pairs(
    temp12k: dict[str, dict[str, Any]],
    pages2k: dict[str, dict[str, Any]],
    *,
    maximum_distance_km: float,
    duplicate_correlation_minimum: float,
    duplicate_minimum_overlap: int,
) -> tuple[pd.DataFrame, dict[str, set[int]]]:
    """Evaluate colocated annual series and return duplicate years by Temp12k ID."""
    rows = []
    years_to_remove: dict[str, set[int]] = {}
    for temp_id, temp in temp12k.items():
        for pages_id, pages in pages2k.items():
            distance = haversine_km(
                temp["lat"], temp["lon"], pages["lat"], pages["lon"]
            )
            if distance > maximum_distance_km:
                continue
            merged = pd.concat(
                [temp["series"].rename("temp12k"), pages["series"].rename("pages2k")],
                axis=1,
            ).dropna()
            if merged.empty:
                continue
            overlap = len(merged)
            corr = float(merged.temp12k.corr(merged.pages2k)) if overlap > 1 else np.nan
            is_duplicate = bool(
                np.isfinite(corr)
                and corr > duplicate_correlation_minimum
                and overlap > duplicate_minimum_overlap
            )
            rows.append(
                {
                    "temp12k_id": temp_id,
                    "pages2k_id": pages_id,
                    "distance_km": distance,
                    "n_overlap_points": overlap,
                    "corr": corr,
                    "abs_corr": abs(corr) if np.isfinite(corr) else np.nan,
                    "temp12k_ptype": temp["ptype"],
                    "pages2k_ptype": pages["ptype"],
                    "temp12k_unit": temp["unit"],
                    "pages2k_unit": pages["unit"],
                    "temp12k_lat": temp["lat"],
                    "temp12k_lon": temp["lon"],
                    "pages2k_lat": pages["lat"],
                    "pages2k_lon": pages["lon"],
                    "is_duplicate_by_rule": int(is_duplicate),
                    "overlap_year_min": int(merged.index.min()),
                    "overlap_year_max": int(merged.index.max()),
                }
            )
            if is_duplicate:
                years_to_remove.setdefault(temp_id, set()).update(
                    int(year) for year in merged.index
                )
    frame = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["is_duplicate_by_rule", "corr", "n_overlap_points", "distance_km"],
            ascending=[False, False, False, True],
        )
    return frame, years_to_remove


def remove_duplicate_years(
    database: Any, years_by_record: dict[str, set[int]], *, minimum_points: int
) -> pd.DataFrame:
    """Remove only samples whose rounded year overlaps a duplicate PAGES2k record."""
    rows = []
    for record_id, years in years_by_record.items():
        record = database.records.get(record_id)
        if record is None:
            continue
        time = np.asarray(record.time, dtype=float)
        value = np.asarray(record.value, dtype=float)
        keep = ~np.isin(np.rint(time).astype(int), list(years))
        if keep.all():
            continue
        remaining = int(keep.sum())
        if remaining < minimum_points:
            del database.records[record_id]
        else:
            record.time = time[keep]
            record.value = value[keep]
        rows.append(
            {
                "temp12k_id": record_id,
                "points_before": len(time),
                "points_removed": int((~keep).sum()),
                "points_after": remaining,
                "years_removed_min": min(years),
                "years_removed_max": max(years),
                "n_years_removed": len(years),
                "dropped_record_for_short_length": int(remaining < minimum_points),
            }
        )
    database.refresh()
    frame = pd.DataFrame(rows, columns=REMOVAL_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["points_removed", "n_years_removed"], ascending=[False, False]
        )
    return frame


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return shortest great-circle distance using longitude wraparound."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lon = math.radians((lon2 - lon1 + 540.0) % 360.0 - 180.0)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def _database_metadata(
    database: Any,
    period: tuple[int, int],
    *,
    anomaly_period: tuple[int, int] | None = None,
) -> dict[str, dict[str, Any]]:
    metadata = {}
    for record_id, record in database.records.items():
        if record.lat is None or record.lon is None:
            continue
        value = np.asarray(record.value, dtype=float)
        if anomaly_period is not None:
            mean = _baseline_mean(record.time, value, *anomaly_period)
            if np.isfinite(mean):
                value = value - mean
        series = _annual_series(record.time, value, *period)
        if series.empty:
            continue
        metadata[str(record_id)] = {
            "lat": float(record.lat),
            "lon": float(record.lon),
            "ptype": str(record.ptype),
            "unit": str(getattr(record, "value_unit", "")),
            "series": series,
        }
    return metadata


def _annual_series(time: Any, value: Any, start: int, end: int) -> pd.Series:
    time = np.asarray(time, dtype=float)
    value = np.asarray(value, dtype=float)
    valid = np.isfinite(time) & np.isfinite(value)
    years = np.rint(time[valid]).astype(int)
    value = value[valid]
    inside = (years >= start) & (years <= end)
    if not np.any(inside):
        return pd.Series(dtype=float)
    return pd.Series(value[inside]).groupby(years[inside]).mean().sort_index()


def _baseline_mean(time: Any, value: Any, start: int, end: int) -> float:
    time = np.asarray(time, dtype=float)
    value = np.asarray(value, dtype=float)
    valid = np.isfinite(time) & np.isfinite(value)
    years = np.rint(time[valid]).astype(int)
    value = value[valid]
    inside = (years >= start) & (years <= end)
    return float(value[inside].mean()) if np.any(inside) else np.nan


def _load_pages2k(config: Temp12kPreparationConfig, cfr: Any):
    source = config.pages2k_source_path
    if source is not None:
        return copy.deepcopy(_load_database(source, cfr))
    return cfr.ProxyDatabase().fetch(config.pages2k_source_name)


def _load_database(path: Path, cfr: Any):
    path = Path(path)
    if path.is_dir():
        job = cfr.ReconJob()
        job.load(str(path))
        return job.proxydb
    with path.open("rb") as stream:
        loaded = pickle.load(stream)
    return loaded.proxydb if hasattr(loaded, "proxydb") else loaded


def _import_cfr():
    try:
        import cfr
    except ImportError as exc:
        raise RuntimeError("cfr==2024.4.4 is required for Temp12k preparation") from exc
    return cfr


def _is_temperature_record(record: dict[str, Any]) -> bool:
    return (
        str(record.get("paleoData_inCompilation", "")).strip() == "Temp12k"
        and str(record.get("paleoData_units", "")).strip() == "degC"
    )


def _extract_time_and_value(record: dict[str, Any]):
    value = _numeric_series(record.get("paleoData_values"))
    if value is None:
        return None, None, ""
    for name, unit_name in (("year", "yearUnits"), ("age", "ageUnits")):
        time = _numeric_series(record.get(name))
        if time is None or time.size != value.size:
            continue
        converted = _time_to_ce(time, str(record.get(unit_name, "")))
        if converted is not None:
            return converted, value, name
    return None, None, ""


def _time_to_ce(time: np.ndarray, raw_unit: str) -> np.ndarray | None:
    unit = raw_unit.strip().lower()
    if "bp" in unit or "bo" in unit or "calibrated" in unit:
        return 1950.0 - time
    if "ad" in unit or "ce" in unit:
        return time.astype(float)
    return None


def _parse_seasonality(record: dict[str, Any], latitude: float) -> list[int]:
    interpretation = record.get("paleoData_interpretation")
    text = None
    if (
        isinstance(interpretation, list)
        and interpretation
        and isinstance(interpretation[0], dict)
    ):
        text = interpretation[0].get("seasonality")
        if text in (None, ""):
            text = interpretation[0].get("seasonalityGeneral")
    if text is None:
        return list(range(1, 13))
    season = str(text).strip().lower()
    if season in {"annual", "ann", "annua", "year", "nan", "not specified", "n/a"}:
        return list(range(1, 13))
    if season in {"summer", "mean summer", "warm season", "growing", "growing season", "summer+"}:
        return [6, 7, 8] if latitude >= 0 else [-12, 1, 2]
    if season in {"winter", "cold season", "winter+"}:
        return [-12, 1, 2] if latitude >= 0 else [6, 7, 8]
    if season in {"warmest", "warmest month", "mtwa"}:
        return [7] if latitude >= 0 else [1]
    if season in {"coldest", "coldest month"}:
        return [1] if latitude >= 0 else [7]
    months = [int(value) for value in re.findall(r"-?\d+", season)]
    months = [month for month in months if 1 <= abs(month) <= 12]
    return months or list(range(1, 13))


def _record_id(record: dict[str, Any], index: int) -> str:
    dataset = str(record.get("dataSetName", "")).strip() or "unknownDataset"
    if record.get("paleoData_TSid") is not None:
        return f"{dataset}::{record['paleoData_TSid']}"
    if record.get("paleoData_number") is not None:
        return f"{dataset}::num{record['paleoData_number']}"
    return f"{dataset}::idx{index}"


def _clean_time_value(time: np.ndarray, value: np.ndarray):
    valid = np.isfinite(time) & np.isfinite(value)
    time, value = time[valid], value[valid]
    order = np.argsort(time)
    time, value = time[order], value[order]
    _, first = np.unique(time, return_index=True)
    return time[first], value[first]


def _numeric_series(value: Any) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return result if result.size else None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
