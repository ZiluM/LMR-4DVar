"""Read-only metadata preflight for configured simulation inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from lmr4dvar.config.models import InputCombine, InputKind, ProjectConfig, QuarterlyMode

from .loader import resolve_input_files, select_horizontal_field
from .monthly_time import label_monthly_intervals


def inspect_simulation_inputs(config: ProjectConfig) -> dict[str, Any]:
    """Inspect source variables, selected sample counts, and grids without field loads."""
    report: dict[str, Any] = {"variables": {}, "errors": [], "warnings": []}
    totals: dict[str, int] = {}
    physical_labels: dict[str, list[np.ndarray]] = {}
    for variable, variable_config in config.state.variables.items():
        pattern = variable_config.input_pattern
        if pattern is None:
            report["errors"].append(f"state variable {variable} has no input_pattern")
            continue
        try:
            root = variable_config.input_root or config.paths.simulation_root
            exclusions = tuple(variable_config.exclude_input_patterns)
            files = resolve_input_files(root, pattern, exclude_patterns=exclusions)
            bounds_files = None
            if variable_config.monthly_time_bounds_pattern:
                bounds_files = resolve_input_files(
                    root, variable_config.monthly_time_bounds_pattern, exclude_patterns=exclusions
                )
                if len(bounds_files) != len(files):
                    raise ValueError(f"{variable}: bounds file count differs from input count")
        except (FileNotFoundError, ValueError) as exc:
            report["errors"].append(str(exc))
            continue
        source_name = variable_config.source_name or variable
        records = []
        for i, path in enumerate(files):
            try:
                records.append(
                    _inspect_file(
                        path,
                        source_name,
                        input_kind=config.preprocess.input_kind,
                        training_period=config.preprocess.training_period,
                        seasonal_months=config.preprocess.seasonal_months,
                        northern_hemisphere=(variable_config.domain.value == "northern_hemisphere"),
                        monthly_time_policy=variable_config.monthly_time_policy,
                        bounds_file=bounds_files[i] if bounds_files else None,
                        drop_partial_boundaries=variable_config.monthly_drop_partial_boundaries,
                        restrict_to_bounds_source=variable_config.monthly_restrict_to_bounds_source,
                        quarterly_mode=config.preprocess.quarterly_mode,
                        require_complete_years=config.preprocess.require_complete_years,
                    )
                )
            except (KeyError, ValueError, OSError) as exc:
                report["errors"].append(f"{variable}: {path}: {exc}")
        if not records:
            continue
        _check_member_compatibility(variable, records, variable_config.input_combine, report)
        total = _training_sample_total(records, variable_config.input_combine)
        if total is not None:
            totals[variable] = total
        if variable_config.input_combine is InputCombine.CONCATENATE_MEMBERS:
            physical_labels[variable] = [record["month_labels"] for record in records]
        report["variables"][variable] = {
            "source_name": source_name,
            "files": len(records),
            "samples_per_file": sorted({record["samples"] for record in records}),
            "training_samples": total,
            "grid": {"lat": records[0]["lat_size"], "lon": records[0]["lon_size"]},
            "units": sorted({record["units"] for record in records if record["units"] is not None}),
            "source_units_override": variable_config.source_units,
            "excluded_input_patterns": variable_config.exclude_input_patterns,
        }
    if physical_labels:
        reference_name = next(iter(physical_labels))
        expected = physical_labels[reference_name]
        for variable, labels in physical_labels.items():
            if len(labels) != len(expected) or any(
                not np.array_equal(left, right)
                for left, right in zip(labels, expected, strict=True)
            ):
                report["errors"].append(
                    f"{variable}: retained physical member times differ from {reference_name}"
                )
    if totals and len(set(totals.values())) != 1:
        details = ", ".join(f"{name}={count}" for name, count in totals.items())
        report["errors"].append(f"state variables have different training sample totals: {details}")
    report["training_samples"] = (
        next(iter(totals.values())) if len(set(totals.values())) == 1 else None
    )
    return report


def _inspect_file(
    path,
    source_name: str,
    *,
    input_kind: InputKind,
    training_period: tuple[int, int] | None,
    seasonal_months: list[int],
    northern_hemisphere: bool,
    monthly_time_policy: str = "coordinate",
    bounds_file=None,
    drop_partial_boundaries: bool = False,
    restrict_to_bounds_source: bool = False,
    quarterly_mode: QuarterlyMode = QuarterlyMode.COMPLETE,
    require_complete_years: bool = False,
) -> dict[str, Any]:
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(path, engine="h5netcdf", decode_times=time_coder) as dataset:
        if monthly_time_policy == "cf_bounds":
            if input_kind is InputKind.PREPARED_QUARTERLY:
                raise ValueError("CF-month correction requires monthly inputs")
            if bounds_file is None:
                dataset = label_monthly_intervals(
                    dataset, drop_partial_boundaries=drop_partial_boundaries
                )
            else:
                with xr.open_dataset(
                    bounds_file, engine="h5netcdf", decode_times=time_coder
                ) as donor:
                    dataset = label_monthly_intervals(
                        dataset,
                        bounds_source=donor,
                        drop_partial_boundaries=drop_partial_boundaries,
                        restrict_to_bounds_source=restrict_to_bounds_source,
                    )
        if source_name not in dataset:
            raise KeyError(f"source variable {source_name!r} is absent")
        field = select_horizontal_field(dataset[source_name])
        missing = {"time", "lat", "lon"} - set(field.dims)
        if missing:
            raise ValueError(f"source field is missing dimensions: {', '.join(sorted(missing))}")
        extra = set(field.dims) - {"time", "lat", "lon"}
        if extra:
            raise ValueError(f"source field has unsupported dimensions: {', '.join(sorted(extra))}")
        years = np.asarray(field.time.dt.year)
        selected = np.ones(years.shape, dtype=bool)
        if training_period is not None:
            selected &= (years >= training_period[0]) & (years <= training_period[1])
        if input_kind is InputKind.PREPARED_QUARTERLY:
            selected &= np.isin(np.asarray(field.time.dt.month), seasonal_months)
        month_labels = years[selected] * 12 + np.asarray(field.time.dt.month)[selected]
        if input_kind is InputKind.MONTHLY:
            if len(np.unique(month_labels)) != month_labels.size:
                raise ValueError("duplicate physical monthly labels")
            quarters, counts = np.unique(month_labels // 3, return_counts=True)
            if quarterly_mode is QuarterlyMode.COMPLETE:
                quarters = quarters[counts == 3]
            q_months = 1 + 3 * (quarters % 4)
            keep = np.isin(q_months, seasonal_months)
            quarters = quarters[keep]
            if require_complete_years:
                if np.any(np.diff(quarters) != 1):
                    raise ValueError("missing internal quarterly training samples")
                unique_years, counts = np.unique(quarters // 4, return_counts=True)
                complete_years = unique_years[counts == 4]
                quarters = quarters[np.isin(quarters // 4, complete_years)]
            month_labels = (quarters // 4) * 12 + 1 + 3 * (quarters % 4)
        samples = int(month_labels.size)
        if samples == 0:
            raise ValueError("configured period and seasons select no samples")
        lat = np.asarray(field.lat)
        if northern_hemisphere:
            lat = lat[lat >= 0]
        if lat.size == 0:
            raise ValueError("configured domain selects no latitude points")
        return {
            "samples": samples,
            "lat": lat,
            "lon": np.asarray(field.lon),
            "lat_size": int(lat.size),
            "lon_size": int(field.lon.size),
            "units": field.attrs.get("units"),
            "month_labels": month_labels,
        }


def _check_member_compatibility(
    variable: str,
    records: list[dict[str, Any]],
    combine: InputCombine,
    report: dict[str, Any],
) -> None:
    if combine is not InputCombine.CONCATENATE_MEMBERS:
        if len(records) > 1:
            report["warnings"].append(
                f"{variable}: by_coords input spans multiple files; "
                "combined sample total is unknown"
            )
        return
    expected = records[0]
    for index, record in enumerate(records[1:], start=1):
        if record["samples"] != expected["samples"]:
            report["errors"].append(
                f"{variable}: member {index} has {record['samples']} samples; "
                f"expected {expected['samples']}"
            )
        for coordinate in ("lat", "lon"):
            if not np.array_equal(record[coordinate], expected[coordinate]):
                report["errors"].append(
                    f"{variable}: member {index} has a different {coordinate} grid"
                )
        if not np.array_equal(record["month_labels"], expected["month_labels"]):
            report["errors"].append(
                f"{variable}: member {index} has different physical month labels"
            )


def _training_sample_total(records: list[dict[str, Any]], combine: InputCombine) -> int | None:
    if combine is InputCombine.CONCATENATE_MEMBERS:
        return sum(record["samples"] for record in records)
    if len(records) == 1:
        return records[0]["samples"]
    return None
