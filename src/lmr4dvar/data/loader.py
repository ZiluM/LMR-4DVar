"""Explicit simulation-file resolution and climate-field loading."""

from __future__ import annotations

from contextlib import ExitStack
from fnmatch import fnmatch
from pathlib import Path

import xarray as xr

from .monthly_time import label_monthly_intervals


def resolve_input_files(
    root: Path, pattern: str, *, exclude_patterns: tuple[str, ...] = ()
) -> tuple[Path, ...]:
    if not pattern or Path(pattern).is_absolute():
        raise ValueError("input_pattern must be a nonempty path relative to simulation_root")
    files = tuple(
        sorted(
            path
            for path in root.glob(pattern)
            if path.is_file() and not any(fnmatch(path.name, item) for item in exclude_patterns)
        )
    )
    if not files:
        raise FileNotFoundError(f"no simulation files match {pattern!r} under {root}")
    return files


def select_horizontal_field(field: xr.DataArray) -> xr.DataArray:
    """Remove only singleton non-horizontal dimensions, never choose a 3D level."""
    extra = set(field.dims) - {"time", "lat", "lon"}
    if any(field.sizes[dim] != 1 for dim in extra):
        raise ValueError("source field has unsupported non-singleton dimensions")
    result = field.isel({dim: 0 for dim in extra}, drop=True)
    result.attrs = field.attrs.copy()
    if extra:
        result.attrs["singleton_dimensions_selected"] = ",".join(sorted(extra))
    return result


def load_climate_field(
    files: tuple[Path, ...],
    source_name: str,
    *,
    monthly_time_policy: str = "coordinate",
    time_bounds_files: tuple[Path, ...] | None = None,
    drop_partial_boundaries: bool = False,
    restrict_to_bounds_source: bool = False,
) -> xr.DataArray:
    """Open one or more coordinate-compatible NetCDF files and load one variable."""
    if not source_name:
        raise ValueError("source_name must be nonempty")
    if monthly_time_policy not in {"coordinate", "cf_bounds"}:
        raise ValueError("unsupported monthly time policy")
    if time_bounds_files is not None and (
        monthly_time_policy != "cf_bounds" or len(time_bounds_files) != len(files)
    ):
        raise ValueError(
            "external monthly bounds require CF policy and one matching file per input"
        )
    if len(files) == 1:
        with ExitStack() as stack:
            dataset = stack.enter_context(xr.open_dataset(files[0], engine="h5netcdf"))
            donor = (
                stack.enter_context(xr.open_dataset(time_bounds_files[0], engine="h5netcdf"))
                if time_bounds_files
                else None
            )
            if monthly_time_policy == "cf_bounds":
                dataset = label_monthly_intervals(
                    dataset,
                    bounds_source=donor,
                    drop_partial_boundaries=drop_partial_boundaries,
                    restrict_to_bounds_source=restrict_to_bounds_source,
                )
            if source_name not in dataset:
                raise KeyError(f"variable {source_name!r} is absent from {files[0]}")
            field = select_horizontal_field(dataset[source_name]).load()
            field.attrs["monthly_time_policy"] = monthly_time_policy
            if time_bounds_files:
                field.attrs["monthly_time_bounds_source"] = str(time_bounds_files[0])
            return field
    with ExitStack() as stack:
        dataset = stack.enter_context(
            xr.open_mfdataset(files, combine="by_coords", engine="h5netcdf")
        )
        donor = (
            stack.enter_context(
                xr.open_mfdataset(time_bounds_files, combine="by_coords", engine="h5netcdf")
            )
            if time_bounds_files
            else None
        )
        if monthly_time_policy == "cf_bounds":
            dataset = label_monthly_intervals(
                dataset,
                bounds_source=donor,
                drop_partial_boundaries=drop_partial_boundaries,
                restrict_to_bounds_source=restrict_to_bounds_source,
            )
        if source_name not in dataset:
            raise KeyError(f"variable {source_name!r} is absent from simulation inputs")
        field = select_horizontal_field(dataset[source_name]).load()
        field.attrs["monthly_time_policy"] = monthly_time_policy
        if time_bounds_files:
            field.attrs["monthly_time_bounds_source"] = ",".join(
                str(path) for path in time_bounds_files
            )
        return field
