"""Artifact I/O for preprocessed simulation fields."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import xarray as xr


def write_preprocessed_artifact(
    directory: str | Path,
    fields: Mapping[str, xr.DataArray],
    climatologies: Mapping[str, xr.DataArray],
    metadata: Mapping[str, Any],
) -> None:
    """Write fields, climatologies, and provenance as a portable artifact."""
    directory = Path(directory)
    fields_dir = directory / "fields"
    climatology_dir = directory / "climatology"
    fields_dir.mkdir(parents=True, exist_ok=True)
    climatology_dir.mkdir(parents=True, exist_ok=True)

    for name, field in fields.items():
        _write_data_array(field, fields_dir / f"{name}.nc", name)
    for name, climatology in climatologies.items():
        _write_data_array(climatology, climatology_dir / f"{name}.nc", name)

    manifest = {
        **metadata,
        "schema_version": 1,
        "variables": sorted(fields),
        "climatologies": sorted(climatologies),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_data_array(field: xr.DataArray, path: Path, fallback_name: str) -> None:
    name = field.name or fallback_name
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    field.rename(name).to_dataset().to_netcdf(
        path, engine="h5netcdf", encoding=encoding
    )
