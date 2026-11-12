"""Reconstruction workflow handler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from lmr4dvar.config import ProjectConfig
from lmr4dvar.eof import EOFModel
from lmr4dvar.reconstruction import (
    decode_variable_trajectory,
    global_mean_ocean_heat_content,
    global_mean_temperature,
    nino34,
    plot_reconstructed_index,
    sea_ice_area,
    sea_ice_volume,
)
from lmr4dvar.reconstruction.indices import latitude_weighted_mean
from lmr4dvar.state import StateLayout

from .planner import artifact_stage_directory, stage_directory
from .stages import Stage


def execute_reconstruction(config: ProjectConfig) -> tuple[Path, ...]:
    analysis_path = stage_directory(config, Stage.ASSIMILATE) / "analysis.nc"
    with xr.open_dataset(analysis_path, engine="h5netcdf") as dataset:
        trajectory = dataset.trajectory.load().values
        time = dataset.time.load().values
    layout = StateLayout.from_mode_counts(
        {name: item.n_modes for name, item in config.state.variables.items()}
    )
    directory = stage_directory(config, Stage.RECONSTRUCT)
    fields_directory = directory / "fields"
    fields_directory.mkdir(parents=True, exist_ok=True)
    fields = {}
    outputs = []
    for variable in config.reconstruction.variables:
        model = EOFModel.load(artifact_stage_directory(config, Stage.EOF) / variable)
        field = decode_variable_trajectory(trajectory, time, variable, model, layout)
        fields[variable] = field
        if config.state.variables[variable].units is not None:
            field.attrs["units"] = config.state.variables[variable].units
        path = fields_directory / f"{variable}.nc"
        field.to_dataset().to_netcdf(path, engine="h5netcdf")
        outputs.append(path)
    indices = _calculate_indices(config, fields)
    indices_path = directory / "indices.nc"
    xr.Dataset(indices).to_netcdf(indices_path, engine="h5netcdf")
    outputs.append(indices_path)
    if config.reconstruction.plot_figures:
        for name, series in indices.items():
            outputs.extend(
                plot_reconstructed_index(
                    name,
                    series,
                    directory / "figures" / "indices" / name,
                    dpi=config.reconstruction.figure_dpi,
                )
            )
    return tuple(outputs)


def _calculate_indices(
    config: ProjectConfig, fields: dict[str, xr.DataArray]
) -> dict[str, xr.DataArray]:
    result = {}
    absolute_sic = None
    absolute_sit = None
    for name in config.reconstruction.indices:
        normalized = name.lower()
        if normalized == "gmt":
            result["GMT"] = global_mean_temperature(_required(fields, "tas", name))
        elif normalized == "gmp":
            result["GMP"] = latitude_weighted_mean(_required(fields, "pr", name))
            result["GMP"].attrs.update(
                long_name="global mean precipitation anomaly on model valid grid",
                units="mm/day",
            )
        elif normalized == "nino34":
            result["Nino34"] = nino34(_required(fields, "tos", name))
        elif normalized == "gmohc":
            result["GMOHC"] = global_mean_ocean_heat_content(_required(fields, "ohc300", name))
        elif normalized == "gmohc2000":
            result["GMOHC2000"] = global_mean_ocean_heat_content(
                _required(fields, "ohc2000", name), name="GMOHC2000"
            )
        elif normalized == "sia":
            if absolute_sic is None:
                absolute_sic = _restore_climatology(
                    _required(fields, "sic", name),
                    config.reconstruction.sea_ice_concentration_climatology,
                    "SIA requires reconstruction.sea_ice_concentration_climatology",
                )
            result["SIA"] = sea_ice_area(
                absolute_sic,
                concentration_scale=config.reconstruction.sea_ice_concentration_scale,
            )
        elif normalized == "siv":
            if absolute_sic is None:
                absolute_sic = _restore_climatology(
                    _required(fields, "sic", name),
                    config.reconstruction.sea_ice_concentration_climatology,
                    "SIV requires reconstruction.sea_ice_concentration_climatology",
                )
            if absolute_sit is None:
                absolute_sit = _restore_climatology(
                    _required(fields, "sit", name),
                    config.reconstruction.sea_ice_thickness_climatology,
                    "SIV requires reconstruction.sea_ice_thickness_climatology",
                )
            result["SIV"] = sea_ice_volume(
                absolute_sic,
                absolute_sit,
                concentration_scale=config.reconstruction.sea_ice_concentration_scale,
            )
        else:
            raise ValueError(f"unsupported reconstruction index: {name}")
    return result


def _required(fields: dict[str, xr.DataArray], variable: str, index: str) -> xr.DataArray:
    try:
        return fields[variable]
    except KeyError as exc:
        raise ValueError(f"index {index} requires reconstructed variable {variable}") from exc


def _restore_climatology(field, climatology_config, missing_message) -> xr.DataArray:
    if climatology_config is None:
        raise ValueError(missing_message)
    with xr.open_dataset(climatology_config.path, engine="h5netcdf") as dataset:
        if climatology_config.variable not in dataset:
            raise KeyError(
                f"variable {climatology_config.variable!r} is absent from {climatology_config.path}"
            )
        climatology = dataset[climatology_config.variable].load()
    for coordinate in ("lat", "lon"):
        source = climatology.get_index(coordinate)
        target = field.get_index(coordinate)
        if not source.is_unique or not target.is_unique or not target.isin(source).all():
            raise ValueError(f"sea-ice climatology has a different {coordinate} grid")
    climatology = climatology.sel(lat=field.lat, lon=field.lon)
    if "month" in climatology.dims:
        months = np.asarray(climatology.month)
        if not np.array_equal(np.sort(months), np.arange(1, 13)):
            raise ValueError("monthly sea-ice climatology requires twelve unique months")
        # New preprocessing saves true monthly means; restore complete DJF-aligned quarters.
        monthly = xr.concat(
            [
                climatology.sel(month=season).mean("month")
                for season in ([12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11])
            ],
            dim=xr.IndexVariable("month", [1, 4, 7, 10]),
        )
    elif "time" in climatology.dims:
        climatology_months = _coordinate_months(climatology.time)
        monthly = climatology.groupby(
            xr.DataArray(climatology_months, dims="time", name="month")
        ).mean("time")
    else:
        raise ValueError("sea-ice climatology must have a month or time dimension")
    field_months = xr.DataArray(
        _coordinate_months(field.time),
        dims="time",
        coords={"time": field.time},
        name="month",
    )
    restored = field + monthly.sel(month=field_months)
    return restored.drop_vars("month", errors="ignore")


def _coordinate_months(time: xr.DataArray):
    try:
        return time.dt.month.values
    except AttributeError:
        values = time.values.astype(float)
        years = values // 1
        return (np.rint((values - years) * 12).astype(int) % 12) + 1
