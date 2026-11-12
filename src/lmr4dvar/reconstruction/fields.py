"""Decode reduced trajectories into self-describing climate fields."""

from __future__ import annotations

import numpy as np
import torch
import xarray as xr

from lmr4dvar.eof import EOFModel
from lmr4dvar.state import StateLayout


def decode_trajectory(
    trajectory: torch.Tensor | np.ndarray,
    time: xr.DataArray | np.ndarray,
    models: dict[str, EOFModel],
    layout: StateLayout,
) -> xr.Dataset:
    """Decode every variable represented by a state layout."""
    values = (
        trajectory.detach().cpu().numpy()
        if isinstance(trajectory, torch.Tensor)
        else np.asarray(trajectory, dtype=float)
    )
    time_values = np.asarray(time.values if isinstance(time, xr.DataArray) else time)
    if values.ndim != 2 or values.shape[1] != layout.size:
        raise ValueError("trajectory must have shape (time, state_layout.size)")
    if time_values.ndim != 1 or time_values.size != values.shape[0]:
        raise ValueError("time must be 1D and match trajectory")
    if set(models) != set(layout.variable_slices):
        raise ValueError("EOF model variables must exactly match the state layout")
    fields = {}
    for variable in layout.variable_slices:
        fields[variable] = decode_variable_trajectory(
            values,
            time_values,
            variable,
            models[variable],
            layout,
        )
    return xr.Dataset(fields)


def decode_variable_trajectory(
    trajectory: torch.Tensor | np.ndarray,
    time: xr.DataArray | np.ndarray,
    variable: str,
    model: EOFModel,
    layout: StateLayout,
) -> xr.DataArray:
    """Decode one variable without aligning its grid to other variables."""
    values = (
        trajectory.detach().cpu().numpy()
        if isinstance(trajectory, torch.Tensor)
        else np.asarray(trajectory, dtype=float)
    )
    time_values = np.asarray(time.values if isinstance(time, xr.DataArray) else time)
    if values.ndim != 2 or values.shape[1] != layout.size:
        raise ValueError("trajectory must have shape (time, state_layout.size)")
    if time_values.ndim != 1 or time_values.size != values.shape[0]:
        raise ValueError("time must be 1D and match trajectory")
    variable_slice = layout.slice_for(variable)
    mode_count = variable_slice.stop - variable_slice.start
    if mode_count > model.components.sizes["mode"]:
        raise ValueError(f"EOF model for {variable} has too few modes")
    pcs = xr.DataArray(
        values[:, variable_slice],
        dims=("time", "mode"),
        coords={"time": time_values, "mode": model.components.mode.values[:mode_count]},
    )
    selected_model = EOFModel(
        components=model.components.isel(mode=slice(0, mode_count)),
        weighted_mean=model.weighted_mean,
        weights=model.weights,
        valid_mask=model.valid_mask,
        pc_scale=model.pc_scale.isel(mode=slice(0, mode_count)),
        singular_values=model.singular_values.isel(mode=slice(0, mode_count)),
        explained_variance_ratio=model.explained_variance_ratio.isel(
            mode=slice(0, mode_count)
        ),
        centered=model.centered,
        missing_value_fill=model.missing_value_fill,
    )
    return selected_model.decode(pcs).rename(variable)
