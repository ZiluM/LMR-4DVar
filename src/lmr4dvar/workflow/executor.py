"""Local handlers for preprocessing, EOF fitting, and LIM training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from lmr4dvar.config import ProjectConfig
from lmr4dvar.config.models import DetrendMethod, InputCombine, InputKind, RegridMethod
from lmr4dvar.data import (
    load_climate_field,
    prepare_quarterly_field,
    preprocess_field,
    resolve_input_files,
    write_preprocessed_artifact,
)
from lmr4dvar.data.precipitation import precipitation_mm_per_day
from lmr4dvar.data.preprocess import complete_quarterly_years
from lmr4dvar.eof import EOFModel
from lmr4dvar.lim import LIMModel
from lmr4dvar.state import StateLayout

from .assimilation import execute_assimilation
from .calibrate_pages2k import execute_pages2k_calibration
from .estimate_borehole_error import execute_borehole_error_estimation
from .planner import StageStatus, build_plan, stage_directory, write_stage_receipt
from .prepare_temp12k import execute_temp12k_preparation
from .reconstruct import execute_reconstruction
from .stages import Stage
from .verify import execute_verification

IMPLEMENTED_STAGES = (
    Stage.PREPROCESS,
    Stage.EOF,
    Stage.TRAIN_LIM,
    Stage.CALIBRATE_PAGES2K,
    Stage.PREPARE_TEMP12K,
    Stage.ESTIMATE_BOREHOLE_ERROR,
    Stage.ASSIMILATE,
    Stage.RECONSTRUCT,
    Stage.VERIFY,
)


def execute_stage(config: ProjectConfig, stage: Stage, *, force: bool = False) -> tuple[Path, ...]:
    """Execute one ready local stage and write its receipt after success."""
    if stage not in IMPLEMENTED_STAGES:
        raise NotImplementedError(f"local handler is not implemented for stage {stage.value}")
    item = build_plan(config, start=stage, end=stage, force=force)[0]
    if item.status is StageStatus.COMPLETE:
        return tuple(Path(record["path"]) for record in _receipt_outputs(item.directory))
    if item.status is not StageStatus.READY:
        raise RuntimeError(f"stage {stage.value} is {item.status.value}: {item.reason}")
    handlers = {
        Stage.PREPROCESS: _execute_preprocess,
        Stage.EOF: _execute_eof,
        Stage.TRAIN_LIM: _execute_train_lim,
        Stage.CALIBRATE_PAGES2K: execute_pages2k_calibration,
        Stage.PREPARE_TEMP12K: execute_temp12k_preparation,
        Stage.ESTIMATE_BOREHOLE_ERROR: execute_borehole_error_estimation,
        Stage.ASSIMILATE: execute_assimilation,
        Stage.RECONSTRUCT: execute_reconstruction,
        Stage.VERIFY: execute_verification,
    }
    outputs = handlers[stage](config)
    write_stage_receipt(config, stage, list(outputs))
    return outputs


def execute_through(
    config: ProjectConfig, end: Stage = Stage.TRAIN_LIM, *, force: bool = False
) -> dict[Stage, tuple[Path, ...]]:
    if end not in IMPLEMENTED_STAGES:
        raise NotImplementedError(f"local execution through {end.value} is not implemented")
    start = Stage.CALIBRATE_PAGES2K if config.paths.prior_run_root is not None else Stage.PREPROCESS
    selected = IMPLEMENTED_STAGES[
        IMPLEMENTED_STAGES.index(start) : IMPLEMENTED_STAGES.index(end) + 1
    ]
    if not selected:
        raise ValueError(f"external-prior workflow cannot end at {end.value}")
    results = {}
    for stage in selected:
        results[stage] = execute_stage(config, stage, force=force)
    return results


def _execute_preprocess(config: ProjectConfig) -> tuple[Path, ...]:
    directory = stage_directory(config, Stage.PREPROCESS)
    fields = {}
    climatologies = {}
    inputs = {}
    time_bound_inputs = {}
    for variable, variable_config in config.state.variables.items():
        if variable_config.input_pattern is None:
            raise ValueError(f"state variable {variable} requires input_pattern for preprocessing")
        source_name = variable_config.source_name or variable
        root = variable_config.input_root or config.paths.simulation_root
        exclusions = tuple(variable_config.exclude_input_patterns)
        files = resolve_input_files(
            root, variable_config.input_pattern, exclude_patterns=exclusions
        )
        bounds_files = None
        if variable_config.monthly_time_bounds_pattern is not None:
            bounds_files = resolve_input_files(
                root, variable_config.monthly_time_bounds_pattern, exclude_patterns=exclusions
            )
            if len(bounds_files) != len(files):
                raise ValueError(f"{variable}: monthly bounds file count differs from source count")
            time_bound_inputs[variable] = [str(path.resolve()) for path in bounds_files]
        loader_options = {
            "monthly_time_policy": variable_config.monthly_time_policy,
            "drop_partial_boundaries": variable_config.monthly_drop_partial_boundaries,
            "restrict_to_bounds_source": variable_config.monthly_restrict_to_bounds_source,
        }
        if (
            config.preprocess.input_kind is InputKind.PREPARED_QUARTERLY
            and variable_config.monthly_time_policy == "cf_bounds"
        ):
            raise ValueError("CF-month correction requires monthly, not prepared-quarterly inputs")
        if variable_config.input_combine is InputCombine.CONCATENATE_MEMBERS:
            sources = (
                load_climate_field(
                    (path,),
                    source_name,
                    **loader_options,
                    time_bounds_files=(bounds_files[i],) if bounds_files else None,
                )
                for i, path in enumerate(files)
            )
        else:
            sources = [
                load_climate_field(
                    files, source_name, **loader_options, time_bounds_files=bounds_files
                )
            ]
        processed = []
        # Process one monthly member at a time instead of retaining all raw fields.
        for source in sources:
            if variable == "pr":
                if variable_config.units != "mm/day":
                    raise ValueError("precipitation preprocessing requires explicit units=mm/day")
                source = precipitation_mm_per_day(source, source_units=variable_config.source_units)
            field, climatology = _prepare_source(config, variable_config, source)
            if config.preprocess.require_complete_years:
                field = complete_quarterly_years(field)
            processed.append((field, climatology))
        fields_for_members = [item[0] for item in processed]
        field = _concatenate_members(fields_for_members) if len(processed) > 1 else processed[0][0]
        member_climatologies = [item[1] for item in processed if item[1] is not None]
        climatology = None
        if member_climatologies:
            climatology = xr.concat(member_climatologies, dim="member").mean(
                "member", keep_attrs=True
            )
        fields[variable] = field.rename(variable)
        template = next(iter(fields.values()))
        for coordinate in ("time", "member", "member_sample", "physical_year", "physical_month"):
            if coordinate in template.coords and (
                coordinate not in field.coords or not field[coordinate].equals(template[coordinate])
            ):
                raise ValueError(f"{variable}: {coordinate} differs from other state variables")
        if climatology is not None:
            climatologies[variable] = climatology.rename(variable)
        inputs[variable] = [str(path.resolve()) for path in files]
    write_preprocessed_artifact(
        directory,
        fields,
        climatologies,
        {
            "input_files": inputs,
            "time_bounds_input_files": time_bound_inputs,
            "project": config.project.name,
            "monthly_time_policies": {
                name: item.monthly_time_policy for name, item in config.state.variables.items()
            },
            "input_selection": {
                name: {
                    "excluded_patterns": item.exclude_input_patterns,
                    "restrict_to_bounds_source": item.monthly_restrict_to_bounds_source,
                    "drop_partial_boundaries": item.monthly_drop_partial_boundaries,
                    "source_units_override": item.source_units,
                }
                for name, item in config.state.variables.items()
            },
        },
    )
    outputs = [directory / "manifest.json"]
    outputs.extend(directory / "fields" / f"{name}.nc" for name in sorted(fields))
    outputs.extend(directory / "climatology" / f"{name}.nc" for name in sorted(climatologies))
    return tuple(outputs)


def _prepare_source(config, variable_config, source):
    if config.preprocess.input_kind is InputKind.PREPARED_QUARTERLY:
        return (
            prepare_quarterly_field(
                source,
                domain=variable_config.domain,
                seasonal_months=config.preprocess.seasonal_months,
                training_period=config.preprocess.training_period,
            ),
            None,
        )
    return preprocess_field(
        source,
        domain=variable_config.domain,
        resolution=config.preprocess.target_grid_degrees,
        regrid=config.preprocess.regrid is not RegridMethod.NONE,
        remove_climatology=config.preprocess.remove_monthly_climatology,
        quarterly_mode=config.preprocess.quarterly_mode,
        seasonal_months=config.preprocess.seasonal_months,
        detrend=config.preprocess.detrend is not DetrendMethod.NONE,
        training_period=config.preprocess.training_period,
    )


def _concatenate_members(fields: list[xr.DataArray]) -> xr.DataArray:
    template = fields[0]
    for member, field in enumerate(fields[1:], start=1):
        if field.sizes["time"] != template.sizes["time"]:
            raise ValueError(
                f"member {member} has {field.sizes['time']} samples; "
                f"expected {template.sizes['time']}"
            )
        if not field.time.equals(template.time):
            raise ValueError(f"member {member} has different physical time coordinates")
        for coordinate in ("lat", "lon"):
            if coordinate in template.coords and not field[coordinate].equals(template[coordinate]):
                raise ValueError(f"member {member} has a different {coordinate} grid")
    lengths = [field.sizes["time"] for field in fields]
    combined = xr.concat(fields, dim="time")
    combined = combined.assign_coords(
        time=np.arange(combined.sizes["time"]),
        member=("time", np.repeat(np.arange(len(fields)), lengths)),
        member_sample=("time", np.concatenate([np.arange(length) for length in lengths])),
        physical_year=(
            "time",
            np.concatenate([np.asarray(field.time.dt.year) for field in fields]),
        ),
        physical_month=(
            "time",
            np.concatenate([np.asarray(field.time.dt.month) for field in fields]),
        ),
    )
    combined.attrs = template.attrs.copy()
    combined.attrs["member_concatenation"] = "sorted input-file order"
    return combined


def _execute_eof(config: ProjectConfig) -> tuple[Path, ...]:
    preprocess_directory = stage_directory(config, Stage.PREPROCESS)
    directory = stage_directory(config, Stage.EOF)
    pcs_by_variable = []
    outputs = []
    layout = StateLayout.from_mode_counts(
        {name: item.n_modes for name, item in config.state.variables.items()}
    )
    common_time = None
    for variable, variable_config in config.state.variables.items():
        path = preprocess_directory / "fields" / f"{variable}.nc"
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            field = dataset[variable].load()
        model, pcs = EOFModel.fit(
            field,
            variable_config.n_modes,
            area_weighted=config.eof.area_weighted,
            center=config.eof.center,
            missing_value_fill=variable_config.eof_fillna_value,
        )
        variable_directory = directory / variable
        model.save(variable_directory, pcs)
        outputs.extend(
            [
                variable_directory / "eofs.nc",
                variable_directory / "metadata.yml",
                variable_directory / "pcs.nc",
            ]
        )
        common_time = pcs.time if common_time is None else common_time
        if not pcs.time.equals(common_time):
            raise ValueError(f"EOF PC time coordinate for {variable} differs from other variables")
        for coordinate in ("member", "member_sample", "physical_year", "physical_month"):
            if coordinate in field.coords:
                pcs = pcs.assign_coords({coordinate: field[coordinate]})
                if pcs_by_variable and not pcs[coordinate].equals(pcs_by_variable[0][coordinate]):
                    raise ValueError(
                        f"EOF PC {coordinate} for {variable} differs from other variables"
                    )
        labels = layout.labels[layout.slice_for(variable)]
        pcs_by_variable.append(
            pcs.rename(mode="state").assign_coords(state=np.asarray(labels, dtype=str))
        )
    coupled = xr.concat(pcs_by_variable, dim="state").transpose("time", "state")
    coupled.name = "pcs"
    coupled_path = directory / "state_pcs.nc"
    coupled.to_dataset().to_netcdf(coupled_path, engine="h5netcdf")
    outputs.append(coupled_path)
    return tuple(outputs)


def _execute_train_lim(config: ProjectConfig) -> tuple[Path, ...]:
    eof_directory = stage_directory(config, Stage.EOF)
    directory = stage_directory(config, Stage.TRAIN_LIM)
    with xr.open_dataset(eof_directory / "state_pcs.nc", engine="h5netcdf") as dataset:
        pcs = dataset.pcs.load()
    if "member" in pcs.coords:
        members = np.asarray(pcs.member)
        lengths = np.unique(members, return_counts=True)[1]
        segment = config.lim.segment_length_steps
        expected = np.repeat(np.unique(members), lengths)
        if not np.array_equal(members, expected) or len(set(lengths)) != 1 or segment != lengths[0]:
            raise ValueError(
                "LIM segment_length_steps must match contiguous physical member blocks"
            )
    model = LIMModel.fit(
        pcs,
        lag_steps=config.lim.lag_steps,
        segment_length_steps=config.lim.segment_length_steps,
        fit_noise=config.lim.fit_noise,
        center=config.lim.center_training,
        require_stable=config.lim.require_stable,
        covariance_floor_relative=config.lim.covariance_floor_relative,
    )
    max_lead = min(20, pcs.sizes["time"] - 1)
    if config.lim.segment_length_steps is not None:
        max_lead = min(max_lead, config.lim.segment_length_steps - 1)
    diagnostics = model.forecast_skill(pcs, max_lead=max_lead)
    diagnostics.attrs["evaluation_scope"] = (
        "in-sample training diagnostics, not held-out forecast skill"
    )
    model.save(directory, diagnostics)
    return (
        directory / "dynamics.nc",
        directory / "metadata.yml",
        directory / "training_metrics.nc",
    )


def _receipt_outputs(directory: Path) -> list[dict]:
    return json.loads((directory / "receipt.json").read_text(encoding="utf-8"))["outputs"]
