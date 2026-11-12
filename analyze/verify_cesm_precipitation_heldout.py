"""Evaluate a quarterly coupled prior on an excluded member without refitting it.

Run through PBS. Inputs are the prior experiment YAML, trained artifacts and raw
monthly simulations. Training-only preprocessing parameters are recovered from
the original monthly training inputs; EOF and LIM artifacts remain read-only.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

from lmr4dvar.config import load_project_config
from lmr4dvar.data.loader import load_climate_field, resolve_input_files, select_horizontal_field
from lmr4dvar.data.monthly_time import label_monthly_intervals
from lmr4dvar.data.precipitation import precipitation_mm_per_day
from lmr4dvar.data.preprocess import regrid_regular, select_domain, select_year_period
from lmr4dvar.data.seasonal_transform import SeasonalTransform
from lmr4dvar.eof import EOFModel
from lmr4dvar.lim import LIMModel
from lmr4dvar.verification.forecast import field_scores, forecast_pairs
from lmr4dvar.workflow.planner import config_fingerprint

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def prepare_monthly(config, name, source):
    variable = config.state.variables[name]
    if name == "pr":
        source = precipitation_mm_per_day(source, source_units=variable.source_units)
    source = select_year_period(source, config.preprocess.training_period)
    if config.preprocess.regrid.value != "none":
        source = regrid_regular(source, config.preprocess.target_grid_degrees)
    return select_domain(source, variable.domain)


def read_evaluation_block(path, name, variable, bounds_path, period):
    """Select full physical Dec-Nov years before validating monthly bounds."""
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with ExitStack() as stack:
        source = stack.enter_context(xr.open_dataset(path, decode_times=coder, engine="h5netcdf"))
        donor = (
            stack.enter_context(xr.open_dataset(bounds_path, decode_times=coder, engine="h5netcdf"))
            if bounds_path
            else source
        )
        bounds = donor[donor.time.attrs["bounds"]]
        bound_dim = next(dim for dim in bounds.dims if dim != "time")
        lower = bounds.isel({bound_dim: 0})
        labels = lower.dt.year * 12 + lower.dt.month - 1
        indices = np.flatnonzero(
            (labels.values >= period[0] * 12 - 1) & (labels.values <= period[1] * 12 + 10)
        )
        expected = 12 * (period[1] - period[0] + 1)
        if len(indices) != expected:
            raise ValueError(f"{name}: evaluation block lacks complete monthly coverage")
        selected_donor = donor.isel(time=indices)
        source_indices = source.indexes["time"].get_indexer(selected_donor.time.values)
        if np.any(source_indices < 0):
            raise ValueError("evaluation source and donor times do not match")
        source = source.isel(time=source_indices)
        corrected = label_monthly_intervals(
            source, bounds_source=selected_donor if bounds_path else None
        )
        return select_horizontal_field(corrected[variable.source_name or name]).load()


def quarter_labels(time):
    return np.asarray(time.dt.year) * 4 + (np.asarray(time.dt.month) - 1) // 3


def assemble(config, member, blocks, output):
    prior = config.paths.output_root / config.project.name / "stages"
    variables, labels, truth, models, provenance = [], None, None, {}, {}
    for name, variable in config.state.variables.items():
        print(f"Preparing training-only transform and held-out {name}", flush=True)
        root = variable.input_root or config.paths.simulation_root
        all_files = resolve_input_files(root, variable.input_pattern)
        held_out = [path for path in all_files if f".{member}." in path.name]
        training = resolve_input_files(
            root, variable.input_pattern, exclude_patterns=tuple(variable.exclude_input_patterns)
        )
        if len(held_out) != 1 or held_out[0] in training:
            raise ValueError(f"{name}: evaluation member must be explicitly excluded from training")
        donors = None
        held_donor = None
        if variable.monthly_time_bounds_pattern:
            donors = resolve_input_files(
                root,
                variable.monthly_time_bounds_pattern,
                exclude_patterns=tuple(variable.exclude_input_patterns),
            )
            held_donors = [
                path
                for path in resolve_input_files(root, variable.monthly_time_bounds_pattern)
                if f".{member}." in path.name
            ]
            if len(held_donors) != 1 or len(donors) != len(training):
                raise ValueError("monthly bounds member selection differs from source selection")
            held_donor = held_donors[0]
        signatures = [
            {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in (*training, *(donors or ()))
        ]
        provenance[name] = {"training": signatures, "held_out": str(held_out[0])}
        parameter_path = output / "preprocessing" / f"{name}.nc"
        parameter_manifest = parameter_path.with_suffix(".json")
        signature = {"config_sha256": config_fingerprint(config), "training": signatures}
        if (
            parameter_path.exists()
            and parameter_manifest.exists()
            and (json.loads(parameter_manifest.read_text()) == signature)
        ):
            transform = SeasonalTransform.load(parameter_path)
        else:
            fits = []
            for index, path in enumerate(training):
                source = load_climate_field(
                    (path,),
                    variable.source_name or name,
                    monthly_time_policy=variable.monthly_time_policy,
                    time_bounds_files=(donors[index],) if donors else None,
                    drop_partial_boundaries=variable.monthly_drop_partial_boundaries,
                    restrict_to_bounds_source=variable.monthly_restrict_to_bounds_source,
                )
                fits.append(
                    SeasonalTransform.fit(
                        prepare_monthly(config, name, source),
                        reference_year=config.preprocess.training_period[0],
                    )
                )
            transform = SeasonalTransform.ensemble_mean(fits)
            transform.save(parameter_path)
            parameter_manifest.write_text(json.dumps(signature, indent=2) + "\n")
        fields = []
        for block in blocks:
            source = read_evaluation_block(held_out[0], name, variable, held_donor, block)
            field = transform.apply(prepare_monthly(config, name, source))
            expected = np.arange(block[0] * 4, (block[1] + 1) * 4)
            if not np.array_equal(quarter_labels(field.time), expected):
                raise ValueError(f"{name}: evaluation quarterly block does not match request")
            fields.append(field)
        field = xr.concat(fields, dim="time")
        current_labels = quarter_labels(field.time)
        if labels is not None and not np.array_equal(current_labels, labels):
            raise ValueError(f"{name}: held-out physical times differ across variables")
        labels = current_labels
        model = EOFModel.load(prior / "eof" / name)
        models[name] = model
        variables.append(model.encode(field).values)
        if name == "pr":
            truth = field.where(model.valid_mask).transpose("time", "lat", "lon")
    if truth is None:
        raise ValueError("precipitation EOF and truth are required")
    state = np.concatenate(variables, axis=1)
    lim = LIMModel.load(prior / "train-lim")
    expected_labels = [
        f"{name}:{index:03d}"
        for name, item in config.state.variables.items()
        for index in range(item.n_modes)
    ]
    if expected_labels != list(lim.training_mean.state.values):
        raise ValueError("held-out state order differs from trained LIM")
    dataset = xr.Dataset(
        {"pcs": (("sample", "state"), state), "physical_quarter": ("sample", labels)},
        coords={"state": expected_labels},
        attrs={"held_out_member": member},
    )
    dataset.to_netcdf(output / "held_out_state.nc", engine="h5netcdf")
    return state, truth, labels, models["pr"], lim, provenance


def evaluate(state, truth, labels, model, lim, pr_slice):
    methods = ["LIM", "persistence", "zero_anomaly", "EOF_ceiling"]
    seasons = ["all", "DJF", "MAM", "JJA", "SON"]
    shape = (4, 4, 5, truth.sizes["lat"], truth.sizes["lon"])
    scores = {
        name: np.full(shape, np.nan) for name in ("correlation", "rmse", "ce", "sample_count")
    }
    for lead in range(1, 5):
        starts, ends = forecast_pairs(labels, lead)
        target = truth.values[ends]
        predictions = [
            model.decode(
                xr.DataArray(
                    lim.forecast(state[starts], lead)[:, pr_slice],
                    dims=("time", "mode"),
                    coords={"time": np.arange(len(starts)), "mode": model.components.mode},
                )
            ).values,
            truth.values[starts],
            np.zeros_like(target),
            model.decode(
                xr.DataArray(
                    state[ends, pr_slice],
                    dims=("time", "mode"),
                    coords={"time": np.arange(len(ends)), "mode": model.components.mode},
                )
            ).values,
        ]
        for method_index, prediction in enumerate(predictions):
            for season_index in range(5):
                selected = (
                    np.ones(len(ends), dtype=bool)
                    if season_index == 0
                    else (labels[ends] % 4 == season_index - 1)
                )
                for name, values in field_scores(prediction[selected], target[selected]).items():
                    scores[name][method_index, lead - 1, season_index] = values
        print(f"Finished {3 * lead}-month forecast scores ({len(starts)} pairs)", flush=True)
    return xr.Dataset(
        {
            name: (("method", "lead_months", "season", "lat", "lon"), values)
            for name, values in scores.items()
        },
        coords={
            "method": methods,
            "lead_months": [3, 6, 9, 12],
            "season": seasons,
            "lat": truth.lat,
            "lon": truth.lon,
        },
        attrs={
            "evaluation_scope": "excluded member, full anomaly truth on training EOF valid mask",
            "preprocessing": "training-only ensemble monthly climatology and quarterly trends",
            "pair_policy": "exclude physical time gaps at every intervening quarter",
            "rmse_units": "mm/day",
            "EOF_ceiling": "target-time projection, not a forecast",
        },
    )


def plot_scores(dataset, output):
    for metric in ("correlation", "rmse"):
        fig, axes = plt.subplots(4, 3, figsize=(13, 9), sharex=True, sharey=True)
        for row, lead in enumerate(dataset.lead_months.values):
            for col, method in enumerate(("LIM", "persistence", "EOF_ceiling")):
                values = dataset[metric].sel(
                    {"method": method, "lead_months": lead, "season": "all"}
                )
                kwargs = (
                    {"vmin": -1, "vmax": 1, "cmap": "RdBu_r"}
                    if metric == "correlation"
                    else {"vmin": 0, "vmax": 2, "cmap": "viridis"}
                )
                mesh = axes[row, col].pcolormesh(dataset.lon, dataset.lat, values, **kwargs)
                axes[row, col].set_title(f"{method}, {lead} months", fontsize=10)
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label=metric, shrink=0.7)
        fig.suptitle(f"Held-out CESM precipitation {metric}")
        fig.savefig(output / f"pr_{metric}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--member", default="011")
    parser.add_argument("--blocks", nargs="+", default=["851:1229", "1231:1849"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_project_config(args.config)
    if config.preprocess.detrend.value != "linear_by_month" or (
        not config.preprocess.remove_monthly_climatology or config.state.timestep_months != 3
    ):
        raise ValueError("this evaluation requires the quarterly training preprocessing contract")
    blocks = [tuple(int(year) for year in block.split(":")) for block in args.blocks]
    if any(len(block) != 2 or block[0] > block[1] for block in blocks):
        raise ValueError("blocks must be increasing start:end years")
    args.output.mkdir(parents=True, exist_ok=True)
    state, truth, labels, model, lim, provenance = assemble(
        config, args.member, blocks, args.output
    )
    offset = 0
    for name, item in config.state.variables.items():
        if name == "pr":
            pr_slice = slice(offset, offset + item.n_modes)
        offset += item.n_modes
    dataset = evaluate(state, truth, labels, model, lim, pr_slice)
    dataset.attrs["held_out_member"] = args.member
    dataset.to_netcdf(args.output / "forecast_skill.nc", engine="h5netcdf")
    weights = np.cos(np.deg2rad(dataset.lat))
    summary = dataset[["correlation", "rmse", "ce"]].weighted(weights).mean(("lat", "lon"))
    summary.to_netcdf(args.output / "area_weighted_gridpoint_skill.nc", engine="h5netcdf")
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "member": args.member,
                "blocks": blocks,
                "inputs": provenance,
                "land_scores": "not computed: no land mask supplied",
                "scope": "all EOF-valid gridpoints; weighted summary is a mean of gridpoint scores",
            },
            indent=2,
        )
        + "\n"
    )
    plot_scores(dataset, args.output)


if __name__ == "__main__":
    main()
