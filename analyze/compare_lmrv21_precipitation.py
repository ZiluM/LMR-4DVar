"""Download official LMR v2.1 prate and compare annual skill on shared coverage.

Run via PBS. References and annual aggregation match the existing quarterly
verification. LMR's original within-year averaging convention is not established
by its public catalog; calendar-year references are also checked as sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from verify_precipitation_products import (
    prepare_20cr,
    prepare_era5,
    prepare_era20c,
    prepare_gpcp,
    standard_grid,
)

from lmr4dvar.config import load_yaml_config
from lmr4dvar.verification.precipitation_annual import (
    annual_years,
    complete_quarter_years,
    verify_annual_pair,
)
from lmr4dvar.verification.precipitation_products import conservative_grid, monthly_to_quarters

URL = "https://atmos.washington.edu/~hakim/lmr/LMRv2/prate_MCruns_ensemble_mean_LMRv2.1.nc"


def load_lmr(path, target):
    with xr.open_dataset(path, decode_times=False, chunks={}) as source:
        if "prate" not in source:
            raise ValueError(f"official file lacks prate: {list(source.data_vars)}")
        field = source.prate
        units = field.attrs.get("units", "")
        if units not in ("kg/m^2/s", "kg/m2/s", "kg m-2 s-1", "kg m**-2 s**-1"):
            raise ValueError(f"unverified LMR precipitation units: {units!r}")
        if "since" in source.time.attrs.get("units", ""):
            decoded = xr.decode_cf(source[["prate"]], use_cftime=True)
            field = decoded.prate
        field = annual_years(field).sel(time=slice(1806, 2000))
        mc_dims = [dim for dim in field.dims if dim.lower() == "mcrun"]
        if len(mc_dims) != 1 or field.sizes[mc_dims[0]] != 20:
            raise ValueError(f"expected 20 Monte Carlo runs: {field.sizes}")
        mean = standard_grid(field.mean(mc_dims[0], skipna=False).compute()) * 86400
        mean = conservative_grid(mean, target)
        mean.attrs.update(
            units="mm/day",
            source=URL,
            reconstruction="20 MCrun grand mean",
            source_dataset_attribute=source.prate.attrs.get("dataset", ""),
            source_experiment=source.attrs.get("experiment", ""),
        )
        return mean


def calendar_annual(monthly):
    month_labels = np.asarray(monthly.time.dt.year) * 12 + np.asarray(monthly.time.dt.month)
    if np.any(np.diff(month_labels) <= 0):
        raise ValueError("calendar averaging requires unique increasing monthly labels")
    labels = monthly.time.dt.year
    counts = monthly.groupby(labels).count("time")
    annual = monthly.groupby(labels).mean("time", skipna=False).where(counts == 12)
    return annual.rename(year="time").dropna("time", how="all")


def plot_comparison(results, path, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for column, (name, (data, scores)) in enumerate(results.items()):
        axes[0, column].plot(data.time, data.reference_mean, color="black", label="Reference")
        axes[0, column].plot(data.time, data.reconstruction_mean, label=name)
        axes[0, column].set_title(
            f"{name}: mean-series r={scores['annual_mean']['correlation']:.3f}"
        )
        axes[0, column].set_ylabel("Annual anomaly (mm/day)")
        axes[0, column].legend()
        image = axes[1, column].pcolormesh(
            data.lon,
            data.lat,
            data.gridpoint_temporal_correlation,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            shading="auto",
        )
        axes[1, column].set_title(
            f"Temporal correlation: area-weighted R={scores['gridpoint_temporal_r']:.3f}"
        )
        axes[1, column].set_xlabel("Longitude")
        axes[1, column].set_ylabel("Latitude")
        fig.colorbar(image, ax=axes[1, column])
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    root = Path(config["output_root"]).parent / "lmrv21_precipitation"
    root.mkdir(parents=True, exist_ok=True)
    raw = root / Path(URL).name
    if not raw.exists():
        temporary = raw.with_suffix(".partial")
        print("Downloading official LMR v2.1 precipitation (~2.14 GB)", flush=True)
        digest = hashlib.sha256()
        size = 0
        with urlopen(URL, timeout=300) as response, temporary.open("wb") as output:
            expected = int(response.headers["Content-Length"])
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
            if size != expected:
                raise ValueError("incomplete LMR download")
        temporary.replace(raw)
        (root / "download_manifest.json").write_text(
            json.dumps(
                {
                    "url": URL,
                    "bytes": size,
                    "sha256": digest.hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
    target = xr.Dataset(
        coords={"lat": np.arange(-88.75, 90, 2.5), "lon": np.arange(1.25, 360, 2.5)}
    )
    lmr = load_lmr(raw, target)
    with xr.open_dataset(config["reconstruction"]) as source:
        ours = complete_quarter_years(source.pr.sel(time=slice(1806, 2000.75)).load())
    ours = conservative_grid(ours, target)
    ours.attrs["units"] = "mm/day"
    with xr.open_dataset(config["land_mask"]) as source:
        land = conservative_grid(standard_grid(source.lsm).load(), target)
    masks = {"global": xr.ones_like(land, dtype=bool), "land": land >= 0.5, "ocean": land < 0.5}
    readers = {
        "GPCPv2.3": prepare_gpcp,
        "20CRv3": prepare_20cr,
        "ERA5": prepare_era5,
        "ERA20C": prepare_era20c,
    }
    summary = {
        "baseline": config["baseline"],
        "source": URL,
        "lmr_estimate": "grand mean of 20 MCrun ensemble-mean fields",
        "lmr_source_metadata": dict(lmr.attrs),
        "fdvar_estimate": "single optimized state trajectory",
        "limitations": [
            "LMR annual within-year convention not established by catalog; "
            "4D-Var uses DJF-year quarters.",
            "Calendar reference sensitivity does not convert the 4D-Var "
            "trajectory into exact calendar years.",
            "Different priors, proxy subsets, and averaging prevent "
            "attributing skill differences to algorithm alone.",
        ],
        "comparisons": {},
    }
    rows = []
    for product, reader in readers.items():
        monthly = reader(config["products"][product], target, Path(config["cache_root"]))
        references = {
            "djf_year_reference": complete_quarter_years(monthly_to_quarters(monthly)),
            "calendar_year_reference_sensitivity": calendar_annual(monthly),
        }
        for protocol, reference in references.items():
            reference.attrs["units"] = "mm/day"
            for interval, period in {"common": [1980, 2000], "longest": config["period"]}.items():
                a, b, r = xr.align(ours, lmr, reference, join="inner")
                selection = (r.time >= period[0]) & (r.time <= period[1])
                a, b, r = (f.isel(time=selection) for f in (a, b, r))
                fixed = (np.isfinite(a) & np.isfinite(b) & np.isfinite(r)).all("time")
                key = f"{product}/{protocol}/{interval}"
                output = root / key
                output.mkdir(parents=True, exist_ok=True)
                regional = {}
                for region, mask in masks.items():
                    results = {}
                    for name, field in {"LMR4DVar": a, "LMRv2.1": b}.items():
                        data, scores = verify_annual_pair(
                            field, r, mask & fixed, baseline=config["baseline"]
                        )
                        data.attrs.update(reference=product, protocol=protocol, estimate=name)
                        data.to_netcdf(output / f"{region}_{name}_metrics.nc", engine="h5netcdf")
                        results[name] = (data, scores)
                        rows.append(
                            {
                                "reference": product,
                                "protocol": protocol,
                                "interval": interval,
                                "region": region,
                                "estimate": name,
                                **{k: v for k, v in scores.items() if k != "annual_mean"},
                                "mean_series_r": scores["annual_mean"]["correlation"],
                                "mean_series_rmse": scores["annual_mean"]["rmse"],
                            }
                        )
                    regional[region] = {name: value[1] for name, value in results.items()}
                    if region == "global" and interval == "longest":
                        plot_comparison(
                            results,
                            output / "global_comparison.png",
                            f"{product}: {int(r.time[0])}-{int(r.time[-1])}; {protocol}",
                        )
                summary["comparisons"][key] = regional
                (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(key, json.dumps(regional["global"]), flush=True)
    with (root / "comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["status"] = "complete"
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("LMR v2.1 comparison complete", flush=True)


if __name__ == "__main__":
    main()
