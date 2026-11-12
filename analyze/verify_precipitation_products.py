"""Verify an existing reconstruction against four global precipitation products.

Run on a PBS compute node with analyze/precipitation_products.yml. Raw public
archives are read-only; reduced monthly fields, figures and metrics are external.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lmr4dvar.config import load_yaml_config
from lmr4dvar.verification.precipitation_products import (
    conservative_grid,
    era5_interval_midpoints,
    monthly_to_quarters,
    temporal_map_title,
    verify_product,
)


def standard_grid(field):
    rename = {
        key: value
        for key, value in (("latitude", "lat"), ("longitude", "lon"))
        if key in field.dims
    }
    return field.rename(rename).assign_coords(lon=lambda a: a.lon % 360).sortby("lon").sortby("lat")


def save_monthly(field, path):
    field = field.rename("pr")
    field.attrs["units"] = "mm/day"
    field.to_dataset().to_netcdf(path, engine="h5netcdf")


def prepare_era5(config, target, cache):
    outputs = []
    first, last = config.get("input_years", [1979, 2000])
    if first < 1979:
        outputs.append(prepare_era5_hourly(config, target, cache, first, 1978))
    for year in range(max(first, 1979), last + 1):
        path = cache / f"era5_{year}.nc"
        if not path.exists():
            candidates = list(Path(config["root"]).glob(f"{year}/*mtpr*.nc"))
            if len(candidates) != 1:
                raise ValueError(f"expected one ERA5 monthly MTPR file for {year}")
            with xr.open_dataset(candidates[0]) as source:
                field = standard_grid(source.MTPR)
                if field.attrs["units"] != "kg m**-2 s**-1":
                    raise ValueError("unexpected ERA5 precipitation units")
                field = conservative_grid(field.load(), target) * 86400
                field.attrs["source_path"] = str(candidates[0])
                field.attrs["conversion"] = "MTPR kg m-2 s-1 multiplied by 86400"
                save_monthly(field, path)
            print(f"Prepared ERA5 {year}", flush=True)
        with xr.open_dataset(path) as dataset:
            outputs.append(dataset.pr.load())
    return xr.concat(outputs, dim="time")


def prepare_era5_hourly(config, target, cache, first, last):
    """Reduce native hourly chunks before spatial remapping, keeping month boundaries."""
    outputs = []
    for year in range(first, last + 1):
        path = cache / f"era5_hourly_month_sums_{year}.nc"
        if not path.exists():
            totals, counts = {}, {}
            for month in range(1, 13):
                files = sorted(Path(config["hourly_root"]).glob(f"{year}{month:02d}/*mtpr*.nc"))
                if not files:
                    raise FileNotFoundError(f"ERA5 hourly MTPR {year}-{month:02d}")
                previous_midpoint = None
                for source_path in files:
                    with xr.open_dataset(
                        source_path, chunks={"forecast_initial_time": 1}
                    ) as source:
                        field = standard_grid(source.MTPR)
                        if field.attrs["units"] != "kg m**-2 s**-1":
                            raise ValueError("unexpected ERA5 hourly precipitation units")
                        midpoints = era5_interval_midpoints(
                            source.forecast_initial_time, source.forecast_hour
                        ).reshape(-1)
                        if previous_midpoint is not None and (
                            midpoints[0] - previous_midpoint != np.timedelta64(1, "h")
                        ):
                            raise ValueError("ERA5 hourly files overlap or have a gap")
                        previous_midpoint = midpoints[-1]
                        months = midpoints.astype("datetime64[M]")
                        hourly = field.stack(sample=("forecast_initial_time", "forecast_hour"))
                        for label in np.unique(months):
                            selected = np.flatnonzero(months == label)
                            reduced = hourly.isel(sample=selected).sum(
                                "sample", skipna=False, dtype=np.float64
                            )
                            reduced = conservative_grid(reduced.compute(), target) * 86400
                            key = pd.Timestamp(label)
                            totals[key] = totals.get(key, 0) + reduced
                            counts[key] = counts.get(key, 0) + len(selected)
            labels = sorted(totals)
            amount = xr.concat(
                [totals[label].expand_dims(time=[label]) for label in labels], dim="time"
            ).rename("rate_sum")
            dataset = amount.to_dataset()
            dataset["hour_count"] = ("time", [counts[label] for label in labels])
            dataset.attrs.update(
                source_root=config["hourly_root"],
                method="MTPR hourly interval midpoints; time sum before conservative remapping",
            )
            dataset.to_netcdf(path, engine="h5netcdf")
            print(f"Prepared ERA5 hourly year {year}", flush=True)
        with xr.open_dataset(path) as source:
            outputs.append(source.load())
    combined = xr.concat(outputs, dim="time").sortby("time")
    rates = combined.rate_sum.groupby("time").sum("time", skipna=False)
    counts = combined.hour_count.groupby("time").sum("time")
    years = rates.time.dt.year
    inside = (years >= first) & (years <= last)
    rates, counts = rates.isel(time=inside), counts.isel(time=inside)
    expected = rates.time.dt.days_in_month * 24
    complete = counts == expected
    incomplete = rates.time.values[~complete.values]
    # ERA5 begins on 1940-01-01 06UTC: the first month lacks its first six hours.
    if any(pd.Timestamp(value) != pd.Timestamp("1940-01-01") for value in incomplete):
        raise ValueError(f"incomplete ERA5 hourly calendar months: {incomplete}")
    result = (rates / counts).where(complete, drop=True)
    result.attrs.update(
        units="mm/day",
        source_root=config["hourly_root"],
        method="exact calendar monthly mean of complete hourly intervals",
    )
    return result


def prepare_gpcp(config, target, cache):
    path = cache / "gpcp_v23_monthly.nc"
    if not path.exists():
        fields = []
        for year in range(1979, 2001):
            for month in range(1, 13):
                source_path = Path(config["root"]) / f"gpcp_cdr_v23rB1_y{year}_m{month:02d}.nc"
                with xr.open_dataset(source_path) as source:
                    if source.attrs.get("product_version") != "v23rB1":
                        raise ValueError("expected GPCP v2.3")
                    field = (
                        source.precip.rename(nlat="lat", nlon="lon")
                        .assign_coords(lat=source.latitude.values, lon=source.longitude.values)
                        .drop_vars(["latitude", "longitude"], errors="ignore")
                    )
                    if field.attrs["units"] not in ("mm/day", "millimeters/day"):
                        raise ValueError("unexpected GPCP precipitation units")
                    field = conservative_grid(field.load(), target)
                    fields.append(
                        field.expand_dims(time=[pd.Timestamp(year=year, month=month, day=1)])
                    )
        field = xr.concat(fields, dim="time")
        field.attrs["source_path"] = config["root"]
        field.attrs["evaluation_scope"] = (
            "satellite-gauge; land incorporates GPCC; not fully independent"
        )
        save_monthly(field, path)
    with xr.open_dataset(path) as source:
        return source.pr.load()


def prepare_20cr(config, target, cache):
    source_key = hashlib.sha256(config["url"].encode()).hexdigest()[:12]
    raw = cache / f"20cr_v3_{source_key}_apcp_monthly_raw.nc"
    if not raw.exists():
        temporary = raw.with_suffix(".partial")
        with urlopen(config["url"], timeout=300) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        with xr.open_dataset(temporary) as check:
            if "apcp" not in check or check.apcp.attrs.get("units") != "kg/m^2":
                raise ValueError("unexpected 20CR monthly accumulation product")
        temporary.rename(raw)
    path = cache / f"20cr_v3_{source_key}_monthly.nc"
    if not path.exists():
        with xr.open_dataset(raw) as source:
            # Monthly means of 3-hourly accumulation, not monthly total.
            field = conservative_grid(standard_grid(source.apcp).load(), target) * 8
            field.attrs["source_url"] = config["url"]
            field.attrs["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
            field.attrs["conversion"] = "mean 3-hour precipitation kg/m2 multiplied by 8 -> mm/day"
            save_monthly(field, path)
    with xr.open_dataset(path) as source:
        return source.pr.load()


def daily_accumulation(values, step):
    """Use complete nonnegative 24-hour TP, never clip bad incremental rainfall."""
    if step != 24:
        raise ValueError("ERA-20C daily total requires the 24-hour forecast step")
    values = np.asarray(values)
    if not np.all(np.isfinite(values)) or np.min(values) < 0:
        raise ValueError("missing or negative ERA-20C daily accumulation")
    return values * 1000


def prepare_era20c(config, target, cache):
    from eccodes import (
        codes_get,
        codes_get_array,
        codes_get_values,
        codes_grib_new_from_file,
        codes_release,
    )

    outputs = []
    first, last = config.get("input_years", [1979, 2000])
    for year in range(first, last + 1):
        path = cache / f"era20c_daily24_{year}.nc"
        if not path.exists():
            decade = f"{year // 10 * 10}_{year // 10 * 10 + 9}"
            candidates = list(Path(config["root"]).glob(f"{decade}/*128_228_tp*{year}010109*"))
            if len(candidates) != 1:
                raise ValueError(f"expected one ERA-20C annual TP file for {year}")
            totals, counts = {}, {}
            previous_init, previous_step = None, 0
            seen_dates = set()
            coordinates = None
            with candidates[0].open("rb") as source:
                while (message := codes_grib_new_from_file(source)) is not None:
                    try:
                        if (
                            codes_get(message, "shortName") != "tp"
                            or codes_get(message, "units") != "m"
                        ):
                            raise ValueError("unexpected ERA-20C GRIB variable/units")
                        date = codes_get(message, "dataDate")
                        hour = codes_get(message, "dataTime")
                        if hour != 600:
                            raise ValueError("expected ERA-20C daily 06 UTC initialization")
                        step = int(codes_get(message, "endStep"))
                        if date != previous_init:
                            if previous_init is not None and previous_step != 24:
                                raise ValueError("incomplete ERA-20C forecast cycle")
                            if date in seen_dates:
                                raise ValueError("duplicate ERA-20C forecast initialization")
                            seen_dates.add(date)
                            previous_step = 0
                        if step != previous_step + 3 or step > 24:
                            raise ValueError("incomplete ERA-20C forecast step sequence")
                        previous_init, previous_step = date, step
                        if step != 24:
                            continue
                        values = codes_get_values(message).reshape(
                            (codes_get(message, "Nj"), codes_get(message, "Ni"))
                        )
                        if codes_get(message, "numberOfMissing"):
                            raise ValueError("ERA-20C GRIB contains missing gridpoints")
                        amount = daily_accumulation(values, step)
                        if coordinates is None:
                            lat = codes_get_array(message, "latitudes").reshape(values.shape)[:, 0]
                            lon = codes_get_array(message, "longitudes").reshape(values.shape)[0]
                            coordinates = {"lat": lat, "lon": lon}
                        midpoint = datetime.strptime(str(date), "%Y%m%d") + timedelta(hours=18)
                        key = (midpoint.year, midpoint.month)
                        if key not in totals:
                            totals[key] = np.zeros(values.shape)
                            counts[key] = 0
                        totals[key] += amount
                        counts[key] += 1
                    finally:
                        codes_release(message)
            if previous_step != 24:
                raise ValueError("incomplete final ERA-20C forecast cycle")
            fields = []
            for (y, m), values in sorted(totals.items()):
                field = conservative_grid(
                    xr.DataArray(values, dims=("lat", "lon"), coords=coordinates), target
                )
                fields.append(field.expand_dims(time=[pd.Timestamp(year=y, month=m, day=1)]))
            result = xr.concat(fields, dim="time").rename("precipitation_amount")
            dataset = result.to_dataset()
            dataset["day_count"] = ("time", [counts[key] for key in sorted(totals)])
            dataset.attrs["source_path"] = str(candidates[0])
            dataset.attrs["conversion"] = (
                "24h TP at step24 of daily 06UTC forecast; meters*1000; midpoint month"
            )
            dataset.attrs["temporal_sampling"] = (
                "06UTC-06UTC days; calendar month boundary shifted by 6 hours"
            )
            dataset.to_netcdf(path, engine="h5netcdf")
            print(f"Prepared ERA-20C {year}", flush=True)
        with xr.open_dataset(path) as source:
            outputs.append(source.load())
    combined = xr.concat(outputs, dim="time").sortby("time")
    amounts = combined.precipitation_amount.groupby("time").sum("time", skipna=False)
    counts = combined.day_count.groupby("time").sum("time")
    expected = amounts.time.dt.days_in_month
    complete = counts == expected
    if not bool(complete.all()):
        raise ValueError("ERA-20C monthly daily totals have incomplete coverage")
    amounts = amounts.where(complete, drop=True)
    field = amounts / amounts.time.dt.days_in_month
    field.attrs.update(
        units="mm/day",
        conversion=combined.attrs["conversion"],
        source_path=config["root"],
        temporal_sampling=combined.attrs["temporal_sampling"],
    )
    return field


def plot_product(product, results, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for row, region in enumerate(("global", "land", "ocean")):
        dataset, summary = results[region]
        axes[row, 0].plot(dataset.year, dataset.reference_annual_mean, color="black", label=product)
        axes[row, 0].plot(
            dataset.year,
            dataset.reconstruction_annual_mean,
            color="#007c91",
            label="Reconstruction",
        )
        metrics = summary["annual_mean"]
        axes[row, 0].set_title(
            f"{region}: mean r={metrics['correlation']:.2f}, RMSE={metrics['rmse']:.3f} mm/day"
        )
        axes[row, 0].set_ylabel("Precipitation anomaly (mm/day)")
        axes[row, 1].plot(dataset.year, dataset.annual_pattern_r, color="#007c91")
        axes[row, 1].axhline(0, color="gray", linewidth=0.7)
        axes[row, 1].set_ylim(-1, 1)
        axes[row, 1].set_title(
            f"{region}: mean spatial pattern r={summary['annual_spatial_pattern_r']:.2f}"
        )
        for axis in axes[row]:
            axis.set_xlabel("Year")
            axis.grid(alpha=0.25)
    axes[0, 0].legend()
    example = results["global"][0]
    fig.suptitle(
        f"{product}: {example.attrs['annual_period']}, baseline {example.attrs['anomaly_period']}"
    )
    fig.savefig(directory / "mean_and_pattern.png", dpi=150)
    plt.close(fig)
    for aggregation in ("annual", "quarterly", "DJF", "MAM", "JJA", "SON"):
        plot_temporal_maps(product, results, directory, aggregation)


def plot_temporal_maps(product, results, directory, aggregation):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for column, region in enumerate(("global", "land", "ocean")):
        dataset, summary = results[region]
        for row, (variable, title, lower) in enumerate(
            (("gridpoint_temporal_correlation", "Temporal r", -1), ("ce", "CE", -1))
        ):
            field = dataset[variable].sel(aggregation=aggregation)
            artist = axes[row, column].pcolormesh(
                field.lon, field.lat, field, cmap="RdBu_r", vmin=lower, vmax=1, shading="auto"
            )
            caption = (
                temporal_map_title(region, summary[aggregation]["gridpoint_temporal_r"])
                if row == 0
                else f"{region}: {title}"
            )
            axes[row, column].set_title(caption)
            axes[row, column].set_xlabel("Longitude")
            axes[row, column].set_ylabel("Latitude")
        fig.colorbar(artist, ax=axes[:, column], shrink=0.8, extend="min")
    example = results["global"][0]
    actual_times = results["global"][1][aggregation]["time_range"]
    period = f"{int(np.floor(actual_times[0]))}-{int(np.floor(actual_times[1]))}"
    fig.suptitle(
        f"{aggregation} precipitation vs {product}: {period}; "
        f"baseline {example.attrs['anomaly_period']}\nCE below -1 clipped in display only"
    )
    fig.savefig(directory / f"{aggregation}_maps.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--product", choices=("GPCPv2.3", "20CRv3", "ERA5", "ERA20C"))
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    directory = Path(config["output_root"])
    directory.mkdir(parents=True, exist_ok=True)
    cache = Path(config.get("cache_root", directory / "prepared"))
    cache.mkdir(parents=True, exist_ok=True)
    target = xr.Dataset(
        coords={"lat": np.arange(-88.75, 90, 2.5), "lon": np.arange(1.25, 360, 2.5)}
    )
    with xr.open_dataset(config["reconstruction"]) as source:
        first, last = config["period"]
        first = min(first, config["baseline"][0])
        last = max(last, config["baseline"][1])
        field = source.pr.sel(time=slice(first, last + 0.75)).load()
    reconstructed = conservative_grid(field, target)
    with xr.open_dataset(config["land_mask"]) as source:
        land_fraction = conservative_grid(standard_grid(source.lsm).load(), target)
    masks = {
        "global": xr.ones_like(land_fraction, dtype=bool),
        "land": land_fraction >= 0.5,
        "ocean": land_fraction < 0.5,
    }
    land_fraction.rename("land_fraction").to_netcdf(
        directory / f"land_fraction_{args.product or 'all'}.nc", engine="h5netcdf"
    )
    readers = {
        "GPCPv2.3": prepare_gpcp,
        "20CRv3": prepare_20cr,
        "ERA5": prepare_era5,
        "ERA20C": prepare_era20c,
    }
    summary = {
        "baseline": config["baseline"],
        "period": config["period"],
        "grid": "global conservative 2.5 degrees",
        "land_definition": "20CR invariant area fraction >=0.5; same mask for all products",
        "products": {},
        "expected_products": list(readers),
    }
    for name, reader in readers.items():
        if args.product and name != args.product:
            continue
        try:
            print(f"Starting {name}", flush=True)
            monthly = reader(config["products"][name], target, cache)
            reference = monthly_to_quarters(monthly)
            results = {}
            for region, mask in masks.items():
                dataset, metrics = verify_product(
                    reconstructed,
                    reference,
                    mask,
                    baseline=config["baseline"],
                    period=config["period"],
                )
                dataset.attrs.update(
                    reference_product=name,
                    reference_type=config["products"][name]["type"],
                    evaluation_scope=config["products"][name]["scope"],
                    region=region,
                )
                product_dir = directory / name
                product_dir.mkdir(exist_ok=True)
                dataset.to_netcdf(product_dir / f"{region}_metrics.nc", engine="h5netcdf")
                results[region] = (dataset, metrics)
            plot_product(name, results, directory / name / "figures")
            summary["products"][name] = {
                "status": "complete",
                "scope": config["products"][name]["scope"],
                "annual_period": results["global"][1]["annual_period"],
                "quarterly_time_range": results["global"][1]["quarterly_time_range"],
                "regions": {region: item[1] for region, item in results.items()},
            }
            print(json.dumps(summary["products"][name]), flush=True)
        except Exception as error:
            import traceback

            traceback.print_exc()
            summary["products"][name] = {"status": "failed", "error": str(error)}
        product_summary = directory / f"summary_{name}.json"
        temporary = product_summary.with_suffix(".partial")
        temporary.write_text(json.dumps(summary["products"][name], indent=2) + "\n")
        temporary.replace(product_summary)
        aggregate_summaries(directory, summary)
    if any(item["status"] != "complete" for item in summary["products"].values()):
        raise SystemExit("one or more reference products failed; see summary.json")


def aggregate_summaries(directory, metadata):
    """Independent PBS jobs publish per-product files; serialize the combined view."""
    with (directory / "summary.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = {key: value for key, value in metadata.items() if key != "products"}
        result["products"] = {name: {"status": "pending"} for name in metadata["expected_products"]}
        result["products"].update(
            {
                path.stem.removeprefix("summary_"): json.loads(path.read_text())
                for path in sorted(directory.glob("summary_*.json"))
            }
        )
        temporary = directory / "summary.partial"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(directory / "summary.json")
        with (directory / "temporal_correlation_summary.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "product",
                    "region",
                    "aggregation",
                    "annual_period",
                    "area_weighted_temporal_R",
                    "time_samples",
                ]
            )
            for product, item in result["products"].items():
                if item["status"] != "complete":
                    continue
                for region, scores in item["regions"].items():
                    for aggregation in ("annual", "quarterly", "DJF", "MAM", "JJA", "SON"):
                        writer.writerow(
                            [
                                product,
                                region,
                                aggregation,
                                item["annual_period"],
                                scores[aggregation]["gridpoint_temporal_r"],
                                scores[aggregation]["time_samples"],
                            ]
                        )


if __name__ == "__main__":
    main()
