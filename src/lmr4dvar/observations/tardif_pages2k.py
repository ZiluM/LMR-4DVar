"""Raw-instrumental PAGES2k calibration with a quarterly Tardif TRW PSM."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lmr4dvar.data.precipitation import precipitation_mm_per_day
from lmr4dvar.data.preprocess import quarterly_mean

from .bivariate_calibration import monthly_season_mean, select_bivariate_seasons
from .calibrated_artifact import load_calibrated_observations, save_calibrated_observations
from .pages2k_calibration import (
    ANNUAL_SEASON,
    QUARTER_SEASONS,
    _load_source_database,
    _normalize_grid,
)
from .quarterly_seasons import complete_quarter_offsets
from .records import ObservationCollection, ObservationRecord


def nearest_instrumental_site(field, latitude, longitude):
    """Nearest great-circle grid center; do not search for the nearest valid site."""
    lat = np.deg2rad(field.lat.values)[:, None]
    lon = np.deg2rad(field.lon.values)[None, :]
    site_lat, site_lon = np.deg2rad([latitude, longitude])
    distance = (
        np.sin((lat - site_lat) / 2) ** 2
        + np.cos(lat) * np.cos(site_lat) * np.sin((lon - site_lon) / 2) ** 2
    )
    i, j = np.unravel_index(np.argmin(distance), distance.shape)
    return field.isel(lat=i, lon=j).load()


def site_anomaly(series, period):
    baseline = series.where(
        (series.time.dt.year >= period[0]) & (series.time.dt.year <= period[1]), drop=True
    )
    if not baseline.sizes["time"] or len(np.unique(baseline.time.dt.month)) != 12:
        raise ValueError("instrumental anomaly baseline must include all calendar months")
    climatology = baseline.groupby("time.month").mean("time")
    return series.groupby("time.month") - climatology


def _candidate_seasons(configured, proxy, *, hemisphere_filter=True):
    candidates = [tuple(s) for s in configured]
    if hemisphere_filter:
        opposite = (
            {(-9, -10, -11, -12, 1, 2), (-12, 1, 2, 3, 4, 5)}
            if float(proxy.lat) >= 0
            else {(3, 4, 5, 6, 7, 8), (6, 7, 8, 9, 10, 11)}
        )
        candidates = [season for season in candidates if season not in opposite]
    metadata_season = getattr(proxy, "seasonality", None)
    if isinstance(metadata_season, (list, tuple, np.ndarray)):
        try:
            if any(
                not isinstance(m, (int, np.integer, float, np.floating)) or m != int(m)
                for m in metadata_season
            ):
                raise ValueError("noninteger metadata season")
            candidate = tuple(int(m) for m in metadata_season)
            complete_quarter_offsets(candidate)
        except (ValueError, TypeError):
            pass
        else:
            candidates.insert(0, candidate)
    return tuple(dict.fromkeys(candidates))


def _fit_univariate(proxy, temperature, candidates, period, minimum_overlap, *, width=None):
    import statsmodels.formula.api as smf

    results = []
    years = np.arange(
        period[0] - (int(np.ceil(width)) if width else 0) - 1,
        period[1] + (int(np.ceil(width)) if width else 0) + 2,
    )
    for season in candidates:
        climate = monthly_season_mean(temperature, season, years)
        if width is None:
            frame = pd.concat(
                [proxy.to_series().rename("y"), climate.to_series().rename("Temperature")], axis=1
            )
        else:
            values = strict_window_mean(climate.values, years, proxy.year.values, width)
            frame = pd.concat(
                [
                    proxy.to_series().rename("y"),
                    pd.Series(values, index=proxy.year.values, name="Temperature"),
                ],
                axis=1,
            )
        frame = frame.sort_index()
        frame = frame.loc[(frame.index >= period[0]) & (frame.index <= period[1])]
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < minimum_overlap:
            continue
        fit = smf.ols("y ~ Temperature", data=frame).fit()
        if np.linalg.matrix_rank(fit.model.exog) != 2 or not np.isfinite(fit.rsquared_adj):
            continue
        results.append((fit, season, frame))
    if not results:
        raise ValueError("no temperature season has sufficient full-rank calibration overlap")
    return max(results, key=lambda item: item[0].rsquared_adj)


def strict_window_mean(values, years, target_time, width):
    """Indexed interval means; missing values outside a window cannot taint it."""
    values, years, target_time = map(np.asarray, (values, years, target_time))
    lower, upper = target_time - width / 2, target_time + width / 2
    left = np.searchsorted(years, lower, side="left")
    right = np.searchsorted(years, upper, side="right")
    finite = np.isfinite(values)
    sums = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0))])
    finite_prefix = np.concatenate([[0], np.cumsum(finite)])
    counts = right - left
    valid_counts = finite_prefix[right] - finite_prefix[left]
    valid = (
        (counts > 0)
        & (valid_counts == counts)
        & (np.ceil(lower) >= years[0])
        & (np.floor(upper) <= years[-1])
    )
    result = np.full(target_time.shape, np.nan)
    result[valid] = (sums[right[valid]] - sums[left[valid]]) / counts[valid]
    return result


def _prepared_records(database):
    for pid, proxy in database.records.items():
        ptype = str(proxy.ptype).lower()
        if ptype == "borehole":
            continue
        if ptype.startswith("coral") and float(proxy.dt) < 1:
            for season in QUARTER_SEASONS:
                prepared = proxy.annualize(months=season)
                yield f"{pid}_q{season[1]}", prepared, (tuple(season),)
        elif float(proxy.dt) <= 1:
            yield str(pid), proxy.annualize(months=ANNUAL_SEASON), None
        else:
            yield str(pid), proxy, (tuple(ANNUAL_SEASON),)


def calibrate_tardif_pages2k(calibration, output_directory):
    """Calibrate without EOF projection; write every accepted/skipped record to audit."""
    import cfr

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    database = _load_source_database(calibration, cfr)
    matched_ids = None
    if calibration.matched_records_path is not None:
        matched_ids = {
            record.record_id
            for record in load_calibrated_observations(calibration.matched_records_path)
        }
        if not matched_ids:
            raise ValueError("matched-record calibration artifact is empty")
    records, diagnostics, candidate_diagnostics = [], [], []
    with ExitStack() as stack:
        t_source = stack.enter_context(xr.open_dataset(calibration.instrumental_path))
        p_source = stack.enter_context(xr.open_dataset(calibration.precipitation_path))
        temperature = _normalize_grid(t_source[calibration.instrumental_variable])
        precipitation = _normalize_grid(p_source[calibration.precipitation_variable])
        if temperature.attrs.get("units") not in {"K", "degC", "degree_Celsius", "degrees Celsius"}:
            raise ValueError("GISTEMP temperature must declare K or Celsius units")
        for pid, proxy, fixed_seasons in _prepared_records(database):
            if matched_ids is not None and pid not in matched_ids:
                diagnostics.append(
                    {"record_id": pid, "accepted": False, "reason": "outside matched network"}
                )
                continue
            try:
                ptype = str(proxy.ptype).lower()
                is_width = ptype == "tree.trw"
                is_bivariate = (
                    is_width and calibration.tree_ring_width_psm == "temperature_precipitation"
                )
                time, value = (
                    np.asarray(proxy.time, dtype=float),
                    np.asarray(proxy.value, dtype=float),
                )
                width = float(proxy.dt) if float(proxy.dt) > 1 else None
                if width is not None and is_width:
                    raise ValueError("TRW bivariate calibration requires annual records")
                if not np.isfinite(time).all() or len(set(time)) != time.size:
                    raise ValueError("proxy years must be finite and unique")
                if width is None and not np.allclose(time, np.rint(time), atol=1e-6, rtol=0):
                    raise ValueError("annualized proxy must have integer year labels")
                proxy_series = xr.DataArray(value, dims="year", coords={"year": time})
                period = (
                    calibration.low_resolution_period if width else calibration.calibration_period
                )
                selected = proxy_series.where(
                    (proxy_series.year >= period[0]) & (proxy_series.year <= period[1]), drop=True
                )
                site_t = nearest_instrumental_site(temperature, float(proxy.lat), float(proxy.lon))
                t = site_anomaly(site_t, calibration.anomaly_period)
                t_seasons = fixed_seasons or _candidate_seasons(
                    calibration.season_candidates,
                    proxy,
                    hemisphere_filter=calibration.hemisphere_season_filter,
                )
                terms = []
                if is_bivariate:
                    site_p = nearest_instrumental_site(
                        precipitation, float(proxy.lat), float(proxy.lon)
                    )
                    p = site_anomaly(precipitation_mm_per_day(site_p), calibration.anomaly_period)
                    p_seasons = _candidate_seasons(
                        calibration.precipitation_season_candidates
                        or calibration.season_candidates,
                        proxy,
                        hemisphere_filter=calibration.hemisphere_season_filter,
                    )
                    result, fits = select_bivariate_seasons(
                        selected,
                        t,
                        p,
                        t_seasons,
                        p_seasons,
                        minimum_overlap=calibration.minimum_overlap,
                    )
                    intercept, variance = result.intercept, result.residual_variance
                    r2, adjusted, bic, fit_years = (
                        result.r_squared,
                        result.adjusted_r_squared,
                        result.bic,
                        result.years,
                    )
                    terms = [
                        {
                            "variable": "tas",
                            "slope": result.slope_temperature,
                            "seasonality": list(result.temperature_months),
                            "units": "K",
                        },
                        {
                            "variable": "pr",
                            "slope": result.slope_precipitation,
                            "seasonality": list(result.precipitation_months),
                            "units": "mm/day",
                        },
                    ]
                    for fit in fits:
                        candidate_diagnostics.append(
                            {
                                "record_id": pid,
                                "temperature_months": json.dumps(fit.temperature_months),
                                "precipitation_months": json.dumps(fit.precipitation_months),
                                "adjusted_r_squared": fit.adjusted_r_squared,
                                "nobs": len(fit.years),
                            }
                        )
                    annual_t = monthly_season_mean(
                        t, result.temperature_months, np.asarray(fit_years)
                    )
                    annual_p = monthly_season_mean(
                        p, result.precipitation_months, np.asarray(fit_years)
                    )
                    residual = selected.sel(year=list(fit_years)).values - result.predict(
                        annual_t.values, annual_p.values
                    )
                else:
                    fit, season, frame = _fit_univariate(
                        selected, t, t_seasons, period, calibration.minimum_overlap, width=width
                    )
                    intercept, variance = (
                        float(fit.params["Intercept"]),
                        float(np.mean(fit.resid**2)),
                    )
                    r2, adjusted, bic, fit_years = (
                        float(fit.rsquared),
                        float(fit.rsquared_adj),
                        float(fit.bic),
                        tuple(float(y) for y in frame.index),
                    )
                    terms = [
                        {
                            "variable": "tas",
                            "slope": float(fit.params["Temperature"]),
                            "seasonality": list(season),
                            "units": "K",
                        }
                    ]
                    residual = np.asarray(fit.resid)
                if not np.isfinite([intercept, variance, r2, adjusted, bic]).all() or variance <= 0:
                    raise ValueError(
                        "regression has nonfinite diagnostics or nonpositive residual variance"
                    )
                lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
                if np.sqrt(max(0.0, r2)) < calibration.psm_correlation_minimum:
                    raise ValueError("calibration correlation below configured threshold")
                if np.isfinite(lag1) and lag1 > calibration.residual_autocorrelation_maximum:
                    raise ValueError("residual lag-1 correlation exceeds configured threshold")
                # A DJF reference spans the previous December, unlike a Jan-Dec
                # monthly climatology. Retain its exact instrumental mean offset.
                reference_years = np.arange(
                    calibration.anomaly_period[0], calibration.anomaly_period[1] + 1
                )
                for term in terms:
                    site = t if term["variable"] == "tas" else p
                    seasonal_reference = monthly_season_mean(
                        site, tuple(term["seasonality"]), reference_years
                    )
                    if not np.isfinite(seasonal_reference.values).all():
                        raise ValueError(
                            "incomplete instrumental seasonal anomaly reference period"
                        )
                    term["anomaly_reference_offset"] = float(seasonal_reference.mean())
                metadata = {
                    "psm_kind": "additive_linear_v1",
                    "proxy_type": str(proxy.ptype),
                    "predictors": terms,
                    "intercept": intercept,
                    "anomaly_reference_period": list(calibration.anomaly_period),
                    "calibration_years": list(fit_years),
                    "adjusted_r_squared": adjusted,
                    "bic": bic,
                    "nearest_temperature_grid": [float(site_t.lat), float(site_t.lon)],
                    "season_policy": "complete_quarters_drop_boundaries",
                }
                if is_bivariate:
                    metadata["nearest_precipitation_grid"] = [float(site_p.lat), float(site_p.lon)]
                if width:
                    metadata["resolution_years"] = width
                records.append(
                    ObservationRecord(
                        pid,
                        "pages2k",
                        "tas",
                        float(proxy.lat),
                        float(proxy.lon),
                        time,
                        value,
                        variance,
                        metadata=metadata,
                    )
                )
                diagnostics.append(
                    {
                        "record_id": pid,
                        "accepted": True,
                        "reason": "",
                        "bivariate": is_bivariate,
                        "nobs": len(fit_years),
                        "r_squared": r2,
                        "adjusted_r_squared": adjusted,
                        "variance": variance,
                        "residual_lag1_autocorrelation": lag1,
                    }
                )
            except (ValueError, KeyError, TypeError) as exc:
                diagnostics.append({"record_id": pid, "accepted": False, "reason": str(exc)})
        reference_path = directory / "reference_pr_quarterly.nc"
        reference = quarterly_mean(precipitation_mm_per_day(precipitation)).rename("pr")
        reference.attrs["source"] = str(calibration.precipitation_path)
        reference.attrs["evaluation_scope"] = (
            "GPCC verification reference; not used in temperature-only PSM regressions"
            if calibration.tree_ring_width_psm == "temperature_only"
            else "GPCC calibration input, not independent precipitation validation"
        )
        reference.to_dataset().to_netcdf(
            reference_path, engine="h5netcdf", encoding={"pr": {"dtype": "float32", "zlib": True}}
        )
    diagnostics_path, candidate_path = (
        directory / "diagnostics.csv",
        directory / "season_candidates.csv",
    )
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    pd.DataFrame(candidate_diagnostics).to_csv(candidate_path, index=False)
    if matched_ids is not None:
        missing = matched_ids - {record.record_id for record in records}
        if missing:
            raise ValueError(
                f"matched-network calibration lost {len(missing)} records: "
                f"{sorted(missing)}; see {diagnostics_path}"
            )
    if not records:
        raise ValueError(
            f"quarterly Tardif calibration produced no accepted records; see {diagnostics_path}"
        )
    artifact = save_calibrated_observations(
        directory / "calibrated_proxies.nc", ObservationCollection(records)
    )
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "tardif2019_quarterly",
                "source": str(calibration.source_path or calibration.source_name),
                "instrumental_path": str(calibration.instrumental_path),
                "precipitation_path": str(calibration.precipitation_path),
                "calibration_config": calibration.model_dump(mode="json"),
                "instrumental_eof_projection": False,
                "tree_ring_width_psm": calibration.tree_ring_width_psm,
                "matched_network_count": len(matched_ids) if matched_ids is not None else None,
                "counts": {
                    "accepted": len(records),
                    "rejected": sum(not row["accepted"] for row in diagnostics),
                    "bivariate": sum(
                        row.get("bivariate", False) for row in diagnostics if row["accepted"]
                    ),
                },
                "limitations": [
                    "Quarterly candidate subset differs from original arbitrary monthly windows.",
                    "Current GISTEMP snapshot is not the original 2019 input.",
                    "The CFR JSON source lacks original record seasonal metadata; "
                    "historical coefficient parity requires that metadata.",
                    "Non-TRW records use a univariate raw GISTEMP temperature PSM.",
                    "Low-resolution records use centered annual temperature windows.",
                    "Seasonal reference offsets preserve the monthly-baseline PSM intercept.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return manifest, diagnostics_path, candidate_path, reference_path, artifact
