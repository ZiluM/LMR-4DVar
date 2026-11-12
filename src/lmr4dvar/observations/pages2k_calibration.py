"""PAGES2k linear PSM calibration in the assimilation EOF subspace."""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from lmr4dvar.config.models import Pages2kCalibrationConfig
from lmr4dvar.eof import EOFModel

ANNUAL_SEASON = [-12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
QUARTER_SEASONS = [[-12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]


def project_instrumental_field(
    path: str | Path,
    variable: str,
    model: EOFModel,
    *,
    anomaly_period: tuple[int, int],
    fill_value: float = 0.0,
) -> xr.DataArray:
    """Project a monthly instrumental anomaly field through one EOF model."""
    with xr.open_dataset(path) as dataset:
        if variable not in dataset:
            raise ValueError(f"instrumental variable {variable!r} is absent from {path}")
        field = dataset[variable].load()
    field = _normalize_grid(field)
    baseline = field.sel(time=slice(str(anomaly_period[0]), str(anomaly_period[1])))
    if baseline.sizes.get("time", 0) == 0:
        raise ValueError(f"instrumental anomaly period {anomaly_period} has no samples")
    anomalies = field.groupby("time.month") - baseline.groupby("time.month").mean("time")
    anomalies = _align_to_eof_grid(anomalies, model)
    source_valid = anomalies.notnull()
    pcs = model.encode(anomalies.fillna(fill_value))
    projected = model.decode(pcs).where(source_valid & model.valid_mask)
    projected.name = variable
    projected.attrs.update(
        {
            "long_name": f"EOF-truncated instrumental {variable} anomaly",
            "units": field.attrs.get("units", "K"),
            "anomaly_period": f"{anomaly_period[0]}-{anomaly_period[1]}",
            "projection_fill_value": fill_value,
        }
    )
    return projected


def calibrate_pages2k(
    calibration: Pages2kCalibrationConfig,
    eof_models: dict[str, EOFModel],
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Calibrate PAGES2k records and save a CFR job plus audit artifacts."""
    if "tas" not in eof_models or "tos" not in eof_models:
        raise ValueError("PAGES2k calibration requires tas and tos EOF models")
    try:
        import cfr
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("PAGES2k calibration requires cfr and statsmodels") from exc

    directory = Path(output_directory)
    fields_directory = directory / "instrumental"
    job_directory = directory / "cfr_job"
    fields_directory.mkdir(parents=True, exist_ok=True)
    projected_paths: dict[str, Path] = {}
    for predictor in ("tas", "tos"):
        field = project_instrumental_field(
            calibration.instrumental_path,
            calibration.instrumental_variable,
            eof_models[predictor],
            anomaly_period=calibration.anomaly_period,
            fill_value=calibration.projection_fill_value,
        ).rename(predictor)
        path = fields_directory / f"{predictor}.nc"
        field.to_dataset(name=predictor).to_netcdf(path, engine="h5netcdf")
        projected_paths[predictor] = path

    source_database = _load_source_database(calibration, cfr)
    source_database = cfr.ProxyDatabase(
        {
            pid: pobj
            for pid, pobj in source_database.records.items()
            if str(pobj.ptype).lower() != "borehole"
        }
    )
    low_resolution, regular = _split_low_resolution(source_database, calibration, cfr)
    prepared, fixed_seasons = _prepare_regular_records(regular, cfr)
    _tag_ocean_records(prepared, eof_models["tos"])

    ocean = cfr.ProxyDatabase(
        {pid: copy.deepcopy(pobj) for pid, pobj in prepared.records.items() if "ocean" in pobj.tags}
    )
    terrestrial = cfr.ProxyDatabase(
        {
            pid: copy.deepcopy(pobj)
            for pid, pobj in prepared.records.items()
            if "ocean" not in pobj.tags
        }
    )
    seasons = {
        pid: fixed_seasons.get(pid, calibration.season_candidates)
        for pid in prepared.records
    }
    calibrated_tas = _calibrate_database(
        terrestrial, projected_paths["tas"], "tas", seasons,
        calibration.calibration_period, calibration.anomaly_period,
        calibration.minimum_overlap, cfr,
    )
    ocean_tos = _calibrate_database(
        copy.deepcopy(ocean), projected_paths["tos"], "tos", seasons,
        calibration.calibration_period, calibration.anomaly_period,
        calibration.minimum_overlap, cfr,
    )
    if calibration.marine_predictor == "best_of_tas_tos":
        ocean_tas = _calibrate_database(
            copy.deepcopy(ocean), projected_paths["tas"], "tas", seasons,
            calibration.calibration_period, calibration.anomaly_period,
            calibration.minimum_overlap, cfr,
        )
        calibrated_ocean = _select_best_predictor(ocean_tas, ocean_tos, cfr)
    else:
        calibrated_ocean = ocean_tos

    low_seasons = {pid: [ANNUAL_SEASON] for pid in low_resolution.records}
    calibrated_low = _calibrate_database(
        low_resolution, projected_paths["tas"], "tas", low_seasons,
        calibration.low_resolution_period, calibration.anomaly_period,
        calibration.minimum_overlap, cfr,
        low_resolution=True, smf=smf,
    )
    for pobj in calibrated_low.records.values():
        pobj.tags.add("low_res")

    combined = calibrated_tas + calibrated_ocean + calibrated_low
    if not combined.records:
        raise ValueError("PAGES2k calibration produced no calibrated records")
    diagnostics = _diagnostics(combined)
    keep = (
        diagnostics.psm_correlation.isna()
        | (diagnostics.psm_correlation >= calibration.psm_correlation_minimum)
    ) & (
        diagnostics.residual_lag1_autocorrelation.isna()
        | (
            diagnostics.residual_lag1_autocorrelation
            <= calibration.residual_autocorrelation_maximum
        )
    )
    kept_ids = set(diagnostics.loc[keep, "record_id"])
    final_database = cfr.ProxyDatabase(
        {pid: pobj for pid, pobj in combined.records.items() if pid in kept_ids}
    )
    diagnostics["accepted"] = diagnostics.record_id.isin(kept_ids)
    diagnostics_path = directory / "diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    final_job = cfr.ReconJob()
    final_job.proxydb = final_database
    final_job.save(str(job_directory))
    job_path = job_directory / "job.pkl"
    manifest = {
        "schema_version": 1,
        "method": "linear PSM calibration in the current tas/tos EOF subspace",
        "source": str(calibration.source_path or calibration.source_name),
        "instrumental_path": str(calibration.instrumental_path),
        "instrumental_variable": calibration.instrumental_variable,
        "anomaly_period": list(calibration.anomaly_period),
        "calibration_period": list(calibration.calibration_period),
        "low_resolution_period": list(calibration.low_resolution_period),
        "minimum_overlap": calibration.minimum_overlap,
        "marine_predictor": calibration.marine_predictor,
        "psm_correlation_minimum": calibration.psm_correlation_minimum,
        "residual_autocorrelation_maximum": calibration.residual_autocorrelation_maximum,
        "counts": {
            "source": len(source_database.records),
            "regular_prepared": len(prepared.records),
            "low_resolution": len(low_resolution.records),
            "accepted": len(final_database.records),
            "rejected_by_qc": len(combined.records) - len(final_database.records),
        },
        "compatibility_notes": [
            "Subannual coral records are converted into four quarterly records.",
            "Low-resolution records use centered temporal averaging of annual tas.",
            "best_of_tas_tos reproduces calib_2026.py rather than the SI's "
            "strict marine-tos wording.",
        ],
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path, diagnostics_path, job_path, *projected_paths.values()


def _normalize_grid(field: xr.DataArray) -> xr.DataArray:
    rename = {}
    for target, candidates in {"lat": ("latitude", "y"), "lon": ("longitude", "x")}.items():
        if target not in field.dims:
            match = next((name for name in candidates if name in field.dims), None)
            if match is not None:
                rename[match] = target
    field = field.rename(rename)
    if not {"time", "lat", "lon"}.issubset(field.dims):
        raise ValueError("instrumental field must have time, lat, and lon dimensions")
    field = field.assign_coords(lon=np.mod(field.lon.values, 360.0))
    return field.sortby("lon").sortby("lat").transpose("time", "lat", "lon")


def _align_to_eof_grid(field: xr.DataArray, model: EOFModel) -> xr.DataArray:
    target_lat = model.valid_mask.lat
    target_lon = model.valid_mask.lon
    if field.lat.equals(target_lat) and field.lon.equals(target_lon):
        return field
    return field.interp(lat=target_lat, lon=target_lon, method="linear")


def _load_source_database(config: Pages2kCalibrationConfig, cfr: Any):
    if config.source_path is None:
        return cfr.ProxyDatabase().fetch(config.source_name)
    path = Path(config.source_path)
    if path.is_dir():
        job = cfr.ReconJob()
        job.load(str(path))
        return job.proxydb
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as stream:
            loaded = pickle.load(stream)
        return loaded.proxydb if hasattr(loaded, "proxydb") else loaded
    return cfr.ProxyDatabase().fetch(str(path))


def _split_low_resolution(database, config, cfr):
    low = {}
    regular = {}
    start, end = config.low_resolution_period
    for pid, pobj in database.records.items():
        is_low = getattr(pobj, "time_unit", "yr") != "yr" or float(pobj.dt) > 1.0
        time = np.asarray(pobj.time)
        overlap = np.count_nonzero(
            np.isfinite(pobj.value) & (time >= start) & (time <= end)
        )
        if is_low and overlap >= config.minimum_overlap:
            low[pid] = copy.deepcopy(pobj)
        elif not is_low:
            regular[pid] = copy.deepcopy(pobj)
    return cfr.ProxyDatabase(low), cfr.ProxyDatabase(regular)


def _prepare_regular_records(database, cfr):
    subannual = {}
    annual = {}
    for pid, pobj in database.records.items():
        if str(pobj.ptype).lower() == "coral" and float(pobj.dt) < 1.0:
            subannual[pid] = pobj
        else:
            annual[pid] = pobj
    prepared = cfr.ProxyDatabase(annual).annualize(months=ANNUAL_SEASON)
    fixed_seasons = {}
    quarterly = {}
    subannual_db = cfr.ProxyDatabase(subannual)
    for season in QUARTER_SEASONS:
        for pid, pobj in subannual_db.annualize(months=season).records.items():
            new_id = f"{pid}_{season[1]}"
            pobj.pid = new_id
            pobj.tags.add("subannually resolved")
            pobj.tags.add("seasonalized")
            quarterly[new_id] = pobj
            fixed_seasons[new_id] = [season]
    return prepared + cfr.ProxyDatabase(quarterly), fixed_seasons


def _tag_ocean_records(database, tos_model: EOFModel) -> None:
    mask = tos_model.valid_mask
    for pobj in database.records.values():
        if str(pobj.ptype).lower() not in {"coral", "marine"}:
            continue
        valid = bool(
            mask.sel(lat=float(pobj.lat), lon=float(pobj.lon) % 360, method="nearest").item()
        )
        if valid:
            pobj.tags.add("ocean")


def _calibrate_database(
    database, climate_path, predictor, seasons_by_id, period, anomaly_period,
    minimum_overlap, cfr,
    *, low_resolution=False, smf=None,
):
    job = cfr.ReconJob()
    job.proxydb = database
    job.load_clim(
        tag="obs", path_dict={predictor: str(climate_path)},
        anom_period=anomaly_period, load=True,
    )
    for pid, pobj in job.proxydb.records.items():
        pobj.tags.discard("calibrated")
        pobj.psm = cfr.psm.Linear(pobj, climate_required=[predictor])
        try:
            pobj.get_clim(job.obs[predictor], tag="obs")
            if low_resolution:
                _calibrate_low_resolution(
                    pobj.psm, seasons_by_id[pid], period, minimum_overlap, smf
                )
            else:
                pobj.psm.calibrate(
                    season_list=seasons_by_id[pid], calib_period=period,
                    nobs_lb=minimum_overlap,
                )
        except (KeyError, TypeError, ValueError, ArithmeticError):
            continue
        if pobj.psm.calib_details is not None:
            pobj.tags.add("calibrated")
            pobj.R = float(pobj.psm.calib_details["PSMmse"])
    return _only_calibrated(job.proxydb, cfr)


def _calibrate_low_resolution(psm, seasons, period, minimum_overlap, smf) -> None:
    candidates = []
    exog = psm.pobj.clim["obs.tas"]
    for season in seasons:
        annual = exog.annualize(months=season).da
        values, times = _window_average(
            np.asarray(annual.values), np.asarray(annual.time.values, dtype=float),
            np.asarray(psm.pobj.time, dtype=float), float(psm.pobj.dt),
        )
        proxy = pd.DataFrame({"time": psm.pobj.time, "proxy": psm.pobj.value})
        climate = pd.DataFrame({"time": times, "tas": values})
        frame = proxy.merge(climate, on="time").dropna().set_index("time").sort_index()
        frame = frame.loc[(frame.index >= period[0]) & (frame.index <= period[1])]
        if len(frame) < minimum_overlap:
            continue
        model = smf.ols("proxy ~ tas", data=frame).fit()
        candidates.append((model.rsquared_adj, model, frame, season))
    if not candidates:
        psm.calib_details = None
        psm.model = None
        return
    _, model, frame, season = max(candidates, key=lambda item: item[0])
    residual = model.resid
    psm.model = model
    psm.calib_details = {
        "df": frame,
        "nobs": model.nobs,
        "fitR2adj": model.rsquared_adj,
        "PSMresid": residual,
        "PSMmse": float(np.mean(residual**2)),
        "SNR": float(np.std(model.predict()) / np.std(residual)),
        "seasonality": season,
    }


def _window_average(values, time, observation_time, resolution):
    mask = np.abs(time[None, :] - observation_time[:, None]) <= resolution / 2
    counts = mask.sum(axis=1)
    valid = counts > 0
    weights = np.divide(
        mask.astype(float), counts[:, None],
        out=np.zeros_like(mask, dtype=float), where=counts[:, None] > 0,
    )
    return weights[valid] @ values, observation_time[valid]


def _only_calibrated(database, cfr):
    return cfr.ProxyDatabase(
        {pid: pobj for pid, pobj in database.records.items() if "calibrated" in pobj.tags}
    )


def _select_best_predictor(tas_database, tos_database, cfr):
    selected = {}
    for pid in set(tas_database.records) | set(tos_database.records):
        candidates = [
            database.records[pid]
            for database in (tas_database, tos_database)
            if pid in database.records
        ]
        selected[pid] = max(
            candidates, key=lambda pobj: float(pobj.psm.calib_details["fitR2adj"])
        )
    return cfr.ProxyDatabase(selected)


def _diagnostics(database) -> pd.DataFrame:
    rows = []
    for pid, pobj in database.records.items():
        details = pobj.psm.calib_details
        predictor = list(details["df"].columns)[1]
        predicted = np.asarray(pobj.psm.model.predict(details["df"][[predictor]]))
        observed = np.asarray(details["df"]["proxy"])
        corr = _correlation(predicted, observed)
        residual = pd.Series(np.asarray(details["PSMresid"]), index=details["df"].index)
        common = residual.index.intersection(residual.index + 1)
        lag1 = _correlation(
            residual.loc[common - 1].to_numpy(), residual.loc[common].to_numpy()
        )
        rows.append(
            {
                "record_id": pid,
                "proxy_type": pobj.ptype,
                "predictor": predictor,
                "seasonality": json.dumps(details["seasonality"]),
                "nobs": int(details["nobs"]),
                "adjusted_r2": float(details["fitR2adj"]),
                "psm_mse": float(details["PSMmse"]),
                "psm_correlation": corr,
                "residual_lag1_autocorrelation": lag1,
                "low_resolution": "low_res" in pobj.tags,
            }
        )
        pobj.psm_corr = corr
        pobj.resid_auto_corr = lag1
    return pd.DataFrame(rows)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])
