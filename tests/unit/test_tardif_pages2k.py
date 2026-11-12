from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from lmr4dvar.config.models import Pages2kCalibrationConfig
from lmr4dvar.data.precipitation import precipitation_mm_per_day
from lmr4dvar.data.preprocess import complete_quarterly_years, quarterly_mean
from lmr4dvar.observations.additive import AdditiveProxyOperator
from lmr4dvar.observations.bivariate_calibration import monthly_season_mean
from lmr4dvar.observations.calibrated_artifact import load_calibrated_observations
from lmr4dvar.observations.linear import PointTransform
from lmr4dvar.observations.records import ObservationCollection
from lmr4dvar.observations.tardif_pages2k import (
    calibrate_tardif_pages2k,
    nearest_instrumental_site,
    site_anomaly,
    strict_window_mean,
)
from lmr4dvar.state import StateLayout


def test_precipitation_monthly_totals_use_calendar_days_and_explicit_units():
    field = xr.DataArray(
        [31.0, 29.0, 31.0],
        dims="time",
        coords={"time": pd.date_range("2000-01-01", periods=3, freq="MS")},
        attrs={"units": "mm/month"},
    )
    np.testing.assert_allclose(precipitation_mm_per_day(field), [1, 1, 1])
    field.attrs["units"] = "m/s"
    np.testing.assert_allclose(precipitation_mm_per_day(field), field.values * 86400000)
    with pytest.raises(ValueError, match="contradict"):
        precipitation_mm_per_day(field, source_units="mm/day")
    field.attrs.clear()
    with pytest.raises(ValueError, match="missing"):
        precipitation_mm_per_day(field)
    assert precipitation_mm_per_day(field, source_units="m/s").attrs["units"] == "mm/day"


def test_nearest_site_handles_longitude_seam_without_searching_valid_values():
    field = xr.DataArray(
        [[1.0, 2.0]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0, 350.0]}
    )
    assert nearest_instrumental_site(field, 0, 359).item() == 1
    field.values[0, 0] = np.nan
    assert np.isnan(nearest_instrumental_site(field, 0, 359).item())


def test_low_resolution_indexed_means_reject_missing_or_truncated_windows():
    values = np.arange(10, dtype=float)
    values[-1] = np.nan
    result = strict_window_mean(values, np.arange(2000, 2010), np.array([2002, 2007, 2008]), 4.0)
    np.testing.assert_allclose(result, [2.0, np.nan, np.nan], equal_nan=True)


def test_quarterly_tardif_calibration_writes_both_coefficients_and_audit(tmp_path, monkeypatch):
    rng = np.random.default_rng(117)
    dates = pd.date_range("1949-01-01", "1991-12-01", freq="MS")
    t = xr.DataArray(
        20 + rng.normal(size=len(dates)), dims="time", coords={"time": dates}, attrs={"units": "K"}
    )
    p = xr.DataArray(
        2 + rng.normal(scale=0.2, size=len(dates)),
        dims="time",
        coords={"time": dates},
        attrs={"units": "mm/day"},
    )
    years = np.arange(1952, 1990)
    t_annual = monthly_season_mean(site_anomaly(t, (1951, 1980)), (6, 7, 8), years)
    p_annual = monthly_season_mean(site_anomaly(p, (1951, 1980)), (-12, 1, 2), years)
    values = (
        0.75 + 2 * t_annual.values - 3 * p_annual.values + rng.normal(scale=0.01, size=len(years))
    )
    proxy = SimpleNamespace(
        ptype="tree.TRW", dt=1.0, lat=0.0, lon=0.0, time=years, value=values, seasonality=[6, 7, 8]
    )
    proxy.annualize = lambda months: proxy
    bad = SimpleNamespace(
        ptype="tree.TRW",
        dt=1.0,
        lat=0.0,
        lon=0.0,
        time=years[:3],
        value=values[:3],
        seasonality=[6, 7, 8],
    )
    bad.annualize = lambda months: bad
    mxd_values = (
        0.5
        + 1.25 * monthly_season_mean(site_anomaly(t, (1951, 1980)), (3, 4, 5), years).values
        + rng.normal(scale=0.01, size=len(years))
    )
    density = SimpleNamespace(
        ptype="tree.MXD",
        dt=1.0,
        lat=0.0,
        lon=0.0,
        time=years,
        value=mxd_values,
        seasonality=[3, 4, 5],
    )
    density.annualize = lambda months: density
    database = SimpleNamespace(records={"tree": proxy, "density": density, "too-short": bad})
    monkeypatch.setattr(
        "lmr4dvar.observations.tardif_pages2k._load_source_database", lambda config, cfr: database
    )
    t_path, p_path = tmp_path / "temperature.nc", tmp_path / "precipitation.nc"
    t.expand_dims(lat=[0.0], lon=[0.0]).rename("tempanomaly").to_dataset().to_netcdf(t_path)
    monthly_total = (p * p.time.dt.days_in_month).assign_attrs(units="mm/month")
    monthly_total.expand_dims(lat=[0.0], lon=[0.0]).rename("p").to_dataset().to_netcdf(p_path)
    config = Pages2kCalibrationConfig(
        enabled=True,
        method="tardif2019_quarterly",
        instrumental_path=t_path,
        precipitation_path=p_path,
        calibration_period=(1952, 1989),
        season_candidates=[[6, 7, 8], [-12, 1, 2], [3, 4, 5]],
        psm_correlation_minimum=0,
        residual_autocorrelation_maximum=1,
    )
    paths = calibrate_tardif_pages2k(config, tmp_path / "calibrated")
    assert all(path.is_file() for path in paths)
    records = list(load_calibrated_observations(paths[-1]))
    assert len(records) == 2
    assert len(records[1].metadata["predictors"]) == 1
    assert records[1].metadata["predictors"][0]["slope"] == pytest.approx(1.25, abs=0.03)
    terms = records[0].metadata["predictors"]
    assert terms[0]["seasonality"] == [6, 7, 8]
    assert terms[1]["seasonality"] == [-12, 1, 2]
    assert terms[0]["slope"] == pytest.approx(2.0, abs=0.03)
    assert terms[1]["slope"] == pytest.approx(-3.0, abs=0.08)
    diagnostics = pd.read_csv(paths[1])
    assert diagnostics.accepted.tolist() == [True, True, False]
    assert "overlap" in diagnostics.reason.iloc[2]
    # Use exact observed quarterly anomalies as an identity EOF state. The
    # reference normalization must preserve the original monthly-baseline fit.
    qt = complete_quarterly_years(quarterly_mean(site_anomaly(t, (1951, 1980))))
    qp = complete_quarterly_years(quarterly_mean(site_anomaly(p, (1951, 1980))))
    trajectory = torch.tensor(np.column_stack([qt.values, qp.values]), dtype=torch.float64)
    model_time = torch.tensor(
        qt.time.dt.year.values + (qt.time.dt.month.values - 1) / 12, dtype=torch.float64
    )
    point_decoder = SimpleNamespace(
        layout=StateLayout.from_mode_counts({"tas": 1, "pr": 1}),
        transform=lambda variable, lat, lon: PointTransform(
            np.array([1.0, 0.0]) if variable == "tas" else np.array([0.0, 1.0]), 0.0, lat, lon
        ),
    )
    forward = AdditiveProxyOperator(point_decoder).predict(
        trajectory, model_time, ObservationCollection([records[0]])
    )[0]
    expected = (
        records[0].metadata["intercept"]
        + terms[0]["slope"] * t_annual.values
        + terms[1]["slope"] * p_annual.values
    )
    np.testing.assert_allclose(forward.numpy(), expected, atol=1e-12)

    temperature_config = config.model_copy(update={
        "tree_ring_width_psm": "temperature_only",
        "matched_records_path": paths[-1],
    })
    temperature_paths = calibrate_tardif_pages2k(temperature_config, tmp_path / "temperature-only")
    temperature_records = list(load_calibrated_observations(temperature_paths[-1]))
    assert [record.record_id for record in temperature_records] == [
        record.record_id for record in records
    ]
    assert all(len(record.metadata["predictors"]) == 1 for record in temperature_records)
    term = temperature_records[0].metadata["predictors"][0]
    assert term["variable"] == "tas"
    predictor = monthly_season_mean(
        site_anomaly(t, (1951, 1980)), tuple(term["seasonality"]), years
    )
    coefficients = np.linalg.lstsq(
        np.column_stack([np.ones(len(years)), predictor.values]), values, rcond=None
    )[0]
    assert term["slope"] == pytest.approx(coefficients[1])
    assert temperature_records[0].metadata["intercept"] == pytest.approx(coefficients[0])
    assert not pd.read_csv(temperature_paths[1]).bivariate.fillna(False).any()
    assert "nearest_precipitation_grid" not in temperature_records[0].metadata
    monkeypatch.setattr(
        "lmr4dvar.observations.tardif_pages2k._load_source_database",
        lambda config, cfr: SimpleNamespace(records={"tree": proxy}),
    )
    with pytest.raises(ValueError, match="matched-network calibration lost 1 records"):
        calibrate_tardif_pages2k(temperature_config, tmp_path / "missing-record")


def test_quarterly_calibration_config_rejects_partial_windows_and_short_overlap():
    with pytest.raises(ValueError, match="tree_ring_width_psm"):
        Pages2kCalibrationConfig(tree_ring_width_psm="invalid")
    with pytest.raises(ValueError, match="require tardif2019_quarterly"):
        Pages2kCalibrationConfig(tree_ring_width_psm="temperature_only")
    with pytest.raises(ValueError, match="complete"):
        Pages2kCalibrationConfig(method="tardif2019_quarterly", season_candidates=[[1, 2, 3]])
    with pytest.raises(ValueError, match="25"):
        Pages2kCalibrationConfig(method="tardif2019_quarterly", minimum_overlap=24)
