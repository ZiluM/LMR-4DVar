import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
import xarray as xr

from lmr4dvar.observations.bivariate_calibration import (
    fit_bivariate,
    monthly_season_mean,
    select_bivariate_seasons,
)


def _annual(values, years):
    return xr.DataArray(values, dims="year", coords={"year": years})


def test_regression_matches_lmr_statsmodels_formula():
    rng = np.random.default_rng(3)
    years = np.arange(1901, 2001)
    temperature, precipitation = rng.normal(size=(2, 100))
    values = 2 + 0.7 * temperature - 0.4 * precipitation + rng.normal(0, 0.1, 100)
    values[1] = np.nan
    temperature[3] = np.nan
    frame = pd.DataFrame({"y": values, "Temperature": temperature, "Moisture": precipitation})
    reference = smf.ols("y ~ Temperature + Moisture", data=frame).fit()
    actual = fit_bivariate(
        _annual(values, years),
        _annual(temperature[::-1], years[::-1]),
        _annual(precipitation, years),
        temperature_months=(6, 7, 8),
        precipitation_months=(-12, 1, 2),
    )
    np.testing.assert_allclose(
        [actual.intercept, actual.slope_temperature, actual.slope_precipitation],
        reference.params,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(actual.residual_variance, np.mean(reference.resid**2))
    np.testing.assert_allclose(actual.adjusted_r_squared, reference.rsquared_adj)
    np.testing.assert_allclose(actual.bic, reference.bic)
    assert len(actual.years) == 98
    np.testing.assert_allclose(actual.predict(temperature, precipitation), reference.predict(frame))


def test_monthly_seasons_use_real_calendar_and_complete_windows():
    time = pd.date_range("1900-01-01", periods=36, freq="MS")
    series = xr.DataArray(np.arange(36.0), dims="time", coords={"time": time})
    result = monthly_season_mean(series[::-1], (-12, 1, 2), np.array([1900, 1901, 1902]))
    np.testing.assert_allclose(result, [np.nan, 12.0, 24.0], equal_nan=True)
    result = monthly_season_mean(series, (12, 13, 14), np.array([1900, 1901, 1902]))
    np.testing.assert_allclose(result, [12.0, 24.0, np.nan], equal_nan=True)
    series[12] = np.nan
    assert np.isnan(monthly_season_mean(series, (-12, 1, 2), np.array([1901])).item())


def test_monthly_duplicate_labels_are_rejected():
    series = xr.DataArray(
        [1, 2], dims="time", coords={"time": pd.to_datetime(["1900-01-01", "1900-01-15"])}
    )
    with pytest.raises(ValueError, match="duplicate"):
        monthly_season_mean(series, (1,), np.array([1900]))


def test_selects_independent_temperature_and_precipitation_seasons():
    rng = np.random.default_rng(5)
    time = pd.date_range("1900-01-01", periods=1200, freq="MS")
    temperature = xr.DataArray(rng.normal(size=1200), dims="time", coords={"time": time})
    precipitation = temperature.copy(data=rng.normal(size=1200))
    years = np.arange(1901, 2000)
    target = (
        2 * monthly_season_mean(temperature, (6, 7, 8), years)
        - 3 * monthly_season_mean(precipitation, (-12, 1, 2), years)
        + rng.normal(0, 0.01, years.size)
    )
    best, candidates = select_bivariate_seasons(
        target,
        temperature,
        precipitation,
        ((6, 7, 8), (3, 4, 5)),
        ((6, 7, 8), (-12, 1, 2)),
    )
    assert best.temperature_months == (6, 7, 8)
    assert best.precipitation_months == (-12, 1, 2)
    assert len(candidates) == 4


def test_rejects_insufficient_overlap_and_rank_deficiency():
    years = np.arange(25)
    series = _annual(np.arange(25.0), years)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_bivariate(series, series, series, temperature_months=(1,), precipitation_months=(1,))
    with pytest.raises(ValueError, match="insufficient"):
        fit_bivariate(
            series[:24], series, series, temperature_months=(1,), precipitation_months=(1,)
        )
