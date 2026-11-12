"""Temperature/precipitation regression core for Tardif et al. (2019).

Predictors must already be anomalies in explicitly documented units. This
module does not project instrumental data through EOFs or choose proxy types.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class BivariateCalibration:
    intercept: float
    slope_temperature: float
    slope_precipitation: float
    residual_variance: float
    r_squared: float
    adjusted_r_squared: float
    bic: float
    years: tuple[int, ...]
    temperature_months: tuple[int, ...]
    precipitation_months: tuple[int, ...]

    def predict(self, temperature, precipitation):
        return (
            self.intercept
            + self.slope_temperature * temperature
            + self.slope_precipitation * precipitation
        )


def monthly_season_mean(
    series: xr.DataArray, months: tuple[int, ...], years: np.ndarray
) -> xr.DataArray:
    """Equal-month average; negative months refer to the previous year.

    Months 13..24 refer to the next year, following LMR's encoding. Incomplete
    or nonfinite seasons return NaN; duplicate year/month labels are errors.
    """
    if series.dims != ("time",):
        raise ValueError("seasonal calibration requires a 1D monthly site series")
    if not months or len(set(months)) != len(months):
        raise ValueError("season months must be nonempty and unique")
    if any(not isinstance(m, (int, np.integer)) or m == 0 or m < -12 or m > 24 for m in months):
        raise ValueError("season months must be integers in -12..-1 or 1..24")
    labels = pd.MultiIndex.from_arrays([series.time.dt.year.values, series.time.dt.month.values])
    if not labels.is_unique:
        raise ValueError("duplicate year/month labels in monthly predictor")
    requested = np.asarray(years)
    if requested.ndim != 1 or not np.isfinite(requested).all():
        raise ValueError("target years must be a finite 1D array")
    if np.any(requested != requested.astype(int)) or len(set(requested)) != requested.size:
        raise ValueError("target years must be unique integers")
    lookup = pd.Series(series.values, index=labels)
    columns = []
    for month in months:
        offset = -1 if month < 0 else (1 if month > 12 else 0)
        actual_month = abs(month) if month < 0 else ((month - 1) % 12 + 1)
        keys = pd.MultiIndex.from_arrays(
            [requested.astype(int) + offset, np.full(requested.size, actual_month)]
        )
        columns.append(lookup.reindex(keys).to_numpy(dtype=float))
    values = np.stack(columns, axis=1)
    complete = np.isfinite(values).all(axis=1)
    means = np.full(requested.size, np.nan)
    means[complete] = values[complete].mean(axis=1)
    return xr.DataArray(means, dims="year", coords={"year": requested.astype(int)})


def fit_bivariate(
    proxy: xr.DataArray,
    temperature: xr.DataArray,
    precipitation: xr.DataArray,
    *,
    temperature_months: tuple[int, ...],
    precipitation_months: tuple[int, ...],
    minimum_overlap: int = 25,
) -> BivariateCalibration:
    """Fit labelled annual series using the original additive OLS formula."""
    import statsmodels.formula.api as smf

    if minimum_overlap < 4:
        raise ValueError("minimum_overlap must be at least four")
    for series in (proxy, temperature, precipitation):
        if series.dims != ("year",) or not series.get_index("year").is_unique:
            raise ValueError("annual calibration series need unique year coordinates")
        years = series.year.values
        if not np.isfinite(years).all() or np.any(years != years.astype(int)):
            raise ValueError("annual calibration years must be finite integers")
    frame = pd.concat(
        [
            proxy.to_series().rename("y"),
            temperature.to_series().rename("Temperature"),
            precipitation.to_series().rename("Moisture"),
        ],
        axis=1,
    ).sort_index()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < minimum_overlap:
        raise ValueError(f"insufficient joint calibration overlap: {len(frame)}")
    regress = smf.ols("y ~ Temperature + Moisture", data=frame).fit()
    if np.linalg.matrix_rank(regress.model.exog) != 3:
        raise ValueError("temperature/precipitation regression is rank deficient")
    if not np.isfinite(regress.rsquared_adj):
        raise ValueError("undefined calibration skill (constant proxy)")
    return BivariateCalibration(
        intercept=float(regress.params["Intercept"]),
        slope_temperature=float(regress.params["Temperature"]),
        slope_precipitation=float(regress.params["Moisture"]),
        residual_variance=float(np.mean(regress.resid**2)),
        r_squared=float(regress.rsquared),
        adjusted_r_squared=float(regress.rsquared_adj),
        bic=float(regress.bic),
        years=tuple(int(y) for y in frame.index),
        temperature_months=temperature_months,
        precipitation_months=precipitation_months,
    )


def select_bivariate_seasons(
    proxy: xr.DataArray,
    temperature: xr.DataArray,
    precipitation: xr.DataArray,
    temperature_candidates: tuple[tuple[int, ...], ...],
    precipitation_candidates: tuple[tuple[int, ...], ...],
    *,
    minimum_overlap: int = 25,
) -> tuple[BivariateCalibration, tuple[BivariateCalibration, ...]]:
    """Search independent seasonal windows; ties retain the first candidate.

    All inputs have explicit time labels. Monthly predictors are site anomalies;
    proxy values are annual and indexed by year. Returned fits retain all valid
    candidates so the selected model can be audited.
    """
    if proxy.dims != ("year",):
        raise ValueError("proxy must be an annual series indexed by year")
    t = {m: monthly_season_mean(temperature, m, proxy.year.values) for m in temperature_candidates}
    p = {
        m: monthly_season_mean(precipitation, m, proxy.year.values)
        for m in precipitation_candidates
    }
    fits = []
    for mt, mp in product(t, p):
        joint = np.isfinite(proxy.values) & np.isfinite(t[mt].values) & np.isfinite(p[mp].values)
        if int(joint.sum()) < minimum_overlap:
            continue
        fits.append(
            fit_bivariate(
                proxy,
                t[mt],
                p[mp],
                temperature_months=mt,
                precipitation_months=mp,
                minimum_overlap=minimum_overlap,
            )
        )
    if not fits:
        raise ValueError("no seasonal pair has sufficient calibration overlap")
    return max(fits, key=lambda fit: fit.adjusted_r_squared), tuple(fits)
