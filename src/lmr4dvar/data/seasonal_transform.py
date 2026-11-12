"""Training-only seasonal preprocessing parameters for held-out forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from .preprocess import monthly_climatology_anomaly, quarterly_mean


@dataclass(frozen=True)
class SeasonalTransform:
    """Apply training monthly means and quarter-specific linear trends unchanged.

    Inputs must already have corrected physical months, target grid and units.
    Trends use physical calendar years, not an index fitted to held-out data.
    """

    parameters: xr.Dataset

    @classmethod
    def fit(cls, monthly: xr.DataArray, *, reference_year: int) -> SeasonalTransform:
        anomaly, climatology = monthly_climatology_anomaly(monthly)
        if not np.array_equal(climatology.month, np.arange(1, 13)):
            raise ValueError("training climatology requires all twelve months")
        quarters = quarterly_mean(anomaly)
        slopes, intercepts = [], []
        for month in (1, 4, 7, 10):
            group = quarters.where(quarters.time.dt.month == month, drop=True)
            years = group.time.dt.year.values.astype(float)
            if years.size < 2 or np.any(np.diff(years) != 1):
                raise ValueError("trend fitting requires consecutive yearly training quarters")
            x = xr.DataArray(years - reference_year, dims="time", coords={"time": group.time})
            centered = x - x.mean()
            slope = (group * centered).sum("time", skipna=False) / (centered**2).sum()
            intercept = group.mean("time", skipna=False) - slope * x.mean()
            slopes.append(slope)
            intercepts.append(intercept)
        quarter = xr.IndexVariable("quarter_month", [1, 4, 7, 10])
        parameters = xr.Dataset(
            {
                "monthly_climatology": climatology,
                "trend_slope": xr.concat(slopes, dim=quarter),
                "trend_intercept": xr.concat(intercepts, dim=quarter),
            },
            attrs={"reference_year": int(reference_year), "fit_scope": "training members only"},
        )
        return cls(parameters)

    @classmethod
    def ensemble_mean(cls, transforms: list[SeasonalTransform]) -> SeasonalTransform:
        if not transforms:
            raise ValueError("at least one training transform is required")
        first = transforms[0].parameters
        for transform in transforms[1:]:
            if transform.parameters.attrs["reference_year"] != first.attrs["reference_year"]:
                raise ValueError("training transforms require the same trend reference year")
            for coordinate in first.coords:
                if not first[coordinate].equals(transform.parameters[coordinate]):
                    raise ValueError("training transform grids differ")
        result = xr.concat([item.parameters for item in transforms], dim="training_member").mean(
            "training_member", skipna=False, keep_attrs=True
        )
        result.attrs["training_member_count"] = len(transforms)
        result.attrs["fit_scope"] = "equal-member mean of training-only fitted parameters"
        return cls(result)

    def apply(self, monthly: xr.DataArray) -> xr.DataArray:
        parameters = self.parameters
        for coordinate in ("lat", "lon"):
            if coordinate in parameters.coords and not monthly[coordinate].equals(
                parameters[coordinate]
            ):
                raise ValueError("evaluation grid differs from training transform")
        anomaly = monthly.groupby("time.month") - parameters.monthly_climatology
        quarters = quarterly_mean(anomaly)
        indices = xr.DataArray(
            quarters.time.dt.month.values, dims="time", coords={"time": quarters.time}
        )
        slope = parameters.trend_slope.sel(quarter_month=indices)
        intercept = parameters.trend_intercept.sel(quarter_month=indices)
        year = quarters.time.dt.year - parameters.attrs["reference_year"]
        result = (quarters - slope * year - intercept).drop_vars("quarter_month")
        result.attrs = monthly.attrs.copy()
        result.attrs["preprocessing"] = "training-only monthly climatology and quarterly trends"
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.parameters.to_netcdf(path, engine="h5netcdf")

    @classmethod
    def load(cls, path: Path) -> SeasonalTransform:
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            return cls(dataset.load())
