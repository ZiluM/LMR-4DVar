import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lmr4dvar.data.preprocess import preprocess_field
from lmr4dvar.data.seasonal_transform import SeasonalTransform


def _monthly(offset=0.0):
    time = pd.date_range("2000-01-01", "2005-12-01", freq="MS")
    rng = np.random.default_rng(12)
    values = rng.normal(size=(len(time), 2, 2))
    values += (np.arange(len(time)) * 0.1 + offset)[:, None, None]
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-1, 1], "lon": [0, 2]},
        attrs={"units": "mm/day"},
    )


def test_fit_and_apply_matches_existing_training_preprocessing(tmp_path):
    monthly = _monthly()
    expected, _ = preprocess_field(monthly, domain="global", regrid=False)
    transform = SeasonalTransform.fit(monthly, reference_year=2000)
    actual = transform.apply(monthly)
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-12)
    xr.testing.assert_equal(actual.time, expected.time)
    path = tmp_path / "parameters.nc"
    transform.save(path)
    xr.testing.assert_allclose(SeasonalTransform.load(path).apply(monthly), actual)


def test_held_out_offset_is_not_removed_by_refitting():
    train = _monthly()
    transform = SeasonalTransform.fit(train, reference_year=2000)
    base = transform.apply(train)
    held_out = transform.apply(_monthly(offset=15.0))
    np.testing.assert_allclose(held_out.values - base.values, 15.0, atol=1e-12)
    assert held_out.attrs["units"] == "mm/day"


def test_ensemble_mean_uses_only_training_member_parameters():
    train = [_monthly(0), _monthly(2)]
    transforms = [SeasonalTransform.fit(field, reference_year=2000) for field in train]
    pooled = SeasonalTransform.ensemble_mean(transforms)
    expected = (transforms[0].apply(train[0]) + transforms[1].apply(train[0])) / 2
    xr.testing.assert_allclose(pooled.apply(train[0]), expected)
    assert pooled.parameters.attrs["training_member_count"] == 2
    with pytest.raises(ValueError, match="same trend reference"):
        SeasonalTransform.ensemble_mean(
            [transforms[0], SeasonalTransform.fit(train[1], reference_year=1999)]
        )
