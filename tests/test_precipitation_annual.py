"""Regression checks for annual-only LMR comparison diagnostics."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from lmr4dvar.verification.precipitation_annual import (
    annual_years,
    complete_quarter_years,
    verify_annual_pair,
)


def annual_field():
    return xr.DataArray(
        np.arange(16, dtype=float).reshape(4, 2, 2),
        dims=("time", "lat", "lon"),
        coords={"time": [1981, 1982, 1983, 1984], "lat": [-30, 30], "lon": [0, 180]},
        attrs={"units": "mm/day"},
    )


def test_identical_annual_anomalies_have_perfect_skill():
    field = annual_field()
    mask = xr.ones_like(field.isel(time=0, drop=True), dtype=bool)
    _, scores = verify_annual_pair(field, field, mask, baseline=[1981, 1984])
    assert scores["gridpoint_temporal_r"] == pytest.approx(1)
    assert scores["gridpoint_ce"] == pytest.approx(1)
    assert scores["gridpoint_rmse_mm_day"] == pytest.approx(0)


def test_shared_annual_mask_and_amplitude_error():
    field = annual_field()
    other = (3 * field).assign_attrs(units="mm/day")
    mask = xr.ones_like(field.isel(time=0, drop=True), dtype=bool)
    mask[0, 0] = False
    _, scores = verify_annual_pair(other, field, mask, baseline=[1981, 1984])
    assert scores["coverage_gridpoints"] == 3
    assert scores["gridpoint_temporal_r"] == pytest.approx(1)
    assert scores["gridpoint_ce"] == pytest.approx(-3)


def test_partial_quarter_year_is_not_annualized():
    field = annual_field().rename(time="sample")
    field = field.assign_coords(sample=[1980.25, 1980.5, 1980.75, 1981.0]).rename(sample="time")
    field = xr.concat(
        [field, field.isel(time=slice(1, 4)).assign_coords(time=[1981.25, 1981.5, 1981.75])],
        dim="time",
    )
    assert complete_quarter_years(field).time.values.tolist() == [1981]


def test_missing_baseline_and_nonfinite_shared_coverage_rejected():
    field = annual_field()
    mask = xr.ones_like(field.isel(time=0, drop=True), dtype=bool)
    with pytest.raises(ValueError, match="baseline"):
        verify_annual_pair(field.isel(time=slice(1, None)), field, mask, baseline=[1981, 1984])
    broken = field.copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        verify_annual_pair(broken, field, mask, baseline=[1981, 1984])


def test_annual_labels_require_unique_integer_years():
    field = annual_field()
    with pytest.raises(ValueError, match="unique"):
        annual_years(field.assign_coords(time=[1981, 1981, 1982, 1983]))
    with pytest.raises(ValueError, match="integer"):
        annual_years(field.assign_coords(time=[1981.25, 1982, 1983, 1984]))


def test_decoded_annual_dates_are_normalized():
    field = annual_field().assign_coords(time=np.arange("1981", "1985", dtype="datetime64[Y]"))
    mask = xr.ones_like(field.isel(time=0, drop=True), dtype=bool)
    _, scores = verify_annual_pair(field, field, mask, baseline=[1981, 1984])
    assert scores["period"] == [1981, 1984]


@pytest.mark.parametrize("cf_dates", [False, True])
def test_lmr_reader_averages_mc_runs_and_converts_units(tmp_path, monkeypatch, cf_dates):
    analyze = Path(__file__).parents[1] / "analyze"
    modules = {}
    for name in ("verify_precipitation_products", "compare_lmrv21_precipitation"):
        spec = importlib.util.spec_from_file_location(name, analyze / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    data = xr.DataArray(
        np.broadcast_to(np.arange(20)[None, :, None, None] * 1e-7, (4, 20, 2, 2)),
        dims=("time", "MCrun", "lat", "lon"),
        coords={
            "time": [1981, 1982, 1983, 1984],
            "MCrun": np.arange(20),
            "lat": [-45, 45],
            "lon": [90, 270],
        },
        attrs={"units": "kg/m^2/s"},
        name="prate",
    )
    path = tmp_path / "lmr.nc"
    if cf_dates:
        data = data.assign_coords(time=np.arange("1981", "1985", dtype="datetime64[Y]"))
        data.time.encoding.update(units="days since 0000-01-01", calendar="noleap")
    data.to_dataset().to_netcdf(path, engine="h5netcdf")
    result = modules["compare_lmrv21_precipitation"].load_lmr(
        path, xr.Dataset(coords={"lat": [-45, 45], "lon": [90, 270]})
    )
    assert result.dims == ("time", "lat", "lon")
    np.testing.assert_allclose(result.values, 9.5e-7 * 86400)
    assert result.attrs["units"] == "mm/day"
    duplicate_months = xr.DataArray(
        [1.0, 2.0],
        dims="time",
        coords={"time": np.array(["1981-01", "1981-01"], dtype="datetime64[M]")},
    )
    with pytest.raises(ValueError, match="unique"):
        modules["compare_lmrv21_precipitation"].calendar_annual(duplicate_months)
