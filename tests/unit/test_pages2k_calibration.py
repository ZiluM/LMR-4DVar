import numpy as np
import xarray as xr

from lmr4dvar.eof import EOFModel
from lmr4dvar.observations.pages2k_calibration import (
    _window_average,
    project_instrumental_field,
)


def test_project_instrumental_field_uses_eof_subspace_and_preserves_missing(tmp_path) -> None:
    time = np.arange("1951-01", "1953-01", dtype="datetime64[M]")
    lat = [-1.0, 1.0]
    lon = [0.0, 2.0]
    training = xr.DataArray(
        np.arange(40, dtype=float).reshape(10, 2, 2),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(10), "lat": lat, "lon": lon},
    )
    model, _ = EOFModel.fit(training, 2, area_weighted=False, center=True)
    values = np.arange(96, dtype=float).reshape(24, 2, 2)
    values[:, 0, 0] = np.nan
    source = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="temp",
    )
    path = tmp_path / "instrumental.nc"
    source.to_dataset().to_netcdf(path, engine="h5netcdf")

    actual = project_instrumental_field(
        path, "temp", model, anomaly_period=(1951, 1951), fill_value=0.0
    )

    assert actual.dims == ("time", "lat", "lon")
    assert actual.sizes["time"] == 24
    assert actual[:, 0, 0].isnull().all()
    assert np.isfinite(actual[:, 1, 1]).all()


def test_window_average_matches_centered_proxy_resolution() -> None:
    values = np.arange(10, dtype=float)
    time = np.arange(2000, 2010, dtype=float)

    actual, matched = _window_average(
        values, time, np.array([2002.0, 2008.0, 2020.0]), 4.0
    )

    np.testing.assert_allclose(actual, [2.0, 7.5])
    np.testing.assert_allclose(matched, [2002.0, 2008.0])
