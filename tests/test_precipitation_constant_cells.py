import numpy as np
import xarray as xr

from lmr4dvar.verification.precipitation_products import verify_product


def test_effectively_constant_reference_cells_have_no_r_or_ce():
    field = xr.DataArray(
        np.arange(64).reshape(16, 2, 2).astype(float),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(1980, 1984, 0.25), "lat": [-45, 45], "lon": [90, 270]},
        attrs={"units": "mm/day"},
    )
    reference = field.copy()
    reference.values[:, 0, 0] = 1e-15 * np.arange(16)
    dataset, summary = verify_product(
        field,
        reference,
        xr.ones_like(field.isel(time=0), dtype=bool),
        baseline=[1981, 1983],
        period=[1980, 1983],
    )
    assert np.isnan(dataset.ce.sel(lat=-45, lon=90)).all()
    assert np.isnan(dataset.gridpoint_temporal_correlation.sel(lat=-45, lon=90)).all()
    assert summary["annual"]["ce_valid_gridpoints"] == 3
