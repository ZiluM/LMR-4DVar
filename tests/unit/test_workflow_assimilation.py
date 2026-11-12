import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from lmr4dvar.config.models import ProjectConfig
from lmr4dvar.workflow.assimilation import (
    _site_climatology,
    _tas_climatology,
    _validate_additive_prior,
    assimilation_time,
)


def _config(start="0850-01", end="0851-12") -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "project": {"name": "assimilation-time"},
            "paths": {"simulation_root": "/tmp", "output_root": "/tmp/output"},
            "state": {"variables": {"tas": {"domain": "global", "n_modes": 1}}},
            "observations": {"pages2k": {"enabled": True}},
            "assimilation": {"start": start, "end": end},
            "reconstruction": {"variables": ["tas"]},
            "verification": {
                "anomaly_period": [850, 851],
                "comparison_period": [850, 851],
            },
        }
    )


def test_assimilation_time_supports_ce_and_negative_years() -> None:
    np.testing.assert_allclose(
        assimilation_time(_config()).numpy(),
        [850, 850.25, 850.5, 850.75, 851, 851.25, 851.5, 851.75],
    )
    negative = assimilation_time(_config("-10-01", "-9-12"))
    np.testing.assert_allclose(negative[:4], [-10, -9.75, -9.5, -9.25])


def test_assimilation_time_rejects_partial_years() -> None:
    with pytest.raises(ValueError, match="complete"):
        assimilation_time(_config("0850-04", "0851-12"))


def test_site_climatology_converts_kelvin_to_celsius() -> None:
    field = xr.DataArray(
        np.full((12, 2, 2), 293.15),
        dims=("month", "lat", "lon"),
        coords={"month": np.arange(1, 13), "lat": [-5, 5], "lon": [0, 180]},
        attrs={"units": "K"},
    )
    record = SimpleNamespace(latitude=4.0, longitude=181.0)
    assert _site_climatology(field, record) == pytest.approx(20.0)
    field.attrs["units"] = "unknown"
    with pytest.raises(ValueError, match="units"):
        _site_climatology(field, record)


def test_tas_climatology_can_use_external_variable(tmp_path) -> None:
    path = tmp_path / "berkeley.nc"
    expected = xr.DataArray(
        np.full((12, 1, 1), 20.0),
        dims=("month_number", "lat", "lon"),
        coords={"month_number": np.arange(1, 13), "lat": [0.0], "lon": [0.0]},
        attrs={"units": "degree C"},
        name="climatology",
    )
    expected.to_dataset(name="climatology").to_netcdf(path, engine="h5netcdf")
    config = _config()
    config.observations.sedpsm.enabled = True
    config.observations.pages2k.enabled = False
    config.paths.sedpsm_root = tmp_path
    config.observations.sedpsm.climatology_path = path

    actual = _tas_climatology(config)

    xr.testing.assert_identical(actual, expected)
    record = SimpleNamespace(latitude=0.0, longitude=0.0)
    assert _site_climatology(actual, record) == pytest.approx(20.0)


def test_additive_binding_rejects_legacy_prior_without_physical_time_manifest(tmp_path):
    config = _config()
    config.paths.prior_run_root = tmp_path
    with pytest.raises(ValueError, match="physical-time preprocessing"):
        _validate_additive_prior(config)
    directory = tmp_path / "stages/preprocess"
    directory.mkdir(parents=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps({"input_files": {}}))
    with pytest.raises(ValueError, match="legacy quarterly prior"):
        _validate_additive_prior(config)
    path.write_text(json.dumps({"monthly_time_policies": {"tas": "cf_bounds", "pr": "cf_bounds"}}))
    _validate_additive_prior(config)
