import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from lmr4dvar.eof import EOFModel


def synthetic_field() -> xr.DataArray:
    time = pd.date_range("2000-01-01", periods=12, freq="QS")
    lat = [-45.0, 45.0]
    lon = [0.0, 90.0, 180.0]
    pattern_1 = np.array([[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]])
    pattern_2 = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    score_1 = np.sin(np.arange(12) * 2 * np.pi / 12)
    score_2 = np.cos(np.arange(12) * 2 * np.pi / 6)
    values = score_1[:, None, None] * pattern_1 + score_2[:, None, None] * pattern_2
    values[:, 0, 0] = np.nan
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="tas",
    )


def test_fit_returns_standardized_pcs_and_deterministic_orientation() -> None:
    model, pcs = EOFModel.fit(synthetic_field(), 2, center=False)

    np.testing.assert_allclose(pcs.std("time"), 1.0)
    for component in model.components.values:
        finite = component[np.isfinite(component)]
        assert finite[np.argmax(np.abs(finite))] > 0


def test_full_rank_encode_decode_roundtrip() -> None:
    field = synthetic_field()
    model, fitted_pcs = EOFModel.fit(field, 2, center=False)

    encoded = model.encode(field)
    decoded = model.decode(encoded)

    xr.testing.assert_allclose(encoded, fitted_pcs)
    xr.testing.assert_allclose(decoded, field.transpose("time", "lat", "lon"), atol=1e-12)


def test_encode_rejects_new_missing_values_inside_valid_mask() -> None:
    field = synthetic_field()
    model, _ = EOFModel.fit(field, 2)
    changed = field.copy()
    changed[0, 1, 1] = np.nan

    with pytest.raises(ValueError, match="inside the EOF valid mask"):
        model.encode(changed)


def test_fillna_value_is_used_for_fit_and_encode() -> None:
    field = synthetic_field()
    model, fitted_pcs = EOFModel.fit(field, 2, missing_value_fill=0.0)

    assert bool(model.valid_mask.all())
    xr.testing.assert_allclose(model.encode(field), fitted_pcs)
    assert model.missing_value_fill == 0.0


def test_fillna_value_survives_artifact_roundtrip(tmp_path) -> None:
    model, pcs = EOFModel.fit(synthetic_field(), 2, missing_value_fill=0.0)
    model.save(tmp_path, pcs)

    loaded = EOFModel.load(tmp_path)

    assert loaded.missing_value_fill == 0.0
    xr.testing.assert_allclose(loaded.encode(synthetic_field()), pcs)


def test_artifact_roundtrip(tmp_path) -> None:
    field = synthetic_field()
    model, pcs = EOFModel.fit(field, 2, center=True)
    model.save(tmp_path, pcs)

    loaded = EOFModel.load(tmp_path)
    decoded = loaded.decode(pcs)

    xr.testing.assert_allclose(loaded.components, model.components)
    xr.testing.assert_allclose(decoded, field, atol=1e-12)
    metadata = yaml.safe_load((tmp_path / "metadata.yml").read_text())
    assert metadata["n_modes"] == 2
    assert metadata["spatial_dimensions"] == ["lat", "lon"]
    with xr.open_dataarray(tmp_path / "pcs.nc", engine="h5netcdf") as stored_pcs:
        xr.testing.assert_allclose(stored_pcs, pcs)
