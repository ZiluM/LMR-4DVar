import numpy as np
import pytest
import xarray as xr
from scipy import linalg

from lmr4dvar.lim import LIMModel


def var_trajectory(samples: int = 4000, seed: int = 4) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    transition = np.array([[0.82, 0.08], [-0.04, 0.68]])
    values = np.zeros((samples, 2))
    for index in range(1, samples):
        values[index] = transition @ values[index - 1] + rng.normal(scale=[0.3, 0.2])
    return xr.DataArray(
        values,
        dims=("time", "state"),
        coords={
            "time": np.arange(samples),
            "state": ["tas:000", "tos:000"],
        },
        name="pcs",
    )


def test_fit_matches_legacy_discrete_equations() -> None:
    pcs = var_trajectory(500)
    values = pcs.values
    x0 = values[:-1]
    x1 = values[1:]
    expected_g = (x1.T @ x0) @ np.linalg.pinv(x0.T @ x0)
    expected_c0 = x0.T @ x0 / (x0.shape[0] - 1)
    expected_qd = expected_c0 - expected_g @ expected_c0 @ expected_g.T

    model = LIMModel.fit(pcs, fit_noise=False)

    np.testing.assert_allclose(model.propagator, expected_g)
    np.testing.assert_allclose(model.stationary_covariance, expected_c0)
    np.testing.assert_allclose(model.raw_model_error_covariance, expected_qd)
    assert model.continuous_operator is None


def test_fit_excludes_transitions_between_equal_length_segments() -> None:
    first = np.arange(8.0)[:, None]
    second = (100 + np.arange(8.0))[:, None]
    values = np.concatenate((first, second), axis=0)
    expected_x0 = np.concatenate((first[:-1], second[:-1]), axis=0)
    expected_x1 = np.concatenate((first[1:], second[1:]), axis=0)
    expected = (expected_x1.T @ expected_x0) @ np.linalg.pinv(expected_x0.T @ expected_x0)

    model = LIMModel.fit(
        values,
        segment_length_steps=8,
        fit_noise=False,
        require_stable=False,
    )

    np.testing.assert_allclose(model.propagator, expected)
    assert model.segment_length_steps == 8


def test_fit_recovers_stable_var_and_continuous_operator() -> None:
    model = LIMModel.fit(var_trajectory(), fit_noise=True)

    expected = np.array([[0.82, 0.08], [-0.04, 0.68]])
    np.testing.assert_allclose(model.propagator, expected, atol=0.025)
    np.testing.assert_allclose(linalg.expm(model.continuous_operator), model.propagator)
    assert model.spectral_radius < 1
    assert np.all(model.decay_times_steps > 0)


def test_corrected_model_error_covariance_is_positive_definite() -> None:
    model = LIMModel.fit(var_trajectory(800), fit_noise=False)

    eigenvalues = np.linalg.eigvalsh(model.model_error_covariance)
    assert np.all(eigenvalues > 0)
    assert eigenvalues.min() >= model.covariance_floor * (1 - 1e-8)


def test_forecast_uses_column_state_dynamics() -> None:
    model = LIMModel.fit(var_trajectory(800), fit_noise=False)
    initial = xr.DataArray(
        [[1.0, -0.5]], dims=("sample", "state"), coords={"state": model.training_mean.state}
    )

    result = model.forecast(initial, 2)
    expected = initial.values @ np.linalg.matrix_power(model.propagator.values, 2).T

    np.testing.assert_allclose(result, expected)
    assert result.name == "forecast"


def test_forecast_skill_has_labeled_state_metrics() -> None:
    model = LIMModel.fit(var_trajectory(800), fit_noise=False)

    skill = model.forecast_skill(var_trajectory(800), max_lead=3)

    assert skill.correlation.dims == ("lead", "state")
    assert skill.lead.values.tolist() == [1, 2, 3]
    assert skill.state.values.tolist() == ["tas:000", "tos:000"]
    assert np.isfinite(skill.rmse).all()


def test_unstable_fit_is_rejected() -> None:
    values = np.column_stack([1.02 ** np.arange(30), 0.8 ** np.arange(30)])

    with pytest.raises(ValueError, match="not damped"):
        LIMModel.fit(values, fit_noise=False)


def test_artifact_roundtrip(tmp_path) -> None:
    pcs = var_trajectory(800)
    model = LIMModel.fit(pcs)
    diagnostics = model.forecast_skill(pcs, 2)

    model.save(tmp_path, diagnostics)
    loaded = LIMModel.load(tmp_path)

    xr.testing.assert_allclose(loaded.propagator, model.propagator)
    xr.testing.assert_allclose(loaded.model_error_covariance, model.model_error_covariance)
    assert loaded.lag_steps == 1
    assert (tmp_path / "training_metrics.nc").is_file()
