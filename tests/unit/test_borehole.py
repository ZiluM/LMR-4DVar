import pickle

import numpy as np
import pytest
import torch
import xarray as xr
from scipy.special import erfc

import lmr4dvar.observations.linear as linear_module
from lmr4dvar.eof import EOFModel
from lmr4dvar.observations import (
    BoreholeOperator,
    BoreholeRecord,
    EOFPointDecoder,
    LinearProxyOperator,
    ObservationCollection,
    conduction_forward,
    gaussian_observation_loss,
    load_borehole_observations,
)
from lmr4dvar.state import StateLayout


def _legacy_numpy(surface, years, depths, logging_year, reference_period):
    reference = (years >= reference_period[0]) & (years <= reference_period[1])
    anomaly = surface - surface[reference].mean()
    use = (years >= reference_period[0]) & (years <= logging_year)
    history = anomaly[use][::-1]
    age = logging_year - years[use][::-1]
    eta = depths[:, None] / (2 * np.sqrt(1.0e-6 * age[None, :] * 365.25 * 86400 + 1e-16))
    kernel = erfc(np.where(age[None, :] > 0, eta, np.inf))
    result = history[0] * kernel[:, 0]
    if history.size > 1:
        age_minus_one = age[1:] - 1
        eta_previous = depths[:, None] / (
            2 * np.sqrt(1.0e-6 * age_minus_one[None, :] * 365.25 * 86400 + 1e-16)
        )
        previous = erfc(np.where(age_minus_one[None, :] > 0, eta_previous, np.inf))
        result += np.sum(history[1:][None, :] * (kernel[:, 1:] - previous), axis=1)
    return result


def _eof_model() -> EOFModel:
    time = np.arange(12)
    pattern = np.array([[1.0, -1.0], [0.5, -0.5]])
    values = 2.0 + np.sin(time * 2 * np.pi / 12)[:, None, None] * pattern
    field = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-45.0, 45.0], "lon": [0.0, 90.0]},
    )
    model, _ = EOFModel.fit(field, 1, center=True)
    return model


def test_conduction_matches_legacy_annual_forward_model() -> None:
    years = np.arange(1200.0, 2001.0)
    surface = np.sin(years / 71) + 0.2 * np.cos(years / 19)
    depths = np.array([100.0, 175.0, 300.0])
    actual = conduction_forward(
        torch.tensor(surface, dtype=torch.float64),
        years,
        depths,
        logging_year=2001,
        reference_period=(1300, 1700),
        history_policy="legacy_reference_start",
    )
    expected = _legacy_numpy(surface, years, depths, 2001, (1300, 1700))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-11, atol=1e-12)


def test_full_history_policy_retains_prebaseline_thermal_signal() -> None:
    years = torch.arange(1000.0, 2001.0, dtype=torch.float64)
    surface = torch.zeros_like(years)
    surface[years < 1300] = 2.0
    full = conduction_forward(
        surface,
        years,
        [200.0],
        logging_year=2001,
        reference_period=(1300, 1700),
    )
    legacy = conduction_forward(
        surface,
        years,
        [200.0],
        logging_year=2001,
        reference_period=(1300, 1700),
        history_policy="legacy_reference_start",
    )
    assert full.item() > 0
    torch.testing.assert_close(legacy, torch.zeros_like(legacy))


def test_conduction_is_differentiable_and_supports_irregular_years() -> None:
    years = torch.tensor([1800.0, 1810.0, 1850.0, 1900.0], dtype=torch.float64)
    surface = torch.tensor([0.2, -0.1, 0.4, 0.8], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda values: conduction_forward(
            values, years, [100.0, 250.0], logging_year=2000, reference_period=None
        ),
        (surface,),
    )


def test_borehole_operator_predicts_profiles_and_backpropagates() -> None:
    model = _eof_model()
    decoder = EOFPointDecoder({"tas": model}, StateLayout.from_mode_counts({"tas": 1}))
    reference_period = (1997, 1998)
    operator = BoreholeOperator(
        decoder,
        reference_period=reference_period,
        history_policy="legacy_reference_start",
    )
    trajectory = torch.linspace(-1, 1, 20, dtype=torch.float64).unsqueeze(1).requires_grad_()
    model_time = torch.tensor(
        [year + quarter / 4 for year in range(1996, 2001) for quarter in range(4)],
        dtype=torch.float64,
    )
    record = BoreholeRecord(
        record_id="test-hole",
        latitude=45,
        longitude=0,
        logging_year=2001,
        depth=np.array([100.0, 200.0]),
        value=np.array([0.1, -0.2]),
        variance=np.array([0.01, 0.02]),
    )
    observations = ObservationCollection([record])
    predicted, observed, variance, families = operator.predict(
        trajectory, model_time, observations
    )
    assert predicted.shape == observed.shape == variance.shape == (2,)
    assert families == ["borehole", "borehole"]
    loss = gaussian_observation_loss(predicted, observed, variance, families=families).total
    loss.backward()
    assert trajectory.grad is not None
    assert torch.isfinite(trajectory.grad).all()
    with pytest.raises(TypeError, match="BoreholeOperator"):
        LinearProxyOperator(decoder).predict(trajectory, model_time, observations)


def test_borehole_operator_batches_sites_and_reuses_fixed_kernels() -> None:
    model = _eof_model()
    decoder = EOFPointDecoder({"tas": model}, StateLayout.from_mode_counts({"tas": 1}))
    records = [
        BoreholeRecord(
            record_id="hole-a",
            latitude=-45,
            longitude=0,
            logging_year=2000,
            depth=np.array([100.0, 200.0]),
            value=np.array([0.1, -0.2]),
            variance=0.1,
        ),
        BoreholeRecord(
            record_id="hole-b",
            latitude=45,
            longitude=0,
            logging_year=2001,
            depth=np.array([150.0]),
            value=np.array([0.3]),
            variance=0.2,
        ),
    ]
    collection = ObservationCollection(records)
    reference_period = (1997, 1998)
    operator = BoreholeOperator(
        decoder,
        reference_period=reference_period,
        history_policy="legacy_reference_start",
    )
    trajectory = torch.linspace(-1, 1, 20, dtype=torch.float64).unsqueeze(1).requires_grad_()
    model_time = torch.tensor(
        [year + quarter / 4 for year in range(1996, 2001) for quarter in range(4)],
        dtype=torch.float64,
    )

    actual = operator.predict(trajectory, model_time, collection)[0]
    plan = operator._plan
    annual_time, annual_state = linear_module.annualize_quarterly(trajectory, model_time)
    expected = []
    for record in records:
        transform = decoder.transform("tas", record.latitude, record.longitude)
        local = annual_state @ torch.as_tensor(transform.weights, dtype=trajectory.dtype)
        local = local + transform.bias
        expected.append(
            conduction_forward(
                local,
                annual_time,
                record.depth,
                logging_year=record.logging_year,
                reference_period=reference_period,
                history_policy="legacy_reference_start",
            )
        )

    torch.testing.assert_close(actual, torch.cat(expected))
    operator.predict(trajectory + 0.1, model_time, collection)
    assert operator._plan is plan


def test_borehole_loader_interpolates_both_depth_coordinates(tmp_path) -> None:
    station = xr.Dataset(
        {
            "perturbation": ("depth", [0.1, 0.2, 0.4]),
            "error": ("depths", [0.01, 0.02, 0.03]),
        },
        coords={
            "depth": [90.0, 100.0, 110.0],
            "depths": [90.0, 100.0, 110.0],
            "lat": 50.0,
            "lon": -95.0,
            "logging_year": 2000,
            "station_id": "station-a",
            "region": "test",
        },
    )
    path = tmp_path / "boreholes.pkl"
    with path.open("wb") as stream:
        pickle.dump([station], stream)
    result = load_borehole_observations(path, target_depths=np.array([100.0, 105.0, 120.0]))
    assert not result.skipped
    record = next(iter(result.observations))
    np.testing.assert_allclose(record.depth, [100.0, 105.0])
    np.testing.assert_allclose(record.value, [0.2, 0.3])
    np.testing.assert_allclose(record.variance, [0.02, 0.025])
    assert record.longitude == 265.0
