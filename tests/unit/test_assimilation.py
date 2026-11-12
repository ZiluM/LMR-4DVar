import numpy as np
import torch
import xarray as xr

from lmr4dvar.assimilation import (
    OperatorBinding,
    WeakConstraintObjective,
    assimilate_lbfgs,
)
from lmr4dvar.lim import LIMModel
from lmr4dvar.observations import ObservationCollection


def _lim_model(
    propagator=None,
    stationary=None,
    model_error=None,
    mean=None,
) -> LIMModel:
    propagator = np.array([[0.8]]) if propagator is None else propagator
    stationary = np.array([[1.0]]) if stationary is None else stationary
    model_error = np.array([[0.1]]) if model_error is None else model_error
    mean = np.array([0.0]) if mean is None else mean
    labels = [f"state_{index}" for index in range(len(mean))]

    def matrix(values, name):
        return xr.DataArray(
            values,
            dims=("state_out", "state_in"),
            coords={"state_out": labels, "state_in": labels},
            name=name,
        )

    return LIMModel(
        propagator=matrix(propagator, "G"),
        model_error_covariance=matrix(model_error, "Qd"),
        raw_model_error_covariance=matrix(model_error, "Qd_raw"),
        stationary_covariance=matrix(stationary, "C0"),
        continuous_operator=None,
        continuous_noise_covariance=None,
        training_mean=xr.DataArray(mean, dims="state", coords={"state": labels}),
        lag_steps=1,
        covariance_floor=1e-10,
    )


class FixedLinearOperator:
    def __init__(self, indices, observed, variance):
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.observed = np.asarray(observed)
        self.variance = np.asarray(variance)

    def predict(self, trajectory, model_time, observations):
        predicted = trajectory[self.indices.to(trajectory.device), 0]
        observed = torch.as_tensor(self.observed, dtype=trajectory.dtype, device=trajectory.device)
        variance = torch.as_tensor(self.variance, dtype=trajectory.dtype, device=trajectory.device)
        return predicted, observed, variance, ["synthetic"] * predicted.numel()


def test_control_is_direct_state_trajectory_and_background_is_prior_mode() -> None:
    objective = WeakConstraintObjective(_lim_model(mean=np.array([2.0])))
    control = torch.tensor([[3.0], [3.0], [2.7]], dtype=torch.float64)
    trajectory = objective.trajectory(control)
    assert trajectory.data_ptr() == control.data_ptr()
    torch.testing.assert_close(trajectory, control)
    torch.testing.assert_close(
        objective.background_control(3)[:, 0],
        torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
    )


def test_state_trajectory_cost_matches_equivalent_forcing_control() -> None:
    model = _lim_model(mean=np.array([2.0]))
    objective = WeakConstraintObjective(model)
    forcing_control = torch.tensor([[3.0], [0.2], [-0.1]], dtype=torch.float64)
    states = [forcing_control[0]]
    for forcing in forcing_control[1:]:
        states.append(
            objective.training_mean
            + objective.propagator @ (states[-1] - objective.training_mean)
            + forcing
        )
    state_control = torch.stack(states)
    binding = OperatorBinding(
        FixedLinearOperator([0], [3.0], [1.0]),
        ObservationCollection([]),
    )

    cost = objective.evaluate(
        state_control, torch.arange(3, dtype=torch.float64), [binding]
    )
    expected_initial = 0.5 * (3.0 - 2.0) ** 2
    expected_model_error = 0.5 * (0.2**2 + (-0.1) ** 2) / 0.1

    torch.testing.assert_close(
        state_control[:, 0], torch.tensor([3.0, 3.0, 2.7], dtype=torch.float64)
    )
    torch.testing.assert_close(
        cost.initial, torch.tensor(expected_initial, dtype=torch.float64)
    )
    torch.testing.assert_close(
        cost.model_error, torch.tensor(expected_model_error, dtype=torch.float64)
    )


def test_cost_terms_match_direct_quadratic_forms_and_gradient() -> None:
    model = _lim_model(
        propagator=np.array([[0.7, 0.1], [0.0, 0.6]]),
        stationary=np.array([[2.0, 0.3], [0.3, 1.0]]),
        model_error=np.array([[0.4, 0.1], [0.1, 0.3]]),
        mean=np.array([0.2, -0.1]),
    )
    objective = WeakConstraintObjective(model, model_error_weight=0.25)
    control = torch.tensor(
        [[0.7, -0.3], [0.1, 0.2], [-0.2, 0.05]],
        dtype=torch.float64,
        requires_grad=True,
    )
    binding = OperatorBinding(
        FixedLinearOperator([0, 2], [0.4, -0.2], [0.5, 0.25]),
        ObservationCollection([]),
    )
    cost = objective.evaluate(control, torch.arange(3, dtype=torch.float64), [binding])
    initial_anomaly = control.detach().numpy()[0] - model.training_mean.values
    expected_initial = 0.5 * initial_anomaly @ np.linalg.solve(
        model.stationary_covariance.values, initial_anomaly
    )
    values = control.detach().numpy()
    mean = model.training_mean.values
    error = values[1:] - mean - (values[:-1] - mean) @ model.propagator.values.T
    expected_model = 0.5 * sum(
        value @ np.linalg.solve(model.model_error_covariance.values, value) for value in error
    )
    np.testing.assert_allclose(cost.initial.detach(), expected_initial)
    np.testing.assert_allclose(cost.model_error.detach(), 0.25 * expected_model)
    assert cost.observation.used_count == 2
    assert torch.autograd.gradcheck(
        lambda value: objective.evaluate(
            value, torch.arange(3, dtype=torch.float64), [binding]
        ).total,
        (control,),
    )


def test_lbfgs_recovers_synthetic_trajectory() -> None:
    objective = WeakConstraintObjective(_lim_model(model_error=np.array([[0.05]])))
    truth = torch.tensor([1.0, 0.8, 0.64, 0.512, 0.4096], dtype=torch.float64)
    binding = OperatorBinding(
        FixedLinearOperator(np.arange(5), truth.numpy(), np.full(5, 1.0e-4)),
        ObservationCollection([]),
    )
    model_time = torch.arange(5, dtype=torch.float64)
    result = assimilate_lbfgs(
        objective,
        model_time,
        [binding],
        max_iterations=8,
        inner_iterations=10,
        minimum_outer_iterations=2,
        tolerance_gradient=1.0e-7,
    )
    assert result.cost_history[-1] < result.cost_history[0]
    assert result.function_evaluations == len(result.cost_history)
    assert 2 <= result.outer_iterations <= 8
    assert len(result.outer_cost_history) == result.outer_iterations
    assert np.isfinite(result.final_gradient_norm)
    assert len(result.observation_by_family_history) == result.function_evaluations
    assert set(result.observation_by_family_history[-1]) == {"synthetic"}
    assert set(result.observation_used_count_history) == {5}
    assert set(result.observation_dropped_count_history) == {0}
    torch.testing.assert_close(result.trajectory[:, 0], truth, atol=3e-3, rtol=0)
    assert np.isfinite(result.gradient_norm_history).all()
