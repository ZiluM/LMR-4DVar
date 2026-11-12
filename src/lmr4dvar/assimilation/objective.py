"""Weak-constraint 4D-Var control mapping and objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lmr4dvar.lim import LIMModel
from lmr4dvar.observations import (
    ObservationCollection,
    ObservationLoss,
    ObservationOperator,
    gaussian_observation_loss,
)


@dataclass(frozen=True)
class OperatorBinding:
    operator: ObservationOperator
    observations: ObservationCollection


@dataclass(frozen=True)
class CostBreakdown:
    total: torch.Tensor
    initial: torch.Tensor
    model_error: torch.Tensor
    observation: ObservationLoss


class WeakConstraintObjective:
    """State-trajectory weak-constraint objective for a discrete LIM."""

    def __init__(
        self,
        model: LIMModel,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        initial_weight: float = 1.0,
        model_error_weight: float = 1.0,
    ) -> None:
        if initial_weight <= 0 or model_error_weight <= 0:
            raise ValueError("prior weights must be positive")
        self.dtype = dtype
        self.device = torch.device(device)
        self.initial_weight = float(initial_weight)
        self.model_error_weight = float(model_error_weight)
        self.propagator = self._matrix(model.propagator.values, "propagator")
        self.device = self.propagator.device
        self.stationary_cholesky = self._cholesky(
            model.stationary_covariance.values, "stationary covariance"
        )
        self.model_error_cholesky = self._cholesky(
            model.model_error_covariance.values, "model-error covariance"
        )
        self.training_mean = torch.as_tensor(
            model.training_mean.values, dtype=dtype, device=self.device
        )
        self.state_size = self.propagator.shape[0]

    def background_control(self, n_steps: int) -> torch.Tensor:
        """Return the prior-mode state trajectory at the LIM training mean."""
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        return self.training_mean.expand(n_steps, -1).clone()

    def trajectory(self, control: torch.Tensor) -> torch.Tensor:
        """Return the directly optimized state trajectory."""
        return self._validate_control(control)

    def evaluate(
        self,
        control: torch.Tensor,
        model_time: torch.Tensor,
        bindings: list[OperatorBinding] | tuple[OperatorBinding, ...],
    ) -> CostBreakdown:
        """Evaluate prior and all bound observation operators."""
        trajectory = self.trajectory(control)
        if model_time.ndim != 1 or model_time.numel() != trajectory.shape[0]:
            raise ValueError("model_time must be 1D and match the control time dimension")
        if not bindings:
            raise ValueError("at least one observation operator binding is required")
        anomaly = trajectory - self.training_mean
        initial_anomaly = anomaly[0]
        initial = 0.5 * torch.sum(
            torch.square(
                torch.linalg.solve_triangular(
                    self.stationary_cholesky,
                    initial_anomaly.unsqueeze(1),
                    upper=False,
                )
            )
        )
        if trajectory.shape[0] > 1:
            model_error_residual = anomaly[1:] - anomaly[:-1] @ self.propagator.T
            whitened_error = torch.linalg.solve_triangular(
                self.model_error_cholesky,
                model_error_residual.T,
                upper=False,
            )
            model_error = 0.5 * torch.sum(torch.square(whitened_error))
        else:
            model_error = trajectory.sum() * 0

        predictions = []
        observed = []
        variances = []
        families = []
        for binding in bindings:
            result = binding.operator.predict(trajectory, model_time, binding.observations)
            prediction, values, variance, result_families = result
            if prediction.numel():
                predictions.append(prediction)
                observed.append(values)
                variances.append(variance)
                families.extend(result_families)
        if predictions:
            observation = gaussian_observation_loss(
                torch.cat(predictions),
                torch.cat(observed),
                torch.cat(variances),
                families=families,
            )
        else:
            empty = trajectory.new_empty((0,))
            observation = gaussian_observation_loss(empty, empty, empty, families=[])
        weighted_initial = self.initial_weight * initial
        weighted_model_error = self.model_error_weight * model_error
        total = weighted_initial + weighted_model_error + observation.total
        return CostBreakdown(total, weighted_initial, weighted_model_error, observation)

    def _validate_control(self, control: torch.Tensor) -> torch.Tensor:
        if control.ndim != 2 or control.shape[1] != self.state_size:
            raise ValueError("control must have shape (time, state_size)")
        if control.device != self.device or control.dtype != self.dtype:
            raise ValueError("control dtype and device must match the objective")
        if not torch.isfinite(control).all():
            raise ValueError("control must be finite")
        return control

    def _matrix(self, values, name: str) -> torch.Tensor:
        matrix = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square")
        if not torch.isfinite(matrix).all():
            raise ValueError(f"{name} must be finite")
        return matrix

    def _cholesky(self, values, name: str) -> torch.Tensor:
        matrix = self._matrix(values, name)
        try:
            return torch.linalg.cholesky(0.5 * (matrix + matrix.T))
        except RuntimeError as exc:
            raise ValueError(f"{name} must be positive definite") from exc
