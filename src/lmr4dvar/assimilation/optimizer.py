"""Deterministic L-BFGS driver for weak-constraint 4D-Var."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .objective import OperatorBinding, WeakConstraintObjective


@dataclass(frozen=True)
class AssimilationResult:
    control: torch.Tensor
    trajectory: torch.Tensor
    cost_history: tuple[float, ...]
    initial_history: tuple[float, ...]
    model_error_history: tuple[float, ...]
    observation_history: tuple[float, ...]
    observation_by_family_history: tuple[dict[str, float], ...]
    observation_used_count_history: tuple[int, ...]
    observation_dropped_count_history: tuple[int, ...]
    gradient_norm_history: tuple[float, ...]
    function_evaluations: int
    outer_iterations: int
    outer_cost_history: tuple[float, ...]
    final_gradient_norm: float
    converged: bool
    convergence_reason: str | None


def assimilate_lbfgs(
    objective: WeakConstraintObjective,
    model_time: torch.Tensor,
    bindings: list[OperatorBinding] | tuple[OperatorBinding, ...],
    *,
    initial_control: torch.Tensor | None = None,
    max_iterations: int = 100,
    inner_iterations: int = 20,
    minimum_outer_iterations: int = 22,
    tolerance_gradient: float = 1.0e-7,
    tolerance_change: float = 1.0e-9,
    outer_tolerance_change: float = 1.0e-6,
    history_size: int = 100,
) -> AssimilationResult:
    """Minimize the objective with persistent, nested L-BFGS iterations.

    ``max_iterations`` counts outer calls to ``LBFGS.step`` and
    ``inner_iterations`` is PyTorch's ``LBFGS(max_iter=...)``. The optimizer is
    retained across outer calls so its curvature history is not discarded.
    Cost-change convergence is enabled only after ``minimum_outer_iterations``.
    """
    if (
        max_iterations < 1
        or inner_iterations < 1
        or minimum_outer_iterations < 1
        or history_size < 1
    ):
        raise ValueError(
            "max_iterations, inner_iterations, minimum_outer_iterations, and "
            "history_size must be positive"
        )
    if (
        tolerance_gradient <= 0
        or tolerance_change <= 0
        or outer_tolerance_change <= 0
    ):
        raise ValueError("optimizer tolerances must be positive")
    if initial_control is None:
        initial_control = objective.background_control(model_time.numel())
    control = initial_control.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [control],
        lr=1.0,
        max_iter=inner_iterations,
        tolerance_grad=tolerance_gradient,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    total_history = []
    initial_history = []
    model_error_history = []
    observation_history = []
    observation_by_family_history = []
    observation_used_count_history = []
    observation_dropped_count_history = []
    gradient_history = []
    outer_cost_history = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        cost = objective.evaluate(control, model_time, bindings)
        cost.total.backward()
        gradient_norm = torch.linalg.vector_norm(control.grad)
        total_history.append(float(cost.total.detach().cpu()))
        initial_history.append(float(cost.initial.detach().cpu()))
        model_error_history.append(float(cost.model_error.detach().cpu()))
        observation_history.append(float(cost.observation.total.detach().cpu()))
        observation_by_family_history.append(
            {
                family: float(value.detach().cpu())
                for family, value in cost.observation.by_family.items()
            }
        )
        observation_used_count_history.append(cost.observation.used_count)
        observation_dropped_count_history.append(cost.observation.dropped_count)
        gradient_history.append(float(gradient_norm.detach().cpu()))
        return cost.total

    converged = False
    convergence_reason = None
    for _ in range(max_iterations):
        optimizer.step(closure)
        with torch.no_grad():
            outer_cost = float(
                objective.evaluate(control, model_time, bindings).total.detach().cpu()
            )
        outer_cost_history.append(outer_cost)
        if (
            len(outer_cost_history) >= minimum_outer_iterations
            and len(outer_cost_history) > 1
            and abs(outer_cost_history[-2] - outer_cost) <= outer_tolerance_change
        ):
            converged = True
            convergence_reason = "cost_change"
            break

    optimizer.zero_grad()
    final_cost = objective.evaluate(control, model_time, bindings)
    final_cost.total.backward()
    final_gradient_norm = float(torch.linalg.vector_norm(control.grad).detach().cpu())
    if final_gradient_norm <= tolerance_gradient:
        converged = True
        convergence_reason = "gradient"

    with torch.no_grad():
        trajectory = objective.trajectory(control)
    state = optimizer.state[control]
    evaluations = int(state.get("func_evals", len(total_history)))
    return AssimilationResult(
        control=control.detach(),
        trajectory=trajectory.detach(),
        cost_history=tuple(total_history),
        initial_history=tuple(initial_history),
        model_error_history=tuple(model_error_history),
        observation_history=tuple(observation_history),
        observation_by_family_history=tuple(observation_by_family_history),
        observation_used_count_history=tuple(observation_used_count_history),
        observation_dropped_count_history=tuple(observation_dropped_count_history),
        gradient_norm_history=tuple(gradient_history),
        function_evaluations=evaluations,
        outer_iterations=len(outer_cost_history),
        outer_cost_history=tuple(outer_cost_history),
        final_gradient_norm=final_gradient_norm,
        converged=converged,
        convergence_reason=convergence_reason,
    )
