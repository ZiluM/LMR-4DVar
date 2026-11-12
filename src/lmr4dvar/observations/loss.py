"""Auditable differentiable observation-loss calculations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class EmptyObservationError(RuntimeError):
    """Raised when no finite matched observations contribute to a loss."""


@dataclass(frozen=True)
class ObservationLoss:
    total: torch.Tensor
    by_family: dict[str, torch.Tensor]
    used_count: int
    dropped_count: int


def gaussian_observation_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    variance: torch.Tensor,
    *,
    families: list[str] | tuple[str, ...] | None = None,
    minimum_variance: float = 1e-8,
    require_nonempty: bool = True,
) -> ObservationLoss:
    """Compute 0.5 sum((H(x)-y)^2/R), retaining family contributions."""
    predicted = predicted.reshape(-1)
    observed = observed.to(device=predicted.device, dtype=predicted.dtype).reshape(-1)
    variance = variance.to(device=predicted.device, dtype=predicted.dtype).reshape(-1)
    if predicted.shape != observed.shape or predicted.shape != variance.shape:
        raise ValueError("predicted, observed, and variance must have equal sizes")
    if minimum_variance <= 0:
        raise ValueError("minimum_variance must be positive")
    if families is None:
        families = ("all",) * predicted.numel()
    if len(families) != predicted.numel():
        raise ValueError("families must identify every observation")
    valid = torch.isfinite(predicted) & torch.isfinite(observed) & torch.isfinite(variance)
    valid &= variance > 0
    used_count = int(valid.sum().item())
    dropped_count = predicted.numel() - used_count
    if used_count == 0 and require_nonempty:
        raise EmptyObservationError("zero finite matched observations contributed to loss")
    terms = 0.5 * torch.square(predicted[valid] - observed[valid]) / torch.clamp(
        variance[valid], min=minimum_variance
    )
    total = terms.sum() if used_count else predicted.sum() * 0
    valid_families = [
        family for family, keep in zip(families, valid.tolist(), strict=True) if keep
    ]
    by_family = {}
    for family in sorted(set(valid_families)):
        family_mask = torch.tensor(
            [item == family for item in valid_families], dtype=torch.bool, device=terms.device
        )
        by_family[family] = terms[family_mask].sum()
    return ObservationLoss(total, by_family, used_count, dropped_count)
