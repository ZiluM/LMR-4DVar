"""Observation-operator protocol shared by all proxy families."""

from __future__ import annotations

from typing import Protocol

import torch

from .records import ObservationCollection


class ObservationOperator(Protocol):
    def predict(
        self,
        trajectory: torch.Tensor,
        model_time: torch.Tensor,
        observations: ObservationCollection,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        """Return prediction, observation, variance, and family arrays."""
        ...
