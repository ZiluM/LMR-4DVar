"""Exact equal-month seasonal means from DJF/MAM/JJA/SON trajectories."""

from __future__ import annotations

import numpy as np
import torch


def complete_quarter_offsets(months: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Translate LMR month encoding to (year offset, DJF-based quarter).

    Only unions of complete quarters are representable. In particular,
    calendar Jan-Dec is not the Dec-Nov annual mean of four DJF-based quarters.
    """
    if not months or len(set(months)) != len(months):
        raise ValueError("season months must be nonempty and unique")
    groups: dict[int, set[int]] = {}
    for month in months:
        if not isinstance(month, (int, np.integer)) or month == 0 or not -12 <= month <= 24:
            raise ValueError("season months must be integers in -12..-1 or 1..24")
        index = abs(month) - 13 if month < 0 else month - 1
        quarter = (index + 1) // 3
        groups.setdefault(quarter, set()).add(index)
    for quarter, indices in groups.items():
        if indices != set(range(3 * quarter - 1, 3 * quarter + 2)):
            raise ValueError("season must contain only complete DJF/MAM/JJA/SON quarters")
    return tuple(divmod(quarter, 4) for quarter in sorted(groups))


def quarterly_season_mean(
    values: torch.Tensor, time: torch.Tensor, months: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average exact seasonal windows; drop missing boundary quarters.

    Times are decimal years at January/April/July/October. Input quarters
    must themselves be complete equal-month means with verified physical
    time labels. This function cannot repair shifted source timestamps.
    """
    if values.ndim not in (1, 2) or time.ndim != 1 or values.shape[0] != time.numel():
        raise ValueError("quarterly values must be 1D/2D and align with 1D time")
    if not time.is_floating_point() or time.device != values.device:
        raise ValueError("time must be floating point and on the values device")
    if time.numel() < 4 or time.numel() % 4:
        raise ValueError("quarterly trajectory must contain complete four-step years")
    if not torch.isfinite(time).all():
        raise ValueError("quarterly time must be finite")
    first_year = torch.floor(time[0])
    expected = first_year + torch.arange(time.numel(), device=time.device, dtype=time.dtype) / 4
    if not torch.allclose(time, expected, atol=1e-6, rtol=0):
        raise ValueError("quarterly time must be consecutive Jan/Apr/Jul/Oct steps")
    offsets = complete_quarter_offsets(months)
    n_years = time.numel() // 4
    first_target = max(0, -min(offset for offset, _ in offsets))
    stop_target = min(n_years, n_years - max(offset for offset, _ in offsets))
    targets = torch.arange(first_target, max(first_target, stop_target), device=time.device)
    quarterly = values.reshape(n_years, 4, *values.shape[1:])
    seasonal = torch.stack(
        [quarterly[targets + offset, quarter] for offset, quarter in offsets], dim=0
    ).mean(dim=0)
    years = torch.floor(time[::4]).to(torch.int64)
    return years[targets], seasonal
