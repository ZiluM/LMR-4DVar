"""Held-out field forecast scores with explicit physical-time pair selection."""

from __future__ import annotations

import numpy as np


def forecast_pairs(labels: np.ndarray, lead: int) -> tuple[np.ndarray, np.ndarray]:
    """Return only starts whose whole lead interval has consecutive quarters."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or lead < 1 or labels.size <= lead:
        raise ValueError("forecast labels must be 1D and longer than a positive lead")
    if np.any(np.diff(labels) <= 0):
        raise ValueError("forecast quarter labels must be strictly increasing")
    starts = np.arange(labels.size - lead)
    valid = np.ones(starts.size, dtype=bool)
    for step in range(lead):
        valid &= labels[starts + step + 1] - labels[starts + step] == 1
    starts = starts[valid]
    if not starts.size:
        raise ValueError("no forecast pairs remain inside continuous physical-time blocks")
    return starts, starts + lead


def field_scores(predicted: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Temporal gridpoint r, RMSE and CE using pairwise finite samples."""
    predicted, truth = np.asarray(predicted, float), np.asarray(truth, float)
    if predicted.shape != truth.shape or predicted.ndim < 2:
        raise ValueError("forecast and truth must have matching (sample, space...) shapes")
    valid = np.isfinite(predicted) & np.isfinite(truth)
    count = valid.sum(axis=0)
    p = np.where(valid, predicted, 0)
    t = np.where(valid, truth, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pm, tm = p.sum(axis=0) / count, t.sum(axis=0) / count
        pa, ta = np.where(valid, p - pm, 0), np.where(valid, t - tm, 0)
        pss, tss = (pa**2).sum(axis=0), (ta**2).sum(axis=0)
        sse = np.where(valid, (predicted - truth) ** 2, 0).sum(axis=0)
        r = (pa * ta).sum(axis=0) / np.sqrt(pss * tss)
        rmse = np.sqrt(sse / count)
        ce = 1 - sse / tss
    return {
        "correlation": np.where((count > 1) & (pss > 0) & (tss > 0), r, np.nan),
        "rmse": np.where(count > 0, rmse, np.nan),
        "ce": np.where((count > 1) & (tss > 0), ce, np.nan),
        "sample_count": count,
    }
