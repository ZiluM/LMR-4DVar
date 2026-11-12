import numpy as np
import pytest
import torch
import xarray as xr

from lmr4dvar.observations.bivariate_calibration import monthly_season_mean
from lmr4dvar.observations.quarterly_seasons import (
    complete_quarter_offsets,
    quarterly_season_mean,
)


@pytest.mark.parametrize(
    "months,expected",
    [
        ((-12, 1, 2), ((0, 0),)),
        ((3, 4, 5, 6, 7, 8), ((0, 1), (0, 2))),
        ((-9, -10, -11, -12, 1, 2), ((-1, 3), (0, 0))),
        ((12, 13, 14), ((1, 0),)),
        ((-3, -4, -5), ((-1, 1),)),
    ],
)
def test_complete_quarter_offsets(months, expected):
    assert complete_quarter_offsets(months) == expected


@pytest.mark.parametrize("months", [(1, 2), (5, 6, 7), tuple(range(1, 13)), (12,), ()])
def test_partial_quarters_are_rejected(months):
    with pytest.raises(ValueError):
        complete_quarter_offsets(months)


@pytest.mark.parametrize(
    "months",
    [(-12, 1, 2), (3, 4, 5, 6, 7, 8), (-9, -10, -11, -12, 1, 2), (12, 13, 14)],
)
def test_quarterly_means_equal_monthly_means(months):
    dates = xr.date_range("1999-12-01", periods=39, freq="MS")
    monthly = xr.DataArray(np.arange(39.0) ** 2, dims="time", coords={"time": dates})
    quarters = torch.tensor(monthly.values[:36].reshape(12, 3).mean(axis=1))
    time = torch.arange(12, dtype=torch.float64) / 4 + 2000
    years, result = quarterly_season_mean(quarters, time, months)
    expected = monthly_season_mean(monthly, months, years.numpy())
    np.testing.assert_allclose(result.numpy(), expected.values)


def test_boundary_drop_and_gradients_into_both_predictors():
    trajectory = torch.arange(24.0, dtype=torch.float64).reshape(12, 2).requires_grad_()
    time = torch.arange(12, dtype=torch.float64) / 4 - 1
    yt, t = quarterly_season_mean(trajectory[:, 0], time, (-9, -10, -11))
    yp, p = quarterly_season_mean(trajectory[:, 1], time, (12, 13, 14))
    assert yt.tolist() == [0, 1]
    assert yp.tolist() == [-1, 0]
    loss = 2 * t[0] + 3 * p[1]
    loss.backward()
    assert trajectory.grad[3, 0] == 2
    assert trajectory.grad[8, 1] == 3
    assert torch.count_nonzero(trajectory.grad) == 2


@pytest.mark.parametrize("bad_time", [[0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.75, 1.0]])
def test_shifted_or_missing_quarter_labels_are_rejected(bad_time):
    with pytest.raises(ValueError, match="consecutive"):
        quarterly_season_mean(torch.ones(4), torch.tensor(bad_time), (-12, 1, 2))
