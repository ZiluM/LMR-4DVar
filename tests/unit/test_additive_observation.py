from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmr4dvar.observations.additive import AdditiveProxyOperator
from lmr4dvar.observations.calibrated_artifact import (
    load_calibrated_observations,
    save_calibrated_observations,
)
from lmr4dvar.observations.linear import PointTransform
from lmr4dvar.observations.records import ObservationCollection, ObservationRecord
from lmr4dvar.state import StateLayout


def decoder():
    return SimpleNamespace(
        layout=StateLayout.from_mode_counts({"tas": 1, "pr": 1}),
        transform=lambda variable, lat, lon: PointTransform(
            np.array([1.0, 0.0]) if variable == "tas" else np.array([0.0, 1.0]),
            0.0,
            lat,
            lon,
        ),
    )


def record(reference=None, previous=False):
    metadata = {
        "psm_kind": "additive_linear_v1",
        "intercept": 0.5,
        "predictors": [
            {"variable": "tas", "slope": 2.0, "seasonality": [6, 7, 8], "units": "K"},
            {
                "variable": "pr",
                "slope": -3.0,
                "seasonality": [-9, -10, -11] if previous else [-12, 1, 2],
                "units": "mm/day",
            },
        ],
    }
    if reference is not None:
        metadata["anomaly_reference_period"] = reference
    return ObservationRecord(
        "tree-a",
        "pages2k",
        "tas",
        45.0,
        0.0,
        np.array([2000.0, 2001.0, 2002.0]),
        np.array([1.0, np.nan, 2.0]),
        0.25,
        metadata=metadata,
    )


def test_independent_predictor_seasons_and_gradients():
    operator = AdditiveProxyOperator(decoder())
    observations = ObservationCollection([record()])
    trajectory = torch.arange(24.0, dtype=torch.float64).reshape(12, 2).requires_grad_()
    time = torch.arange(12, dtype=torch.float64) / 4 + 2000
    predicted, observed, variance, families = operator.predict(trajectory, time, observations)
    expected = 0.5 + 2 * trajectory[[2, 6, 10], 0] - 3 * trajectory[[0, 4, 8], 1]
    torch.testing.assert_close(predicted, expected)
    torch.testing.assert_close(variance, torch.full((3,), 0.25, dtype=torch.float64))
    assert torch.isnan(observed[1])
    assert families == ["pages2k"] * 3
    assert torch.autograd.gradcheck(
        lambda x: operator.predict(x, time, observations)[0], (trajectory,)
    )
    predicted.sum().backward()
    assert torch.count_nonzero(trajectory.grad[:, 0]) == 3
    assert torch.count_nonzero(trajectory.grad[:, 1]) == 3


def test_previous_season_drops_boundary_in_both_terms():
    trajectory = torch.arange(24.0, dtype=torch.float64).reshape(12, 2)
    time = torch.arange(12, dtype=torch.float64) / 4 + 2000
    prediction, observed, _, _ = AdditiveProxyOperator(decoder()).predict(
        trajectory, time, ObservationCollection([record(previous=True)])
    )
    expected = 0.5 + 2 * trajectory[[6, 10], 0] - 3 * trajectory[[3, 7], 1]
    torch.testing.assert_close(prediction, expected)
    assert observed.numel() == 2


def test_reference_mean_is_differentiable_and_complete():
    trajectory = torch.randn(12, 2, dtype=torch.float64, requires_grad=True)
    time = torch.arange(12, dtype=torch.float64) / 4 + 2000
    observations = ObservationCollection([record(reference=[2000, 2001])])
    operator = AdditiveProxyOperator(decoder())
    assert torch.autograd.gradcheck(
        lambda x: operator.predict(x, time, observations)[0], (trajectory,)
    )
    predicted = operator.predict(trajectory, time, observations)[0]
    torch.testing.assert_close(predicted[:2].mean(), torch.tensor(0.5, dtype=torch.float64))
    with pytest.raises(ValueError, match="incomplete anomaly"):
        operator.predict(trajectory, time, ObservationCollection([record(reference=[1999, 2001])]))


def test_no_time_match_or_partial_season_is_not_silently_accepted():
    time = torch.arange(12, dtype=torch.float64) / 4 + 1900
    operator = AdditiveProxyOperator(decoder())
    with pytest.raises(ValueError, match="no observations matched"):
        operator.predict(torch.zeros(12, 2), time, ObservationCollection([record()]))
    invalid = record()
    invalid.metadata["predictors"][0]["seasonality"] = [5, 6, 7]
    with pytest.raises(ValueError, match="complete"):
        operator.predict(torch.zeros(12, 2), time, ObservationCollection([invalid]))


def test_calibrated_netcdf_roundtrip_and_schema_validation(tmp_path):
    path = save_calibrated_observations(tmp_path / "proxies.nc", ObservationCollection([record()]))
    restored = list(load_calibrated_observations(path))[0]
    assert restored.metadata == record().metadata
    np.testing.assert_equal(restored.value, record().value)
    np.testing.assert_equal(restored.variance, record().variance)
    assert restored.time_bounds is None


def test_temporal_windows_do_not_use_truncated_boundary_means():
    proxy = record()
    proxy.metadata["resolution_years"] = 2.0
    observations = ObservationCollection([proxy])
    trajectory = torch.arange(24.0, dtype=torch.float64).reshape(12, 2).requires_grad_()
    time = torch.arange(12, dtype=torch.float64) / 4 + 2000
    operator = AdditiveProxyOperator(decoder())
    result, observed, _, _ = operator.predict(trajectory, time, observations)
    assert result.numel() == observed.numel() == 1
    expected = 0.5 + 2 * trajectory[[2, 6, 10], 0].mean() - 3 * trajectory[[0, 4, 8], 1].mean()
    torch.testing.assert_close(result[0], expected)
    assert torch.autograd.gradcheck(
        lambda x: operator.predict(x, time, observations)[0], (trajectory,)
    )
