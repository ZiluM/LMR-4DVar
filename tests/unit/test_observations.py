from dataclasses import dataclass

import numpy as np
import pytest
import torch

from lmr4dvar.observations import (
    EmptyObservationError,
    ObservationCollection,
    ObservationFamily,
    ObservationRecord,
    average_indexed_windows,
    average_to_windows,
    build_window_index,
    gaussian_observation_loss,
    match_nearest_times,
    sedpsm_record_to_observation,
)


def record(record_id: str = "proxy-1") -> ObservationRecord:
    return ObservationRecord(
        record_id=record_id,
        family=ObservationFamily.PAGES2K,
        variable="tas",
        latitude=45,
        longitude=-10,
        time=np.array([1000, 1001]),
        value=np.array([0.1, np.nan]),
        variance=0.25,
    )


def test_record_normalizes_longitude_and_broadcasts_variance() -> None:
    observation = record()

    assert observation.longitude == 350
    np.testing.assert_array_equal(observation.variance, [0.25, 0.25])
    np.testing.assert_array_equal(observation.valid_value_mask, [True, False])


def test_collection_rejects_duplicate_ids_and_counts_families() -> None:
    with pytest.raises(ValueError, match="unique"):
        ObservationCollection([record(), record()])

    collection = ObservationCollection([record("one"), record("two")])
    assert collection.observation_count == 4
    assert collection.counts_by_family()["pages2k"] == 4
    assert len(collection.by_family("sedpsm")) == 0


def test_nearest_time_matching_reports_unmatched_observations() -> None:
    match = match_nearest_times(
        np.array([1000.0, 1001.0, 1002.0]),
        np.array([1000.01, 1001.98, 1005.0]),
        tolerance=0.05,
    )

    np.testing.assert_array_equal(match.model_indices, [0, 2])
    np.testing.assert_array_equal(match.observation_indices, [0, 1])
    np.testing.assert_array_equal(match.unmatched_observation_indices, [2])


def test_window_averaging_is_differentiable_and_reports_counts() -> None:
    values = torch.arange(5.0, requires_grad=True)
    averaged, indices, counts = average_to_windows(
        values,
        torch.arange(5.0),
        torch.tensor([1.0, 3.5, 9.0]),
        widths=torch.tensor([2.0, 1.0, 1.0]),
    )

    torch.testing.assert_close(averaged, torch.tensor([1.0, 3.5]))
    torch.testing.assert_close(indices, torch.tensor([0, 1]))
    torch.testing.assert_close(counts, torch.tensor([3, 2]))
    averaged.sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_indexed_windows_match_dense_values_and_gradients() -> None:
    model_time = torch.tensor([0.0, 1.0, 2.0, 4.0, 7.0], dtype=torch.float64)
    observation_time = torch.tensor([1.0, 3.0, 8.0], dtype=torch.float64)
    bounds = torch.tensor([[0.0, 2.0], [4.0, 1.0], [8.0, 9.0]], dtype=torch.float64)
    indexed_values = torch.arange(10.0, dtype=torch.float64).reshape(5, 2)
    indexed_values.requires_grad_()
    dense_values = indexed_values.detach().clone().requires_grad_()

    index = build_window_index(model_time, observation_time, bounds=bounds)
    actual, indices, counts = average_indexed_windows(indexed_values, index)
    lower = torch.minimum(bounds[:, 0], bounds[:, 1])
    upper = torch.maximum(bounds[:, 0], bounds[:, 1])
    membership = (model_time[None, :] >= lower[:, None]) & (
        model_time[None, :] <= upper[:, None]
    )
    dense_counts = membership.sum(1)
    valid = dense_counts > 0
    weights = membership[valid].to(dense_values.dtype) / dense_counts[valid, None]
    expected = weights @ dense_values

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(indices, torch.tensor([0, 1]))
    torch.testing.assert_close(counts, torch.tensor([3, 3]))
    actual_gradient = torch.autograd.grad(actual.square().sum(), indexed_values)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), dense_values)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_window_index_rejects_unsorted_model_time() -> None:
    with pytest.raises(ValueError, match="sorted"):
        build_window_index(
            torch.tensor([0.0, 2.0, 1.0]),
            torch.tensor([1.0]),
            widths=1.0,
        )


def test_gaussian_loss_reports_family_terms_and_gradient() -> None:
    predicted = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    observed = torch.tensor([0.0, 1.0, float("nan")])
    variance = torch.tensor([1.0, 0.5, 1.0])

    result = gaussian_observation_loss(
        predicted, observed, variance, families=["pages2k", "sedpsm", "sedpsm"]
    )

    torch.testing.assert_close(result.total, torch.tensor(1.5))
    torch.testing.assert_close(result.by_family["pages2k"], torch.tensor(0.5))
    torch.testing.assert_close(result.by_family["sedpsm"], torch.tensor(1.0))
    assert result.used_count == 2
    assert result.dropped_count == 1
    result.total.backward()
    torch.testing.assert_close(predicted.grad, torch.tensor([1.0, 2.0, 0.0]))


def test_gaussian_loss_rejects_empty_valid_set() -> None:
    with pytest.raises(EmptyObservationError, match="zero finite matched"):
        gaussian_observation_loss(
            torch.tensor([float("nan")]), torch.tensor([1.0]), torch.tensor([1.0])
        )


@dataclass
class FakeSedPSMRecord:
    time: np.ndarray
    values: np.ndarray
    meta: dict
    name: str = "Uk37"


def test_sedpsm_adapter_uses_ensemble_mean_and_variance() -> None:
    source = FakeSedPSMRecord(
        time=np.array([1000.0, 1100.0]),
        values=np.array([[1.0, 3.0], [2.0, 2.0]]),
        meta={"lat": -5.0, "lon": 190.0, "dataSetName": "site-a", "proxy_type": "Uk37"},
    )

    observation = sedpsm_record_to_observation(source, index=0, experiment=2)

    assert observation.record_id == "SedPSM_PPE2_site-a"
    assert observation.family is ObservationFamily.SEDPSM
    np.testing.assert_allclose(observation.value, [2.0, 2.0])
    np.testing.assert_allclose(observation.variance, [1.0, 1e-8])
    assert observation.metadata["ensemble_size"] == 2
