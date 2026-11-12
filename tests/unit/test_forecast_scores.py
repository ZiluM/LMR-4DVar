import numpy as np

from lmr4dvar.verification.forecast import field_scores, forecast_pairs


def test_forecast_pairs_never_cross_physical_time_gaps():
    labels = np.array([0, 1, 2, 3, 8, 9, 10, 11])
    starts, ends = forecast_pairs(labels, 2)
    np.testing.assert_array_equal(starts, [0, 1, 4, 5])
    np.testing.assert_array_equal(ends, [2, 3, 6, 7])


def test_perfect_persistence_and_zero_forecast_metrics():
    truth = np.arange(5.0)[:, None]
    perfect = field_scores(truth, truth)
    assert perfect["correlation"][0] == 1
    assert perfect["rmse"][0] == 0
    assert perfect["ce"][0] == 1
    zero = field_scores(np.zeros_like(truth), truth)
    assert np.isnan(zero["correlation"][0])
    np.testing.assert_allclose(zero["rmse"], np.sqrt(6))
    np.testing.assert_allclose(zero["ce"], -2)


def test_missing_values_are_matched_per_point():
    truth = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
    predicted = np.array([[1.0, 4.0], [np.nan, 3.0], [3.0, 4.0]])
    result = field_scores(predicted, truth)
    np.testing.assert_array_equal(result["sample_count"], [2, 2])
    np.testing.assert_allclose(result["ce"], 1)
