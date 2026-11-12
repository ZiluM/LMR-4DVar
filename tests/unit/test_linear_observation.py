import numpy as np
import pytest
import torch
import xarray as xr

import lmr4dvar.observations.linear as linear_module
from lmr4dvar.eof import EOFModel
from lmr4dvar.observations import (
    EOFPointDecoder,
    LinearProxyOperator,
    ObservationCollection,
    ObservationFamily,
    ObservationRecord,
    annualize_quarterly,
    gaussian_observation_loss,
)
from lmr4dvar.state import StateLayout


def eof_and_pcs() -> tuple[EOFModel, xr.DataArray]:
    time = np.arange(12)
    score_1 = np.sin(np.arange(12) * 2 * np.pi / 12)
    score_2 = np.cos(np.arange(12) * 2 * np.pi / 6)
    pattern_1 = np.array([[1.0, -1.0], [0.5, -0.5]])
    pattern_2 = np.array([[0.2, 0.8], [-0.2, -0.8]])
    values = 2.0 + score_1[:, None, None] * pattern_1 + score_2[:, None, None] * pattern_2
    field = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-45.0, 45.0], "lon": [0.0, 90.0]},
    )
    return EOFModel.fit(field, 2, center=True)


def test_state_layout_preserves_variable_order_and_labels() -> None:
    layout = StateLayout.from_mode_counts({"tas": 2, "tos": 1})
    assert layout.size == 3
    assert layout.labels == ("tas:000", "tas:001", "tos:000")
    assert layout.slice_for("tos") == slice(2, 3)
    with pytest.raises(KeyError, match="absent"):
        layout.slice_for("sic")


def test_point_transform_matches_full_eof_decode() -> None:
    model, pcs = eof_and_pcs()
    decoder = EOFPointDecoder({"tas": model}, StateLayout.from_mode_counts({"tas": 2}))
    transform = decoder.transform("tas", latitude=40, longitude=5)
    local = pcs.values @ transform.weights + transform.bias
    full = model.decode(pcs).sel(lat=45, lon=0).values
    np.testing.assert_allclose(local, full, atol=1e-12)
    assert (transform.grid_latitude, transform.grid_longitude) == (45, 0)
    assert decoder.transform("tas", latitude=40, longitude=5) is transform


def test_annualization_selects_quarters_and_explicit_edge_policy() -> None:
    values = torch.arange(8.0)
    time = torch.tensor([2000.0, 2000.25, 2000.5, 2000.75, 2001.0, 2001.25, 2001.5, 2001.75])
    years, mam = annualize_quarterly(values, time, [3, 4, 5])
    _, legacy = annualize_quarterly(values, time, [-9, -10, -11])
    drop_years, dropped = annualize_quarterly(values, time, [-9, -10, -11], edge_policy="drop")
    torch.testing.assert_close(years, torch.tensor([2000, 2001]))
    torch.testing.assert_close(mam, torch.tensor([1.0, 5.0]))
    torch.testing.assert_close(legacy, torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(drop_years, torch.tensor([2001]))
    torch.testing.assert_close(dropped, torch.tensor([3.0]))


def test_great_circle_point_selection_wraps_longitude_and_preserves_legacy_default():
    model, _ = eof_and_pcs()
    layout = StateLayout.from_mode_counts({"tas": 2})
    legacy = EOFPointDecoder({"tas": model}, layout).transform("tas", 45, 359)
    corrected = EOFPointDecoder({"tas": model}, layout, spatial_policy="great_circle").transform(
        "tas", 45, 359
    )
    assert legacy.grid_longitude == 90
    assert corrected.grid_longitude == 0


def test_annualization_supports_same_year_december() -> None:
    values = torch.arange(8.0)
    time = torch.tensor([2000.0, 2000.25, 2000.5, 2000.75, 2001.0, 2001.25, 2001.5, 2001.75])

    years, december = annualize_quarterly(values, time, [12])

    torch.testing.assert_close(years, torch.tensor([2000, 2001]))
    torch.testing.assert_close(december, torch.tensor([3.0, 7.0]))


def test_linear_proxy_operator_predicts_and_backpropagates() -> None:
    model, _ = eof_and_pcs()
    decoder = EOFPointDecoder({"tas": model}, StateLayout.from_mode_counts({"tas": 2}))
    operator = LinearProxyOperator(decoder)
    trajectory = torch.arange(16.0, dtype=torch.float64).reshape(8, 2).requires_grad_()
    model_time = torch.tensor(
        [2000.0, 2000.25, 2000.5, 2000.75, 2001.0, 2001.25, 2001.5, 2001.75],
        dtype=torch.float64,
    )
    observation = ObservationRecord(
        record_id="pages-a",
        family=ObservationFamily.PAGES2K,
        variable="tas",
        latitude=45,
        longitude=0,
        time=np.array([2000.0, 2001.0]),
        value=np.array([1.0, 2.0]),
        variance=np.array([0.5, 0.5]),
        metadata={"seasonality": [3, 4, 5], "slope": 2.0, "intercept": -1.0},
    )
    predicted, observed, variance, families = operator.predict(
        trajectory, model_time, ObservationCollection([observation])
    )
    transform = decoder.transform("tas", 45, 0)
    expected = (
        trajectory[[1, 5]] @ torch.tensor(transform.weights, dtype=trajectory.dtype)
        + transform.bias
    ) * 2 - 1
    torch.testing.assert_close(predicted, expected)
    torch.testing.assert_close(observed, torch.tensor([1.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(variance, torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert families == ["pages2k", "pages2k"]
    loss = gaussian_observation_loss(predicted, observed, variance, families=families).total
    loss.backward()
    assert trajectory.grad is not None
    assert torch.isfinite(trajectory.grad).all()


def test_linear_proxy_operator_returns_empty_for_no_time_match() -> None:
    model, _ = eof_and_pcs()
    operator = LinearProxyOperator(
        EOFPointDecoder({"tas": model}, StateLayout.from_mode_counts({"tas": 2}))
    )
    observation = ObservationRecord(
        "outside",
        ObservationFamily.TEMP12K,
        "tas",
        45,
        0,
        np.array([1500.0]),
        np.array([0.0]),
        1.0,
    )
    predicted, observed, variance, families = operator.predict(
        torch.zeros((4, 2)),
        torch.tensor([2000.0, 2000.25, 2000.5, 2000.75]),
        ObservationCollection([observation]),
    )
    assert predicted.numel() == observed.numel() == variance.numel() == 0
    assert families == []


def test_temp12k_model_estimate_is_centered_on_reference_period() -> None:
    model, _ = eof_and_pcs()
    decoder = EOFPointDecoder({"tas": model}, StateLayout.from_mode_counts({"tas": 2}))
    operator = LinearProxyOperator(decoder)
    trajectory = torch.arange(24.0, dtype=torch.float64).reshape(12, 2).requires_grad_()
    model_time = torch.tensor(
        [year + quarter / 4 for year in range(1999, 2002) for quarter in range(4)],
        dtype=torch.float64,
    )
    observation = ObservationRecord(
        "temp-a",
        ObservationFamily.TEMP12K,
        "tas",
        45,
        0,
        np.array([1999.0, 2000.0, 2001.0]),
        np.zeros(3),
        1.0,
        metadata={"anomaly_reference_period": [1999.0, 2000.0]},
    )

    predicted, *_ = operator.predict(trajectory, model_time, ObservationCollection([observation]))

    torch.testing.assert_close(predicted[:2].mean(), torch.tensor(0.0, dtype=torch.float64))
    predicted.sum().backward()
    assert trajectory.grad is not None


def test_linear_proxy_operator_caches_window_indices(monkeypatch) -> None:
    model, _ = eof_and_pcs()
    decoder = EOFPointDecoder({"tas": model}, StateLayout.from_mode_counts({"tas": 2}))
    calls = 0
    original = linear_module.build_window_index

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(linear_module, "build_window_index", counted)
    operator = LinearProxyOperator(decoder)
    trajectory = torch.arange(16.0, dtype=torch.float64).reshape(8, 2)
    model_time = torch.tensor(
        [2000.0, 2000.25, 2000.5, 2000.75, 2001.0, 2001.25, 2001.5, 2001.75],
        dtype=torch.float64,
    )
    observation = ObservationRecord(
        "temp-window",
        ObservationFamily.TEMP12K,
        "tas",
        45,
        0,
        np.array([2000.0, 2001.0]),
        np.zeros(2),
        1.0,
        metadata={"resolution_years": 2.0},
    )
    collection = ObservationCollection([observation])

    first = operator.predict(trajectory, model_time, collection)[0]
    second = operator.predict(trajectory + 1, model_time, collection)[0]

    assert calls == 1
    assert first.shape == second.shape == (2,)


def test_linear_proxy_operator_batches_records_and_caches_exact_matches(monkeypatch) -> None:
    model, _ = eof_and_pcs()
    decoder = EOFPointDecoder({"tas": model}, StateLayout.from_mode_counts({"tas": 2}))
    annualize_calls = 0
    match_calls = 0
    original_annualize = linear_module.annualize_quarterly
    original_match = linear_module.match_nearest_times

    def counted_annualize(*args, **kwargs):
        nonlocal annualize_calls
        annualize_calls += 1
        return original_annualize(*args, **kwargs)

    def counted_match(*args, **kwargs):
        nonlocal match_calls
        match_calls += 1
        return original_match(*args, **kwargs)

    monkeypatch.setattr(linear_module, "annualize_quarterly", counted_annualize)
    monkeypatch.setattr(linear_module, "match_nearest_times", counted_match)
    records = [
        ObservationRecord(
            f"pages-{index}",
            ObservationFamily.PAGES2K,
            "tas",
            latitude,
            0,
            np.array([2000.0, 2001.0]),
            np.array([1.0, 2.0]),
            0.5,
            metadata={"seasonality": [6, 7, 8]},
        )
        for index, latitude in enumerate((-45.0, 45.0))
    ]
    collection = ObservationCollection(records)
    operator = LinearProxyOperator(decoder)
    trajectory = torch.arange(16.0, dtype=torch.float64).reshape(8, 2)
    model_time = torch.tensor(
        [2000.0, 2000.25, 2000.5, 2000.75, 2001.0, 2001.25, 2001.5, 2001.75],
        dtype=torch.float64,
    )

    first = operator.predict(trajectory, model_time, collection)[0]
    second = operator.predict(trajectory + 1, model_time, collection)[0]

    assert first.shape == second.shape == (4,)
    assert annualize_calls == 2
    assert match_calls == 2
