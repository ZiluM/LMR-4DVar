import numpy as np
import pytest
import torch
import xarray as xr

from lmr4dvar.eof import EOFModel
from lmr4dvar.observations import (
    EOFPointDecoder,
    ObservationCollection,
    ObservationFamily,
    ObservationRecord,
    SedPSMOperator,
    gaussian_observation_loss,
    sedpsm_experiment_parameters,
)
from lmr4dvar.state import StateLayout


def _decoder() -> EOFPointDecoder:
    time = np.arange(12)
    values = 2.0 + np.sin(time * 2 * np.pi / 12)[:, None, None] * np.array(
        [[[1.0, -1.0], [0.5, -0.5]]]
    )
    field = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-45.0, 45.0], "lon": [0.0, 90.0]},
    )
    model, _ = EOFModel.fit(field, 1, center=True)
    return EOFPointDecoder({"tas": model}, StateLayout.from_mode_counts({"tas": 1}))


class FakeClimate:
    def __init__(self, values, years, **metadata):
        self.values = values
        self.years = np.asarray(years)
        self.metadata = metadata


class FakeFinal:
    def __init__(self, time, values):
        self.time = np.asarray(time)
        self.values = values

    def select_member(self, member):
        assert member == "mean"
        return self


def test_experiment_parameters_follow_ppe_rundown() -> None:
    exp0 = sedpsm_experiment_parameters(0, "Uk37", "site", 100)
    exp1 = sedpsm_experiment_parameters(1, "Uk37", "site", 100)
    exp2 = sedpsm_experiment_parameters(2, "Uk37", "site", 100)
    exp3 = sedpsm_experiment_parameters(3, "Uk37", "site", 100)
    exp4 = sedpsm_experiment_parameters(4, "Uk37", "site", 100, n_age_realizations=7)
    assert exp0 == exp1
    assert exp0["bio_depth"] == 0.5
    assert exp2["bio_depth"] == 5
    assert exp2["sigma_meas"] == 1.0e-5
    assert exp3["sigma_meas"] == 0.23
    assert "age_model" not in exp3
    assert exp4["age_model"] == "baconish"
    assert exp4["n_age_realizations"] == 7
    mgca = sedpsm_experiment_parameters(
        3, "MgCa", "Ocn-CariacoBasin.Black.2007", 10
    )
    assert mgca["species"] == "bulloides"
    assert mgca["sigma_ind"] == 2.0


def test_sedpsm_operator_runs_pipeline_and_backpropagates() -> None:
    calls = []

    def fake_pipeline(climate, sample_timepoints, **parameters):
        calls.append((climate, parameters))
        indices = [int(np.argmin(np.abs(climate.years - year))) for year in sample_timepoints]
        values = torch.sigmoid(climate.values[indices])
        return {}, FakeFinal(sample_timepoints, values)

    operator = SedPSMOperator(
        _decoder(),
        climatology={"SedPSM_PPE2_site-a": 20.0},
        run_pipeline=fake_pipeline,
        tensor_climate_type=FakeClimate,
    )
    trajectory = torch.linspace(-1, 1, 20, dtype=torch.float64).unsqueeze(1).requires_grad_()
    model_time = torch.tensor(
        [year + quarter / 4 for year in range(1996, 2001) for quarter in range(4)],
        dtype=torch.float64,
    )
    record = ObservationRecord(
        record_id="SedPSM_PPE2_site-a",
        family=ObservationFamily.SEDPSM,
        variable="tas",
        latitude=45,
        longitude=0,
        time=np.array([1997.0, 1999.0]),
        value=np.array([0.2, 0.3]),
        variance=np.array([0.1, 0.2]),
        metadata={
            "experiment": 2,
            "proxy_type": "Uk37",
            "dataset_name": "site-a",
            "ensemble_size": 5,
        },
    )
    predicted, observed, variance, families = operator.predict(
        trajectory, model_time, ObservationCollection([record])
    )
    assert predicted.shape == observed.shape == variance.shape == (2,)
    assert families == ["sedpsm", "sedpsm"]
    assert calls[0][0].metadata["units"] == "degC"
    assert calls[0][1]["nens"] == 5
    assert calls[0][1]["bio_depth"] == 5
    gaussian_observation_loss(predicted, observed, variance, families=families).total.backward()
    assert trajectory.grad is not None
    assert torch.isfinite(trajectory.grad).all()


def test_sedpsm_operator_requires_site_climatology() -> None:
    operator = SedPSMOperator(
        _decoder(),
        climatology={},
        run_pipeline=lambda *args, **kwargs: None,
        tensor_climate_type=FakeClimate,
    )
    record = ObservationRecord(
        "missing",
        ObservationFamily.SEDPSM,
        "tas",
        0,
        0,
        np.array([2000.0]),
        np.array([0.0]),
        1.0,
        metadata={"experiment": 0, "proxy_type": "Uk37", "ensemble_size": 1},
    )
    with pytest.raises(ValueError, match="climatology"):
        operator.predict(
            torch.zeros((4, 1)),
            torch.tensor([2000.0, 2000.25, 2000.5, 2000.75]),
            ObservationCollection([record]),
        )
