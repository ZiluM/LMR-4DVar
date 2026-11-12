import json

import numpy as np
import pandas as pd
import torch
import xarray as xr

from lmr4dvar.assimilation import OperatorBinding
from lmr4dvar.config.models import ProjectConfig
from lmr4dvar.lim import LIMModel
from lmr4dvar.observations import ObservationCollection
from lmr4dvar.workflow import Stage, StageStatus, build_plan, execute_stage, execute_through


def test_preprocess_eof_lim_local_workflow(tmp_path, monkeypatch) -> None:
    simulation = tmp_path / "simulation"
    simulation.mkdir()
    rng = np.random.default_rng(8)
    coefficient = np.zeros(240)
    for index in range(1, coefficient.size):
        coefficient[index] = 0.85 * coefficient[index - 1] + rng.normal(scale=0.4)
    pattern = np.array([[1.0, -0.6], [0.4, -0.2]])
    field = xr.DataArray(
        coefficient[:, None, None] * pattern,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("1800-01-01", periods=240, freq="MS"),
            "lat": [-45.0, 45.0],
            "lon": [0.0, 180.0],
        },
        name="temperature",
    )
    field.to_dataset().to_netcdf(simulation / "tas.nc", engine="h5netcdf")
    config = ProjectConfig.model_validate(
        {
            "project": {"name": "local-integration"},
            "paths": {"simulation_root": simulation, "output_root": tmp_path / "output"},
            "state": {
                "variables": {
                    "tas": {
                        "domain": "global",
                        "n_modes": 1,
                        "source_name": "temperature",
                        "input_pattern": "tas.nc",
                    }
                }
            },
            "preprocess": {
                "regrid": "none",
                "remove_monthly_climatology": False,
                "quarterly_mode": "complete",
                "seasonal_months": [1, 4, 7, 10],
                "detrend": "none",
            },
            "observations": {"pages2k": {"enabled": True}},
            "assimilation": {"start": "1800-01", "end": "1819-12"},
            "reconstruction": {"variables": ["tas"], "indices": ["GMT"]},
            "verification": {
                "anomaly_period": [1800, 1809],
                "comparison_period": [1800, 1819],
                "plot_figures": False,
                "reference_fields": {
                    "tas": tmp_path
                    / "output/local-integration/stages/reconstruct/fields/tas.nc"
                },
            },
        }
    )

    outputs = execute_through(config)

    assert set(outputs) == {Stage.PREPROCESS, Stage.EOF, Stage.TRAIN_LIM}
    plan = build_plan(config)
    assert [item.status for item in plan[:4]] == [
        StageStatus.COMPLETE,
        StageStatus.COMPLETE,
        StageStatus.COMPLETE,
        StageStatus.READY,
    ]
    eof_path = tmp_path / "output/local-integration/stages/eof/state_pcs.nc"
    with xr.open_dataset(eof_path, engine="h5netcdf") as dataset:
        assert dataset.pcs.state.values.tolist() == ["tas:000"]
        assert dataset.pcs.sizes["time"] > 50
    lim = LIMModel.load(tmp_path / "output/local-integration/stages/train-lim")
    assert lim.spectral_radius < 1

    repeated = execute_through(config)
    assert repeated[Stage.TRAIN_LIM] == outputs[Stage.TRAIN_LIM]

    class ZeroObservationOperator:
        def predict(self, trajectory, model_time, observations):
            return (
                trajectory[:1, 0],
                torch.zeros(1, dtype=trajectory.dtype, device=trajectory.device),
                torch.ones(1, dtype=trajectory.dtype, device=trajectory.device),
                ["synthetic"],
            )

    monkeypatch.setattr(
        "lmr4dvar.workflow.assimilation.build_observation_bindings",
        lambda workflow_config, decoder: [
            OperatorBinding(ZeroObservationOperator(), ObservationCollection([]))
        ],
    )
    calibration_outputs = execute_stage(config, Stage.CALIBRATE_PAGES2K)
    assert {path.name for path in calibration_outputs} == {"manifest.json"}
    temp12k_outputs = execute_stage(config, Stage.PREPARE_TEMP12K)
    assert {path.name for path in temp12k_outputs} == {"manifest.json"}
    borehole_outputs = execute_stage(config, Stage.ESTIMATE_BOREHOLE_ERROR)
    assert {path.name for path in borehole_outputs} == {"manifest.json"}
    assimilation_outputs = execute_stage(config, Stage.ASSIMILATE)
    assert {path.name for path in assimilation_outputs} == {"analysis.nc", "diagnostics.json"}
    with xr.open_dataset(assimilation_outputs[0], engine="h5netcdf") as dataset:
        assert dataset.trajectory.shape == (80, 1)
        assert dataset.state.values.tolist() == ["tas:000"]
        assert dataset.attrs["control_parameterization"] == "state_trajectory"
        assert dataset.attrs["numeric_precision"] == "float32"
        assert dataset.trajectory.dtype == np.float32
        np.testing.assert_array_equal(dataset.control, dataset.trajectory)
    diagnostics = json.loads(assimilation_outputs[1].read_text(encoding="utf-8"))
    assert diagnostics["control_parameterization"] == "state_trajectory"
    assert diagnostics["numeric_precision"] == "float32"
    assert build_plan(config)[7].status is StageStatus.READY
    reconstruction_outputs = execute_stage(config, Stage.RECONSTRUCT)
    assert {path.name for path in reconstruction_outputs} == {
        "tas.nc",
        "indices.nc",
        "seasonal.png",
        "annual.png",
    }
    assert all(path.stat().st_size > 0 for path in reconstruction_outputs)
    with xr.open_dataset(reconstruction_outputs[0], engine="h5netcdf") as dataset:
        assert dataset.tas.shape == (80, 2, 2)
    assert build_plan(config)[8].status is StageStatus.READY
    verification_outputs = execute_stage(config, Stage.VERIFY)
    assert {path.name for path in verification_outputs} == {"pattern_metrics.nc", "summary.json"}
    with xr.open_dataset(verification_outputs[0], engine="h5netcdf") as dataset:
        np.testing.assert_allclose(dataset.pattern_rmse, 0.0)
        assert "pattern_correlation" not in dataset
        assert bool(dataset.gridpoint_temporal_correlation.isnull().all())
        assert bool(dataset.area_weighted_mean_correlation.isnull().all())
        assert dataset.aggregation.values.tolist() == [
            "quarterly",
            "annual",
            "DJF",
            "MAM",
            "JJA",
            "SON",
        ]
    verification_summary = json.loads(verification_outputs[1].read_text())
    assert np.isnan(
        verification_summary["spatial_correlation"]["tas"]["annual"][
            "area_weighted_mean_correlation"
        ]
    )
    assert all(item.status is StageStatus.COMPLETE for item in build_plan(config))
