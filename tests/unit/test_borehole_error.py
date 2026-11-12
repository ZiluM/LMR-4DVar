import numpy as np
import xarray as xr

from lmr4dvar.config.models import BoreholeErrorEstimationConfig
from lmr4dvar.observations import (
    effective_error_options,
    estimate_borehole_error_artifact,
    estimate_profile_error,
    load_borehole_observations,
)


def _profiles(depths: np.ndarray) -> dict[str, np.ndarray]:
    mean = 0.8 * np.exp(-depths / 180) - 0.1
    return {
        "mean": mean,
        "plus_2sigma": mean + 0.02 * (depths.max() - depths) / depths.max(),
        "minus_2sigma": mean - 0.02 * (depths.max() - depths) / depths.max(),
    }


def test_method_profiles_resolve_documented_candidate_ensembles() -> None:
    depths = np.arange(20.0, 305.0, 5.0)
    base = BoreholeErrorEstimationConfig()

    si = estimate_profile_error(
        _profiles(depths),
        depths,
        logging_year=2000,
        history_start_year=1300,
        time_steps=base.time_steps,
        thermal_conductivities=base.thermal_conductivities,
        volumetric_heat_capacities=base.volumetric_heat_capacities,
        options=effective_error_options(base),
        variance_floor=base.variance_floor,
    )
    legacy_config = BoreholeErrorEstimationConfig(method_profile="legacy_notebook")
    legacy = estimate_profile_error(
        _profiles(depths),
        depths,
        logging_year=2000,
        history_start_year=1300,
        time_steps=legacy_config.time_steps,
        thermal_conductivities=legacy_config.thermal_conductivities,
        volumetric_heat_capacities=legacy_config.volumetric_heat_capacities,
        options=effective_error_options(legacy_config),
        variance_floor=legacy_config.variance_floor,
    )

    assert si.candidate_count == 243
    assert legacy.candidate_count == 189
    assert si.retained_count > 0
    assert legacy.retained_count > 1
    assert np.all(si.variance >= base.variance_floor)


def test_error_artifact_is_consumed_by_borehole_loader(tmp_path) -> None:
    depth = np.arange(20.0, 305.0, 5.0)
    profiles = _profiles(depth)
    source = xr.Dataset(
        {
            "perturbation": (("station", "depth"), [profiles["mean"]]),
            "perturbation_plus_2sigma": (
                ("station", "depth"),
                [profiles["plus_2sigma"]],
            ),
            "perturbation_minus_2sigma": (
                ("station", "depth"),
                [profiles["minus_2sigma"]],
            ),
        },
        coords={
            "station": ["test-station"],
            "depth": depth,
            "lat": ("station", [45.0]),
            "lon": ("station", [-100.0]),
            "logging_year": ("station", [2000]),
            "region": ("station", ["test"]),
        },
    )
    source_path = tmp_path / "perturbations.nc"
    source.to_netcdf(source_path, engine="h5netcdf")
    config = BoreholeErrorEstimationConfig(
        enabled=True,
        perturbation_path=source_path,
        time_steps=[50],
        thermal_conductivities=[3.0],
        volumetric_heat_capacities=[3.0e6],
        svd_modes_by_step={50: [2]},
        gradient_threshold_k_per_year=10.0,
    )

    outputs = estimate_borehole_error_artifact(config, tmp_path / "output")
    loaded = load_borehole_observations(
        outputs[0], target_depths=np.array([100.0, 200.0, 300.0])
    )

    assert not loaded.skipped
    record = next(iter(loaded.observations))
    assert record.record_id == "test-station"
    assert record.longitude == 260.0
    assert np.all(record.variance > 0)
