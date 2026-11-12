from pathlib import Path

import pytest
import yaml

from lmr4dvar.config import load_project_config


def base_config(tmp_path: Path) -> dict:
    return {
        "project": {"name": "test"},
        "paths": {
            "simulation_root": str(tmp_path / "sim"),
            "output_root": str(tmp_path / "out"),
            "sedpsm_root": str(tmp_path / "sedpsm"),
        },
        "state": {
            "timestep_months": 3,
            "variables": {"tas": {"domain": "global", "n_modes": 2}},
        },
        "observations": {
            "sedpsm": {
                "enabled": True,
                "experiment": 0,
                "path": str(tmp_path / "data" / "seddb.pkl"),
            }
        },
        "assimilation": {"start": "0850-01", "end": "1850-12"},
        "reconstruction": {"variables": ["tas"]},
        "verification": {
            "anomaly_period": [850, 1849],
            "comparison_period": [850, 1850],
        },
    }


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_loads_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    write_yaml(path, base_config(tmp_path))

    config = load_project_config(path)

    assert config.project.name == "test"
    assert config.state.variables["tas"].n_modes == 2
    assert config.state.variables["tas"].eof_fillna_value is None
    assert config.paths.prior_run_root is None
    assert config.observations.sedpsm.enabled
    assert config.observations.sedpsm.path == tmp_path / "data" / "seddb.pkl"
    assert config.observations.sedpsm.n_age_realizations == 1
    assert config.observations.sedpsm.variance_floor == 1e-8
    assert config.preprocess.quarterly_mode.value == "complete"
    assert config.preprocess.input_kind.value == "monthly"
    assert config.preprocess.detrend.value == "linear_by_month"
    assert not config.eof.center
    assert config.assimilation.initial_weight == 1.0
    assert config.assimilation.model_error_weight == 1.0
    assert config.assimilation.precision.value == "float32"


def test_sedpsm_legacy_relative_path_resolves_below_code_root(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["observations"]["sedpsm"].pop("path")
    data["observations"]["sedpsm"]["relative_path"] = "data/PPE/proxy.pkl"
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    config = load_project_config(path)

    assert config.observations.sedpsm.resolve_data_path(
        config.paths.sedpsm_root
    ) == tmp_path / "sedpsm" / "data" / "PPE" / "proxy.pkl"


def test_sedpsm_rejects_both_data_path_forms(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["observations"]["sedpsm"]["relative_path"] = "data/PPE/proxy.pkl"
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_project_config(path)


def test_loads_member_and_external_climatology_options(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["state"]["variables"]["tas"].update(
        {"input_combine": "concatenate_members"}
    )
    data["preprocess"] = {
        "input_kind": "prepared_quarterly",
        "training_period": [850, 1849],
    }
    data["observations"]["sedpsm"].update(
        {
            "climatology_path": str(tmp_path / "climatology.nc"),
            "climatology_variable": "climatology",
        }
    )
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    config = load_project_config(path)

    assert config.state.variables["tas"].input_combine.value == "concatenate_members"
    assert config.preprocess.input_kind.value == "prepared_quarterly"
    assert config.preprocess.training_period == (850, 1849)
    assert config.observations.sedpsm.climatology_variable == "climatology"


def test_loads_variable_eof_fillna_value(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["state"]["variables"]["tas"]["eof_fillna_value"] = 0.0
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    config = load_project_config(path)

    assert config.state.variables["tas"].eof_fillna_value == 0.0


def test_loads_pages2k_calibration_contract(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["observations"] = {
        "pages2k": {
            "enabled": True,
            "calibration": {
                "enabled": True,
                "source_path": str(tmp_path / "pages2k.json"),
                "source_name": None,
                "instrumental_path": str(tmp_path / "gistemp.nc"),
                "marine_predictor": "tos",
            },
        }
    }
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    config = load_project_config(path)
    calibration = config.observations.pages2k.calibration

    assert calibration.enabled
    assert calibration.anomaly_period == (1951, 1980)
    assert calibration.minimum_overlap == 25
    assert calibration.marine_predictor == "tos"
    assert len(calibration.season_candidates) == 15


def test_loads_borehole_error_estimation_contract(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["observations"] = {
        "borehole": {
            "enabled": True,
            "error_estimation": {
                "enabled": True,
                "perturbation_path": str(tmp_path / "perturbations.nc"),
                "method_profile": "legacy_notebook",
            },
        }
    }
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    config = load_project_config(path)
    estimation = config.observations.borehole.error_estimation

    assert estimation.enabled
    assert estimation.time_steps == [25, 40, 50]
    assert estimation.method_profile == "legacy_notebook"
    assert estimation.depth_start == 20


def test_loads_temp12k_preparation_contract(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["observations"] = {
        "temp12k": {
            "enabled": True,
            "preparation": {
                "enabled": True,
                "source_path": str(tmp_path / "Temp12k1_1_0.pkl"),
                "pages2k_source_path": str(tmp_path / "pages2k.pkl"),
            },
        }
    }
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    config = load_project_config(path)
    preparation = config.observations.temp12k.preparation

    assert preparation.enabled
    assert preparation.source_format == "raw_temp12k"
    assert preparation.anomaly_period == (0, 2000)
    assert preparation.duplicate_correlation_minimum == 0.98
    assert preparation.duplicate_minimum_overlap == 11


def test_loads_structured_reference_field(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["verification"]["reference_fields"] = {
        "tas": {
            "path": str(tmp_path / "reference.nc"),
            "variable": "tas_mean",
            "comparison_period": [1900, 1999],
        }
    }
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    config = load_project_config(path)
    reference = config.verification.reference_fields["tas"]

    assert reference.variable == "tas_mean"
    assert reference.comparison_period == (1900, 1999)


def test_loads_structured_reference_indices(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["verification"]["reference_indices"] = {
        "Nino34": {
            "path": str(tmp_path / "nino.nc"),
            "comparison_period": [1880, 1999],
        }
    }
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    config = load_project_config(path)
    reference = config.verification.reference_indices["Nino34"]

    assert reference.path == tmp_path / "nino.nc"
    assert reference.comparison_period == (1880, 1999)


def test_profile_deep_merge(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    profile_path = tmp_path / "profile.yml"
    write_yaml(config_path, base_config(tmp_path))
    write_yaml(
        profile_path,
        {
            "assimilation": {
                "device": "cpu",
                "max_iterations": 5,
                "inner_iterations": 3,
                "minimum_outer_iterations": 2,
            }
        },
    )

    config = load_project_config(config_path, profile_path)

    assert config.assimilation.device.value == "cpu"
    assert config.assimilation.max_iterations == 5
    assert config.assimilation.inner_iterations == 3
    assert config.assimilation.minimum_outer_iterations == 2
    assert config.assimilation.start == "0850-01"


def test_rejects_reconstruction_variable_not_in_state(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["reconstruction"]["variables"] = ["tas", "tos"]
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    with pytest.raises(ValueError, match="absent from state.variables"):
        load_project_config(path)


def test_requires_climatologies_for_absolute_sea_ice_indices(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["reconstruction"]["indices"] = ["SIA", "SIV"]
    path = tmp_path / "config.yml"
    write_yaml(path, data)

    with pytest.raises(ValueError, match="requires a SIC climatology"):
        load_project_config(path)

    data["reconstruction"]["sea_ice_concentration_climatology"] = {
        "path": str(tmp_path / "sic_climatology.nc"),
        "variable": "sic",
    }
    write_yaml(path, data)
    with pytest.raises(ValueError, match="requires a SIT climatology"):
        load_project_config(path)


def test_rejects_unresolved_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = base_config(tmp_path)
    data["paths"]["simulation_root"] = "${MISSING_LMR4DVAR_TEST_ROOT}"
    path = tmp_path / "config.yml"
    write_yaml(path, data)
    monkeypatch.delenv("MISSING_LMR4DVAR_TEST_ROOT", raising=False)

    with pytest.raises(ValueError, match="unresolved environment variable"):
        load_project_config(path)
