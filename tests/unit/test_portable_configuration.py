from pathlib import Path

import pytest
import yaml

from lmr4dvar.config import load_project_config, load_yaml_config

ROOT = Path(__file__).resolve().parents[2]


def test_analysis_config_expands_nested_paths_and_preserves_urls(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIMATE_INPUT", str(tmp_path / "archive"))
    path = tmp_path / "analysis.yml"
    data = {
        "products": {"rain": {"root": "${CLIMATE_INPUT}/rain"}},
        "inputs": ["${CLIMATE_INPUT}/tas.nc"],
        "url": "https://example.org/data?var=pr&year=2000",
        "baseline": [1981, 2000],
    }
    path.write_text(yaml.safe_dump(data))
    loaded = load_yaml_config(path)
    assert loaded["products"]["rain"]["root"] == str(tmp_path / "archive/rain")
    assert loaded["inputs"] == [str(tmp_path / "archive/tas.nc")]
    assert loaded["url"] == data["url"]
    assert loaded["baseline"] == [1981, 2000]


def test_analysis_config_rejects_unset_path_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_CLIMATE_INPUT", raising=False)
    path = tmp_path / "analysis.yml"
    path.write_text("root: ${MISSING_CLIMATE_INPUT}/rain\n")
    with pytest.raises(ValueError, match="unresolved environment variable"):
        load_yaml_config(path)


def _set_example_environment(monkeypatch, tmp_path):
    import re

    for directory in (ROOT / "configs", ROOT / "analyze"):
        for path in directory.rglob("*.yml"):
            for name in re.findall(r"\$\{([A-Z0-9_]+)\}", path.read_text()):
                monkeypatch.setenv(name, str(tmp_path / name.lower()))


@pytest.mark.parametrize(
    "path",
    sorted(path for path in (ROOT / "configs").glob("*/*.yml") if path.parent.name != "profiles"),
    ids=lambda p: p.name,
)
def test_public_project_templates_validate_without_private_storage(path, tmp_path, monkeypatch):
    _set_example_environment(monkeypatch, tmp_path)
    config = load_project_config(path)
    assert config.paths.output_root.is_relative_to(tmp_path)


@pytest.mark.parametrize(
    "profile", sorted((ROOT / "configs/profiles").glob("*.yml")), ids=lambda p: p.name
)
def test_public_site_profiles_overlay_project_templates(profile, tmp_path, monkeypatch):
    _set_example_environment(monkeypatch, tmp_path)
    config = load_project_config(ROOT / "configs/examples/cesm_sedpsm_exp0.yml", profile)
    assert config.hpc.system == "pbs"
    assert Path(config.hpc.python_executable).is_relative_to(tmp_path)
    assert config.paths.output_root.is_relative_to(tmp_path)


@pytest.mark.parametrize("path", sorted((ROOT / "analyze").glob("*.yml")), ids=lambda p: p.name)
def test_public_analysis_templates_expand_paths(path, tmp_path, monkeypatch):
    _set_example_environment(monkeypatch, tmp_path)
    loaded = load_yaml_config(path)
    assert "${" not in str(loaded)
