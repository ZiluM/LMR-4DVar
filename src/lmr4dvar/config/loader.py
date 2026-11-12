"""Load YAML configuration with environment expansion and profile overlays."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .models import ProjectConfig


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if "$" in expanded:
            raise ValueError(f"unresolved environment variable in configuration value: {value}")
        return expanded
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")
    return data


def load_yaml_config(
    config_path: str | Path,
    profile_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a YAML mapping with the same path expansion used by project configs."""
    config = _read_yaml(Path(config_path))
    if profile_path is not None:
        config = _deep_merge(config, _read_yaml(Path(profile_path)))
    return _expand_environment(config)


def load_project_config(
    config_path: str | Path,
    profile_path: str | Path | None = None,
) -> ProjectConfig:
    return ProjectConfig.model_validate(load_yaml_config(config_path, profile_path))
