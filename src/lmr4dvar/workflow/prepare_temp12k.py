"""Workflow handler for Temp12k observation preparation."""

from __future__ import annotations

import json
from pathlib import Path

from lmr4dvar.config import ProjectConfig
from lmr4dvar.observations import prepare_temp12k

from .planner import stage_directory
from .stages import Stage


def execute_temp12k_preparation(config: ProjectConfig) -> tuple[Path, ...]:
    """Prepare Temp12k or record use of an external legacy database."""
    directory = stage_directory(config, Stage.PREPARE_TEMP12K)
    directory.mkdir(parents=True, exist_ok=True)
    temp12k = config.observations.temp12k
    preparation = temp12k.preparation
    if not temp12k.enabled or not preparation.enabled:
        manifest = directory / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "preparation_executed": False,
                    "temp12k_enabled": temp12k.enabled,
                    "external_temp12k_path": (
                        str(temp12k.path) if temp12k.path is not None else None
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs = [manifest]
        if temp12k.enabled and temp12k.path is not None:
            outputs.append(temp12k.path)
        return tuple(outputs)
    outputs = list(prepare_temp12k(preparation, directory))
    outputs.append(preparation.source_path)
    if preparation.pages2k_source_path is not None:
        pages = preparation.pages2k_source_path
        outputs.append(pages / "job.pkl" if pages.is_dir() else pages)
    return tuple(outputs)
