"""Workflow handler for depth-dependent borehole error estimation."""

from __future__ import annotations

import json
from pathlib import Path

from lmr4dvar.config import ProjectConfig
from lmr4dvar.observations import estimate_borehole_error_artifact

from .planner import stage_directory
from .stages import Stage


def execute_borehole_error_estimation(config: ProjectConfig) -> tuple[Path, ...]:
    """Estimate errors or record a configured legacy borehole artifact."""
    directory = stage_directory(config, Stage.ESTIMATE_BOREHOLE_ERROR)
    directory.mkdir(parents=True, exist_ok=True)
    borehole = config.observations.borehole
    estimation = borehole.error_estimation
    if not borehole.enabled or not estimation.enabled:
        manifest = directory / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "estimation_executed": False,
                    "borehole_enabled": borehole.enabled,
                    "external_borehole_path": (
                        str(borehole.path) if borehole.path is not None else None
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs = [manifest]
        if borehole.enabled and borehole.path is not None:
            outputs.append(borehole.path)
        return tuple(outputs)
    outputs = list(estimate_borehole_error_artifact(estimation, directory))
    outputs.append(estimation.perturbation_path)
    return tuple(outputs)
