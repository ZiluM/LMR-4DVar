"""Workflow handler for PAGES2k PSM calibration."""

from __future__ import annotations

import json
from pathlib import Path

from lmr4dvar.config import ProjectConfig
from lmr4dvar.eof import EOFModel
from lmr4dvar.observations import calibrate_pages2k
from lmr4dvar.observations.tardif_pages2k import calibrate_tardif_pages2k

from .planner import artifact_stage_directory, stage_directory
from .stages import Stage


def execute_pages2k_calibration(config: ProjectConfig) -> tuple[Path, ...]:
    """Run calibration or record use of an external calibrated artifact."""
    directory = stage_directory(config, Stage.CALIBRATE_PAGES2K)
    directory.mkdir(parents=True, exist_ok=True)
    pages2k = config.observations.pages2k
    if not pages2k.enabled or not pages2k.calibration.enabled:
        manifest = directory / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "calibration_executed": False,
                    "pages2k_enabled": pages2k.enabled,
                    "external_calibrated_path": (
                        str(pages2k.path) if pages2k.path is not None else None
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs = [manifest]
        if pages2k.enabled and pages2k.path is not None:
            artifact = pages2k.path / "job.pkl" if pages2k.path.is_dir() else pages2k.path
            outputs.append(artifact)
        return tuple(outputs)
    if pages2k.calibration.method == "tardif2019_quarterly":
        outputs = list(calibrate_tardif_pages2k(pages2k.calibration, directory))
        outputs.extend(
            [pages2k.calibration.instrumental_path, pages2k.calibration.precipitation_path]
        )
        if pages2k.calibration.source_path is not None:
            source = pages2k.calibration.source_path
            outputs.append(source / "job.pkl" if source.is_dir() else source)
        return tuple(outputs)
    eof_directory = artifact_stage_directory(config, Stage.EOF)
    models = {variable: EOFModel.load(eof_directory / variable) for variable in ("tas", "tos")}
    outputs = list(calibrate_pages2k(pages2k.calibration, models, directory))
    outputs.append(pages2k.calibration.instrumental_path)
    source = pages2k.calibration.source_path
    if source is not None:
        outputs.append(source / "job.pkl" if source.is_dir() else source)
    return tuple(outputs)
