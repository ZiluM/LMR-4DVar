"""Dependency-aware local workflow planning and resume receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from lmr4dvar.config import ProjectConfig

from .manifest import fingerprint_path
from .stages import Stage


class StageStatus(str, Enum):
    COMPLETE = "complete"
    READY = "ready"
    STALE = "stale"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class StagePlanItem:
    stage: Stage
    status: StageStatus
    reason: str
    directory: Path


def run_directory(config: ProjectConfig) -> Path:
    return config.paths.output_root / config.project.name


def stage_directory(config: ProjectConfig, stage: Stage) -> Path:
    return run_directory(config) / "stages" / stage.value


def artifact_stage_directory(config: ProjectConfig, stage: Stage) -> Path:
    """Return a local stage directory or the configured external prior stage."""
    if config.paths.prior_run_root is not None and stage in (Stage.EOF, Stage.TRAIN_LIM):
        return config.paths.prior_run_root / "stages" / stage.value
    return stage_directory(config, stage)


def config_fingerprint(config: ProjectConfig) -> str:
    canonical = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_plan(
    config: ProjectConfig,
    *,
    start: Stage | None = None,
    end: Stage | None = None,
    force: bool = False,
) -> tuple[StagePlanItem, ...]:
    stages = list(Stage.ordered())
    if config.paths.prior_run_root is not None and start is None:
        start = Stage.CALIBRATE_PAGES2K
    start_index = stages.index(start) if start is not None else 0
    end_index = stages.index(end) if end is not None else len(stages) - 1
    if start_index > end_index:
        raise ValueError("workflow start stage must not follow end stage")
    selected = stages[start_index : end_index + 1]
    expected_config = config_fingerprint(config)
    items = []
    if start_index == 0:
        upstream_complete = True
    elif selected[0] is Stage.CALIBRATE_PAGES2K and config.paths.prior_run_root is not None:
        upstream_complete = _external_prior_valid(config)
    else:
        upstream_complete = _receipt_valid(config, stages[start_index - 1], expected_config)
    for stage in selected:
        directory = stage_directory(config, stage)
        receipt = directory / "receipt.json"
        valid = _receipt_valid(config, stage, expected_config)
        if valid and not force:
            status = StageStatus.COMPLETE
            reason = "receipt, configuration, dependency, and outputs match"
        elif not upstream_complete:
            status = StageStatus.BLOCKED
            reason = "upstream stage is incomplete or stale"
        elif receipt.exists() and not force:
            status = StageStatus.STALE
            reason = "receipt no longer matches configuration, dependency, or outputs"
        else:
            status = StageStatus.READY
            reason = "no valid receipt" if not force else "forced rerun"
        items.append(StagePlanItem(stage, status, reason, directory))
        upstream_complete = status is StageStatus.COMPLETE
    return tuple(items)


def write_stage_receipt(
    config: ProjectConfig,
    stage: Stage,
    outputs: list[Path],
) -> Path:
    if not outputs:
        raise ValueError("stage receipt requires at least one output")
    directories = [path for path in outputs if path.exists() and not path.is_file()]
    if directories:
        raise ValueError("stage receipts require concrete output files, not directories")
    upstream = _upstream_receipt_path(config, stage)
    if upstream is not None and not upstream.is_file():
        raise ValueError(f"upstream receipt is missing: {upstream}")
    data = {
        "schema_version": 1,
        "stage": stage.value,
        "config_sha256": config_fingerprint(config),
        "upstream_receipt_sha256": _file_sha256(upstream) if upstream else None,
        "outputs": [fingerprint_path(path) for path in outputs],
    }
    receipt = _receipt_path(config, stage)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return receipt


def _receipt_valid(config: ProjectConfig, stage: Stage, expected_config: str) -> bool:
    receipt_path = _receipt_path(config, stage)
    if not receipt_path.is_file():
        return False
    try:
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        if data["stage"] != stage.value or data["config_sha256"] != expected_config:
            return False
        upstream = _upstream_receipt_path(config, stage)
        expected_upstream = _file_sha256(upstream) if upstream and upstream.is_file() else None
        if data.get("upstream_receipt_sha256") != expected_upstream:
            return False
        for saved in data["outputs"]:
            current = fingerprint_path(Path(saved["path"]))
            if current != saved:
                return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(data["outputs"])


def _receipt_path(config: ProjectConfig, stage: Stage) -> Path:
    return stage_directory(config, stage) / "receipt.json"


def _upstream_receipt_path(config: ProjectConfig, stage: Stage) -> Path | None:
    if stage is Stage.PREPROCESS:
        return None
    if stage is Stage.CALIBRATE_PAGES2K and config.paths.prior_run_root is not None:
        return config.paths.prior_run_root / "stages" / Stage.TRAIN_LIM.value / "receipt.json"
    stages = list(Stage.ordered())
    return _receipt_path(config, stages[stages.index(stage) - 1])


def _external_prior_valid(config: ProjectConfig) -> bool:
    root = config.paths.prior_run_root
    if root is None:
        return False
    eof_receipt = root / "stages" / Stage.EOF.value / "receipt.json"
    lim_receipt = root / "stages" / Stage.TRAIN_LIM.value / "receipt.json"
    if not _saved_outputs_current(eof_receipt, Stage.EOF):
        return False
    if not _saved_outputs_current(lim_receipt, Stage.TRAIN_LIM):
        return False
    try:
        lim_data = json.loads(lim_receipt.read_text(encoding="utf-8"))
        return lim_data.get("upstream_receipt_sha256") == _file_sha256(eof_receipt)
    except (OSError, json.JSONDecodeError):
        return False


def _saved_outputs_current(receipt: Path, stage: Stage) -> bool:
    try:
        data = json.loads(receipt.read_text(encoding="utf-8"))
        if data["stage"] != stage.value or not data["outputs"]:
            return False
        return all(fingerprint_path(Path(item["path"])) == item for item in data["outputs"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
