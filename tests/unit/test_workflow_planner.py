from pathlib import Path

from lmr4dvar.config.models import ProjectConfig
from lmr4dvar.workflow import Stage, StageStatus, build_plan, write_stage_receipt


def _config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "project": {"name": "workflow-test"},
            "paths": {
                "simulation_root": tmp_path / "simulations",
                "output_root": tmp_path / "output",
            },
            "state": {
                "variables": {"tas": {"domain": "global", "n_modes": 1}}
            },
            "observations": {"pages2k": {"enabled": True, "path": tmp_path / "proxy"}},
            "assimilation": {"start": "0850-01", "end": "2000-12"},
            "reconstruction": {"variables": ["tas"]},
            "verification": {
                "anomaly_period": [850, 1849],
                "comparison_period": [850, 2000],
            },
        }
    )


def test_new_plan_has_one_ready_stage_and_blocked_dependents(tmp_path) -> None:
    plan = build_plan(_config(tmp_path))
    assert plan[0].status is StageStatus.READY
    assert all(item.status is StageStatus.BLOCKED for item in plan[1:])


def test_receipt_enables_resume_and_detects_changed_output(tmp_path) -> None:
    config = _config(tmp_path)
    output = tmp_path / "preprocessed.nc"
    output.write_text("first", encoding="utf-8")
    write_stage_receipt(config, Stage.PREPROCESS, [output])

    plan = build_plan(config)
    assert plan[0].status is StageStatus.COMPLETE
    assert plan[1].status is StageStatus.READY

    output.write_text("changed-size", encoding="utf-8")
    stale = build_plan(config)
    assert stale[0].status is StageStatus.STALE
    assert stale[1].status is StageStatus.BLOCKED


def test_receipt_depends_on_upstream_receipt_hash(tmp_path) -> None:
    config = _config(tmp_path)
    preprocess_output = tmp_path / "preprocessed.nc"
    eof_output = tmp_path / "eofs.nc"
    preprocess_output.write_text("field", encoding="utf-8")
    eof_output.write_text("eofs", encoding="utf-8")
    preprocess_receipt = write_stage_receipt(config, Stage.PREPROCESS, [preprocess_output])
    write_stage_receipt(config, Stage.EOF, [eof_output])
    assert build_plan(config)[1].status is StageStatus.COMPLETE

    preprocess_receipt.write_text(
        preprocess_receipt.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    plan = build_plan(config, start=Stage.EOF, end=Stage.TRAIN_LIM)
    assert plan[0].status is StageStatus.STALE
    assert plan[1].status is StageStatus.BLOCKED


def test_external_prior_starts_plan_at_pages2k_calibration(tmp_path) -> None:
    source = _config(tmp_path)
    outputs = {}
    for stage, name in (
        (Stage.PREPROCESS, "field.nc"),
        (Stage.EOF, "eofs.nc"),
        (Stage.TRAIN_LIM, "dynamics.nc"),
    ):
        output = tmp_path / name
        output.write_text(stage.value, encoding="utf-8")
        outputs[stage] = output
        write_stage_receipt(source, stage, [output])

    target = source.model_copy(deep=True)
    target.project.name = "external-prior-test"
    target.paths.prior_run_root = tmp_path / "output" / "workflow-test"
    plan = build_plan(target)

    assert [item.stage for item in plan] == [
        Stage.CALIBRATE_PAGES2K,
        Stage.PREPARE_TEMP12K,
        Stage.ESTIMATE_BOREHOLE_ERROR,
        Stage.ASSIMILATE,
        Stage.RECONSTRUCT,
        Stage.VERIFY,
    ]
    assert plan[0].status is StageStatus.READY
    assert all(item.status is StageStatus.BLOCKED for item in plan[1:])

    calibration = tmp_path / "calibration.json"
    calibration.write_text("calibration", encoding="utf-8")
    write_stage_receipt(target, Stage.CALIBRATE_PAGES2K, [calibration])
    temp12k = tmp_path / "temp12k.json"
    temp12k.write_text("temp12k", encoding="utf-8")
    write_stage_receipt(target, Stage.PREPARE_TEMP12K, [temp12k])
    borehole_error = tmp_path / "borehole-error.json"
    borehole_error.write_text("borehole-error", encoding="utf-8")
    write_stage_receipt(target, Stage.ESTIMATE_BOREHOLE_ERROR, [borehole_error])
    assimilation = tmp_path / "analysis.nc"
    assimilation.write_text("analysis", encoding="utf-8")
    write_stage_receipt(target, Stage.ASSIMILATE, [assimilation])
    resumed = build_plan(target)
    assert resumed[0].status is StageStatus.COMPLETE
    assert resumed[1].status is StageStatus.COMPLETE
    assert resumed[2].status is StageStatus.COMPLETE
    assert resumed[3].status is StageStatus.COMPLETE
    assert resumed[4].status is StageStatus.READY
