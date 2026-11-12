"""Command-line interface for LMR4DVar."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from pydantic import ValidationError

from lmr4dvar.config import load_project_config
from lmr4dvar.config.validation import path_errors
from lmr4dvar.data import inspect_simulation_inputs
from lmr4dvar.hpc import render_pbs_script, submit_pbs, write_pbs_script
from lmr4dvar.workflow import IMPLEMENTED_STAGES, Stage, build_plan, execute_stage, execute_through
from lmr4dvar.workflow.manifest import build_manifest, write_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmr4dvar")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a project configuration")
    validate.add_argument("config", type=Path)
    validate.add_argument("--profile", type=Path)
    validate.add_argument("--check-paths", action="store_true")
    validate.add_argument("--json", action="store_true", dest="as_json")

    preflight = subparsers.add_parser(
        "preflight", help="inspect simulation metadata without running preprocessing"
    )
    preflight.add_argument("config", type=Path)
    preflight.add_argument("--profile", type=Path)
    preflight.add_argument("--json", action="store_true", dest="as_json")

    subparsers.add_parser("stages", help="list workflow stages in execution order")

    snapshot = subparsers.add_parser(
        "snapshot", help="record fingerprints for external or legacy artifacts"
    )
    snapshot.add_argument("paths", nargs="+", type=Path)
    snapshot.add_argument("--output", required=True, type=Path)
    snapshot.add_argument("--checksum", action="store_true")

    plan = subparsers.add_parser("plan", help="inspect stage readiness without executing")
    plan.add_argument("config", type=Path)
    plan.add_argument("--profile", type=Path)
    plan.add_argument("--from", dest="start", choices=[stage.value for stage in Stage])
    plan.add_argument("--to", dest="end", choices=[stage.value for stage in Stage])
    plan.add_argument("--force", action="store_true")
    plan.add_argument("--json", action="store_true", dest="as_json")

    run = subparsers.add_parser("run", help="execute implemented scientific stages locally")
    run.add_argument("config", type=Path)
    run.add_argument("--profile", type=Path)
    selection = run.add_mutually_exclusive_group(required=True)
    implemented = [stage.value for stage in IMPLEMENTED_STAGES]
    selection.add_argument("--stage", choices=implemented)
    selection.add_argument("--through", choices=implemented)
    run.add_argument("--force", action="store_true")

    render = subparsers.add_parser("render-pbs", help="render a PBS script without submitting")
    render.add_argument("config", type=Path)
    render.add_argument("--profile", type=Path)
    render.add_argument(
        "--through",
        choices=[stage.value for stage in IMPLEMENTED_STAGES],
        default=Stage.TRAIN_LIM.value,
    )
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--force", action="store_true")

    submit = subparsers.add_parser("submit-pbs", help="submit an already rendered PBS script")
    submit.add_argument("script", type=Path)
    return parser


def _validate(args: argparse.Namespace) -> int:
    try:
        config = load_project_config(args.config, args.profile)
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        print(f"Configuration invalid: {exc}")
        return 2

    errors = path_errors(config) if args.check_paths else []
    if errors:
        print("Configuration schema is valid, but path checks failed:")
        for error in errors:
            print(f"- {error}")
        return 2

    if args.as_json:
        print(json.dumps(config.model_dump(mode="json"), indent=2))
    else:
        print(f"Configuration valid: {config.project.name}")
        print(f"State dimension: {sum(item.n_modes for item in config.state.variables.values())}")
        enabled = [
            name
            for name in ("pages2k", "sedpsm", "borehole", "temp12k")
            if getattr(config.observations, name).enabled
        ]
        print(f"Observation sources: {', '.join(enabled)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        return _validate(args)
    if args.command == "preflight":
        try:
            config = load_project_config(args.config, args.profile)
            errors = path_errors(config)
            report = inspect_simulation_inputs(config) if not errors else {
                "variables": {},
                "errors": errors,
                "warnings": [],
                "training_samples": None,
            }
        except (OSError, TypeError, ValueError, ValidationError) as exc:
            print(f"Preflight failed: {exc}")
            return 2
        if args.as_json:
            print(json.dumps(report, indent=2))
        else:
            for variable, item in report["variables"].items():
                print(
                    f"{variable}: {item['files']} file(s), "
                    f"samples={item['training_samples']}, "
                    f"grid={item['grid']['lat']}x{item['grid']['lon']}"
                )
            for warning in report["warnings"]:
                print(f"Warning: {warning}")
            for error in report["errors"]:
                print(f"Error: {error}")
        return 2 if report["errors"] else 0
    if args.command == "stages":
        for index, stage in enumerate(Stage.ordered(), start=1):
            print(f"{index}. {stage.value}")
        return 0
    if args.command == "snapshot":
        try:
            manifest = build_manifest(args.paths, checksum=args.checksum)
            write_manifest(manifest, args.output)
        except (OSError, ValueError) as exc:
            print(f"Snapshot failed: {exc}")
            return 2
        print(f"Wrote manifest: {args.output}")
        return 0
    if args.command == "plan":
        try:
            config = load_project_config(args.config, args.profile)
            plan = build_plan(
                config,
                start=Stage(args.start) if args.start else None,
                end=Stage(args.end) if args.end else None,
                force=args.force,
            )
        except (OSError, TypeError, ValueError, ValidationError) as exc:
            print(f"Planning failed: {exc}")
            return 2
        records = [
            {
                "stage": item.stage.value,
                "status": item.status.value,
                "reason": item.reason,
                "directory": str(item.directory),
            }
            for item in plan
        ]
        if args.as_json:
            print(json.dumps(records, indent=2))
        else:
            for item in records:
                print(f"{item['stage']}: {item['status']} ({item['reason']})")
        return 0
    if args.command == "run":
        try:
            config = load_project_config(args.config, args.profile)
            if args.stage:
                stage = Stage(args.stage)
                results = {stage: execute_stage(config, stage, force=args.force)}
            else:
                results = execute_through(config, Stage(args.through), force=args.force)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, ValidationError) as exc:
            print(f"Execution failed: {exc}")
            return 2
        for stage, outputs in results.items():
            print(f"{stage.value}: complete")
            for output in outputs:
                print(f"  {output}")
        return 0
    if args.command == "render-pbs":
        try:
            config = load_project_config(args.config, args.profile)
            script = render_pbs_script(
                config,
                args.config,
                through=Stage(args.through),
                profile_path=args.profile,
                force=args.force,
            )
            output = write_pbs_script(script, args.output)
        except (OSError, TypeError, ValueError, ValidationError) as exc:
            print(f"PBS rendering failed: {exc}")
            return 2
        print(f"Wrote PBS script: {output}")
        return 0
    if args.command == "submit-pbs":
        try:
            job_id = submit_pbs(args.script)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"PBS submission failed: {exc}")
            return 2
        print(f"Submitted PBS job: {job_id}")
        return 0
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
