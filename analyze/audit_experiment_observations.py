"""Audit configured observation adapters without running assimilation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lmr4dvar.config import load_project_config
from lmr4dvar.observations import load_borehole_observations, load_cfr_observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(result) -> dict:
    records = tuple(result.observations)
    variables = sorted({record.variable for record in records})
    return {
        "records": len(records),
        "values": result.observations.observation_count,
        "variables": variables,
        "skipped": len(result.skipped),
        "skipped_examples": [
            {"record_id": item.record_id, "reason": item.reason}
            for item in result.skipped[:10]
        ],
    }


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    report = {}
    source_strict = {}
    for family in ("pages2k", "temp12k"):
        source = getattr(config.observations, family)
        if source.enabled:
            if source.path is None:
                raise ValueError(f"observations.{family}.path is required")
            report[family] = summarize(
                load_cfr_observations(
                    source.path,
                    family=family,
                    calibrated_only=family == "pages2k",
                    strict=False,
                )
            )
            source_strict[family] = source.strict
    borehole = config.observations.borehole
    if borehole.enabled:
        if borehole.path is None:
            raise ValueError("observations.borehole.path is required")
        report["borehole"] = summarize(
            load_borehole_observations(borehole.path, strict=False)
        )
        source_strict["borehole"] = borehole.strict
    report["strict_compatible"] = all(item["skipped"] == 0 for item in report.values())
    report["configuration_compatible"] = all(
        not source_strict[name] or item["skipped"] == 0
        for name, item in report.items()
        if isinstance(item, dict)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
