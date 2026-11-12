"""Audit CESM-LME TAS/PRECT member and physical-month alignment, without fields.

Only coordinates and bounds are loaded. The JSON output belongs in an external
input/output root, not the source repository. This is an input audit, not a
forecast evaluation or a claim that existing quarterly PCs are aligned.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from lmr4dvar.data.monthly_time import label_monthly_intervals


def inspect_months(path: Path, variable: str) -> tuple[dict, np.ndarray]:
    with xr.open_dataset(path) as dataset:
        if variable not in dataset:
            raise ValueError(f"{variable} missing from {path}")
        corrected = label_monthly_intervals(dataset)
        months = corrected.time.dt.year.values * 12 + corrected.time.dt.month.values
        description = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "variable": variable,
            "dimensions": dict(dataset[variable].sizes),
            "calendar": dataset.time.encoding.get("calendar", "standard"),
            "declared_units": dataset[variable].attrs.get("units"),
            "coordinate_first": str(dataset.time.values[0]),
            "coordinate_last": str(dataset.time.values[-1]),
            "physical_month_first": str(corrected.time.values[0]),
            "physical_month_last": str(corrected.time.values[-1]),
            "time_bounds": dataset.time.attrs["bounds"],
        }
    return description, months


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tas-directory", type=Path, required=True)
    parser.add_argument("--prect-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    members = []
    common_months = None
    for member in range(1, 14):
        prefix = f"b.e11.BLMTRC5CN.f19_g16.{member:03d}.cam.h0"
        temperature, t_months = inspect_months(
            args.tas_directory / f"{prefix}.TREFHT.085001-184912.nc", "TREFHT"
        )
        precipitation, p_months = inspect_months(
            args.prect_directory / f"{prefix}.PRECT.085001-184912.nc", "PRECT"
        )
        if not np.array_equal(t_months, p_months):
            raise ValueError(f"physical TAS/PRECT month mismatch for member {member:03d}")
        if common_months is not None and not np.array_equal(t_months, common_months):
            raise ValueError(f"physical month coverage differs for member {member:03d}")
        common_months = t_months
        members.append({"member": f"{member:03d}", "tas": temperature, "pr": precipitation})
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "raw monthly coordinates/bounds, not data values or existing PCs",
        "physical_month_alignment": "passed",
        "units_status": "PRECT units require PRECC/PRECL provenance; no unit inference here",
        "members": members,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(f"Physical monthly alignment passed for {len(members)} members: {args.output}")


if __name__ == "__main__":
    main()
