"""Paired skill-map differences on exactly matched reference coverage."""

import numpy as np
import xarray as xr


def paired_skill_difference(fdvar, lmr, variable):
    """Return untruncated 4D-Var minus LMR skill and paired-area summaries."""
    if variable not in ("ce", "gridpoint_temporal_correlation"):
        raise ValueError("only CE and temporal correlation are skill-map variables")
    left, right = xr.align(fdvar[variable], lmr[variable], join="exact")
    left_mask, right_mask = xr.align(fdvar.coverage_mask, lmr.coverage_mask, join="exact")
    valid = left_mask & right_mask & np.isfinite(left) & np.isfinite(right)
    if not bool(valid.any()):
        raise ValueError("skill difference has no paired finite coverage")
    left, right = left.where(valid), right.where(valid)
    delta = left - right
    weights = xr.DataArray(np.cos(np.deg2rad(left.lat)), dims="lat", coords={"lat": left.lat})
    area = float((valid * weights).sum())
    summary = {
        "fdvar_area_mean": float(left.weighted(weights).mean()),
        "lmr_area_mean": float(right.weighted(weights).mean()),
        "difference_area_mean": float(delta.weighted(weights).mean()),
        "fdvar_better_area_fraction": float(((delta > 0) * weights).sum()) / area,
        "lmr_better_area_fraction": float(((delta < 0) * weights).sum()) / area,
        "valid_gridpoints": int(valid.sum()),
    }
    dataset = xr.Dataset({"fdvar": left, "lmr": right, "difference": delta, "valid": valid})
    dataset.attrs.update(
        variable=variable,
        difference_definition="LMR4DVar minus official LMRv2.1 grand mean",
        positive_difference="higher skill for LMR4DVar",
        statistical_significance="not evaluated",
        numeric_clipping="none",
    )
    for name in ("fdvar", "lmr", "difference"):
        dataset[name].attrs["units"] = "1"
    return dataset, summary
