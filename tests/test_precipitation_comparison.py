import numpy as np
import pytest
import xarray as xr

from lmr4dvar.verification.precipitation_comparison import paired_skill_difference


def dataset(values):
    field = xr.DataArray(values, dims=("lat", "lon"), coords={"lat": [0, 60], "lon": [0, 180]})
    return xr.Dataset(
        {
            "ce": field,
            "gridpoint_temporal_correlation": field,
            "coverage_mask": xr.ones_like(field, dtype=bool),
        }
    )


def test_difference_sign_and_ce_not_clipped():
    maps, scores = paired_skill_difference(
        dataset([[2, -50], [1, 0]]), dataset([[1, -1], [1, 0]]), "ce"
    )
    assert maps.difference[0, 1] == -49
    assert scores["difference_area_mean"] == pytest.approx(-16)
    assert scores["fdvar_better_area_fraction"] == pytest.approx(1 / 3)
    assert scores["lmr_better_area_fraction"] == pytest.approx(1 / 3)


def test_difference_only_uses_shared_valid_area():
    left, right = dataset([[2, np.nan], [1, 0]]), dataset([[1, 1], [1, 0]])
    right.coverage_mask[1, 1] = False
    maps, scores = paired_skill_difference(left, right, "ce")
    assert scores["valid_gridpoints"] == 2
    assert scores["difference_area_mean"] == pytest.approx(2 / 3)
    assert np.isnan(maps.difference[0, 1])


def test_misaligned_grids_and_empty_overlap_rejected():
    left = dataset([[1, 1], [1, 1]])
    with pytest.raises(ValueError):
        paired_skill_difference(left, left.assign_coords(lon=[1, 181]), "ce")
    right = left.copy(deep=True)
    right.coverage_mask[:] = False
    with pytest.raises(ValueError, match="coverage"):
        paired_skill_difference(left, right, "ce")
