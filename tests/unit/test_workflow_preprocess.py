import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lmr4dvar.workflow.executor import _concatenate_members


def _member(offset: float) -> xr.DataArray:
    return xr.DataArray(
        np.arange(3, dtype=float)[:, None, None] + offset,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=3, freq="QS-JAN"),
            "lat": [1.0],
            "lon": [0.0],
        },
    )


def test_concatenate_members_preserves_sorted_member_blocks() -> None:
    result = _concatenate_members([_member(0), _member(10)])

    np.testing.assert_array_equal(result[:, 0, 0], [0, 1, 2, 10, 11, 12])
    np.testing.assert_array_equal(result.time, np.arange(6))
    np.testing.assert_array_equal(result.member, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(result.member_sample, [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(result.physical_year, [2000] * 6)
    np.testing.assert_array_equal(result.physical_month, [1, 4, 7, 1, 4, 7])


def test_member_concatenation_rejects_equal_length_but_shifted_time():
    first, shifted = _member(0), _member(10)
    shifted = shifted.assign_coords(time=pd.date_range("2000-04-01", periods=3, freq="QS-JAN"))
    with pytest.raises(ValueError, match="physical time"):
        _concatenate_members([first, shifted])
