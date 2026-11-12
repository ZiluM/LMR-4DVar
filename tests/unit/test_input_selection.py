import numpy as np
import pytest
import xarray as xr

from lmr4dvar.data.loader import resolve_input_files, select_horizontal_field


def test_member_exclusions_are_explicit(tmp_path):
    for member in ("001", "002", "011"):
        (tmp_path / f"cesm.{member}.nc").touch()
    files = resolve_input_files(tmp_path, "*.nc", exclude_patterns=("*.011.*",))
    assert [path.name for path in files] == ["cesm.001.nc", "cesm.002.nc"]


def test_only_singleton_extra_dimensions_are_selected():
    field = xr.DataArray(np.ones((2, 1, 3, 4)), dims=("time", "z_t", "lat", "lon"))
    result = select_horizontal_field(field)
    assert result.dims == ("time", "lat", "lon")
    assert result.attrs["singleton_dimensions_selected"] == "z_t"
    with pytest.raises(ValueError, match="non-singleton"):
        select_horizontal_field(xr.DataArray(np.ones((2, 2, 3, 4)), dims=field.dims))
