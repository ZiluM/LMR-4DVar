import numpy as np
import pytest
import xarray as xr

from lmr4dvar.data.monthly_time import label_monthly_intervals
from lmr4dvar.data.preprocess import quarterly_mean


def monthly_dataset(calendar="standard", start="2000-01-01"):
    dates = xr.date_range(start, periods=13, freq="MS", calendar=calendar, use_cftime=True)
    return xr.Dataset(
        {
            "tas": ("time", np.arange(12.0)),
            "time_bnds": (("time", "bnds"), np.stack([dates[:-1], dates[1:]], axis=1)),
        },
        coords={"time": ("time", dates[1:], {"bounds": "time_bnds"})},
    )


@pytest.mark.parametrize("calendar", ["standard", "noleap", "360_day"])
def test_interval_labels_restore_physical_months_without_changing_data(calendar):
    original = monthly_dataset(calendar)
    corrected = label_monthly_intervals(original)
    assert original.time.dt.month.values[0] == 2
    assert corrected.time.dt.month.values[0] == 1
    xr.testing.assert_equal(
        corrected.tas.reset_coords(drop=True).drop_vars("time"),
        original.tas.reset_coords(drop=True).drop_vars("time"),
    )
    np.testing.assert_array_equal(corrected.time_bnds.values, original.time_bnds.values)
    assert corrected.time.attrs["bounds"] == "time_bnds"
    quarter = quarterly_mean(corrected.tas)
    assert quarter.time.dt.month.values.tolist() == [4, 7, 10]
    np.testing.assert_allclose(quarter.values, [3, 6, 9])


def test_negative_years_are_preserved():
    corrected = label_monthly_intervals(monthly_dataset("noleap", "-0500-01-01"))
    assert corrected.time.dt.year.values[0] == -500


def test_missing_bounds_and_nonmonthly_intervals_fail():
    dataset = monthly_dataset()
    with pytest.raises(ValueError, match="explicit CF"):
        label_monthly_intervals(dataset.drop_vars("time_bnds"))
    dataset.time_bnds.values[0, 1] = dataset.time_bnds.values[1, 1]
    with pytest.raises(ValueError, match="exactly one"):
        label_monthly_intervals(dataset)


def test_nonconsecutive_intervals_and_outside_labels_fail():
    with pytest.raises(ValueError, match="consecutive"):
        label_monthly_intervals(monthly_dataset().isel(time=[0, 2, 3]))
    dataset = monthly_dataset()
    dataset = dataset.assign_coords(time=dataset.time.values[::-1])
    dataset.time.attrs["bounds"] = "time_bnds"
    with pytest.raises(ValueError, match="inside"):
        label_monthly_intervals(dataset)


def test_external_bounds_require_matching_coordinates_and_do_not_modify_source():
    original = monthly_dataset("noleap")
    derived = original.drop_vars("time_bnds")
    corrected = label_monthly_intervals(derived, bounds_source=original)
    assert corrected.time.dt.month.values[0] == 1
    assert original.time.attrs["bounds"] == "time_bnds"
    assert derived.time.attrs["bounds"] == "time_bnds"
    with pytest.raises(ValueError, match="different time"):
        label_monthly_intervals(derived.isel(time=slice(1, None)), bounds_source=original)


def test_partial_initial_month_requires_explicit_drop_and_internal_partial_fails():
    from datetime import timedelta

    source = monthly_dataset("noleap")
    source.time_bnds.values[0, 0] += timedelta(days=1, hours=1)
    with pytest.raises(ValueError, match="midnight"):
        label_monthly_intervals(source)
    corrected = label_monthly_intervals(source, drop_partial_boundaries=True)
    assert corrected.sizes["time"] == 11
    assert corrected.time.dt.month.values[0] == 2
    assert corrected.attrs["dropped_partial_boundary_months"] == 1
    source.time_bnds.values[5, 0] += timedelta(days=1)
    with pytest.raises(ValueError, match="internal"):
        label_monthly_intervals(source, drop_partial_boundaries=True)


def test_external_bounds_coverage_requires_explicit_selection():
    original = monthly_dataset("noleap")
    source = original.drop_vars("time_bnds")
    donor = original.isel(time=slice(0, 8))
    with pytest.raises(ValueError, match="different time"):
        label_monthly_intervals(source, bounds_source=donor)
    result = label_monthly_intervals(source, bounds_source=donor, restrict_to_bounds_source=True)
    np.testing.assert_array_equal(result.tas.values, source.tas.values[:8])
    assert result.time.dt.month.values.tolist() == list(range(1, 9))
    with pytest.raises(ValueError, match="contiguous"):
        label_monthly_intervals(
            source,
            bounds_source=original.isel(time=[0, 2, 3]),
            restrict_to_bounds_source=True,
        )
