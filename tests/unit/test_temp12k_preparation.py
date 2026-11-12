import pickle
from collections import Counter
from types import SimpleNamespace

import numpy as np

from lmr4dvar.observations.temp12k_preparation import (
    ANNUAL_SEASON,
    _database_metadata,
    apply_anomaly_reference,
    build_temp12k_database,
    find_candidate_pairs,
    normalize_seasonality,
    remove_duplicate_years,
)


class FakeRecord:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.tags = set()
        time = np.asarray(self.time, dtype=float)
        self.dt = float(np.median(np.diff(time))) if len(time) > 1 else np.nan


class FakeDatabase:
    def __init__(self, records=None, source=None):
        self.records = records or {}
        self.source = source

    def refresh(self):
        return None


FAKE_CFR = SimpleNamespace(
    ProxyRecord=FakeRecord,
    ProxyDatabase=FakeDatabase,
    proxy=SimpleNamespace(get_ptype=lambda archive, proxy: f"{archive}.{proxy}".lower()),
)


def _record(record_id, values, *, lat=0.0, lon=0.0, seasonality=None):
    return FakeRecord(
        pid=record_id,
        time=np.arange(len(values), dtype=float),
        value=np.asarray(values, dtype=float),
        lat=lat,
        lon=lon,
        ptype="test.temperature",
        value_unit="degC",
        seasonality=seasonality or list(range(1, 13)),
    )


def test_build_filters_converts_time_and_uses_reported_uncertainty(tmp_path) -> None:
    raw = {
        "TS": [
            {
                "paleoData_inCompilation": "Temp12k",
                "paleoData_units": "degC",
                "paleoData_values": [2.0, 1.0, 3.0],
                "age": [50.0, 100.0, 50.0],
                "ageUnits": "yr BP",
                "geo_meanLat": -20,
                "geo_meanLon": 150,
                "archiveType": "LakeSediment",
                "paleoData_proxy": "temperature12k",
                "paleoData_temperature12kUncertainty": 0.5,
                "dataSetName": "site",
                "paleoData_TSid": "series",
                "paleoData_interpretation": [{"seasonality": "warmest"}],
            },
            {
                "paleoData_inCompilation": "Temp12k",
                "paleoData_units": "permil",
            },
        ]
    }
    path = tmp_path / "raw.pkl"
    with path.open("wb") as stream:
        pickle.dump(raw, stream)
    summary = Counter()

    database = build_temp12k_database(path, FAKE_CFR, summary)

    record = database.records["site::series"]
    np.testing.assert_allclose(record.time, [1850, 1900])
    np.testing.assert_allclose(record.value, [1, 2])
    assert record.R == 0.25
    assert record.seasonality == [1]
    assert summary == {"kept": 1, "dropped_not_temp12k_degC": 1}


def test_anomaly_and_seasonality_match_legacy_final_product() -> None:
    with_baseline = _record("a", [10.0, 12.0, 14.0])
    with_baseline.time = np.array([0.1, 0.2, 1.0])
    without_baseline = _record("b", [3.0, 4.0], seasonality=[7])
    without_baseline.time = np.array([-2.0, -1.0])
    database = FakeDatabase({"a": with_baseline, "b": without_baseline})

    dropped = apply_anomaly_reference(database, (0, 2), require_baseline=False)
    changed = normalize_seasonality(database)

    assert dropped == 0
    # The legacy baseline weights native samples, not annual grouped means.
    np.testing.assert_allclose(with_baseline.value, [-2, 0, 2])
    np.testing.assert_allclose(without_baseline.value, [3, 4])
    assert with_baseline.seasonality == ANNUAL_SEASON
    assert without_baseline.seasonality == [6, 7, 8]
    assert changed == 2


def test_seasonality_distinguishes_wrapped_and_same_year_december() -> None:
    database = FakeDatabase(
        {
            "djf": _record("djf", [1, 2], seasonality=[12, 1, 2]),
            "sep_jul": _record(
                "sep_jul",
                [1, 2],
                seasonality=[9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7],
            ),
            "may_dec": _record(
                "may_dec", [1, 2], seasonality=[5, 6, 7, 8, 9, 10, 11, 12]
            ),
            "repeated": _record(
                "repeated", [1, 2], seasonality=[7, 8, 7, 8, 9]
            ),
        }
    )

    changed = normalize_seasonality(database)

    assert changed == 2
    assert database.records["djf"].seasonality == [-12, 1, 2]
    assert database.records["sep_jul"].seasonality == [
        -9,
        -10,
        -11,
        -12,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    assert database.records["may_dec"].seasonality == [5, 6, 7, 8, 9, 10, 11, 12]
    assert database.records["repeated"].seasonality == [7, 8, 7, 8, 9]


def test_duplicate_rule_is_strict_and_removes_only_overlap_years() -> None:
    values = np.arange(14, dtype=float)
    temp_record = _record("temp", values)
    temp_record.time = np.arange(14, dtype=float)
    pages_record = _record("pages", values[:12])
    pages_record.time = np.arange(12, dtype=float)
    temp_database = FakeDatabase({"temp": temp_record})
    pages_database = FakeDatabase({"pages": pages_record})

    temp = _database_metadata(temp_database, (0, 20))
    pages = _database_metadata(pages_database, (0, 20))
    pairs, years = find_candidate_pairs(
        temp,
        pages,
        maximum_distance_km=50,
        duplicate_correlation_minimum=0.98,
        duplicate_minimum_overlap=11,
    )
    removals = remove_duplicate_years(temp_database, years, minimum_points=2)

    assert pairs.is_duplicate_by_rule.tolist() == [1]
    assert removals.points_removed.tolist() == [12]
    np.testing.assert_allclose(temp_record.time, [12, 13])
    np.testing.assert_allclose(temp_record.value, [12, 13])

    _, strict_years = find_candidate_pairs(
        temp,
        pages,
        maximum_distance_km=50,
        duplicate_correlation_minimum=1.0,
        duplicate_minimum_overlap=11,
    )
    assert strict_years == {}
