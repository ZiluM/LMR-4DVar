from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lmr4dvar.observations import (
    ObservationFamily,
    cfr_proxy_to_observation,
)


def fake_proxy() -> SimpleNamespace:
    details = {
        "df": pd.DataFrame({"proxy": [1.0], "tos": [2.0]}),
        "seasonality": [-12, 1, 2],
    }
    return SimpleNamespace(
        ptype="coral.d18O",
        tags={"calibrated"},
        psm=SimpleNamespace(
            calib_details=details,
            model=SimpleNamespace(params={"Intercept": -4.0, "tos": -0.25}),
        ),
        lat=5.0,
        lon=-160.0,
        time=np.array([1000.0, 1001.0]),
        value=np.array([0.1, 0.2]),
        R=0.04,
        dt=1.0,
    )


def test_pages2k_adapter_extracts_linear_psm_contract() -> None:
    observation = cfr_proxy_to_observation(
        "coral-a", fake_proxy(), family=ObservationFamily.PAGES2K
    )

    assert observation.variable == "tos"
    assert observation.longitude == 200
    assert observation.metadata["slope"] == -0.25
    assert observation.metadata["intercept"] == -4.0
    assert observation.metadata["seasonality"] == [-12, 1, 2]
    np.testing.assert_array_equal(observation.variance, [0.04, 0.04])


def test_temp12k_adapter_uses_tas_and_resolution() -> None:
    proxy = fake_proxy()
    proxy.tags = {"low_res"}
    proxy.seasonality = [6, 7, 8]
    proxy.dt = 50.0

    observation = cfr_proxy_to_observation("temp-a", proxy, family=ObservationFamily.TEMP12K)

    assert observation.variable == "tas"
    assert observation.metadata["slope"] == 1
    assert observation.metadata["resolution_years"] == 50
    assert observation.metadata["seasonality"] == [6, 7, 8]
    assert observation.metadata["anomaly_reference_period"] == [0.0, 2000.0]


def test_low_resolution_requires_positive_dt() -> None:
    proxy = fake_proxy()
    proxy.tags = {"low_res"}
    proxy.dt = np.nan

    with pytest.raises(ValueError, match="dt must be"):
        cfr_proxy_to_observation("bad", proxy, family="pages2k")


def test_multiple_predictors_are_not_silently_truncated() -> None:
    proxy = fake_proxy()
    proxy.psm.calib_details["df"]["pr"] = [0.5]
    with pytest.raises(ValueError, match="multi-predictor"):
        cfr_proxy_to_observation("tree", proxy, family="pages2k")
