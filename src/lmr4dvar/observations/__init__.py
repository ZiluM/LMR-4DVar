"""Proxy observation records and operators."""

from .borehole import BoreholeOperator, conduction_forward, load_borehole_observations
from .borehole_error import (
    BoreholeErrorOptions,
    BoreholeErrorResult,
    effective_error_options,
    estimate_borehole_error_artifact,
    estimate_profile_error,
    heat_conduction_matrix,
)
from .cfr import (
    ObservationLoadResult,
    SkippedObservation,
    cfr_proxy_to_observation,
    load_cfr_observations,
)
from .linear import (
    ANNUAL_MONTHS,
    EOFPointDecoder,
    LinearProxyOperator,
    PointTransform,
    annualize_quarterly,
)
from .loss import EmptyObservationError, ObservationLoss, gaussian_observation_loss
from .matching import (
    TimeMatch,
    WindowIndex,
    average_indexed_windows,
    average_to_windows,
    build_window_index,
    match_nearest_times,
)
from .pages2k_calibration import calibrate_pages2k, project_instrumental_field
from .protocol import ObservationOperator
from .records import BoreholeRecord, ObservationCollection, ObservationFamily, ObservationRecord
from .sedpsm import (
    SedPSMOperator,
    load_sedpsm_ppe,
    sedpsm_experiment_parameters,
    sedpsm_record_to_observation,
)
from .temp12k_preparation import (
    apply_anomaly_reference,
    build_temp12k_database,
    find_candidate_pairs,
    normalize_seasonality,
    prepare_temp12k,
    remove_duplicate_years,
)

__all__ = [
    "EmptyObservationError",
    "ANNUAL_MONTHS",
    "BoreholeOperator",
    "BoreholeErrorOptions",
    "BoreholeErrorResult",
    "BoreholeRecord",
    "EOFPointDecoder",
    "LinearProxyOperator",
    "ObservationLoadResult",
    "ObservationCollection",
    "ObservationFamily",
    "ObservationLoss",
    "ObservationOperator",
    "ObservationRecord",
    "PointTransform",
    "SkippedObservation",
    "SedPSMOperator",
    "TimeMatch",
    "WindowIndex",
    "average_indexed_windows",
    "average_to_windows",
    "apply_anomaly_reference",
    "annualize_quarterly",
    "cfr_proxy_to_observation",
    "calibrate_pages2k",
    "build_temp12k_database",
    "build_window_index",
    "conduction_forward",
    "effective_error_options",
    "estimate_borehole_error_artifact",
    "estimate_profile_error",
    "find_candidate_pairs",
    "gaussian_observation_loss",
    "load_sedpsm_ppe",
    "load_cfr_observations",
    "load_borehole_observations",
    "heat_conduction_matrix",
    "match_nearest_times",
    "normalize_seasonality",
    "prepare_temp12k",
    "project_instrumental_field",
    "remove_duplicate_years",
    "sedpsm_record_to_observation",
    "sedpsm_experiment_parameters",
]
