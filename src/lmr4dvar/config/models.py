"""Pydantic models for the complete LMR4DVar workflow."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Domain(str, Enum):
    GLOBAL = "global"
    NORTHERN_HEMISPHERE = "northern_hemisphere"


class Device(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class Precision(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class RegridMethod(str, Enum):
    NONE = "none"
    LINEAR = "linear"


class QuarterlyMode(str, Enum):
    LEGACY_QS_DEC = "legacy_qs_dec"
    COMPLETE = "complete"


class DetrendMethod(str, Enum):
    NONE = "none"
    LINEAR_BY_MONTH = "linear_by_month"


class InputCombine(str, Enum):
    BY_COORDS = "by_coords"
    CONCATENATE_MEMBERS = "concatenate_members"


class InputKind(str, Enum):
    MONTHLY = "monthly"
    PREPARED_QUARTERLY = "prepared_quarterly"


class ProjectInfo(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    description: str = ""


class PathsConfig(StrictModel):
    simulation_root: Path
    output_root: Path
    sedpsm_root: Path | None = None
    prior_run_root: Path | None = None


class VariableConfig(StrictModel):
    domain: Domain
    n_modes: PositiveInt
    eof_fillna_value: float | None = Field(default=None, allow_inf_nan=False)
    source_name: str | None = None
    input_pattern: str | None = None
    input_root: Path | None = None
    exclude_input_patterns: list[str] = Field(default_factory=list)
    monthly_time_policy: str = "coordinate"
    monthly_time_bounds_pattern: str | None = None
    monthly_drop_partial_boundaries: bool = False
    monthly_restrict_to_bounds_source: bool = False
    source_units: str | None = None
    input_combine: InputCombine = InputCombine.BY_COORDS
    units: str | None = None

    @model_validator(mode="after")
    def validate_monthly_time_policy(self) -> VariableConfig:
        if self.monthly_time_policy not in {"coordinate", "cf_bounds"}:
            raise ValueError("unsupported variable monthly_time_policy")
        if self.monthly_time_bounds_pattern is not None and self.monthly_time_policy != "cf_bounds":
            raise ValueError("monthly_time_bounds_pattern requires monthly_time_policy=cf_bounds")
        if self.monthly_drop_partial_boundaries and self.monthly_time_policy != "cf_bounds":
            raise ValueError("dropping partial months requires monthly_time_policy=cf_bounds")
        if self.monthly_restrict_to_bounds_source and self.monthly_time_bounds_pattern is None:
            raise ValueError("restricting to bounds coverage requires monthly_time_bounds_pattern")
        return self


class StateConfig(StrictModel):
    timestep_months: PositiveInt = 3
    variables: dict[str, VariableConfig]

    @model_validator(mode="after")
    def require_variables(self) -> StateConfig:
        if not self.variables:
            raise ValueError("state.variables must contain at least one variable")
        return self


class PreprocessConfig(StrictModel):
    input_kind: InputKind = InputKind.MONTHLY
    training_period: tuple[int, int] | None = None
    target_grid_degrees: float = Field(default=2.0, gt=0, le=180)
    regrid: RegridMethod = RegridMethod.LINEAR
    remove_monthly_climatology: bool = True
    quarterly_mode: QuarterlyMode = QuarterlyMode.COMPLETE
    seasonal_months: list[int] = Field(default_factory=lambda: [1, 4, 7, 10])
    detrend: DetrendMethod = DetrendMethod.LINEAR_BY_MONTH
    require_complete_years: bool = False

    @model_validator(mode="after")
    def validate_months(self) -> PreprocessConfig:
        if len(set(self.seasonal_months)) != len(self.seasonal_months):
            raise ValueError("preprocess.seasonal_months contains duplicates")
        if any(month < 1 or month > 12 for month in self.seasonal_months):
            raise ValueError("preprocess.seasonal_months must be in 1..12")
        if self.training_period is not None and self.training_period[0] > self.training_period[1]:
            raise ValueError("preprocess.training_period must be increasing")
        if self.require_complete_years and set(self.seasonal_months) != {1, 4, 7, 10}:
            raise ValueError("complete quarterly years require Jan/Apr/Jul/Oct seasons")
        return self


class EOFConfig(StrictModel):
    area_weighted: bool = True
    solver: str = "deterministic"
    center: bool = False


class LIMConfig(StrictModel):
    lag_steps: PositiveInt = 1
    segment_length_steps: PositiveInt | None = None
    fit_noise: bool = True
    center_training: bool = False
    require_stable: bool = True
    covariance_floor_relative: float = Field(default=1e-10, gt=0, lt=1)


class ObservationSourceConfig(StrictModel):
    enabled: bool = False
    path: Path | None = None
    strict: bool = True


def _default_pages2k_seasons() -> list[list[int]]:
    return [
        [-12, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [-12, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [3, 4, 5, 6, 7, 8],
        [-9, -10, -11, -12, 1, 2],
        [-12, 1, 2, 3, 4, 5, 6, 7, 8],
        [3, 4, 5, 6, 7, 8, 9, 10, 11],
        [-9, -10, -11, -12, 1, 2, 3, 4, 5],
        [-6, -7, -8, -9, -10, -11, -12, 1, 2],
        [-12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [-6, -7, -8, -9, -10, -11, -12, 1, 2, 3, 4, 5],
        [-9, -10, -11, -12, 1, 2, 3, 4, 5, 6, 7, 8],
    ]


class Pages2kCalibrationConfig(StrictModel):
    enabled: bool = False
    method: str = "eof_projected_linear"
    source_name: str | None = "PAGES2kv2"
    source_path: Path | None = None
    instrumental_path: Path | None = None
    instrumental_variable: str = "tempanomaly"
    precipitation_path: Path | None = None
    precipitation_variable: str = "p"
    tree_ring_width_psm: str = "temperature_precipitation"
    matched_records_path: Path | None = None
    precipitation_season_candidates: list[list[int]] | None = None
    hemisphere_season_filter: bool = True
    anomaly_period: tuple[int, int] = (1951, 1980)
    calibration_period: tuple[int, int] = (1850, 2015)
    low_resolution_period: tuple[int, int] = (1850, 2000)
    minimum_overlap: PositiveInt = 25
    psm_correlation_minimum: float = Field(default=0.1, ge=-1, le=1)
    residual_autocorrelation_maximum: float = Field(default=0.9, ge=-1, le=1)
    projection_fill_value: float = Field(default=0.0, allow_inf_nan=False)
    season_candidates: list[list[int]] = Field(default_factory=_default_pages2k_seasons)
    marine_predictor: str = "best_of_tas_tos"

    @model_validator(mode="after")
    def validate_calibration(self) -> Pages2kCalibrationConfig:
        if self.method not in {"eof_projected_linear", "tardif2019_quarterly"}:
            raise ValueError("unsupported pages2k calibration method")
        if self.tree_ring_width_psm not in {"temperature_precipitation", "temperature_only"}:
            raise ValueError("unsupported tree_ring_width_psm")
        if self.method != "tardif2019_quarterly" and (
            self.tree_ring_width_psm != "temperature_precipitation"
            or self.matched_records_path is not None
        ):
            raise ValueError(
                "TRW predictor selection and matched records require tardif2019_quarterly"
            )
        for name, period in (
            ("anomaly_period", self.anomaly_period),
            ("calibration_period", self.calibration_period),
            ("low_resolution_period", self.low_resolution_period),
        ):
            if period[0] > period[1]:
                raise ValueError(f"pages2k calibration {name} must be increasing")
        if self.enabled:
            if self.instrumental_path is None:
                raise ValueError("pages2k calibration instrumental_path is required when enabled")
            if self.source_name is None and self.source_path is None:
                raise ValueError("pages2k calibration requires source_name or source_path")
        if self.marine_predictor not in {"tos", "best_of_tas_tos"}:
            raise ValueError("unsupported pages2k calibration marine_predictor")
        if not self.season_candidates or any(not season for season in self.season_candidates):
            raise ValueError("pages2k calibration season_candidates must not be empty")
        invalid_month = any(
            month == 0 or abs(month) > 12 for season in self.season_candidates for month in season
        )
        if invalid_month:
            raise ValueError("pages2k calibration season months must be in -12..-1 or 1..12")
        if self.method == "tardif2019_quarterly":
            from lmr4dvar.observations.quarterly_seasons import complete_quarter_offsets

            if self.enabled and self.precipitation_path is None:
                raise ValueError("tardif2019_quarterly requires precipitation_path")
            if self.minimum_overlap < 25:
                raise ValueError("Tardif calibration requires at least 25 joint samples")
            precipitation = self.precipitation_season_candidates
            if precipitation is not None and not precipitation:
                raise ValueError("precipitation season candidates must not be empty")
            for season in self.season_candidates + (precipitation or []):
                complete_quarter_offsets(tuple(season))
        return self


class Pages2kConfig(ObservationSourceConfig):
    artifact_format: str = "cfr"
    calibration: Pages2kCalibrationConfig = Field(default_factory=Pages2kCalibrationConfig)

    @model_validator(mode="after")
    def validate_artifact_format(self) -> Pages2kConfig:
        if self.artifact_format not in {"cfr", "calibrated_netcdf"}:
            raise ValueError("unsupported PAGES2k artifact_format")
        return self


class SedPSMConfig(ObservationSourceConfig):
    experiment: int = Field(default=0, ge=0, le=4)
    path: Path | None = None
    relative_path: Path | None = None
    n_age_realizations: PositiveInt = 1
    variance_floor: float = Field(default=1e-8, gt=0)
    climatology_path: Path | None = None
    climatology_variable: str = Field(default="climatology", min_length=1)

    @model_validator(mode="after")
    def validate_data_path(self) -> SedPSMConfig:
        if self.path is not None and self.relative_path is not None:
            raise ValueError("SedPSM path and relative_path are mutually exclusive")
        if self.relative_path is not None and self.relative_path.is_absolute():
            raise ValueError("SedPSM relative_path must be relative to paths.sedpsm_root")
        return self

    def resolve_data_path(self, sedpsm_root: Path) -> Path:
        if self.path is not None:
            return self.path
        if self.relative_path is not None:
            return sedpsm_root / self.relative_path
        raise ValueError("SedPSM path or relative_path is required")


class BoreholeErrorEstimationConfig(StrictModel):
    enabled: bool = False
    perturbation_path: Path | None = None
    method_profile: str = "si_2026"
    history_start_year: int = 1300
    depth_start: float = Field(default=20.0, ge=0)
    depth_end: float = Field(default=300.0, gt=0)
    depth_step: float = Field(default=5.0, gt=0)
    minimum_depth_points: PositiveInt = 5
    time_steps: list[PositiveInt] = Field(default_factory=lambda: [25, 40, 50])
    thermal_conductivities: list[float] = Field(default_factory=lambda: [2.5, 3.0, 3.5])
    volumetric_heat_capacities: list[float] = Field(default_factory=lambda: [2.5e6, 3.0e6, 3.5e6])
    svd_modes_by_step: dict[int, list[PositiveInt]] | None = None
    gradient_threshold_k_per_year: float | None = Field(default=None, gt=0)
    deduplicate_diffusivities: bool | None = None
    variance_ddof: int | None = Field(default=None, ge=0, le=1)
    variance_floor: float = Field(default=1e-8, gt=0)

    @model_validator(mode="after")
    def validate_error_estimation(self) -> BoreholeErrorEstimationConfig:
        if self.enabled and self.perturbation_path is None:
            raise ValueError("borehole error_estimation perturbation_path is required when enabled")
        if self.method_profile not in {"si_2026", "legacy_notebook"}:
            raise ValueError("unsupported borehole error_estimation method_profile")
        if self.depth_start >= self.depth_end:
            raise ValueError("borehole error_estimation depth_start must precede depth_end")
        if not self.time_steps or len(set(self.time_steps)) != len(self.time_steps):
            raise ValueError("borehole error_estimation time_steps must be unique")
        if any(value <= 0 for value in self.thermal_conductivities):
            raise ValueError("thermal conductivities must be positive")
        if any(value <= 0 for value in self.volumetric_heat_capacities):
            raise ValueError("volumetric heat capacities must be positive")
        if self.svd_modes_by_step is not None:
            missing = set(self.time_steps) - set(self.svd_modes_by_step)
            if missing:
                raise ValueError(
                    "svd_modes_by_step is missing time steps: "
                    + ", ".join(str(value) for value in sorted(missing))
                )
        return self


class BoreholeConfig(ObservationSourceConfig):
    reference_period: tuple[float, float] = (1300, 1700)
    history_policy: str = "full"
    error_estimation: BoreholeErrorEstimationConfig = Field(
        default_factory=BoreholeErrorEstimationConfig
    )

    @model_validator(mode="after")
    def validate_borehole_options(self) -> BoreholeConfig:
        if self.reference_period[0] > self.reference_period[1]:
            raise ValueError("borehole reference_period must be increasing")
        if self.history_policy not in {"full", "legacy_reference_start"}:
            raise ValueError("unsupported borehole history_policy")
        return self


class Temp12kPreparationConfig(StrictModel):
    enabled: bool = False
    source_path: Path | None = None
    source_format: str = "raw_temp12k"
    pages2k_source_name: str | None = "PAGES2kv2"
    pages2k_source_path: Path | None = None
    anomaly_period: tuple[int, int] = (0, 2000)
    require_anomaly_baseline: bool = False
    comparison_period: tuple[int, int] = (0, 2000)
    maximum_distance_km: float = Field(default=50.0, gt=0)
    duplicate_correlation_minimum: float = Field(default=0.98, ge=-1, le=1)
    duplicate_minimum_overlap: PositiveInt = 11
    diagnostic_absolute_correlation_minimum: float = Field(default=0.8, ge=0, le=1)
    diagnostic_minimum_overlap: PositiveInt = 20
    minimum_record_points: PositiveInt = 2

    @model_validator(mode="after")
    def validate_preparation(self) -> Temp12kPreparationConfig:
        if self.source_format not in {"raw_temp12k", "cfr_proxydb"}:
            raise ValueError("unsupported Temp12k preparation source_format")
        for name, period in (
            ("anomaly_period", self.anomaly_period),
            ("comparison_period", self.comparison_period),
        ):
            if period[0] > period[1]:
                raise ValueError(f"Temp12k preparation {name} must be increasing")
        if self.enabled:
            if self.source_path is None:
                raise ValueError("Temp12k preparation source_path is required when enabled")
            if self.pages2k_source_name is None and self.pages2k_source_path is None:
                raise ValueError(
                    "Temp12k preparation requires pages2k_source_name or pages2k_source_path"
                )
        return self


class Temp12kConfig(ObservationSourceConfig):
    preparation: Temp12kPreparationConfig = Field(default_factory=Temp12kPreparationConfig)


class ObservationsConfig(StrictModel):
    pages2k: Pages2kConfig = Field(default_factory=Pages2kConfig)
    sedpsm: SedPSMConfig = Field(default_factory=SedPSMConfig)
    borehole: BoreholeConfig = Field(default_factory=BoreholeConfig)
    temp12k: Temp12kConfig = Field(default_factory=Temp12kConfig)

    @model_validator(mode="after")
    def require_observations(self) -> ObservationsConfig:
        if not any(
            source.enabled for source in (self.pages2k, self.sedpsm, self.borehole, self.temp12k)
        ):
            raise ValueError("at least one observation source must be enabled")
        return self


class AssimilationConfig(StrictModel):
    start: str
    end: str
    optimizer: str = "lbfgs"
    max_iterations: PositiveInt = 100
    inner_iterations: PositiveInt = 20
    minimum_outer_iterations: PositiveInt = 22
    device: Device = Device.AUTO
    precision: Precision = Precision.FLOAT32
    initial_weight: float = Field(default=1.0, gt=0)
    model_error_weight: float = Field(default=1.0, gt=0)
    tolerance_gradient: float = Field(default=1e-7, gt=0)
    tolerance_change: float = Field(default=1e-9, gt=0)
    outer_tolerance_change: float = Field(default=1e-6, gt=0)
    history_size: PositiveInt = 100


class ReconstructionClimatologyConfig(StrictModel):
    path: Path
    variable: str


class ReconstructionConfig(StrictModel):
    variables: list[str]
    indices: list[str] = Field(default_factory=list)
    legacy_layout: bool = False
    sea_ice_concentration_scale: float = Field(default=1.0, gt=0)
    sea_ice_concentration_climatology: ReconstructionClimatologyConfig | None = None
    sea_ice_thickness_climatology: ReconstructionClimatologyConfig | None = None
    plot_figures: bool = True
    figure_dpi: PositiveInt = 150

    @model_validator(mode="after")
    def validate_sea_ice_climatologies(self) -> ReconstructionConfig:
        indices = {name.lower() for name in self.indices}
        if indices & {"sia", "siv"} and self.sea_ice_concentration_climatology is None:
            raise ValueError("SIA/SIV reconstruction requires a SIC climatology")
        if "siv" in indices and self.sea_ice_thickness_climatology is None:
            raise ValueError("SIV reconstruction requires a SIT climatology")
        return self


class ReferenceFieldConfig(StrictModel):
    path: Path
    variable: str | None = None
    comparison_period: tuple[int, int] | None = None
    regrid_to_reconstruction: bool = False
    evaluation_scope: str | None = None

    @model_validator(mode="after")
    def validate_period(self) -> ReferenceFieldConfig:
        if (
            self.comparison_period is not None
            and self.comparison_period[0] > self.comparison_period[1]
        ):
            raise ValueError("reference field comparison_period must be increasing")
        return self


class ReferenceSeriesConfig(StrictModel):
    path: Path
    variable: str | None = None
    comparison_period: tuple[int, int] | None = None

    @model_validator(mode="after")
    def validate_period(self) -> ReferenceSeriesConfig:
        if (
            self.comparison_period is not None
            and self.comparison_period[0] > self.comparison_period[1]
        ):
            raise ValueError("reference series comparison_period must be increasing")
        return self


class VerificationConfig(StrictModel):
    anomaly_period: tuple[int, int]
    comparison_period: tuple[int, int]
    running_mean_years: PositiveInt = 31
    plot_figures: bool = True
    figure_dpi: PositiveInt = 150
    reference_fields: dict[str, Path | ReferenceFieldConfig] = Field(default_factory=dict)
    reference_indices: Path | dict[str, ReferenceSeriesConfig] | None = None

    @model_validator(mode="after")
    def validate_periods(self) -> VerificationConfig:
        if self.anomaly_period[0] > self.anomaly_period[1]:
            raise ValueError("verification.anomaly_period must be increasing")
        if self.comparison_period[0] > self.comparison_period[1]:
            raise ValueError("verification.comparison_period must be increasing")
        return self


class HPCConfig(StrictModel):
    system: str = "pbs"
    queue: str = "casper"
    account: str
    python_executable: str = "python"
    walltime: str = "12:00:00"
    memory_gb: PositiveInt = 80
    cpus: PositiveInt = 1
    gpus: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_pbs_values(self) -> HPCConfig:
        if self.system != "pbs":
            raise ValueError("only the pbs HPC system is currently supported")
        if not self.python_executable.strip():
            raise ValueError("hpc.python_executable must not be empty")
        return self


class ProjectConfig(StrictModel):
    project: ProjectInfo
    paths: PathsConfig
    state: StateConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    eof: EOFConfig = Field(default_factory=EOFConfig)
    lim: LIMConfig = Field(default_factory=LIMConfig)
    observations: ObservationsConfig
    assimilation: AssimilationConfig
    reconstruction: ReconstructionConfig
    verification: VerificationConfig
    hpc: HPCConfig | None = None

    @model_validator(mode="after")
    def validate_cross_section_contracts(self) -> ProjectConfig:
        state_variables = set(self.state.variables)
        missing = set(self.reconstruction.variables) - state_variables
        if missing:
            raise ValueError(
                "reconstruction.variables are absent from state.variables: "
                + ", ".join(sorted(missing))
            )
        sedpsm = self.observations.sedpsm
        if sedpsm.enabled:
            if self.paths.sedpsm_root is None:
                raise ValueError("paths.sedpsm_root is required when SedPSM is enabled")
            if sedpsm.path is None and sedpsm.relative_path is None:
                raise ValueError(
                    "observations.sedpsm.path or relative_path is required when enabled"
                )
        pages2k = self.observations.pages2k
        if pages2k.calibration.enabled and not pages2k.enabled:
            raise ValueError("PAGES2k calibration requires observations.pages2k.enabled=true")
        if pages2k.calibration.enabled and pages2k.calibration.method == "tardif2019_quarterly":
            if not {"tas", "pr"}.issubset(state_variables) or self.state.timestep_months != 3:
                raise ValueError(
                    "quarterly bivariate PSM requires tas/pr states and a 3-month timestep"
                )
        borehole = self.observations.borehole
        if borehole.error_estimation.enabled and not borehole.enabled:
            raise ValueError(
                "borehole error estimation requires observations.borehole.enabled=true"
            )
        temp12k = self.observations.temp12k
        if temp12k.preparation.enabled and not temp12k.enabled:
            raise ValueError("Temp12k preparation requires observations.temp12k.enabled=true")
        return self
