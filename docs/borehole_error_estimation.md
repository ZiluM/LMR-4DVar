# Borehole Error Estimation

## Workflow Position

The borehole preparation stage runs after the other proxy preparation stages
and before assimilation:

```text
train-lim -> calibrate-pages2k -> prepare-temp12k
  -> estimate-borehole-error -> assimilate
```

When `observations.borehole.error_estimation.enabled` is false, the stage
records and fingerprints the legacy `observations.borehole.path`. When enabled,
assimilation reads the generated `estimate-borehole-error/boreholes.nc`.

## Inputs

The estimator consumes the combined perturbation NetCDF produced after fitting
and removing each station's linear geothermal profile over 200--300 m. It
requires:

- `perturbation`: raw temperature minus the mean geothermal fit.
- `perturbation_plus_2sigma`: perturbation using slope plus two standard errors.
- `perturbation_minus_2sigma`: perturbation using slope minus two standard errors.
- Station coordinates `lat`, `lon`, `logging_year`, and `region`.

This replaces the file-by-file error merge in `combine_error0.py`. The source
perturbation file remains external and is fingerprinted rather than copied into
Git.

## Inverse Ensemble

For each perturbation profile, the conductive step-response matrix is

```text
M[i,j] = erfc(z_i / (2 sqrt(kappa t_j)))
       - erfc(z_i / (2 sqrt(kappa (t_j - delta_t))))
```

The truncated SVD inverse is applied to each profile:

```text
T_surface = V S_k^-1 U^T y_borehole
```

Each candidate surface history is projected forward through the same matrix.
Candidates whose maximum absolute temporal gradient exceeds the configured
threshold are rejected. The retained depth residuals are measured relative to
the mean perturbation profile, matching `cal_error.py`:

```text
residual[m, depth] = reconstructed[m, depth] - mean_perturbation[depth]
R[depth] = sum(residual[:, depth]^2) / (N_retained - ddof)
```

A positive `variance_floor` prevents zero variance from creating an infinite
assimilation weight.

## Method Profiles

`method_profile: si_2026` is the scientific default:

- Nine conductivity/heat-capacity combinations are retained, even where two
  combinations yield the same diffusivity.
- Time steps are 25, 40, and 50 years.
- SVD modes are `[3,4,5]`, `[2,3,4]`, and `[2,3,4]`, respectively.
- The gradient threshold is 4 K per decade, or 0.4 K per year.
- Variance uses `ddof=0`, matching the SI mean-squared-residual equation.
- Three perturbations times nine thermal combinations times nine time-step/SVD
  choices produce 243 candidates before gradient filtering.

`method_profile: legacy_notebook` reproduces active details in `cal_error.py`:

- Duplicate diffusivities are removed, leaving seven distinct values.
- Every time step uses SVD modes `[2,3,4]`.
- The gradient threshold is `2.57/50 = 0.0514` K per year.
- Variance uses `ddof=1`.
- The resulting pre-filter ensemble contains 189 candidates.

Every profile default can be overridden explicitly. Overrides and effective
values are written to `manifest.json`; the profile name alone is not treated as
sufficient provenance.

## Configuration

```yaml
observations:
  borehole:
    enabled: true
    strict: false
    reference_period: [1300, 1700]
    history_policy: legacy_reference_start
    error_estimation:
      enabled: true
      perturbation_path: /path/to/CV21_profile_perturbations.nc
      method_profile: si_2026
      history_start_year: 1300
      depth_start: 20
      depth_end: 300
      depth_step: 5
      minimum_depth_points: 5
      time_steps: [25, 40, 50]
      thermal_conductivities: [2.5, 3.0, 3.5]
      volumetric_heat_capacities: [2.5e6, 3.0e6, 3.5e6]
      variance_floor: 1.0e-8
```

## Artifacts

`boreholes.nc` contains:

- Mean and plus/minus-two-sigma perturbation profiles on the configured grid.
- `error`, the diagonal observation-error variance in K2 by station and depth.
- Candidate and retained ensemble counts per station.
- A status string for successful or skipped stations.
- Latitude, longitude, logging year, region, method, and units metadata.

`manifest.json` records source path, method profile, all thermal and inversion
parameters, effective profile defaults, and successful/skipped station counts.

## Regression Evidence

For station `CA_0110`, `legacy_notebook` generated all 189 candidates and
retained all 189. Against the existing `error_data/CA_0110.nc`, the 44 common
finite depth variances have correlation 1.0, RMSE
`1.35e-18 K2`, and maximum absolute difference `3.47e-18 K2`. This confirms
numerical reproduction of `cal_error.py` for that station. The `si_2026`
profile generated and retained 243 candidates; its mean variance was
`0.0019919 K2`, compared with `0.0020011 K2` in legacy mode.
