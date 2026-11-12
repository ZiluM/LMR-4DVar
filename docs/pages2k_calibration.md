# PAGES2k Calibration

## Position In The Workflow

PAGES2k calibration is an explicit stage:

```text
preprocess -> eof -> train-lim -> calibrate-pages2k
  -> prepare-temp12k -> estimate-borehole-error -> assimilate
```

The stage uses the `tas` and `tos` EOF artifacts from the current run. With an
external prior, it uses those artifacts from `paths.prior_run_root`. This
prevents a calibrated PSM from silently using a different EOF basis than the
one decoded by the observation operator.

When `observations.pages2k.calibration.enabled` is false, the stage records the
legacy `observations.pages2k.path` and assimilation reads that calibrated CFR
job directly. This is the compatibility path for existing experiments.

## Scientific Procedure

The implementation follows `NewOBS/truncate_data.py`, `NewOBS/calib_2026.py`,
and SI Section "PAGES2k proxy system models in the reduced LIM space":

1. Read the monthly instrumental temperature field and form monthly anomalies
   relative to `anomaly_period`, normally 1951-1980.
2. For `tas` and `tos` separately, fill missing source values only for the EOF
   projection, encode with the experiment EOF, and decode back to the truncated
   physical field. Restore the instrumental missing mask afterward.
3. Remove borehole records from PAGES2k. Annualize ordinary records on the
   December-to-November year used by the legacy workflow. Convert each
   subannual coral into four quarterly records.
4. Identify ocean records from coral/marine type and the nearest valid `tos`
   EOF grid cell. Terrestrial records use `tas`. `marine_predictor=tos` follows
   the SI wording; `best_of_tas_tos` reproduces `calib_2026.py` by selecting the
   predictor with the larger adjusted R-squared.
5. Fit `proxy = intercept + slope * temperature` for every configured seasonal
   candidate and retain the candidate with the largest adjusted R-squared.
   At least `minimum_overlap` instrumental observations are required.
6. For records coarser than annual resolution, average annual `tas` over a
   centered window of width `ProxyRecord.dt` before fitting.
7. Set the scalar observation variance `R` to the mean squared calibration
   residual. Retain records satisfying the PSM-correlation and lag-1 residual
   autocorrelation thresholds. NaN diagnostics are retained for compatibility
   with the old filter behavior and are visible in `diagnostics.csv`.

The diagonal `R` assumption follows the SI: the instrumental interval is too
short and unevenly overlapping to estimate a stable cross-record covariance.

## Configuration

```yaml
observations:
  pages2k:
    enabled: true
    strict: true
    calibration:
      enabled: true
      source_name: PAGES2kv2
      source_path: null
      instrumental_path: /path/to/gistemp1200_GHCNv4_ERSSTv5.2x2.nc
      instrumental_variable: tempanomaly
      anomaly_period: [1951, 1980]
      calibration_period: [1850, 2015]
      low_resolution_period: [1850, 2000]
      minimum_overlap: 25
      psm_correlation_minimum: 0.1
      residual_autocorrelation_maximum: 0.9
      projection_fill_value: 0.0
      marine_predictor: best_of_tas_tos
```

Set `source_path` to a local CFR-supported JSON/NetCDF file, ProxyDatabase
pickle, or ReconJob directory for network-independent runs. If it is absent,
`source_name: PAGES2kv2` uses CFR's registered source URL.

## Artifacts

The stage writes:

- `instrumental/tas.nc` and `instrumental/tos.nc`: EOF-truncated predictors.
- `cfr_job/job.pkl`: calibrated CFR compatibility artifact consumed by the
  existing adapter.
- `diagnostics.csv`: predictor, seasonality, overlap, adjusted R-squared, MSE,
  PSM correlation, residual lag-1 correlation, and acceptance per record.
- `manifest.json`: source paths, periods, thresholds, method choices, and
  record counts.

The CFR pickle is confined to this compatibility boundary. Assimilation still
converts every accepted proxy immediately into the standalone
`ObservationRecord` contract.

## Differences Exposed From The Legacy Script

The old scripts hard-code the model name and paths, depend on SSNLIM pickle
classes, and contain unused expert-season mappings. The active code assigns the
same 15 statistical candidate seasons to ordinary records. Those candidates
are now the explicit `season_candidates` configuration default.

The SI says marine records use ERSST/TOS, while `calib_2026.py` fits both TAS
and TOS for ocean-tagged records and chooses adjusted R-squared. Both behaviors
are supported explicitly. The existing experiment is configured with
`best_of_tas_tos` for regression compatibility.

The old script appends calibrated low-resolution records after applying its
two QC filters to regular records. The integrated stage applies the SI's stated
screening rule to every calibrated record, including low-resolution records.
This choice is explicit in `diagnostics.csv` and may change that small subset
relative to the legacy job.
