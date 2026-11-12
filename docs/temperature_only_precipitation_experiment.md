# Temperature-only PSM precipitation ablation

Run `configs/experiments/cesm_pages2k_temperature_only_500_2000.yml` through
`verify` on a GPU compute node. No preprocessing, EOF fitting or LIM training
is performed: the exact seven-variable prior is reused from
`CESM_tardif_quarterly_prior`, including the 15 precipitation EOF modes.

The experiment retains the bivariate run's accepted proxy IDs (719 records),
raw proxy values, instrumental temperature input, anomaly baseline (1951-1980),
calibration period (1901-2000), quarterly season candidates, reconstruction
period (-500 to 2000), float32 precision and LBFGS settings. All records are
refitted with temperature only. TRW seasonal selection, temperature coefficients,
intercepts and residual variances are refitted, not obtained by dropping the
precipitation coefficient. Missing matched records cause calibration to fail.
GPCC is loaded to prepare verification fields but never enters the temperature
regressions. No accepted PSM has a precipitation predictor. Precipitation remains
indirectly constrained through multivariate prior covariance and LIM dynamics.
This compares the complete recalibrated PSM, not just one regression coefficient;
temperature season choices and observation-error weights can therefore change.

Outputs are under the configured results root in
`CESM_pages2k_temperature_only_500_2000`. The calibration manifest and diagnostics
record predictor selection and network counts. GPCC verification runs as part
of the workflow. `analyze/precipitation_products_temperature_only.yml` compares
GPCPv2.3, 20CRv3, ERA5 and ERA20C over each longest complete overlap;
`analyze/precipitation_products_temperature_only_common.yml` uses 1980-2000.
Both reuse the bivariate experiment's prepared reference cache, grids, baseline,
land/ocean masks and scoring implementation. Metrics include temporal
correlation maps and area-weighted mean R, annual spatial-pattern correlation,
global-mean series correlation, CE and RMSE.

Render the workflow with `lmr4dvar render-pbs` and a local GPU profile.
Create a separate CPU batch job for the product-comparison script; it must
run after the reconstruction succeeds. Scheduler wrappers and job identifiers
are site-specific and are not required by the scientific configuration.
