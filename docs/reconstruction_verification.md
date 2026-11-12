# Reconstruction and Verification

`decode_trajectory` maps the optimized `(time, state)` trajectory back through
each variable's EOF artifact. `StateLayout` is the source of state slices and
the EOF mode coordinate is checked for every variable. The output is an xarray
Dataset on the original physical grids and preserves the supplied analysis
time coordinate.

The built-in diagnostics are:

- `GMT`: cosine-latitude weighted global mean of `tas`.
- `Nino34`: cosine-latitude weighted `tos` over 5S--5N and 170W--120W.
- `GMOHC`: cosine-latitude weighted global mean of `ohc300` (0--300 m).
- `GMOHC2000`: cosine-latitude weighted global mean of `ohc2000`
  (300--2000 m).
- `SIA`: concentration times spherical grid-cell area, optionally restricted
  to the Northern Hemisphere.
- `SIV`: concentration times thickness times grid-cell area.

Every requested index is written to `indices.nc`. Reconstruction also writes
unreferenced quarterly and annual curves beneath
`stages/reconstruct/figures/indices/<name>/seasonal.png` and `annual.png`.
These figures show the reconstructed values directly; anomaly baselines and
observational comparisons remain part of verification. The PNGs are first-class
reconstruction receipt outputs. Set
`reconstruction.plot_figures: false` to disable them or control resolution with
`reconstruction.figure_dpi`.

Sea-ice concentration scaling is explicit. Fractional fields use
`concentration_scale=1`; percentage fields use `concentration_scale=100`.
The code does not infer units from value ranges. An externally supplied cell
area is preferred for irregular model grids; otherwise area calculation is
limited to increasing one-dimensional regular latitude/longitude coordinates.
SIA and SIV are absolute indices, while the LIM state contains SIC and SIT
anomalies. Their reconstruction therefore requires explicit monthly SIC and
SIT climatology files. The workflow restores each analysis quarter's matching
calendar-month climatology before integration and rejects SIA/SIV configs that
omit the required climatology. It never approximates SIV as
`SIC_anomaly * SIT_anomaly`.

Verification aligns exact time coordinate labels with an inner join and rejects
different spatial grids. It never compares arrays by positional index. Baseline
anomalies support numeric CE years and decoded datetime/cftime coordinates.
Series metrics report correlation, RMSE, and finite common count. Centered
running means default to complete windows; alternate `min_periods` must be
requested explicitly. Spatial verification computes Pearson correlation through
time at each grid point, then reports its cosine-latitude weighted spatial mean
for quarterly, annual, DJF, MAM, JJA, and SON aggregations. Time-resolved spatial
RMSE and valid-grid count are retained separately.

The GMT implementation was checked on the iCESM 850--1849 quarterly source:
all 4,000 values were finite. Relative to the existing verification GMT made
after interpolation to the 2-degree grid, correlation was 0.999996 and RMSE was
0.025 K. The nonzero offset is expected because area averaging and spatial
interpolation do not commute; production comparisons must use the same grid on
both sides.

Workflow reference files are configured under
`verification.reference_fields` and `verification.reference_indices`.
`validate --check-paths` requires these external references to exist. Field
files must contain the configured state-variable name; the index reference and
reconstruction are compared for every common index variable.

A reference field may instead use the structured form with `path`, `variable`,
and `comparison_period`. This supports observational products whose NetCDF
variable name differs from the reconstructed state variable and preserves
variable-specific verification windows. A bare path remains shorthand for the
state-variable name and the global comparison period.

`reference_indices` likewise accepts either one multi-index NetCDF path or a
mapping from reconstructed index name to `path`, optional `variable`, and
optional `comparison_period`. The mapping form avoids creating merged copies of
independent observational index products.

The verify stage writes legacy-compatible correlation maps beneath
`stages/verify/figures/pattern/<variable>/`: `all.png`, `ANN_annual.png`, and
the four seasonal `*_annual.png` files. Index comparisons are written beneath
`stages/verify/figures/indices/<name>/` for annual and centered-running-mean
series. Every PNG is listed in `summary.json` and in the stage receipt. Set
`verification.plot_figures: false` to disable figures or use
`verification.figure_dpi` to control raster resolution.
