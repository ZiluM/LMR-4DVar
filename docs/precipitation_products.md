# Global precipitation reference comparisons

Run `python analyze/verify_precipitation_products.py analyze/precipitation_products.yml`
on a PBS CPU node. This is post-processing of the existing CESM real-PAGES2k
bivariate reconstruction, not another assimilation or a pseudo-proxy experiment.

## Inputs

- GPCP v2.3: NCAR GDEX d728004, monthly v23rB1 satellite-gauge files.
- ERA5: NCAR GDEX d633001, monthly mean MTPR (kg m-2 s-1), multiplied by 86400.
- ERA-20C: NCAR GDEX d626000, daily 06 UTC TP forecast cycles, steps 3,...,24 h.
  Although these GRIB files label stepType as instant, TP is cumulative from
  initialization. Select the full step24 accumulation once per day, convert
  m to mm, group by its midpoint (18 UTC on the initialization day), then divide
  monthly totals by calendar days. Require all daily cycles and reject missing
  or negative daily totals. No negative incremental rainfall is clipped.
  This uses 06UTC-06UTC days: each monthly window is shifted six hours relative
  to strict midnight-to-midnight calendar months. This approximation is recorded
  in reference artifacts and the evaluation scope.
- 20CR v3: NOAA PSL monthly ensemble mean APCP, averages of 3-hour accumulation
  (kg/m2), multiplied by eight, NOT divided by days/month.
- Fixed land fraction: NCAR d131003 invariant lsm, area-remapped to the same grid;
  land cells have fraction >=0.5; ocean cells <0.5, including coastal mixtures.

## Evaluation contract

Use a global 2.5-degree grid matching GPCP centers, conservative spherical
rectilinear cell overlaps, inferred center-to-edge latitude bounds with polar
edges at +/-90, and periodic longitude. Gaussian source latitude edges are
inferred too (not native Gaussian quadrature weights). Missing contributing
source cells invalidate target cells; never fill reference gaps with zero.

Compare 1980-2000 with separate DJF/MAM/JJA/SON anomaly baselines over 1981-2000.
GPCP begins January 1979, so DJF 1979 is incomplete. Monthly references use
complete, equally weighted three-month seasons. Annual means average the four
complete quarterly states (DJF-year convention), matching the assimilation
time convention rather than strict January-December annual means. Reject
duplicate/missing monthly samples and incomplete baseline quarters.

Report global, land and ocean separately, using a fixed paired-finite mask in
each comparison. Metrics include area-mean annual series r/RMSE, spatial anomaly
pattern r at each quarter/year, and gridpoint temporal r/CE/RMSE for annual,
quarterly and each season. Do not confuse the latter two correlation definitions.
No 31-year smoothing is possible for the 21-year comparison.
Gridpoint r and CE are undefined where the relevant temporal standard
deviation is <=1e-6 mm/day. This avoids reporting enormous CE or spurious r
from numerical noise at effectively constant/dry cells. The threshold and
valid-gridpoint counts are recorded; reference gaps are still never filled.

ERA5, ERA-20C and 20CR precipitation is model-generated reanalysis output, not
direct observation truth. GPCP includes GPCC gauge information over land and is
not fully independent of GPCC-calibrated PSMs. Differences between these
references are part of uncertainty, not automatic reconstruction errors.

Reduced inputs are cached externally for reruns. Raw archive files remain
read-only. Product failures are recorded independently in summary.json, without
preventing the remaining products from being attempted. Failed runs exit nonzero.

## Sources

- https://gdex.ucar.edu/datasets/d728004/
- https://gdex.ucar.edu/datasets/d633001/
- https://gdex.ucar.edu/datasets/d626000/
- https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.html
- https://psl.noaa.gov/thredds/dodsC/Projects/20CRv3/accumsSI/apcp.1867.nc.html

## Input handling and output locations

Cyclic neighbor midpoints define longitude edges, ensuring exact 360-degree
closure for rounded GRIB coordinates. GPCP's equivalent `millimeters/day`
unit spelling is accepted. ERA5 monthly reads start in 1979, including the
months needed for DJF 1980. ERA-20C packing tolerances are derived from each
GRIB message. Its cumulative fields can decrease beyond packing uncertainty,
so exact nonnegative three-hour deaccumulation is not assumed. Complete daily
step24 accumulations use the explicitly recorded 06UTC day convention.

Results are under the reconstruction run's
`verification/precipitation_products/`, while the GPCC workflow verification
remains under `stages/verify/`. Run the analysis in a CPU batch job with local
scheduler resources and the environment variables in `configs/paths.example.sh`.
No site-specific wrapper is needed by the analysis script.

## Recorded reference results

The following tables record an existing experiment snapshot. These additional
precipitation comparisons are not asserted to reproduce the accompanying
paper's exact production runs. Regenerate them with matching inputs and
configurations before using them as regression evidence.

All four products produced the three regional NetCDF metric files and figures.
Values below use 1980-2000, with quarter-specific 1981-2000 anomalies. Spatial
pattern r is computed across gridpoints for each year, then averaged over time;
gridpoint temporal r is computed through time, then area-averaged over space.

| Reference | Global mean series r | Ocean mean series r | Global spatial pattern r | Global gridpoint temporal r |
| --- | ---: | ---: | ---: | ---: |
| GPCP v2.3 | 0.106 | 0.289 | 0.344 | 0.186 |
| 20CR v3 | 0.511 | 0.437 | 0.319 | 0.206 |
| ERA5 | 0.144 | 0.072 | 0.336 | 0.204 |
| ERA-20C | 0.423 | 0.492 | 0.363 | 0.207 |

Global area-averaged annual gridpoint CE is negative for every product:
-0.352 (GPCP), -0.910 (20CR), -0.424 (ERA5), -0.317 (ERA-20C).
Global mean agreement is reference-sensitive; modest positive spatial
correlation does not establish reliable local rainfall amplitudes.

## Longest Overlap Comparisons

Use `analyze/precipitation_products_longest.yml`, with `--product PRODUCT` to
run separate CPU jobs. The original 1980-2000 output directory is preserved.
The new output directory is `verification/precipitation_products_longest/`.
Completed short-window input reductions are reused, without shortening the
new comparisons to the shared satellite era. Keep the common 1981-2000 anomaly
baseline; do not change the reconstruction or assimilation.

| Product | First complete quarter | Expected annual comparison |
| --- | --- | --- |
| GPCP v2.3 | MAM 1979 | 1980-2000 |
| 20CR v3 | MAM 1806 | 1807-2000 |
| ERA-20C | MAM 1900 | 1901-2000 |
| ERA5 | MAM 1940 | 1941-2000 |

These ranges depend on verified complete input coverage. Include all complete
edge quarters in quarterly/seasonal correlations, but never average partial
years for annual metrics. Save actual time ranges for every aggregation.
The 1806-1835 part of 20CR is its experimental extension, not its main
1836-onward production period.

Completed longest-window annual results:

| Product | Actual annual period | Global temporal-map R | Land R | Ocean R |
| --- | --- | ---: | ---: | ---: |
| GPCP v2.3 | 1980-2000 | 0.186 | 0.113 | 0.216 |
| 20CR v3 | 1807-2000 | 0.082 | 0.063 | 0.090 |
| ERA5 | 1941-2000 | 0.129 | 0.072 | 0.151 |
| ERA-20C | 1901-2000 | 0.127 | 0.100 | 0.138 |

All four extended-input comparisons completed. These R values are not
comparable as a ranking of reference products because the periods differ;
the original common-period results above remain available.

## Official LMR v2.1 Benchmark

`analyze/compare_lmrv21_precipitation.py` downloads the official annual
`prate_MCruns_ensemble_mean_LMRv2.1.nc` and averages its 20 MCrun fields.
Units are verified as kg/m^2/s and converted to mm/day. CF time is decoded
with its noleap calendar and year-zero origin. The source's internal dataset
attribute still says version 1.1; identify it by the official v2.1 URL and
retain this metadata caveat and the input SHA-256 checksum.

The recorded comparison covers four products, two reference-annual
conventions, two interval selections, and three regions. The reader preserves
native input chunks before scientific averaging and comparison.
Outputs are below `verification/lmrv21_precipitation/` (96 CSV rows).
Do not replicate annual LMR values into four synthetic quarterly values.

Both methods are rebased to 1981-2000 on identical comparison years, target
grid, and shared finite coverage. Primary reference annual means preserve
the existing DJF-year convention. Exact LMR within-year averaging is not
established by its public catalog; the extra calendar-reference sensitivity
does not make the 4D-Var annual estimates exact calendar means.

| Product / longest annual interval | 4D-Var temporal-map R | LMR temporal-map R | 4D-Var mean CE | LMR mean CE |
| --- | ---: | ---: | ---: | ---: |
| GPCP / 1980-2000 | 0.186 | 0.211 | -0.352 | 0.017 |
| 20CR / 1807-2000 | 0.082 | 0.153 | -0.945 | -0.490 |
| ERA5 / 1941-2000 | 0.129 | 0.195 | -0.555 | -0.124 |
| ERA-20C / 1901-2000 | 0.127 | 0.191 | -0.542 | -0.198 |

The LMR grand mean also has higher global mean annual spatial-pattern r
and lower area-mean local RMSE in these matched comparisons. It does not
always have higher global-mean-series correlation. This is not a controlled
algorithm comparison: 20 MCrun means versus one trajectory, different
priors, proxy subsets, and seasonal candidates all affect the result.
Eight focused annual-only tests passed, including CF-date year labeling,
MCrun averaging, unit conversion, missing baselines, shared masks, and CE
amplitude sensitivity.

Data source: https://atmos.washington.edu/~hakim/lmr/LMRv2/

### Paired Correlation And CE Maps

Run `analyze/plot_precipitation_skill_comparison.py COMPARISON_ROOT` on
the saved LMR v2.1 comparison directory. No raw archives or assimilation
rerun are needed. The script verifies matched years and reference mean
series before plotting. It creates six overview PNGs under `figures/`:
correlation maps, CE maps, and four-metric bar charts for both `common`
and `longest` intervals. All maps include both LMR v2.1 and LMR4D-Var.

Differences are **LMR4D-Var minus LMR v2.1** on paired finite skill cells.
Positive delta-r or delta-CE favors 4D-Var. Absolute CE and delta-CE
colors saturate outside [-1,1], but raw CE is never clipped in saved
differences or area-weighted summaries. Correlation differences display
[-0.5,0.5]. Titles state area means over the paired skill-map domain.
The sign-based better-area fractions in `difference_summary.json` are
not statistical-significance results. Each product/interval directory
also stores `ce_difference.nc` and `correlation_difference.nc`.
Three focused tests verify signs, raw CE preservation,
latitude weighting, valid coverage, and rejection of mismatched grids.

ERA5 before 1979 is read from the formal hourly archive d633000, not the
preliminary 1950-1978 monthly back extension. Forecast initializations occur
at 06/18 UTC, with steps 1-12. MTPR represents the hour preceding validity;
assign each sample to its interval midpoint, then its calendar month. Reduce
float64 sums in native hourly chunks before conservative spatial remapping.
Merge cross-month/year contributions and require days_in_month*24 samples.
Only the initial January 1940 (first six hours unavailable) may be incomplete;
drop it rather than extrapolating. All interior missing hours/months are errors.

Every annual, quarterly and seasonal temporal-correlation map now labels
`Area-weighted R`, computed as the cosine-latitude weighted mean of the
gridpoint temporal Pearson r values on fixed paired-finite coverage. This is
neither r of the global mean precipitation series nor spatial pattern r.
The same statistic is shown for land and ocean. Save the complete values in
`temporal_correlation_summary.csv` and regional NetCDF files.

Independent product jobs publish atomic per-product JSON summaries; a locked
merge updates summary.json and reports uncompleted products as pending.
Land-mask output files are product-specific to prevent concurrent writes.
20CR raw/reduced caches are keyed by source URL, preventing accidental reuse
of the shorter downloaded time range.

Use a separate CPU batch job per product, sized for the selected archive and
period. Local job scripts and scheduler identifiers are not part of the
scientific contract. The tests cover physical hourly month boundaries,
partial-year exclusions, and summary merging.

References for the long inputs and hourly convention:

- https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.pressure.html
- https://gdex.ucar.edu/datasets/d633000/
- https://confluence.ecmwf.int/pages/viewpage.action?pageId=111151514
