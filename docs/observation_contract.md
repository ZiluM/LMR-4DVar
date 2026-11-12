# Observation Contract

Every observation family is converted to dependency-free records before
assimilation. Time-indexed proxies use `ObservationRecord`, with a globally
unique ID, family, observed climate variable, location, time, value, error
variance, optional time bounds, and provenance metadata. Depth-indexed
borehole profiles use `BoreholeRecord`; depth is not represented as fake time.

`variance` always means variance, not standard deviation. Scalar variances are
broadcast to each observation. Invalid or non-positive variances are rejected
when records are constructed. Missing observed values may remain in a record,
but the loss reports how many values were dropped.

Time matching returns model indices, observation indices, distances, and all
unmatched observation indices. Low-resolution records use differentiable window
averaging and report the number of model samples in every accepted window.
Inclusive window boundaries are converted once to half-open model index ranges
with `searchsorted`. Each objective evaluation then uses a differentiable
cumulative sum rather than materializing an observation-by-model-time matrix.
The linear proxy operator caches those fixed ranges across optimizer closures.
Quarterly-to-seasonal aggregation is vectorized and shared by records with the
same seasonality. Records are grouped by variable so EOF point decoding uses
batched matrix multiplication. EOF point transforms, observation tensors, and
exact time matches are fixed plans rather than closure-time xarray work.
Extrapolation is not implicit.

For linear proxy models, `EOFPointDecoder` constructs a full-state affine
transform at the nearest valid EOF grid cell. The transform includes PC
standard-deviation scaling, area-weight removal, and any fitted EOF mean. Only
the requested variable's state slice is nonzero. A proxy slope and intercept
are applied after local climate reconstruction.

Quarterly states are converted to proxy seasonality with an explicit edge
policy. `legacy_repeat` reproduces the old behavior of repeating the first
year's quarter when a season requests the previous year; `drop` omits that
incomplete first annual value. Low-resolution observations are then averaged
over their time bounds or configured resolution without implicit extrapolation.

The Gaussian observation objective is:

```text
Jo = 0.5 * sum((H(x) - y)^2 / R)
```

It reports total loss, loss by observation family, used count, and dropped
count. Assimilation diagnostics retain each of these values for every optimizer
closure. If no finite matched observations contribute, it raises
`EmptyObservationError`; a zero observational loss cannot silently stand in for
an empty assimilation.

SedPSM PPE records are reduced exactly as in the legacy integration: the
ensemble mean is the observed value and population ensemble variance is `R`,
with an explicit configurable floor. SedPSM remains a separately installed
local dependency. The standalone repository does not mutate `sys.path` to load
it.

The SedPSM observation operator annualizes and decodes local `tas`, adds a
required site climatology to convert EOF anomalies back to degrees Celsius,
and then calls the external tensor-backed pipeline. A configured external
climatology, such as the Berkeley Earth field used by regression experiments,
takes precedence over the prior simulation climatology artifact. Missing climatology is an
error rather than an implicit zero. Experiments 0--4 use the sensor, archive,
observation-noise, and age-model parameter progression documented by the
SedPSM PPE branch. The fixed pipeline seed makes repeated objective evaluations
deterministic; experiment 4 alone receives age-model arguments.

The adapter was checked against SedPSM `pytorch_age_model` revision `9ae40d3`.
PPE experiment 0 loaded 24 records and 27,744 individual observations; the
first record contained 1,156 times and 10 ensemble members. These counts are a
data preflight reference, not hard-coded loader assumptions.

A real experiment-2 Mg/Ca smoke test produced 97 of 97 finite values from an
850--2000 tensor climate history. Back propagation produced finite gradients
at 1,151 annual inputs, confirming that the installed external pipeline keeps
the DA graph connected.

PAGES2k and Temp12k serialized databases are read with `cfr==2024.4.4` and
immediately converted to standalone records. PAGES2k adapters extract the
calibrated predictor, slope, intercept, seasonality, and `R`; Temp12k adapters
use local `tas`, the record seasonality, and temporal resolution. Temp12k is
loaded without PAGES2k's `calibrated` tag filter because its serialized database
is already selected, and model estimates are centered on 0--2000 CE before
comparison to its anomaly records. Strict mode
rejects the complete load when any record cannot be converted. Audit mode
returns every skipped record ID and reason rather than silently dropping it.

PAGES2k calibration may be produced inside this workflow before conversion.
The instrumental predictor is projected through the exact EOF artifact used by
the assimilation, and calibration diagnostics and thresholds are retained as
stage artifacts. The complete procedure and legacy-compatibility choices are
documented in [pages2k_calibration.md](pages2k_calibration.md).

Temp12k preparation may likewise be produced inside the workflow. Raw records
are converted to direct-temperature observations, seasonality is normalized,
values are referenced to 0--2000 CE, and colocated PAGES2k duplicate years are
removed with auditable pair and removal tables. See
[temp12k_preparation.md](temp12k_preparation.md).

The CESM ensemble-001 PAGES2k-like regression database passed an audit with
425 of 425 records converted, 260,614 values retained, and zero skipped
records. Its calibrated predictors comprise 421 `tas` records and four `tos`
records.

Borehole profiles are interpolated to 100--300 m at 5 m spacing by default,
matching the retained legacy analysis depths. Surface `tas` is annualized and
decoded at the nearest valid EOF grid point before the complementary-error-
function heat kernel is applied. The implementation preserves tensor dtype,
device, and gradients and supports irregularly spaced annual histories by
differencing kernels at the actual sample ages.
For a fixed network and time axis, site decoding weights and heat-conduction
coefficients are precomputed once. Optimizer closures then decode every site
with one matrix multiplication and evaluate all profiles with a batched tensor
contraction.

Depth-dependent borehole variance may be estimated inside the workflow before
assimilation. The estimator reads mean and geothermal-slope-envelope
perturbation profiles, builds a configurable SVD inverse ensemble, rejects
histories with excessive temporal gradients, projects retained histories back
to depth space, and stores mean squared residual by depth. The compact NetCDF
artifact is read directly into `BoreholeRecord`; legacy pickles remain
supported. See [borehole_error_estimation.md](borehole_error_estimation.md).

Reference-period handling is explicit. The default `full` history policy
subtracts the reference-period mean while retaining older surface-temperature
history, because old thermal signals still affect the profile. The
`legacy_reference_start` policy additionally removes all history before the
reference-period start and reproduces the former FDVar operator for regression
experiments. This legacy truncation is not treated as the scientific default.

The CESM ensemble-001 pseudo-borehole pickle passed an audit with 1,071 usable
stations and 41,862 retained depth values. Eight additional stations contain a
scalar string `NAN` instead of depth-indexed error variance; they are reported
as skipped in audit mode and make strict loading fail. No replacement variance
is invented silently.
