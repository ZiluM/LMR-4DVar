# Workflow Receipts

The workflow is a linear dependency graph:

```text
preprocess -> eof -> train-lim -> calibrate-pages2k -> prepare-temp12k
  -> estimate-borehole-error -> assimilate -> reconstruct -> verify
```

`lmr4dvar plan CONFIG` is side-effect free. For a new run it marks preprocess
`ready` and all dependent stages `blocked`. A stage is `complete` only when its
receipt matches the canonical resolved configuration hash, the exact upstream
receipt hash, and every recorded output fingerprint. An existing mismatched
receipt is `stale`; subsequent stages remain blocked.

Stage directories follow:

```text
${output_root}/${project.name}/stages/${stage}/
```

An assimilation-only project may set `paths.prior_run_root` to a completed
project run. Its EOF and train-LIM artifacts are loaded in place rather than
copied or retrained. Planning starts at PAGES2k calibration, validates the
external EOF/LIM output fingerprints and receipt chain, and records the
external LIM receipt hash as the calibration dependency. Temp12k preparation
and borehole error estimation follow PAGES2k calibration. Assimilation depends
on all three preparation stages, including when one uses a legacy artifact.

After successful execution, a stage writes `receipt.json` in its directory.
Receipts list concrete files rather than artifact directories, because a
directory timestamp does not reliably change when an existing child file is
modified. Large files use size and modification time for routine resume checks;
the generic `snapshot --checksum` command remains available when a full SHA-256
baseline is required.

`--from` and `--to` select a contiguous planning range but do not bypass
dependencies. `--force` marks an otherwise complete first selected stage ready
for rerun and blocks its dependents until a new receipt is written. Planning
never invokes a scientific stage or PBS command.

Local `run` handlers implement all nine stages. The
preprocess handler resolves only configured relative NetCDF patterns and uses
the `h5netcdf` backend. Monthly inputs run the complete preprocessing chain.
`prepared_quarterly` inputs are only cropped, season-selected, and
domain-selected, so existing products are not transformed twice. Configured
multi-member inputs are processed separately and concatenated in sorted file
order. EOF artifacts are written per variable, then PCs are
concatenated in `StateLayout` order into `state_pcs.nc`. LIM training consumes
that labeled coupled state and writes dynamics, metadata, and forecast-skill
artifacts. Each handler writes its receipt only after every declared output
exists.

PAGES2k calibration consumes the `tas` and `tos` EOF artifacts and writes the
EOF-projected instrumental fields, a CFR compatibility job, record-level CSV
diagnostics, and a JSON manifest. When calibration is disabled, the stage
writes a manifest identifying the external calibrated artifact instead. Local
instrumental, raw-proxy, and legacy calibrated input files are also recorded in
the receipt so changing them makes the stage stale.

Temp12k preparation converts raw compilation entries to direct-temperature CFR
records, normalizes seasonality, references values to a configured climatology,
and removes only years duplicated in colocated, highly correlated PAGES2k
records. Candidate and removal diagnostics are included in its receipt.

Borehole error estimation consumes geothermal-gradient-removed mean and
plus/minus-two-sigma perturbation profiles. It writes a compact NetCDF artifact
containing profile values, depth-dependent effective error variance, station
metadata, and candidate/retained ensemble counts. A local source NetCDF or
legacy precomputed pickle is fingerprinted by the stage receipt.

Assimilation rebuilds `StateLayout`, loads all EOF and LIM artifacts, and
rejects different state labels. It constructs the configured family-specific
observation bindings, directly optimizes the complete state trajectory in
float64, and writes `analysis.nc` plus full
cost/gradient diagnostics. SedPSM uses a configured external climatology when
present and otherwise requires the `tas` monthly climatology from preprocessing.
It recognizes explicit Kelvin or Celsius units; missing or
unknown units fail before optimization. Borehole audit behavior and thermal
history policy are explicit configuration values.

Reconstruction writes each variable separately so global and hemispheric grids
are not expanded through xarray coordinate union. Requested indices are stored
in a separate file. Quarterly and annual figures for each requested index are
also tracked by the reconstruction receipt. Absolute SIA/SIV restore configured
monthly SIC/SIT climatologies before area integration. Verification normalizes
reference time coordinates and
applies quarter-specific climatologies over the configured anomaly period.
Spatial-field correlation is
computed through time at every grid point, then summarized with cosine-latitude
area weighting for quarterly, annual, and individual-season aggregations. The
gridpoint correlation maps and weighted means are written to
`pattern_metrics.nc`; time-resolved spatial RMSE is retained. Index metrics use
annual means, so a running window is measured in years, not seasonal steps.
When `verification.plot_figures` is enabled, legacy-named spatial maps and index
comparison PNGs are recorded as first-class outputs in the verify receipt.

PBS scripts invoke the same `lmr4dvar run` command as local execution. Queue,
account, walltime, CPU, memory, GPU count, and Python interpreter come from the
local profile. `render-pbs` only writes an executable script; `submit-pbs` is a
separate explicit scheduler action. Rendered jobs export the submission
environment so configuration variables such as simulation and output roots
remain local rather than being committed to the shared experiment YAML.
