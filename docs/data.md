# Data and external dependencies

The repository distributes source, configuration templates, and tests.
The synthetic example generates its own small inputs. Real-data experiments
require separately obtained simulations, proxy compilations, and verification
products. No large climate fields or proxy databases are bundled.

## Configure locations

Copy `configs/paths.example.sh` to the ignored `configs/paths.local.sh`, edit
it, and source it in the environment that will run the commands. Only set
the variables used by your selected configuration. Shell expansion is
supported in project YAML files and precipitation analysis YAML files;
unresolved variables raise an error. Relative paths resolve from the process
working directory, so run the documented commands from the repository root.

| Variable | Contents |
| --- | --- |
| `LMR4DVAR_SIMULATION_ROOT` | CESM/MRI input hierarchy used by `input_pattern` |
| `LMR4DVAR_OUTPUT_ROOT` | Writable root for experiment artifacts and logs |
| `LMR4DVAR_OBSERVATION_ROOT` | Prepared observation and climatology hierarchy |
| `LMR4DVAR_REFERENCE_ROOT` | Verification fields and indices |
| `LMR4DVAR_PRIOR_ROOT` | One compatible completed prior run, including `stages/` |
| `LMR4DVAR_TARDIF_OBSERVATION_ROOT` | Raw PAGES2k, GISTEMP, and GPCC files below |
| `LMR4DVAR_CESM_PRECIPITATION_ROOT` | Monthly CESM-LME PRECT member files |
| `LMR4DVAR_INPUT_ROOT` | Optional input provenance and audit reports |

Additional variables for sediment, Temp12k, boreholes, and precipitation
verification are listed in the path template. Paths ending in `_ARTIFACT`
refer to prepared files or a CFR job directory; they are not raw compilations.

## Climate simulations

CESM-LME files use the member-specific names in
`configs/experiments/cesm_tardif_quarterly_prior.yml`, for example:

```text
simulation-root/
  CESM_LME/remap/tas/*.nc
  CESM_LME/remap/tos/*.nc
  CESM_LME/remap/thetao/*.nc
  CESM_LME/remap/ohc2000/*.nc
  CESM_LME/remap/sic/*.nc
  CESM_LME/remap/sit/*.nc
```

Download simulations through their data providers, such as the
[CESM-LME GDEX archive](https://gdex.ucar.edu/datasets/d651058/) or
[ESGF](https://esgf.llnl.gov/) for MRI-ESM2-0. Preserve each provider's
citation and use conditions. Input patterns describe prepared/remapped
files: acquisition, ocean heat-content integration, and production of
those derived source files are separate prerequisites. This workflow
does not derive OHC directly from raw three-dimensional ocean temperature.

Fields must expose `time`, `lat`, and `lon` coordinates, declared source
variables and units, and CF calendars. The physically aligned CESM
configuration requires monthly CF interval bounds. OHC files without bounds
use explicitly paired SST bounds files; do not infer calendar months from
shifted end-of-interval timestamps. See the manifest and
[preprocessing contract](legacy_preprocessing_eof.md).

The seven-variable CESM prior excludes member 011 and uses 999 complete
years (851–1849), or 3,996 quarterly states, per retained member. Preserve
the configured member selection and segment boundaries. Legacy six-variable
regression configurations instead describe prepared quarterly inputs and
their historical convention; these are separate scientific experiments.

## Proxy calibration and instrumental data

The raw quarterly calibration expects these files under
`LMR4DVAR_TARDIF_OBSERVATION_ROOT`:

| File | Source |
| --- | --- |
| `pages2kv2.json` | [CFR data export](https://github.com/fzhu2e/cfr-data/blob/main/pages2kv2.json) |
| `gistemp1200_GHCNv4_ERSSTv5.nc` | [NASA GISTEMP v4](https://data.giss.nasa.gov/gistemp/) |
| `full_data_v6_precip_10.nc` | [GPCC Full Data v6](https://doi.org/10.5676/DWD_GPCC/FD_M_V6_100) |

Acquire and decompress files outside the repository. Verify versions and
checksums against `analyze/tardif2019_input_manifest.yml`. That manifest
records one acquisition snapshot; live upstream files can change. It
explicitly identifies GISTEMP as a current snapshot rather than a 2019
snapshot and does not establish exact historical PSM numerical parity.

Other supported inputs are calibrated CFR PAGES2k jobs, prepared Temp12k
compilations, borehole profiles with estimated errors, and SedPSM PPE products.
Their required fields and time/variance conventions are documented in
[observation_contract.md](observation_contract.md),
[temp12k_preparation.md](temp12k_preparation.md), and
[borehole_error_estimation.md](borehole_error_estimation.md).
Prepared pickle inputs should come from a trusted producer and use a
compatible CFR/dependency version. Public output contracts prefer structured
NetCDF artifacts.

SedPSM is an optional, separately installed external checkout selected with
`SEDPSM_ROOT`. Its tested revision and setup are in
[collaborator_cesm_ppe.md](collaborator_cesm_ppe.md); its database is selected
separately with `SEDPSM_DATA_PATH`.

## Precipitation verification

`analyze/precipitation_products*.yml` provide the scientific comparison
periods and anomaly baselines. Set the land-mask and product-root variables
in the path template. Existing readers expect the provider file layouts:
GPCP monthly files under the root, ERA5 monthly files in `YEAR/`, ERA5 hourly
files in `YYYYMM/`, and ERA-20C forecast archives in the layout checked by
the reader. Inspect `analyze/verify_precipitation_products.py` when arranging
downloaded subsets; a provider's alternate export format may need an adapter.

Dataset IDs, acquisition links, units, temporal conventions, and independence
limitations are documented in [precipitation_products.md](precipitation_products.md).
The 20CR and LMR comparison scripts can download large public files during
execution. Run these analyses on compute nodes with external storage.

## Before a production run

Run `lmr4dvar validate CONFIG --check-paths`, then `lmr4dvar preflight CONFIG`
and inspect `lmr4dvar plan CONFIG`. Confirm matched files, state order, mode
counts, source units, calendars, member boundaries, anomaly periods, and
reference coverage. Configuration validity alone does not establish
scientific comparability of independently prepared input datasets.
