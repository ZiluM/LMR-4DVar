# Reproducibility and validation

## Scientific scope

This public distribution implements the workflow described by Meng et al.
(2026), [arXiv:2608.19469](https://arxiv.org/abs/2608.19469), and includes
current CESM/MRI configurations and additional precipitation comparisons.
The arXiv record is a preprint; no journal publication information is
asserted here. `CITATION.cff` supplies the paper's public bibliographic record.

The distribution is not an archive of every paper run. Exact numerical
reproduction also requires the corresponding input snapshots, derived
climate fields, proxy preparation, external dependency revisions, resolved
configuration, and numerical environment. Those data and run artifacts are
not bundled. Current experiment templates and later precipitation results
must not be assumed to identify the paper's exact production configuration.

## Numerical behavior

Release preparation preserves the existing numerical implementations:
monthly/quarterly time handling, area-weighted EOFs, normalized PCs, LIM
covariance handling, proxy operators, direct state-trajectory optimization,
field reconstruction, and verification metrics. Configuration path
expansion is shared with analysis scripts. Scheduler resources are provided
through site profiles.

Preserve variable order, masks, units, calendars, training-member boundaries,
anomaly references, proxy seasonality, smoothing, and observation variances
when comparing runs. EOF sign differences alone do not imply changed
reconstructed fields. The legacy behavior contracts are in
`legacy_preprocessing_eof.md`, `legacy_lim.md`, and `observation_contract.md`.

## Environment

The small-data validation environment uses:

| Component | Version |
| --- | --- |
| Python | 3.10.21 |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| xarray | 2025.6.1 |
| SciPy | 1.13.1 |
| PyTorch | 2.5.1+cu121, CPU execution for tests |
| CFR | 2024.4.4 |
| Pydantic | 2.13.4 |
| h5netcdf / netCDF4 | 1.8.1 / 1.7.3 |

`environment.yml` pins the principal numerical versions and uses portable
PyTorch 2.5.1 without requiring a particular CUDA wheel. Its complete
cross-platform conda solve has not been independently validated. For a
compatible Linux CUDA 12.1 installation matching the validation environment:

```bash
python -m pip install 'torch==2.5.1+cu121' \
  --index-url https://download.pytorch.org/whl/cu121
```

GPU production runs additionally need compatible drivers, a GPU scheduler
request, and `assimilation.device: cuda`. Floating-point behavior can differ
with devices, BLAS libraries, and dependency versions; record those choices.
For a completed research run, archive `conda env export` or `pip freeze`
alongside the stage receipts after reviewing machine-specific paths.

## Validation

The source baseline passed all 195 small-data tests before release
preparation. The public checkout passed all 218 tests, including portable
configuration loading and the complete synthetic workflow. Its editable
installation, standalone wheel build, lint, shell syntax, and local
documentation links were also checked. A separate invocation of
`examples/synthetic_workflow.py` completed all nine stages through verification,
using the actual observation operator and optimizer and assimilating all 20
synthetic observations. These checks performed no network downloads of data
or scheduler submissions.

Existing environment warnings concern the local PROJ database, a NumPy
extension binary-size warning, and deprecated xarray `use_cftime` usage.
They do not fail the baseline suite. Production-scale climate runs, GPU
assimilation, and exact comparisons with archived paper outputs are outside
this small-data release validation.

## Record and cite a run

Record the software version or commit, input URLs/DOIs and checksums,
dependency versions (including SedPSM where enabled), resolved YAML,
anomaly/calibration/training periods, time and weighting policies, and
receipt/diagnostic files. Preserve optimization convergence diagnostics and
observation counts. Cite the paper and all external datasets and methods
used in the experiment. Update the paper's citation metadata when a journal
reference becomes available.
