# LMR4D-Var

LMR4D-Var reconstructs climate trajectories by combining area-weighted EOFs,
a coupled Linear Inverse Model (LIM) prior, proxy observation operators, and
deterministic weak-constraint four-dimensional variational data assimilation.
The Python package and command are named `lmr4dvar`.

Companion paper: Meng et al. (2026),
[Coupled multiscale paleoclimate reconstruction with four-dimensional variational data assimilation](https://arxiv.org/abs/2608.19469).
The linked article is an arXiv preprint. Citation metadata are in
[CITATION.cff](CITATION.cff).

## Install

Use Python 3.10 and the supplied conda environment:

```bash
git clone https://github.com/ZiluM/LMR-4DVar.git
cd LMR-4DVar
conda env create -f environment.yml
conda activate lmr4dvar
python -m pip install -e . --no-deps
lmr4dvar --help
```

The default environment runs on a CPU; CUDA is optional. The validated
numerical versions and CUDA installation are documented in
[reproducibility.md](docs/reproducibility.md).
For a pip-managed environment, use `python -m pip install -e ".[dev]"`.
Precipitation GRIB readers additionally require `".[grib]"`.

SedPSM is needed only for sediment proxy experiments. Install its separately
versioned checkout as described in [the SedPSM guide](docs/collaborator_cesm_ppe.md)
and set `SEDPSM_ROOT` and `SEDPSM_DATA_PATH`. Other proxy workflows do not
require SedPSM.

## Try a small complete example

This deterministic CPU example creates two tiny synthetic climate fields and
a noisy synthetic proxy, then runs preprocessing, EOF/LIM fitting, assimilation,
reconstruction, and verification:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python examples/synthetic_workflow.py --output runs/synthetic
```

No external data or scheduler are required. Inputs, a reusable YAML
configuration, stage receipts, and results are written under `runs/synthetic/`.
The synthetic proxy has known coefficients, so the optional calibration and
other proxy preparation stages write skipped-stage manifests.
The example demonstrates software operation; its scores are not evidence of
real-data reconstruction skill.

## Configure a real experiment

Copy and edit the path template, then source it:

```bash
cp configs/paths.example.sh configs/paths.local.sh
# Edit configs/paths.local.sh for your local data and output locations.
source configs/paths.local.sh
```

Configurations use environment variables and fail if a required variable is
unresolved. They do not download or redistribute simulations or proxy databases.
See [data.md](docs/data.md) for input contracts, acquisition references,
directory layouts, and preprocessing requirements.

| Configuration directory | Purpose |
| --- | --- |
| `configs/examples/` | SedPSM examples and reuse of compatible EOF/LIM priors |
| `configs/experiments/` | CESM/MRI scientific experiment settings |
| `configs/regression/` | Compatibility checks using prepared quarterly inputs |
| `configs/profiles/` | Optional scheduler and device overlays |

The physically aligned seven-variable CESM example starts with
`configs/experiments/cesm_tardif_quarterly_prior.yml`. For example:

```bash
lmr4dvar validate configs/experiments/cesm_tardif_quarterly_prior.yml --check-paths
lmr4dvar preflight configs/experiments/cesm_tardif_quarterly_prior.yml
lmr4dvar plan configs/experiments/cesm_tardif_quarterly_prior.yml
```

`validate`, `preflight`, and `plan` do not execute scientific stages.
Before running large data, inspect input dimensions, memory requirements,
time bounds, units, and the configured anomaly and training periods.
The older prepared-quarterly regression templates retain their documented
seasonal convention and must not substitute for a physically aligned prior
in the additive temperature/precipitation PSM workflow.

## Run and resume

```bash
lmr4dvar stages
lmr4dvar run project.yml --through train-lim
lmr4dvar run project.yml --through verify
lmr4dvar run project.yml --stage assimilate
```

The nine stages are `preprocess`, `eof`, `train-lim`, `calibrate-pages2k`,
`prepare-temp12k`, `estimate-borehole-error`, `assimilate`, `reconstruct`,
and `verify`. Valid receipts allow completed stages to be reused;
`--force` explicitly reruns selected stages. A configured
`paths.prior_run_root` reuses compatible, completed EOF/LIM artifacts.

Outputs live under `output_root/project.name/stages/`: prepared fields and
climatologies, EOF components and PCs, LIM matrices, prepared observations,
analysis trajectories, reconstructed fields and indices, and verification
metrics. Stage receipts record configuration and dependency fingerprints.
See [workflow_receipts.md](docs/workflow_receipts.md) for the artifact contract.

For PBS, copy a site profile and supply your own account, queue, and interpreter.
A CUDA experiment needs a GPU profile such as
`configs/profiles/pbs-gpu.example.yml`. CPU runs can overlay
`assimilation.device: cpu` in their local profile.

```bash
lmr4dvar render-pbs project.yml --profile configs/profiles/local.yml \
  --through verify --output runs/run.pbs
# Review the rendered job and resource requests before submitting it yourself.
lmr4dvar submit-pbs runs/run.pbs
```

Rendering does not submit a job. Large preprocessing, EOF fitting, training,
and reconstruction runs belong on compute nodes.

## Documentation

- [Workflow diagram](docs/lmr4d_workflow.md) and [architecture](docs/architecture.md)
- [Weak-constraint objective and optimization](docs/weak_constraint_4dvar.md)
- [Observation contracts](docs/observation_contract.md)
- [PAGES2k calibration](docs/pages2k_calibration.md),
  [Temp12k preparation](docs/temp12k_preparation.md), and
  [borehole errors](docs/borehole_error_estimation.md)
- [Reconstruction and verification](docs/reconstruction_verification.md)
- [Precipitation reference comparisons](docs/precipitation_products.md)
- [Data and dependencies](docs/data.md) and
  [reproducibility and validation](docs/reproducibility.md)

## Development

```bash
python -m pip install -e ".[dev]"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for scientific regression requirements.
Report reproducible issues through
[GitHub Issues](https://github.com/ZiluM/LMR-4DVar/issues).

## License and citation

Code is available under the [MIT License](LICENSE). External libraries,
SedPSM, simulations, proxy compilations, and reference products retain their
own licenses and citation requirements.

Please cite the companion paper and record the software version or commit,
configuration, data versions, and preprocessing choices used in your work.
The public repository provides the workflow and scientific configurations;
exact paper results also require the corresponding archived inputs and run
artifacts, as explained in [reproducibility.md](docs/reproducibility.md).
