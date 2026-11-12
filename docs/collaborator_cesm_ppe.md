# SedPSM PPE with shared CESM or MRI priors

This setup runs SedPSM assimilation, reconstruction, and verification while
reusing compatible, previously trained CESM or MRI-ESM2-0 EOF and LIM artifacts.
It does not preprocess the simulation or retrain the prior.

## Code

Clone and install LMR4DVar from `https://github.com/ZiluM/LMR-4DVar.git`. Install the
tested external SedPSM implementation:

```bash
git clone --branch pytorch_age_model https://github.com/ZiluM/SedPSM.git
cd SedPSM
git checkout 9ae40d3
python -m pip install -e . --no-deps
```

The LMR4DVar-specific adapter is included in this repository.

## Paths

Set these in the shell before validation or submission:

```bash
cp configs/paths.example.sh configs/paths.local.sh
# Edit the local file, including simulation, observation, reference, and SedPSM paths.
source configs/paths.local.sh
export LMR4DVAR_PRIOR_ROOT="${LMR4DVAR_OUTPUT_ROOT}/cesm_lme_sedpsm_exp0"
```

`SEDPSM_ROOT` is the code checkout. `SEDPSM_DATA_PATH` is an independent input
and may be anywhere readable. Keep configurations and all generated output in
the collaborator's own directories.

For the MRI-ESM2-0 prior, replace only the prior root and use the MRI template:

```bash
export LMR4DVAR_PRIOR_ROOT="${LMR4DVAR_OUTPUT_ROOT}/mri_esm2_sedpsm_exp0"
```

- CESM: `configs/examples/cesm_sedpsm_ppe_reuse_prior.yml`
- MRI-ESM2-0: `configs/examples/mri_esm2_sedpsm_ppe_reuse_prior.yml`

Both priors use the same 120-dimensional state layout. The MRI SIC and SIT EOFs
were trained with missing values filled by zero, and the MRI template records
that choice explicitly.

The shared iCESM quarterly verification product currently covers 850--1849.
The assimilation and reconstruction can continue through 2000, but the
template computes verification metrics only over the available 850--1849
reference interval.

## Configure and run

Copy the matching model template to a local experiment configuration. Give
each experiment a unique `project.name`, and set
`observations.sedpsm.experiment` to the PPE definition represented by the
sedDB file.

Validate the complete input contract before submitting:

```bash
lmr4dvar validate my_ppe.yml --profile my_ncar.yml --check-paths
lmr4dvar plan my_ppe.yml --profile my_ncar.yml
```

With `prior_run_root` set, the plan starts at the observation preparation
stages. It loads the existing EOF and LIM artifacts in place. It does not read
the CESM training fields; `LMR4DVAR_SIMULATION_ROOT` supplies only the SIC and
SIT climatologies used for SIA and SIV.

Render and explicitly submit a PBS job using a private NCAR profile containing
the collaborator's account and Python interpreter. Because the template sets
`assimilation.device: cuda`, the profile must request a GPU, for example:

```yaml
hpc:
  system: pbs
  queue: casper
  account: REPLACE_WITH_PROJECT
  python_executable: ${LMR4DVAR_PYTHON}
  walltime: "24:00:00"
  memory_gb: 80
  cpus: 8
  gpus: 1
```

Then render and submit:

```bash
lmr4dvar render-pbs my_ppe.yml --profile my_ncar.yml --through verify \
  --output run_ppe.pbs
lmr4dvar submit-pbs run_ppe.pbs
```
