# Real-data regression preflight

The first executable regression configurations are:

- `configs/regression/cesm_lme_sedpsm_exp0.yml`
- `configs/regression/mri_esm2_sedpsm_exp0.yml`

They use local simulation products but run EOF fitting, LIM fitting, 4D-Var,
reconstruction, and verification inside this repository.

## Required roots

```bash
cp configs/paths.example.sh configs/paths.local.sh
# Edit the local file to point to your simulations, observations, and outputs.
source configs/paths.local.sh
cp configs/profiles/ncar.example.yml configs/profiles/local.yml
```

Validate without executing a scientific stage:

```bash
lmr4dvar validate configs/regression/cesm_lme_sedpsm_exp0.yml \
  --profile configs/profiles/local.yml --check-paths
lmr4dvar validate configs/regression/mri_esm2_sedpsm_exp0.yml \
  --profile configs/profiles/local.yml --check-paths
```

Inspect every matched NetCDF variable, selected sample count, and grid without
loading field values or writing workflow outputs:

```bash
lmr4dvar preflight configs/regression/cesm_lme_sedpsm_exp0.yml \
  --profile configs/profiles/local.yml
```

## Compatibility decisions

The state order is exactly the legacy LIM order:
`tas, ohc300, tos, sic, sit, ohc2000`. The retained-mode counts are
`15, 30, 15, 15, 15, 30`, for a 120-dimensional state.

CESM-LME inputs contain 13 forced members. The new loader processes members in
sorted filename order, checks equal grids and sample counts, and concatenates
their samples in member blocks. The explicit 850-1849 training period trims
CESM `ohc2000`, whose local product extends through 2005, to the same 4,000
quarters per member as the other variables.

MRI seasonal products contain a final December 1849 partial quarter. Selecting
only January, April, July, and October removes that edge, yielding 4,000 common
quarterly samples. This deliberately corrects the legacy MRI EOF input length
of 4,001 samples.

The inputs are marked `prepared_quarterly`: they have already undergone local
OHC calculation, 2-degree regridding, monthly anomaly removal, quarterly
averaging, and seasonal detrending. The repository therefore starts scientific
transformation at EOF fitting for these regression runs. Raw monthly inputs
remain supported by the default `monthly` preprocessing mode.

SedPSM uses the same Berkeley Earth `climatology` field as the legacy FDVar
configuration. Falling back to the prior simulation's monthly climatology is
supported, but is not scientifically equivalent to these regressions.

Verification uses the iCESM TAS field and GMT reference products already used
by the legacy experiment. No expensive stage or PBS job is launched by this
preflight.
