# Copy to configs/paths.local.sh, edit the paths, then source that file.
# Set only the variables required by the configuration you use.
export LMR4DVAR_SIMULATION_ROOT=/path/to/simulations
export LMR4DVAR_OUTPUT_ROOT=/path/to/outputs
export LMR4DVAR_INPUT_ROOT=/path/to/inputs
export LMR4DVAR_OBSERVATION_ROOT=/path/to/observations
export LMR4DVAR_REFERENCE_ROOT=/path/to/reference-fields
export LMR4DVAR_PRIOR_ROOT=/path/to/completed-prior-run
export LMR4DVAR_TARDIF_OBSERVATION_ROOT=/path/to/tardif2019/observations
export LMR4DVAR_CESM_PRECIPITATION_ROOT=/path/to/CESM-LME/PRECT

# Optional proxy families and prepared artifacts.
export SEDPSM_ROOT=/path/to/SedPSM
export SEDPSM_DATA_PATH=/path/to/sediment-database.pkl
export LMR4DVAR_BOREHOLE_ROOT=/path/to/borehole-inputs
export LMR4DVAR_TEMP12K_SOURCE=/path/to/Temp12k1_1_0.pkl
export LMR4DVAR_TEMP12K_ARTIFACT=/path/to/prepared-temp12k.pkl
export LMR4DVAR_PAGES2K_ARTIFACT=/path/to/calibrated-cfr-job

# Optional precipitation verification products, in the layouts in data.md.
export LMR4DVAR_LAND_MASK=/path/to/land_mask.nc
export LMR4DVAR_GPCP_ROOT=/path/to/gpcp-monthly
export LMR4DVAR_ERA5_ROOT=/path/to/era5-monthly
export LMR4DVAR_ERA5_HOURLY_ROOT=/path/to/era5-hourly
export LMR4DVAR_ERA20C_ROOT=/path/to/era20c

# Optional PBS profile. The active environment supplies the interpreter.
export LMR4DVAR_PYTHON="$(command -v python)"
export LMR4DVAR_PBS_ACCOUNT=REPLACE_WITH_PROJECT
export LMR4DVAR_PBS_QUEUE=REPLACE_WITH_QUEUE
