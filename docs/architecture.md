# Architecture

## Boundaries

LMR4DVar separates scientific stages through self-describing artifacts instead
of Python-object pickle files. A stage consumes validated configuration and a
fingerprinted artifact from its predecessor. It must not discover inputs by
searching another source checkout.

## Components

- `config`: typed project configuration, profile overlays, and preflight checks
- `data`: local simulation catalog and preprocessing
- `eof`: weighted EOF fit, encode, decode, and artifact I/O
- `lim`: stochastic linear dynamics fit, forecast, diagnostics, and artifact I/O
- `observations`: common records and PAGES2k/SedPSM/borehole/Temp12k operators
- `assimilation`: weak-constraint control transform, objective, and optimization
- `reconstruction`: EOF decoding and climate indices
- `verification`: time-series and field metrics
- `workflow`: stage ordering, manifests, resume, and provenance
- `hpc`: site-independent PBS rendering

## Dependency Direction

Lower-level numerical modules do not import workflow or CLI code. Observation
operators may depend on state metadata but not on experiment-directory layouts.
The CLI delegates to workflow stages and contains no scientific calculations.

## Compatibility

Legacy paths and file names are handled by explicit adapters. New internals do
not reproduce the old implicit assumptions, such as quarter ordering inferred
from array position or a model class inferred from a pickle import path.

The exact preprocessing and EOF assumptions discovered in the legacy workflow
are recorded in [legacy_preprocessing_eof.md](legacy_preprocessing_eof.md).
The legacy discrete LIM equations and covariance handling are recorded in
[legacy_lim.md](legacy_lim.md).
The common proxy-record, time-matching, and loss rules are recorded in
[observation_contract.md](observation_contract.md).
The PAGES2k instrumental calibration and EOF-projection workflow is recorded in
[pages2k_calibration.md](pages2k_calibration.md).
The borehole inverse/forward ensemble and effective error construction is
recorded in [borehole_error_estimation.md](borehole_error_estimation.md).
