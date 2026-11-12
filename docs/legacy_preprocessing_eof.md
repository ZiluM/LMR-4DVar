# Legacy Preprocessing and EOF Reference

This document records the behavior that the standalone preprocessing and EOF
implementations must reproduce or intentionally replace. It is based on the
the historical MRI preprocessing scripts and the reference EOF implementation.
Their relevant scientific behavior is specified below; those external source
checkouts are not runtime dependencies.

## Preprocessing Definition

The legacy chain applies the following operations:

1. CDO bilinear remapping to `r180x90`. The resulting cell centers are longitude
   `0, 2, ..., 358` and latitude `-89, -87, ..., 89`.
2. A full-record monthly climatology (`cdo ymonmean`) is removed from each
   monthly field.
3. `xarray.resample(time="QS-DEC").mean()` forms DJF-aligned quarters.
4. The MRI script replaces the resample labels with every third original time
   plus the final original time. For a January-to-December input this retains
   incomplete January-February and December edge averages. This behavior is
   available as `quarterly_mode: legacy_qs_dec`.
5. `sacpy.get_anom(method=1)` linearly detrends each calendar-month group
   separately. A spatial point with any missing value in one group is missing
   for the complete group.
6. `sic` and `sit` retain the Northern Hemisphere; other state fields are
   global.

The standalone default is `quarterly_mode: complete`, which drops incomplete
edge quarters and labels complete means by their center month. Legacy
regression configs must request `legacy_qs_dec` explicitly.

The Python `linear` interpolation backend is scientifically equivalent in
purpose but is not assumed bitwise identical to CDO `remapbil`. Production
parity tests compare resulting fields with an explicit tolerance.

## Legacy EOF Definition

The legacy EOF class:

- multiplies the physical field by `sqrt(cos(latitude))`;
- retains only spatial points finite at every training time;
- solves the covariance eigenproblem and obtains spatial vectors;
- scales every PC by its population standard deviation (`ddof=0`);
- decodes by restoring the PC scale and dividing by the latitude weight.

`lmr4dvar.eof.EOFModel` preserves those mathematical contracts but uses direct,
deterministic SVD. Modes are ordered by decreasing singular value. Each mode is
oriented so its largest-absolute component is positive. Encoding rejects new
missing values inside the training mask instead of silently replacing them by
zero by default. A variable can set `eof_fillna_value` explicitly when missing
values have a defined physical value. The MRI regression sets it to zero for
`sic` and `sit`, reproducing the legacy sea-ice EOF treatment during both
fitting and later projection.

EOF signs and rotations within a numerically degenerate eigenspace are not
scientifically identifiable. Regression therefore compares reconstructed
fields, explained variance, and weighted subspaces rather than raw component
signs alone.

## Production Regression

The MRI `tas.pkl` legacy EOF artifact is approximately 1.68 GB. Loading it and
fitting production EOFs are batch-compute tasks and must not run on a login
node. A PBS regression should report:

- maximum and RMS difference of preprocessed fields;
- retained-subspace projection error;
- explained-variance difference;
- PC correlations after mode matching and sign alignment;
- reconstruction RMS error in physical units.

The current unit suite validates these formulas on compact synthetic fields.
Full-scale CESM and MRI agreement requires separate production regression
evidence with documented tolerances; the small-data suite alone does not
establish that agreement.
