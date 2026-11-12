# Contributing

Install the environment in `environment.yml`, then install the package in
editable mode. Keep importable code in `src/lmr4dvar/` and tests in `tests/`.
Use explicit configuration and declared dependencies for local inputs.

Before changing scientific behavior, trace the implementation, callers,
downstream artifacts, units, masks, calendars, and provenance. Capture
regression evidence before replacing EOF, LIM, observation, or reconstruction
behavior. EOF sign equivalence requires reconstructed-field or subspace
agreement. Empty observation matches must fail visibly.

Document anomaly baselines, time matching, weighting, smoothing, and any
intentional change in numerical behavior. Keep scheduler resources in local
profiles. Run large workloads on compute nodes; use small synthetic tests
for local validation.

Run focused scientific tests, the full small-data suite when appropriate,
`python -m ruff check .`, and `git diff --check` before submitting a pull
request. Explain the problem, resulting behavior, and validation performed.

Keep simulations, proxy databases, generated NetCDF/Zarr files, experiment
outputs, logs, credentials, and local site profiles outside version control.
Third-party data and dependencies retain their own distribution terms.
