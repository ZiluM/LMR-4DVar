# Weak-Constraint 4D-Var

The control has shape `(time, state)` and is the complete state trajectory:

```text
v = (x0, x1, ..., xK-1)
n(k) = x(k+1) - mean - G @ (x(k) - mean)
```

This state form is a one-to-one reparameterization of the forcing form. The
initial state and inferred increments produce the same weak-constraint cost,
but all increments are evaluated as one batched matrix operation. No serial
LIM propagation is required inside an optimizer closure. The affine form
matters when a LIM was trained with centering. The prior mode is the training
mean at every step, which implies zero model-error increments. For the legacy
uncentered LIM, `training_mean` is zero.

The two parameterizations have the same objective values and minimizer under
their one-to-one mapping. Their coordinate gradients and L-BFGS iteration paths
need not be identical, so convergence diagnostics should be compared by cost
terms and reconstructed states rather than iteration number alone.

The objective is:

```text
J = initial_weight * 0.5 * (x0-mean)' C0^-1 (x0-mean)
  + model_error_weight * 0.5 * sum(n(k)' Qd^-1 n(k))
  + 0.5 * sum((H(x)-y)^2 / R)
```

`initial_weight` and `model_error_weight` are separate configuration values.
The manuscript's model sensitivity parameter `Jm` maps to
`model_error_weight` only. The old implementation's `jb_weight` multiplied
both prior terms and should only be reproduced intentionally by setting both
new weights to that value.

Covariance inverses are never formed. The objective factors `C0` and corrected
`Qd` with Cholesky decomposition and computes whitened quadratic norms using
triangular solves. A non-positive-definite covariance is rejected at objective
construction rather than modified during assimilation; covariance correction
belongs to LIM training and is recorded in the LIM artifact.

Production precision is explicit in `assimilation.precision` and defaults to
`float32`. Both the analysis NetCDF and diagnostics JSON record the selected
precision. `float64` remains available for numerical regression and sensitivity
tests.

All family-specific operators are bound to their observation collections and
their predictions are concatenated before computing `Jo`. Consequently, an
empty match across the complete network raises `EmptyObservationError`, while
the returned `ObservationLoss` retains family-level cost and used/dropped
counts.

`assimilate_lbfgs` starts at the prior-mode trajectory unless an explicit state
trajectory is provided. It uses deterministic strong-Wolfe L-BFGS and records
total, initial-state, model-error, observation, and gradient-norm histories at
every closure evaluation. For output compatibility, the result preserves both
`control` and `trajectory`; under this parameterization they contain the same
optimized states.
