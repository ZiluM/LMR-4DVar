# Legacy LIM Reference

The production regression model was built by
`~/SI_recon/LIM/main_lim.py` with `type: LIM`. It concatenates the retained,
standardized PCs in YAML variable order and calls `slim.LIM(pcs,
fit_noise=True)` with the default one-sample lag. Because the inputs are
quarterly PCs, one model step is one quarter.

For lagged pairs `X0 = X[:-1]` and `X1 = X[1:]`, the legacy discrete equations
are:

```text
G      = (X1.T @ X0) @ pinv(X0.T @ X0)
C0     = (X0.T @ X0) / (number_of_pairs - 1)
Qd_raw = C0 - G @ C0 @ G.T
```

The legacy code does not subtract a new training mean because each PC was
already standardized from an anomaly field. Corresponding regression configs
must therefore use `center_training: false`.

The 4D-Var code consumes `G`, `Q_d`, and `C0`. It uses `C0` for initial-state
uncertainty and `Q_d` for every subsequent weak-constraint model increment.
The continuous operator `L = log(G)` and continuous noise covariance
`Q = -(L C0 + C0 L.T)` are not used by the current assimilation objective.

`LIMModel` saves both `Qd_raw`, which preserves the legacy equation for
regression, and a symmetric positive-definite `Qd`, whose eigenvalues are
floored before the DA inverse is taken. Artifact metadata reports the floor and
the minimum eigenvalue before and after correction. This makes a numerical
repair explicit instead of applying it inside the assimilation constructor.

Production parity must compare `G`, `C0`, and `Qd_raw` directly, then compare
one-to-four-quarter forecasts and decoded field skill. Full CESM and MRI
training remains a PBS task.

Concatenated ensemble members must set `lim.segment_length_steps` to one
member's sample count. Lagged pairs crossing a member boundary are then
excluded from `G`, `C0`, and `Qd`. CESM-LME has 4,000 quarterly samples per
member. Its weak-constraint assimilation uses the discrete model directly, so
`fit_noise: false` intentionally omits the complex-valued continuous `log(G)`
diagnostics.
