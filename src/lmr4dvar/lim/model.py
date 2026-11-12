"""Independent linear inverse model fitting and artifact I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from scipy import linalg


@dataclass(frozen=True)
class LIMModel:
    """Discrete LIM dynamics with raw and invertible model-error covariance."""

    propagator: xr.DataArray
    model_error_covariance: xr.DataArray
    raw_model_error_covariance: xr.DataArray
    stationary_covariance: xr.DataArray
    continuous_operator: xr.DataArray | None
    continuous_noise_covariance: xr.DataArray | None
    training_mean: xr.DataArray
    lag_steps: int
    covariance_floor: float
    segment_length_steps: int | None = None

    @classmethod
    def fit(
        cls,
        pcs: xr.DataArray | np.ndarray,
        *,
        lag_steps: int = 1,
        segment_length_steps: int | None = None,
        fit_noise: bool = True,
        center: bool = False,
        require_stable: bool = True,
        covariance_floor_relative: float = 1e-10,
        state_labels: list[str] | None = None,
    ) -> LIMModel:
        """Fit the legacy LIM equations to a sequential state trajectory."""
        values, labels = _training_values(pcs, state_labels)
        if lag_steps < 1 or lag_steps >= values.shape[0]:
            raise ValueError("lag_steps must be positive and smaller than sample count")
        if segment_length_steps is not None:
            if segment_length_steps <= lag_steps:
                raise ValueError("segment_length_steps must exceed lag_steps")
            if values.shape[0] % segment_length_steps:
                raise ValueError("sample count must be divisible by segment_length_steps")
        if not np.isfinite(values).all():
            raise ValueError("LIM training data must be finite")
        mean = values.mean(axis=0) if center else np.zeros(values.shape[1])
        anomalies = values - mean
        starts = np.arange(anomalies.shape[0] - lag_steps)
        if segment_length_steps is not None:
            valid = starts // segment_length_steps == (starts + lag_steps) // segment_length_steps
            starts = starts[valid]
        x0 = anomalies[starts]
        x1 = anomalies[starts + lag_steps]
        if x0.shape[0] < 2:
            raise ValueError("LIM fitting requires at least two lagged sample pairs")

        c0_sum = x0.T @ x0
        propagator = (x1.T @ x0) @ np.linalg.pinv(c0_sum)
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(propagator))))
        if require_stable and spectral_radius >= 1.0:
            raise ValueError(f"fitted LIM is not damped: spectral radius={spectral_radius:.6g}")

        c0 = c0_sum / (x0.shape[0] - 1)
        raw_qd = _symmetrize(c0 - propagator @ c0 @ propagator.T)
        qd, floor = _positive_definite(raw_qd, covariance_floor_relative)

        continuous_operator = None
        continuous_noise = None
        if fit_noise:
            continuous_operator = _real_matrix_log(propagator) / lag_steps
            continuous_noise = _symmetrize(-(continuous_operator @ c0 + c0 @ continuous_operator.T))

        state = xr.IndexVariable("state", labels)
        state_out = xr.IndexVariable("state_out", labels)
        state_in = xr.IndexVariable("state_in", labels)

        def matrix(values: np.ndarray, name: str) -> xr.DataArray:
            return xr.DataArray(
                values,
                dims=("state_out", "state_in"),
                coords={"state_out": state_out, "state_in": state_in},
                name=name,
            )

        return cls(
            propagator=matrix(propagator, "G"),
            model_error_covariance=matrix(qd, "Qd"),
            raw_model_error_covariance=matrix(raw_qd, "Qd_raw"),
            stationary_covariance=matrix(c0, "C0"),
            continuous_operator=(
                matrix(continuous_operator, "L") if continuous_operator is not None else None
            ),
            continuous_noise_covariance=(
                matrix(continuous_noise, "Q") if continuous_noise is not None else None
            ),
            training_mean=xr.DataArray(mean, dims="state", coords={"state": state}, name="mean"),
            lag_steps=lag_steps,
            covariance_floor=floor,
            segment_length_steps=segment_length_steps,
        )

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.propagator.values))))

    @property
    def decay_times_steps(self) -> np.ndarray:
        """Return modal e-folding times in model steps; invalid modes are infinite."""
        magnitude = np.abs(np.linalg.eigvals(self.propagator.values))
        times = np.full(magnitude.shape, np.inf, dtype=float)
        damped = (magnitude > 0) & (magnitude < 1)
        times[damped] = -self.lag_steps / np.log(magnitude[damped])
        return times

    def forecast(
        self, initial: xr.DataArray | np.ndarray, lead_steps: int
    ) -> xr.DataArray | np.ndarray:
        """Forecast one or more row-oriented states by an integer lead."""
        if lead_steps < 0:
            raise ValueError("lead_steps must be non-negative")
        is_xarray = isinstance(initial, xr.DataArray)
        values = np.asarray(initial.values if is_xarray else initial, dtype=float)
        if values.shape[-1] != self.propagator.sizes["state_in"]:
            raise ValueError("initial state dimension does not match LIM state")
        if not np.isfinite(values).all():
            raise ValueError("initial state must be finite")
        transition = np.linalg.matrix_power(self.propagator.values, lead_steps)
        mean = self.training_mean.values
        result = (values - mean) @ transition.T + mean
        if not is_xarray:
            return result
        return xr.DataArray(
            result,
            dims=initial.dims,
            coords=initial.coords,
            attrs=initial.attrs,
            name="forecast",
        )

    def forecast_skill(self, pcs: xr.DataArray | np.ndarray, max_lead: int) -> xr.Dataset:
        """Calculate per-state correlation and RMSE for sequential forecasts."""
        values, labels = _training_values(pcs, list(self.training_mean.state.values))
        if max_lead < 1 or max_lead >= values.shape[0]:
            raise ValueError("max_lead must be positive and smaller than sample count")
        if self.segment_length_steps is not None and max_lead >= self.segment_length_steps:
            raise ValueError("forecast lead must be shorter than each member segment")
        correlations = np.empty((max_lead, values.shape[1]))
        rmse = np.empty_like(correlations)
        pair_counts = []
        for lead in range(1, max_lead + 1):
            starts = np.arange(values.shape[0] - lead)
            if self.segment_length_steps is not None:
                segment = self.segment_length_steps
                starts = starts[starts // segment == (starts + lead) // segment]
            predicted = self.forecast(values[starts], lead)
            target = values[starts + lead]
            pair_counts.append(starts.size)
            rmse[lead - 1] = np.sqrt(np.mean(np.square(predicted - target), axis=0))
            correlations[lead - 1] = [
                _correlation(predicted[:, index], target[:, index])
                for index in range(values.shape[1])
            ]
        return xr.Dataset(
            {
                "correlation": (("lead", "state"), correlations),
                "rmse": (("lead", "state"), rmse),
                "pair_count": ("lead", pair_counts),
            },
            coords={"lead": np.arange(1, max_lead + 1), "state": labels},
            attrs={"member_boundary_policy": "exclude cross-segment forecast pairs"},
        )

    def save(self, directory: str | Path, diagnostics: xr.Dataset | None = None) -> None:
        """Write the LIM model as self-describing NetCDF and YAML artifacts."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        variables = {
            "G": self.propagator,
            "Qd": self.model_error_covariance,
            "Qd_raw": self.raw_model_error_covariance,
            "C0": self.stationary_covariance,
            "training_mean": self.training_mean,
        }
        if self.continuous_operator is not None:
            variables["L"] = self.continuous_operator
        if self.continuous_noise_covariance is not None:
            variables["Q"] = self.continuous_noise_covariance
        xr.Dataset(variables).to_netcdf(directory / "dynamics.nc", engine="h5netcdf")
        metadata = {
            "schema_version": 1,
            "lag_steps": self.lag_steps,
            "segment_length_steps": self.segment_length_steps,
            "state_labels": [str(value) for value in self.training_mean.state.values],
            "spectral_radius": self.spectral_radius,
            "stable": self.spectral_radius < 1.0,
            "covariance_floor": self.covariance_floor,
            "raw_qd_min_eigenvalue": float(
                np.linalg.eigvalsh(self.raw_model_error_covariance.values).min()
            ),
            "qd_min_eigenvalue": float(
                np.linalg.eigvalsh(self.model_error_covariance.values).min()
            ),
        }
        (directory / "metadata.yml").write_text(
            yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
        )
        if diagnostics is not None:
            diagnostics.to_netcdf(directory / "training_metrics.nc", engine="h5netcdf")

    @classmethod
    def load(cls, directory: str | Path) -> LIMModel:
        """Load a LIM artifact into memory."""
        directory = Path(directory)
        with xr.open_dataset(directory / "dynamics.nc", engine="h5netcdf") as dataset:
            loaded = dataset.load()
        metadata = yaml.safe_load((directory / "metadata.yml").read_text(encoding="utf-8"))
        return cls(
            propagator=loaded.G,
            model_error_covariance=loaded.Qd,
            raw_model_error_covariance=loaded.Qd_raw,
            stationary_covariance=loaded.C0,
            continuous_operator=loaded.L if "L" in loaded else None,
            continuous_noise_covariance=loaded.Q if "Q" in loaded else None,
            training_mean=loaded.training_mean,
            lag_steps=int(metadata["lag_steps"]),
            covariance_floor=float(metadata["covariance_floor"]),
            segment_length_steps=metadata.get("segment_length_steps"),
        )


def _training_values(
    pcs: xr.DataArray | np.ndarray, state_labels: list[str] | None
) -> tuple[np.ndarray, list[str]]:
    if isinstance(pcs, xr.DataArray):
        if pcs.ndim != 2 or "time" not in pcs.dims or "state" not in pcs.dims:
            raise ValueError("PC DataArray must have time and state dimensions")
        ordered = pcs.transpose("time", "state")
        values = np.asarray(ordered.values, dtype=float)
        labels = [str(value) for value in ordered.state.values]
        if state_labels is not None and labels != [str(value) for value in state_labels]:
            raise ValueError("state_labels disagree with PC state coordinates")
    else:
        values = np.asarray(pcs, dtype=float)
        if values.ndim != 2:
            raise ValueError("LIM training data must be two-dimensional")
        labels = state_labels or [f"state_{index:03d}" for index in range(values.shape[1])]
    if len(labels) != values.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("state labels must be unique and match the state dimension")
    return values, labels


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    real = np.asarray(matrix).real
    return 0.5 * (real + real.T)


def _positive_definite(covariance: np.ndarray, floor_relative: float) -> tuple[np.ndarray, float]:
    if not 0 < floor_relative < 1:
        raise ValueError("covariance_floor_relative must be between zero and one")
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(covariance))
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = floor_relative * scale
    corrected = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return _symmetrize(corrected), floor


def _real_matrix_log(matrix: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    result = linalg.logm(matrix)
    imaginary = float(np.max(np.abs(result.imag))) if np.iscomplexobj(result) else 0.0
    scale = max(float(np.max(np.abs(result.real))), 1.0)
    if imaginary > tolerance * scale:
        raise ValueError(
            f"continuous LIM operator has a non-negligible imaginary part: {imaginary:.3e}"
        )
    return np.asarray(result.real)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])
