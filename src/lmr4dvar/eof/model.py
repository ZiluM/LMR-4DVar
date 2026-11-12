"""Deterministic masked, area-weighted EOF model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import yaml


@dataclass(frozen=True)
class EOFModel:
    """EOF basis with standardized principal components."""

    components: xr.DataArray
    weighted_mean: xr.DataArray
    weights: xr.DataArray
    valid_mask: xr.DataArray
    pc_scale: xr.DataArray
    singular_values: xr.DataArray
    explained_variance_ratio: xr.DataArray
    centered: bool
    missing_value_fill: float | None

    @classmethod
    def fit(
        cls,
        field: xr.DataArray,
        n_modes: int,
        *,
        area_weighted: bool = True,
        center: bool = False,
        missing_value_fill: float | None = None,
    ) -> tuple[EOFModel, xr.DataArray]:
        """Fit an EOF basis and return the model and standardized PCs."""
        if missing_value_fill is not None:
            field = field.fillna(missing_value_fill)
        spatial_dims = _spatial_dims(field)
        if n_modes < 1:
            raise ValueError("n_modes must be positive")
        weights = _weights(field, spatial_dims, area_weighted)
        stacked = field.stack(space=spatial_dims).transpose("time", "space")
        stacked_weights = weights.stack(space=spatial_dims)
        valid = stacked.notnull().all("time") & (stacked_weights > np.finfo(float).eps)
        n_valid = int(valid.sum())
        if n_valid == 0:
            raise ValueError("no spatial points are finite for the complete training period")
        if n_modes > min(field.sizes["time"], n_valid):
            raise ValueError("n_modes exceeds the numerical rank dimensions")

        values = stacked.values[:, valid.values] * stacked_weights.values[valid.values]
        mean = values.mean(axis=0) if center else np.zeros(n_valid)
        matrix = values - mean
        u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        components_valid = vt[:n_modes].copy()
        raw_pcs = u[:, :n_modes] * singular_values[:n_modes]
        _orient_modes(components_valid, raw_pcs)

        scales = raw_pcs.std(axis=0, ddof=0)
        if np.any(scales <= np.finfo(float).eps):
            raise ValueError("at least one retained mode has zero PC variance")
        standardized_pcs = raw_pcs / scales

        component_full = np.full((n_modes, stacked.sizes["space"]), np.nan)
        component_full[:, valid.values] = components_valid
        mean_full = np.full(stacked.sizes["space"], np.nan)
        mean_full[valid.values] = mean

        mode = np.arange(n_modes)
        components = xr.DataArray(
            component_full,
            dims=("mode", "space"),
            coords={"mode": mode, "space": stacked.space},
        ).unstack("space")
        weighted_mean = xr.DataArray(
            mean_full, dims="space", coords={"space": stacked.space}
        ).unstack("space")
        valid_mask = valid.unstack("space")
        total_variance = np.square(singular_values).sum()
        variance_ratio = np.square(singular_values[:n_modes]) / total_variance

        model = cls(
            components=components,
            weighted_mean=weighted_mean,
            weights=weights,
            valid_mask=valid_mask,
            pc_scale=xr.DataArray(scales, dims="mode", coords={"mode": mode}),
            singular_values=xr.DataArray(
                singular_values[:n_modes], dims="mode", coords={"mode": mode}
            ),
            explained_variance_ratio=xr.DataArray(
                variance_ratio, dims="mode", coords={"mode": mode}
            ),
            centered=center,
            missing_value_fill=missing_value_fill,
        )
        pcs = xr.DataArray(
            standardized_pcs,
            dims=("time", "mode"),
            coords={"time": field.time, "mode": mode},
            name="pcs",
        )
        return model, pcs

    def encode(self, field: xr.DataArray) -> xr.DataArray:
        """Project fields into standardized PC space."""
        aligned = _align_spatial(field, self.valid_mask)
        if self.missing_value_fill is not None:
            aligned = aligned.fillna(self.missing_value_fill)
        valid = self.valid_mask.stack(space=self.valid_mask.dims).values
        stacked = aligned.stack(space=self.valid_mask.dims).transpose("time", "space")
        values = stacked.values[:, valid]
        if not np.isfinite(values).all():
            raise ValueError("field contains non-finite values inside the EOF valid mask")
        weights = self.weights.stack(space=self.valid_mask.dims).values[valid]
        mean = self.weighted_mean.stack(space=self.valid_mask.dims).values[valid]
        components = self.components.stack(space=self.valid_mask.dims).values[:, valid]
        scores = ((values * weights) - mean) @ components.T
        scores /= self.pc_scale.values
        return xr.DataArray(
            scores,
            dims=("time", "mode"),
            coords={"time": field.time, "mode": self.components.mode},
            name="pcs",
        )

    def decode(self, pcs: xr.DataArray) -> xr.DataArray:
        """Decode standardized PCs to the original physical grid."""
        if "time" not in pcs.dims or "mode" not in pcs.dims:
            raise ValueError("pcs must have time and mode dimensions")
        pcs = pcs.sel(mode=self.components.mode).transpose("time", "mode")
        valid = self.valid_mask.stack(space=self.valid_mask.dims).values
        components = self.components.stack(space=self.valid_mask.dims).values[:, valid]
        weighted = (pcs.values * self.pc_scale.values) @ components
        weighted += self.weighted_mean.stack(space=self.valid_mask.dims).values[valid]
        values = weighted / self.weights.stack(space=self.valid_mask.dims).values[valid]
        full = np.full((pcs.sizes["time"], valid.size), np.nan)
        full[:, valid] = values
        stacked_mask = self.valid_mask.stack(space=self.valid_mask.dims)
        decoded = xr.DataArray(
            full,
            dims=("time", "space"),
            coords={"time": pcs.time, "space": stacked_mask.space},
        ).unstack("space")
        decoded.name = "reconstruction"
        return decoded

    def save(self, directory: str | Path, pcs: xr.DataArray | None = None) -> None:
        """Write a self-describing NetCDF EOF artifact."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dataset = xr.Dataset(
            {
                "components": self.components,
                "weighted_mean": self.weighted_mean,
                "weights": self.weights,
                "valid_mask": self.valid_mask,
                "pc_scale": self.pc_scale,
                "singular_values": self.singular_values,
                "explained_variance_ratio": self.explained_variance_ratio,
            },
            attrs={
                "centered": int(self.centered),
                "weighting": "sqrt(cos(latitude))",
                "sign_convention": "largest-absolute component is positive",
                "missing_value_fill": (
                    "none" if self.missing_value_fill is None else self.missing_value_fill
                ),
            },
        )
        dataset.to_netcdf(directory / "eofs.nc", engine="h5netcdf")
        metadata = {
            "schema_version": 1,
            "n_modes": self.components.sizes["mode"],
            "spatial_dimensions": [dim for dim in self.components.dims if dim != "mode"],
            "centered": self.centered,
            "weighting": "sqrt(cos(latitude))",
            "pc_scaling": "population_standard_deviation",
            "sign_convention": "largest-absolute component is positive",
            "missing_value_fill": self.missing_value_fill,
        }
        (directory / "metadata.yml").write_text(
            yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
        )
        if pcs is not None:
            pcs.to_dataset(name="pcs").to_netcdf(
                directory / "pcs.nc", engine="h5netcdf"
            )

    @classmethod
    def load(cls, directory: str | Path) -> EOFModel:
        """Load an EOF artifact into memory."""
        with xr.open_dataset(
            Path(directory) / "eofs.nc", engine="h5netcdf"
        ) as dataset:
            loaded = dataset.load()
        return cls(
            components=loaded.components,
            weighted_mean=loaded.weighted_mean,
            weights=loaded.weights,
            valid_mask=loaded.valid_mask.astype(bool),
            pc_scale=loaded.pc_scale,
            singular_values=loaded.singular_values,
            explained_variance_ratio=loaded.explained_variance_ratio,
            centered=bool(loaded.attrs["centered"]),
            missing_value_fill=(
                None
                if loaded.attrs.get("missing_value_fill", "none") == "none"
                else float(loaded.attrs["missing_value_fill"])
            ),
        )


def _spatial_dims(field: xr.DataArray) -> tuple[str, ...]:
    if "time" not in field.dims:
        raise ValueError("field must have a time dimension")
    dims = tuple(dim for dim in field.dims if dim != "time")
    if not dims:
        raise ValueError("field must have at least one spatial dimension")
    return dims


def _weights(
    field: xr.DataArray, spatial_dims: tuple[str, ...], area_weighted: bool
) -> xr.DataArray:
    template = field.isel(time=0, drop=True)
    if not area_weighted:
        return xr.ones_like(template, dtype=float)
    if "lat" not in field.coords or "lat" not in spatial_dims:
        raise ValueError("area-weighted EOF requires a lat dimension")
    latitude_weight = np.sqrt(np.clip(np.cos(np.deg2rad(field.lat)), 0.0, None))
    return xr.broadcast(latitude_weight, template)[0].transpose(*spatial_dims)


def _orient_modes(components: np.ndarray, pcs: np.ndarray) -> None:
    for mode in range(components.shape[0]):
        anchor = np.argmax(np.abs(components[mode]))
        if components[mode, anchor] < 0:
            components[mode] *= -1
            pcs[:, mode] *= -1


def _align_spatial(field: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    if "time" not in field.dims:
        raise ValueError("field must have a time dimension")
    expected = tuple(template.dims)
    actual = tuple(dim for dim in field.dims if dim != "time")
    if set(actual) != set(expected):
        raise ValueError(f"spatial dimensions differ: expected {expected}, found {actual}")
    aligned = field.transpose("time", *expected)
    for dim in expected:
        if not aligned[dim].equals(template[dim]):
            raise ValueError(f"coordinate differs for spatial dimension {dim}")
    return aligned
