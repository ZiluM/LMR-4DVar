"""Filesystem preflight checks for validated project configuration."""

from __future__ import annotations

from pathlib import Path

from .models import ProjectConfig


def path_errors(config: ProjectConfig) -> list[str]:
    errors: list[str] = []
    prior_root = config.paths.prior_run_root
    if prior_root is not None:
        for relative in (
            "stages/eof/receipt.json",
            "stages/train-lim/receipt.json",
            "stages/eof/state_pcs.nc",
            "stages/train-lim/dynamics.nc",
        ):
            path = prior_root / relative
            if not path.is_file():
                errors.append(f"external prior artifact does not exist: {path}")
    elif not config.paths.simulation_root.is_dir():
        errors.append(f"simulation root does not exist: {config.paths.simulation_root}")
    elif prior_root is None:
        for variable, variable_config in config.state.variables.items():
            pattern = variable_config.input_pattern
            root = variable_config.input_root or config.paths.simulation_root
            if pattern is None:
                errors.append(f"state variable {variable} has no input_pattern")
            elif Path(pattern).is_absolute():
                errors.append(f"state variable {variable} input_pattern must be relative")
            elif not any(root.glob(pattern)):
                errors.append(
                    f"state variable {variable} input_pattern matches no files: {pattern}"
                )
            bounds_pattern = variable_config.monthly_time_bounds_pattern
            if bounds_pattern is not None and (
                Path(bounds_pattern).is_absolute() or not any(root.glob(bounds_pattern))
            ):
                errors.append(f"{variable}: monthly bounds pattern is missing or absolute")

    output_parent = _existing_parent(config.paths.output_root)
    if output_parent is None:
        errors.append(f"no existing parent for output root: {config.paths.output_root}")

    sedpsm = config.observations.sedpsm
    if sedpsm.enabled:
        root = config.paths.sedpsm_root
        if root is None or not root.is_dir():
            errors.append(f"SedPSM root does not exist: {root}")
        elif not sedpsm.resolve_data_path(root).is_file():
            errors.append(f"SedPSM input does not exist: {sedpsm.resolve_data_path(root)}")
        if sedpsm.climatology_path is not None and not sedpsm.climatology_path.is_file():
            errors.append(f"SedPSM climatology does not exist: {sedpsm.climatology_path}")

    for name in ("pages2k", "borehole", "temp12k"):
        source = getattr(config.observations, name)
        generated_pages2k = name == "pages2k" and source.calibration.enabled
        generated_borehole = name == "borehole" and source.error_estimation.enabled
        generated_temp12k = name == "temp12k" and source.preparation.enabled
        generated = generated_pages2k or generated_borehole or generated_temp12k
        if source.enabled and not generated and (source.path is None or not source.path.exists()):
            errors.append(f"{name} input does not exist: {source.path}")
    pages2k_calibration = config.observations.pages2k.calibration
    if pages2k_calibration.enabled:
        matched = pages2k_calibration.matched_records_path
        if matched is not None and not matched.is_file():
            errors.append(f"PAGES2k matched-record artifact missing: {matched}")
        if not pages2k_calibration.instrumental_path.is_file():
            errors.append(
                "PAGES2k instrumental input does not exist: "
                f"{pages2k_calibration.instrumental_path}"
            )
        if pages2k_calibration.method == "tardif2019_quarterly" and (
            pages2k_calibration.precipitation_path is None
            or not pages2k_calibration.precipitation_path.is_file()
        ):
            errors.append(
                f"PAGES2k precipitation input missing: {pages2k_calibration.precipitation_path}"
            )
        if (
            pages2k_calibration.source_path is not None
            and not pages2k_calibration.source_path.exists()
        ):
            errors.append(f"PAGES2k source input does not exist: {pages2k_calibration.source_path}")
    borehole_estimation = config.observations.borehole.error_estimation
    if borehole_estimation.enabled and not borehole_estimation.perturbation_path.is_file():
        errors.append(
            f"borehole perturbation input does not exist: {borehole_estimation.perturbation_path}"
        )
    temp12k_preparation = config.observations.temp12k.preparation
    if temp12k_preparation.enabled:
        if not temp12k_preparation.source_path.is_file():
            errors.append(f"Temp12k source input does not exist: {temp12k_preparation.source_path}")
        pages2k_source = temp12k_preparation.pages2k_source_path
        if pages2k_source is not None and not pages2k_source.exists():
            errors.append(f"Temp12k PAGES2k source does not exist: {pages2k_source}")
    local_stages = config.paths.output_root / config.project.name / "stages"
    for name in (
        "sea_ice_concentration_climatology",
        "sea_ice_thickness_climatology",
    ):
        climatology = getattr(config.reconstruction, name)
        climate_variable = "sic" if name == "sea_ice_concentration_climatology" else "sit"
        generated = (
            prior_root is None
            and climate_variable in config.state.variables
            and config.preprocess.remove_monthly_climatology
            and config.preprocess.input_kind.value == "monthly"
            and climatology is not None
            and climatology.path
            == local_stages / "preprocess/climatology" / f"{climate_variable}.nc"
        )
        if climatology is not None and not climatology.path.is_file() and not generated:
            errors.append(f"reconstruction climatology does not exist ({name}): {climatology.path}")
    for variable, reference in config.verification.reference_fields.items():
        path = reference if isinstance(reference, Path) else reference.path
        generated = (
            variable == "pr"
            and pages2k_calibration.enabled
            and pages2k_calibration.method == "tardif2019_quarterly"
            and path == local_stages / "calibrate-pages2k/reference_pr_quarterly.nc"
        )
        if not path.is_file() and not generated:
            errors.append(f"verification reference field does not exist ({variable}): {path}")
    reference_indices = config.verification.reference_indices
    if isinstance(reference_indices, Path) and not reference_indices.is_file():
        errors.append(f"verification reference indices do not exist: {reference_indices}")
    elif isinstance(reference_indices, dict):
        for name, reference in reference_indices.items():
            if not reference.path.is_file():
                errors.append(
                    f"verification reference index does not exist ({name}): {reference.path}"
                )
    return errors


def _existing_parent(path: Path) -> Path | None:
    candidate = path
    while candidate != candidate.parent:
        if candidate.exists():
            return candidate
        candidate = candidate.parent
    return candidate if candidate.exists() else None
