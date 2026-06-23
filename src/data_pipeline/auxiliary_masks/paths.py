from __future__ import annotations

from pathlib import Path

from data_pipeline.pipeline_orchestrator.orchestration import (
    PATH_MODE_MERGED,
    PATH_MODE_PER_WELL,
    artifact_path,
    step_dir,
    validated_path,
)


AUXILIARY_MASKS_ROOT_NAME = "auxiliary_masks"
AUXILIARY_MASK_FAMILIES = ("via", "yolk", "focus", "bubble")


def auxiliary_masks_root(data_root: Path, experiment_id: str) -> Path:
    return step_dir(
        data_root,
        AUXILIARY_MASKS_ROOT_NAME,
        str(experiment_id),
        path_mode=PATH_MODE_MERGED,
    )


def auxiliary_mask_subdir(data_root: Path, experiment_id: str, family: str) -> Path:
    if family not in AUXILIARY_MASK_FAMILIES:
        raise KeyError(f"Unknown auxiliary mask family: {family}")
    return auxiliary_masks_root(data_root, experiment_id) / family


def auxiliary_mask_well_root(data_root: Path, experiment_id: str, well_id: str) -> Path:
    return step_dir(
        data_root,
        AUXILIARY_MASKS_ROOT_NAME,
        str(experiment_id),
        path_mode=PATH_MODE_PER_WELL,
        well_id=str(well_id),
    )


def auxiliary_mask_path(
    data_root: Path,
    experiment_id: str,
    family: str,
    image_id: str,
) -> Path:
    return auxiliary_mask_subdir(data_root, experiment_id, family) / f"{image_id}_{family}.png"


def auxiliary_mask_manifest_path(data_root: Path, experiment_id: str) -> Path:
    return artifact_path(
        data_root,
        AUXILIARY_MASKS_ROOT_NAME,
        "manifest",
        str(experiment_id),
        path_mode=PATH_MODE_MERGED,
    )


def auxiliary_mask_sentinel_path(data_root: Path, experiment_id: str) -> Path:
    return validated_path(
        data_root,
        AUXILIARY_MASKS_ROOT_NAME,
        "manifest",
        str(experiment_id),
        path_mode=PATH_MODE_MERGED,
    )
