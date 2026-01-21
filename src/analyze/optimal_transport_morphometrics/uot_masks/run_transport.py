"""Run unbalanced OT on a single mask pair."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.analyze.utils.optimal_transport import (
    UOTConfig,
    UOTFramePair,
    UOTProblem,
    UOTResult,
    mask_to_density,
    enforce_min_mass,
    mask_to_density_uniform,
    downsample_density,
    build_support,
    compute_transport_maps,
    summarize_metrics,
    POTBackend,
    UOTBackend,
)
from .preprocess import preprocess_pair, preprocess_pair_canonical
from .frame_mask_io import load_mask_pair_from_csv
from .uot_grid import CanonicalGridConfig, rescale_velocity_to_um, rescale_distance_to_um


def build_problem(
    mask_src: np.ndarray,
    mask_tgt: np.ndarray,
    config: UOTConfig,
) -> Tuple[UOTProblem, dict]:
    src_proc, tgt_proc, preprocess_meta = preprocess_pair(mask_src, mask_tgt, config)

    src_density = mask_to_density(src_proc, config.mass_mode)
    tgt_density = mask_to_density(tgt_proc, config.mass_mode)

    if config.mass_mode.name == "DISTANCE_TRANSFORM":
        src_density = enforce_min_mass(src_density, fallback=mask_to_density_uniform(src_proc))
        tgt_density = enforce_min_mass(tgt_density, fallback=mask_to_density_uniform(tgt_proc))

    if config.downsample_factor > 1:
        src_density = downsample_density(src_density, config.downsample_factor)
        tgt_density = downsample_density(tgt_density, config.downsample_factor)

    src_support, src_meta = build_support(
        src_density,
        max_points=config.max_support_points,
        sampling_mode=config.sampling_mode,
        sampling_strategy=config.sampling_strategy,
        random_seed=config.random_seed,
    )
    tgt_support, tgt_meta = build_support(
        tgt_density,
        max_points=config.max_support_points,
        sampling_mode=config.sampling_mode,
        sampling_strategy=config.sampling_strategy,
        random_seed=config.random_seed,
    )

    transform_meta = {
        "preprocess": preprocess_meta,
        "downsample_factor": config.downsample_factor,
        "support_src": src_meta,
        "support_tgt": tgt_meta,
    }

    problem = UOTProblem(
        src=src_support,
        tgt=tgt_support,
        work_shape_hw=src_density.shape,
        transform_meta=transform_meta,
    )
    return problem, transform_meta


def run_uot_pair(
    pair: UOTFramePair,
    config: Optional[UOTConfig] = None,
    backend: Optional[UOTBackend] = None,
) -> UOTResult:
    """
    Run UOT on a mask pair.

    If config.use_canonical_grid is True, will use canonical grid preprocessing
    with physical units. Otherwise uses legacy preprocessing.
    """
    if config is None:
        config = UOTConfig()
    if backend is None:
        backend = POTBackend()

    if pair.src.embryo_mask is None or pair.tgt.embryo_mask is None:
        raise ValueError("UOTFramePair must contain embryo masks.")

    # Choose preprocessing mode
    if config.use_canonical_grid:
        # Canonical grid mode: transform to standardized physical grid
        canonical_config = CanonicalGridConfig(
            reference_um_per_pixel=config.canonical_grid_um_per_pixel,
            grid_shape_hw=config.canonical_grid_shape_hw,
            align_mode=config.canonical_grid_align_mode,
            downsample_factor=1,  # No additional downsampling on canonical grid
        )

        src_canonical, tgt_canonical, preprocess_meta = preprocess_pair_canonical(
            pair.src,
            pair.tgt,
            config,
            canonical_config,
        )

        # Build problem from canonical grids
        problem, transform_meta = build_problem(src_canonical, tgt_canonical, config)

        # Merge preprocessing metadata
        transform_meta["preprocess"] = preprocess_meta
        transform_meta["canonical_grid"] = True

    else:
        # Legacy mode: use original preprocessing
        problem, transform_meta = build_problem(pair.src.embryo_mask, pair.tgt.embryo_mask, config)
        transform_meta["canonical_grid"] = False

    # Solve transport problem
    backend_result = backend.solve(problem.src, problem.tgt, config)

    mass_created_hw, mass_destroyed_hw, velocity_field = compute_transport_maps(
        backend_result.coupling,
        problem.src.coords_yx,
        problem.tgt.coords_yx,
        problem.src.weights,
        problem.tgt.weights,
        problem.work_shape_hw,
    )

    # If using canonical grid, rescale velocities to micrometers
    if config.use_canonical_grid:
        src_transform = preprocess_meta["src_transform"]
        velocity_field = rescale_velocity_to_um(velocity_field, src_transform)
        # Note: distances in metrics are still in pixels for now
        # Could rescale them here if needed

    metrics = summarize_metrics(
        backend_result.cost,
        backend_result.coupling,
        mass_created_hw,
        mass_destroyed_hw,
        config.metric,
    )

    diagnostics = {
        "metrics": metrics,
        "backend": backend_result.diagnostics,
    }

    return UOTResult(
        cost=backend_result.cost,
        coupling=backend_result.coupling,
        mass_created_hw=mass_created_hw,
        mass_destroyed_hw=mass_destroyed_hw,
        velocity_field_yx_hw2=velocity_field,
        support_src_yx=problem.src.coords_yx,
        support_tgt_yx=problem.tgt.coords_yx,
        weights_src=problem.src.weights,
        weights_tgt=problem.tgt.weights,
        transform_meta=transform_meta,
        diagnostics=diagnostics,
    )


def run_from_csv(
    csv_path: Path,
    embryo_id: str,
    frame_index_src: int,
    frame_index_tgt: int,
    config: Optional[UOTConfig] = None,
    backend: Optional[UOTBackend] = None,
) -> UOTResult:
    pair = load_mask_pair_from_csv(csv_path, embryo_id, frame_index_src, frame_index_tgt)
    return run_uot_pair(pair, config=config, backend=backend)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run UOT on a single mask pair from CSV.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--embryo-id", required=True)
    parser.add_argument("--frame-src", type=int, required=True)
    parser.add_argument("--frame-tgt", type=int, required=True)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--mass-mode", type=str, default="uniform")
    args = parser.parse_args()

    cfg = UOTConfig(downsample_factor=args.downsample, mass_mode=args.mass_mode)
    result = run_from_csv(args.csv, args.embryo_id, args.frame_src, args.frame_tgt, config=cfg)
    metrics = result.diagnostics.get("metrics", {})
    print("UOT cost:", result.cost)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
