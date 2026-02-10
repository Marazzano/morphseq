"""Run unbalanced OT on a single mask pair."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from analyze.utils.optimal_transport import (
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
    compute_cost_maps,
    summarize_metrics,
)
from analyze.utils.optimal_transport.backends.base import UOTBackend
from analyze.utils.optimal_transport.pair_frame import create_pair_frame_geometry
from .preprocess import preprocess_pair, preprocess_pair_canonical
from .uot_grid import CanonicalGridConfig, rescale_velocity_to_um, rescale_distance_to_um


def build_problem(
    mask_src: np.ndarray,
    mask_tgt: np.ndarray,
    config: UOTConfig,
    px_size_um: float = 7.8,  # NEW PARAMETER
) -> Tuple[UOTProblem, dict]:
    """Build UOT problem with optional pair-frame geometry."""

    # NEW: Create pair frame if enabled
    pair_frame = None
    if config.use_pair_frame:
        # P0 CORRECTNESS: Ensure inputs are actually canonical-space masks
        assert mask_src.shape == mask_tgt.shape, \
            f"Mask shapes must match for pair-frame: {mask_src.shape} vs {mask_tgt.shape}"
        if config.use_canonical_grid:
            assert mask_src.shape == tuple(config.canonical_grid_shape_hw), \
                f"Pair-frame requires canonical grid: {mask_src.shape} vs {config.canonical_grid_shape_hw}"
        # Note: If not using canonical_grid, we trust the caller has provided canonical masks

        pair_frame = create_pair_frame_geometry(
            mask_src,
            mask_tgt,
            downsample_factor=config.downsample_factor,
            padding_px=config.padding_px,
            px_size_um=px_size_um,
        )

        # Use pair frame for cropping
        bbox = pair_frame.pair_crop_box_yx
        src_crop = mask_src[bbox.y0:bbox.y1, bbox.x0:bbox.x1]
        tgt_crop = mask_tgt[bbox.y0:bbox.y1, bbox.x0:bbox.x1]

        # CRITICAL: Apply padding using pair_frame's computed pad values
        # This ensures both crops use EXACTLY the same padding
        pad_h, pad_w = pair_frame.crop_pad_hw
        src_proc = np.pad(src_crop, ((0, pad_h), (0, pad_w)), mode="constant")
        tgt_proc = np.pad(tgt_crop, ((0, pad_h), (0, pad_w)), mode="constant")

        preprocess_meta = {
            "orig_shape": tuple(mask_src.shape),
            "bbox_y0y1x0x1": (bbox.y0, bbox.y1, bbox.x0, bbox.x1),
            "pad_hw": pair_frame.crop_pad_hw,
            "pair_frame_used": True,
        }
    else:
        # LEGACY PATH: Use existing preprocessing
        src_proc, tgt_proc, preprocess_meta = preprocess_pair(mask_src, mask_tgt, config)
        preprocess_meta["pair_frame_used"] = False

    # Rest of function (density, downsampling, support building) unchanged
    src_density = mask_to_density(src_proc, config.mass_mode)
    tgt_density = mask_to_density(tgt_proc, config.mass_mode)

    if config.mass_mode.name == "DISTANCE_TRANSFORM":
        src_density = enforce_min_mass(src_density, fallback=mask_to_density_uniform(src_proc))
        tgt_density = enforce_min_mass(tgt_density, fallback=mask_to_density_uniform(tgt_proc))

    # Golden test 6.2: Mass conservation during downsampling
    # Note: downsample_density() already uses sum pooling (verified in multiscale_sampling.py)
    if config.downsample_factor > 1:
        src_mass_before = float(src_density.sum())
        tgt_mass_before = float(tgt_density.sum())

        src_density = downsample_density(src_density, config.downsample_factor)
        tgt_density = downsample_density(tgt_density, config.downsample_factor)

        # Verify mass conservation (should be exact with sum pooling)
        assert np.isclose(src_density.sum(), src_mass_before, rtol=1e-6, atol=1e-6), \
            "Source mass not conserved during downsampling"
        assert np.isclose(tgt_density.sum(), tgt_mass_before, rtol=1e-6, atol=1e-6), \
            "Target mass not conserved during downsampling"

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
        pair_frame=pair_frame,  # NEW
    )

    # CRITICAL: Verify that solver's work shape matches pair_frame's work shape
    if pair_frame is not None:
        assert problem.work_shape_hw == pair_frame.work_shape_hw, \
            f"Work shape mismatch: solver={problem.work_shape_hw} vs pair_frame={pair_frame.work_shape_hw}"

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
        from analyze.utils.optimal_transport.backends.pot_backend import POTBackend

        backend = POTBackend()

    if pair.src.embryo_mask is None or pair.tgt.embryo_mask is None:
        raise ValueError("UOTFramePair must contain embryo masks.")

    # Choose preprocessing mode
    full_mask_src = None
    full_mask_tgt = None

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
        full_mask_src = src_canonical
        full_mask_tgt = tgt_canonical

        # Build problem from canonical grids
        problem, transform_meta = build_problem(
            src_canonical, tgt_canonical, config,
            px_size_um=config.canonical_grid_um_per_pixel
        )

        # Merge preprocessing metadata
        transform_meta["preprocess"] = preprocess_meta
        transform_meta["canonical_grid"] = True

    else:
        # Legacy mode: use original preprocessing
        problem, transform_meta = build_problem(
            pair.src.embryo_mask, pair.tgt.embryo_mask, config,
            px_size_um=7.8  # Default for legacy mode
        )
        transform_meta["canonical_grid"] = False
        full_mask_src = pair.src.embryo_mask
        full_mask_tgt = pair.tgt.embryo_mask

    # Solve transport problem
    backend_result = backend.solve(problem.src, problem.tgt, config)

    mass_created_hw, mass_destroyed_hw, velocity_field = compute_transport_maps(
        backend_result.coupling,
        problem.src.coords_yx,
        problem.tgt.coords_yx,
        problem.src.weights,
        problem.tgt.weights,
        problem.work_shape_hw,
        pair_frame=problem.pair_frame,  # NEW
    )

    cost_src_support = backend_result.cost_per_src
    cost_tgt_support = backend_result.cost_per_tgt
    cost_src_px = None
    cost_tgt_px = None
    if cost_src_support is not None and cost_tgt_support is not None:
        cost_src_px, cost_tgt_px = compute_cost_maps(
            cost_src_support,
            cost_tgt_support,
            problem.src.coords_yx,
            problem.tgt.coords_yx,
            problem.work_shape_hw,
            pair_frame=problem.pair_frame,
        )

    # P0 CORRECTNESS: If pair-frame used, verify padded regions are truly empty
    if problem.pair_frame is not None:
        pad_h, pad_w = problem.pair_frame.crop_pad_hw
        work_h, work_w = mass_created_hw.shape[:2]

        # Before rasterization, mass maps should be work-shaped (not canonical yet)
        # Padded bands should contain ~0 (solver shouldn't put mass in fake padding)
        if pad_h > 0 and mass_created_hw.shape == problem.work_shape_hw:
            # Padded band is at bottom of work grid
            pad_band_y = mass_created_hw[-pad_h:, :]
            assert np.allclose(pad_band_y, 0, atol=1e-8), \
                f"Mass created in padded Y band (bottom {pad_h} rows): max={pad_band_y.max()}"
            pad_band_y_destroyed = mass_destroyed_hw[-pad_h:, :]
            assert np.allclose(pad_band_y_destroyed, 0, atol=1e-8), \
                f"Mass destroyed in padded Y band: max={pad_band_y_destroyed.max()}"

        if pad_w > 0 and mass_created_hw.shape == problem.work_shape_hw:
            # Padded band is at right of work grid
            pad_band_x = mass_created_hw[:, -pad_w:]
            assert np.allclose(pad_band_x, 0, atol=1e-8), \
                f"Mass created in padded X band (right {pad_w} cols): max={pad_band_x.max()}"
            pad_band_x_destroyed = mass_destroyed_hw[:, -pad_w:]
            assert np.allclose(pad_band_x_destroyed, 0, atol=1e-8), \
                f"Mass destroyed in padded X band: max={pad_band_x_destroyed.max()}"

    # If using canonical grid without pair frame, apply legacy rescaling
    if config.use_canonical_grid and not config.use_pair_frame:
        src_transform = preprocess_meta["src_transform"]
        velocity_field = rescale_velocity_to_um(velocity_field, src_transform)
    # Note: If pair_frame is used, velocity is already in μm from rasterization

    # Extract source and target masses from backend diagnostics
    m_src = backend_result.diagnostics.get("m_src") if backend_result.diagnostics else None
    m_tgt = backend_result.diagnostics.get("m_tgt") if backend_result.diagnostics else None

    metrics = summarize_metrics(
        backend_result.cost,
        backend_result.coupling,
        mass_created_hw,
        mass_destroyed_hw,
        config.metric,
        m_src=m_src,
        m_tgt=m_tgt,
        pair_frame=problem.pair_frame,  # NEW - for um² calculation
        coord_scale=float(config.coord_scale),
    )

    def _mask_area(mask: np.ndarray) -> float:
        return float((np.asarray(mask) > 0).sum())

    if full_mask_src is not None and full_mask_tgt is not None:
        m_src_full = _mask_area(full_mask_src)
        m_tgt_full = _mask_area(full_mask_tgt)

        if problem.pair_frame is not None:
            bbox = problem.pair_frame.pair_crop_box_yx
            m_src_crop = _mask_area(full_mask_src[bbox.y0:bbox.y1, bbox.x0:bbox.x1])
            m_tgt_crop = _mask_area(full_mask_tgt[bbox.y0:bbox.y1, bbox.x0:bbox.x1])
        else:
            m_src_crop = m_src_full
            m_tgt_crop = m_tgt_full

        mass_delta_crop = m_tgt_crop - m_src_crop
        mass_delta_full = m_tgt_full - m_src_full
        mass_ratio_crop = float("nan") if m_src_crop <= 0 else (m_tgt_crop / m_src_crop)
        mass_ratio_full = float("nan") if m_src_full <= 0 else (m_tgt_full / m_src_full)

        metrics.update({
            "m_src_crop": m_src_crop,
            "m_tgt_crop": m_tgt_crop,
            "m_src_full": m_src_full,
            "m_tgt_full": m_tgt_full,
            "mass_delta_crop": mass_delta_crop,
            "mass_delta_full": mass_delta_full,
            "mass_ratio_crop": mass_ratio_crop,
            "mass_ratio_full": mass_ratio_full,
        })

    diagnostics = {
        "metrics": metrics,
        "backend": backend_result.diagnostics,
    }

    return UOTResult(
        cost=backend_result.cost,
        coupling=backend_result.coupling,
        mass_created_px=mass_created_hw,
        mass_destroyed_px=mass_destroyed_hw,
        velocity_px_per_frame_yx=velocity_field,
        support_src_yx=problem.src.coords_yx,
        support_tgt_yx=problem.tgt.coords_yx,
        weights_src=problem.src.weights,
        weights_tgt=problem.tgt.weights,
        transform_meta=transform_meta,
        cost_src_support=cost_src_support,
        cost_tgt_support=cost_tgt_support,
        cost_src_px=cost_src_px,
        cost_tgt_px=cost_tgt_px,
        diagnostics=diagnostics,
        pair_frame=problem.pair_frame,  # NEW: Enables property conversions
    )


def run_from_csv(
    csv_path: Path,
    embryo_id: str,
    frame_index_src: int,
    frame_index_tgt: int,
    config: Optional[UOTConfig] = None,
    backend: Optional[UOTBackend] = None,
    data_root: Optional[Path] = None,
) -> UOTResult:
    from .frame_mask_io import load_mask_pair_from_csv

    pair = load_mask_pair_from_csv(csv_path, embryo_id, frame_index_src, frame_index_tgt, data_root=data_root)
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
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()

    cfg = UOTConfig(downsample_factor=args.downsample, mass_mode=args.mass_mode)
    result = run_from_csv(args.csv, args.embryo_id, args.frame_src, args.frame_tgt, config=cfg, data_root=args.data_root)
    metrics = result.diagnostics.get("metrics", {})
    print("UOT cost:", result.cost)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
