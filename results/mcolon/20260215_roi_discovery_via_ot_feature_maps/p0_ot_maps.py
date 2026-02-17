"""
Phase 0 Step 1: Generate OT maps (fixed WT reference → many targets).

Wraps the existing UOT pipeline to produce per-sample feature maps
on the canonical grid (256×576 at 10 µm/px). Outputs cost_density, 
displacement field, and delta_mass per sample.

Usage:
    from p0_ot_maps import generate_ot_maps
    results = generate_ot_maps(mask_ref, target_masks, config)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from roi_config import (
    FeatureDatasetConfig,
    Phase0FeatureSet,
    PHASE0_CHANNEL_SCHEMAS,
)

logger = logging.getLogger(__name__)


@dataclass
class OTMapResult:
    """Result from a single ref→target OT computation."""
    sample_id: str
    cost_density: np.ndarray       # (H, W) float32 — canonical grid shape
    displacement_yx: np.ndarray    # (H, W, 2) float32 — (v, u) convention
    delta_mass: np.ndarray         # (H, W) float32 — created - destroyed
    total_cost_C: float
    diagnostics: Dict


def _compute_ot_params_hash(config_dict: dict) -> str:
    """Deterministic hash of OT parameters for provenance."""
    raw = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_single_ot(
    mask_ref: np.ndarray,
    mask_target: np.ndarray,
    sample_id: str,
    raw_um_per_px_ref: float,
    raw_um_per_px_tgt: float,
    yolk_ref: Optional[np.ndarray] = None,
    yolk_tgt: Optional[np.ndarray] = None,
    uot_config=None,
    backend=None,
) -> OTMapResult:
    """
    Run unbalanced OT: ref → target, return canonical-grid maps.

    Masks are provided in raw resolution; the UOT pipeline automatically
    aligns them to canonical grid (256×576 at 10 µm/px) when
    use_canonical_grid=True.

    Parameters
    ----------
    mask_ref : (H_ref, W_ref) uint8 — reference mask at raw resolution
    mask_target : (H_tgt, W_tgt) uint8 — target mask at raw resolution
    sample_id : str
    raw_um_per_px_ref : float — physical resolution of reference mask
    raw_um_per_px_tgt : float — physical resolution of target mask
    uot_config : UOTConfig, optional
    backend : UOTBackend, optional

    Returns
    -------
    OTMapResult with maps on canonical grid (256, 576)
    """
    from analyze.utils.optimal_transport import UOTConfig, UOTFramePair, UOTFrame, MassMode
    from analyze.utils.optimal_transport.backends.pot_backend import POTBackend
    from analyze.optimal_transport_morphometrics.uot_masks.run_transport import run_uot_pair

    if uot_config is None:
        # Use validated config from Stream D but with POT backend
        # Parameters: epsilon=1e-4, reg_m=10, max_support_points=3000
        uot_config = UOTConfig(
            # Relaxation parameters (validated in Stream D)
            epsilon=1e-4,
            marginal_relaxation=10.0,
            max_support_points=3000,
            # Canonical grid configuration
            use_canonical_grid=True,
            output_grid="canonical",
            canonical_grid_shape_hw=(256, 576),
            canonical_grid_um_per_pixel=10.0,
            canonical_grid_align_mode="yolk",
            canonical_grid_center_mode="joint_centering",
            # No downsampling - compute on full canonical grid
            downsample_factor=1,
            downsample_divisor=1,
            # Other settings
            padding_px=16,
            mass_mode=MassMode.UNIFORM,
            align_mode="none",  # Canonical alignment handles it
            store_coupling=True,
            random_seed=42,
            metric="sqeuclidean",
            coord_scale=1.0 / max((256, 576)),
        )
    
    if backend is None:
        backend = POTBackend()  # Using POT instead of OTT

    # Create frames with metadata for physical units
    meta_ref = {"um_per_pixel": raw_um_per_px_ref}
    if yolk_ref is not None:
        meta_ref["yolk_mask"] = yolk_ref
    meta_tgt = {"um_per_pixel": raw_um_per_px_tgt}
    if yolk_tgt is not None:
        meta_tgt["yolk_mask"] = yolk_tgt

    pair = UOTFramePair(
        src=UOTFrame(
            embryo_mask=mask_ref,
            meta=meta_ref,
        ),
        tgt=UOTFrame(
            embryo_mask=mask_target,
            meta=meta_tgt,
        ),
    )

    result = run_uot_pair(pair, config=uot_config, backend=backend)

    # Extract maps (should be canonical-shaped)
    cost_density = result.cost_src_px if result.cost_src_px is not None else np.zeros((256, 576), dtype=np.float32)
    displacement_yx = result.velocity_px_per_frame_yx  # (H, W, 2)
    delta_mass = result.mass_created_px - result.mass_destroyed_px

    # Verify canonical shape with a clear error message
    expected_hw = (256, 576)
    bad_shapes = []
    if cost_density.shape != expected_hw:
        bad_shapes.append(f"cost_density={cost_density.shape}")
    if displacement_yx.shape != (expected_hw[0], expected_hw[1], 2):
        bad_shapes.append(f"displacement_yx={displacement_yx.shape}")
    if delta_mass.shape != expected_hw:
        bad_shapes.append(f"delta_mass={delta_mass.shape}")
    if bad_shapes:
        cfg = uot_config or {}
        raise ValueError(
            "OT output shapes are not canonical. This usually happens when OT is returning work-grid "
            "maps (e.g., downsampled) instead of full canonical grid. "
            f"Expected={expected_hw}, got: {', '.join(bad_shapes)}. "
            f"Config: use_canonical_grid={getattr(cfg, 'use_canonical_grid', None)}, "
            f"output_grid={getattr(cfg, 'output_grid', None)}, "
            f"use_pair_frame_geometry={getattr(cfg, 'use_pair_frame_geometry', None)}, "
            f"use_pair_frame(deprecated)={getattr(cfg, 'use_pair_frame', None)}, "
            f"downsample_factor={getattr(cfg, 'downsample_factor', None)}, "
            f"downsample_divisor={getattr(cfg, 'downsample_divisor', None)}."
        )

    return OTMapResult(
        sample_id=sample_id,
        cost_density=cost_density.astype(np.float32),
        displacement_yx=displacement_yx.astype(np.float32),
        delta_mass=delta_mass.astype(np.float32),
        total_cost_C=float(result.cost),
        diagnostics=result.diagnostics or {},
    )


def generate_ot_maps(
    mask_ref: np.ndarray,
    target_masks: List[np.ndarray],
    sample_ids: List[str],
    raw_um_per_px_ref: float,
    raw_um_per_px_targets: np.ndarray,
    yolk_ref: Optional[np.ndarray] = None,
    yolk_targets: Optional[List[np.ndarray]] = None,
    feature_set: Phase0FeatureSet = Phase0FeatureSet.V0_COST,
    uot_config=None,
    backend=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run OT for all targets against fixed reference, return feature array X and total_cost_C.

    Masks are provided at raw resolution; UOT pipeline aligns to canonical grid.

    Parameters
    ----------
    mask_ref : (H_ref, W_ref) uint8 — reference mask at raw resolution
    target_masks : list of (H_i, W_i) uint8, length N — raw resolution
    sample_ids : list of str, length N
    raw_um_per_px_ref : float — physical resolution of reference mask
    raw_um_per_px_targets : (N,) array — physical resolution per target
    feature_set : Phase0FeatureSet

    Returns
    -------
    X : (N, 256, 576, C) float32 — feature maps per sample on canonical grid
    total_cost_C : (N,) float32 — total OT cost per sample
    """
    N = len(target_masks)
    assert len(sample_ids) == N
    assert len(raw_um_per_px_targets) == N
    if yolk_targets is not None:
        assert len(yolk_targets) == N
    
    # Canonical grid shape
    H, W = 256, 576
    channel_schemas = PHASE0_CHANNEL_SCHEMAS[feature_set]
    C = len(channel_schemas)

    X = np.zeros((N, H, W, C), dtype=np.float32)
    total_cost_C = np.zeros(N, dtype=np.float32)

    for i, (mask_tgt, sid) in enumerate(zip(target_masks, sample_ids)):
        logger.info(f"OT map {i+1}/{N}: {sid}")

        ot_result = run_single_ot(
            mask_ref, mask_tgt, sid,
            raw_um_per_px_ref=raw_um_per_px_ref,
            raw_um_per_px_tgt=raw_um_per_px_targets[i],
            yolk_ref=yolk_ref,
            yolk_tgt=yolk_targets[i] if yolk_targets is not None else None,
            uot_config=uot_config, backend=backend,
        )

        total_cost_C[i] = ot_result.total_cost_C

        if feature_set == Phase0FeatureSet.V0_COST:
            X[i, :, :, 0] = ot_result.cost_density
        elif feature_set == Phase0FeatureSet.V1_DYNAMICS:
            X[i, :, :, 0] = ot_result.cost_density
            X[i, :, :, 1] = ot_result.displacement_yx[:, :, 1]  # disp_u (x)
            X[i, :, :, 2] = ot_result.displacement_yx[:, :, 0]  # disp_v (y)
            X[i, :, :, 3] = np.sqrt(
                ot_result.displacement_yx[:, :, 0]**2
                + ot_result.displacement_yx[:, :, 1]**2
            )  # disp_mag
            X[i, :, :, 4] = ot_result.delta_mass

    logger.info(f"Generated OT maps: X shape={X.shape}, mean cost={total_cost_C.mean():.4f}")
    return X, total_cost_C


__all__ = [
    "OTMapResult",
    "run_single_ot",
    "generate_ot_maps",
]
