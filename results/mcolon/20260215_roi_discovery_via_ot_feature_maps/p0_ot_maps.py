"""
Phase 0 Step 1: Generate OT maps (fixed WT reference → many targets).

Wraps the existing UOT pipeline to produce per-sample feature maps
on the 512×512 canonical grid. Outputs cost_density, displacement field,
and delta_mass per sample.

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
    cost_density: np.ndarray       # (512, 512) float32
    displacement_yx: np.ndarray    # (512, 512, 2) float32 — (v, u) convention
    delta_mass: np.ndarray         # (512, 512) float32 — created - destroyed
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
    uot_config=None,
    backend=None,
) -> OTMapResult:
    """
    Run unbalanced OT: ref → target, return canonical-grid maps.

    Uses the existing run_uot_pair() pipeline. mask_ref and mask_target
    must already be on the 512×512 canonical grid.
    """
    from analyze.utils.optimal_transport import UOTConfig, UOTFramePair, UOTFrame
    from analyze.optimal_transport_morphometrics.uot_masks.run_transport import run_uot_pair

    if uot_config is None:
        uot_config = UOTConfig(
            use_pair_frame=True,
            use_canonical_grid=False,  # masks already canonical
            downsample_factor=4,
        )

    pair = UOTFramePair(
        src=UOTFrame(embryo_mask=mask_ref),
        tgt=UOTFrame(embryo_mask=mask_target),
    )

    result = run_uot_pair(pair, config=uot_config, backend=backend)

    # Extract maps (already canonical-shaped if pair_frame was used)
    cost_density = result.cost_src_px if result.cost_src_px is not None else np.zeros_like(mask_ref, dtype=np.float32)
    displacement_yx = result.velocity_px_per_frame_yx  # (H, W, 2)
    delta_mass = result.mass_created_px - result.mass_destroyed_px

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
    feature_set: Phase0FeatureSet = Phase0FeatureSet.V0_COST,
    uot_config=None,
    backend=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run OT for all targets against fixed reference, return feature array X and total_cost_C.

    Parameters
    ----------
    mask_ref : (512, 512) uint8
    target_masks : list of (512, 512) uint8, length N
    sample_ids : list of str, length N
    feature_set : Phase0FeatureSet

    Returns
    -------
    X : (N, 512, 512, C) float32 — feature maps per sample
    total_cost_C : (N,) float32 — total OT cost per sample
    """
    N = len(target_masks)
    assert len(sample_ids) == N
    H, W = mask_ref.shape
    channel_schemas = PHASE0_CHANNEL_SCHEMAS[feature_set]
    C = len(channel_schemas)

    X = np.zeros((N, H, W, C), dtype=np.float32)
    total_cost_C = np.zeros(N, dtype=np.float32)

    for i, (mask_tgt, sid) in enumerate(zip(target_masks, sample_ids)):
        logger.info(f"OT map {i+1}/{N}: {sid}")

        ot_result = run_single_ot(
            mask_ref, mask_tgt, sid,
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
