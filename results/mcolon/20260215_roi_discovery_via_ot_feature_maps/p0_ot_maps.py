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
    aligned_ref_mask: Optional[np.ndarray]    # (H, W) uint8 — exact OT-aligned source mask
    aligned_target_mask: Optional[np.ndarray] # (H, W) uint8 — exact OT-aligned target mask
    total_cost_C: float
    diagnostics: Dict
    alignment_debug: Optional[Dict] = None


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
    source_id: Optional[str] = None,
    target_id: Optional[str] = None,
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

    # Fail fast on mask/yolk shape mismatches. Passing a canonicalized embryo mask
    # with a raw-resolution yolk mask silently breaks canonical alignment.
    if yolk_ref is not None and yolk_ref.shape != mask_ref.shape:
        raise ValueError(
            f"Reference yolk shape {yolk_ref.shape} does not match reference mask shape {mask_ref.shape}. "
            "Both must be in the same coordinate system (raw input expected)."
        )
    if yolk_tgt is not None and yolk_tgt.shape != mask_target.shape:
        raise ValueError(
            f"Target yolk shape {yolk_tgt.shape} does not match target mask shape {mask_target.shape}. "
            "Both must be in the same coordinate system (raw input expected)."
        )

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

    # Heuristic warning for accidental double-canonicalization:
    # if input already looks canonical but raw_um_per_px indicates otherwise,
    # preprocess_pair_canonical will rescale it again.
    if (
        getattr(uot_config, "use_canonical_grid", False)
        and tuple(mask_ref.shape) == tuple(getattr(uot_config, "canonical_grid_shape_hw", (256, 576)))
        and not np.isclose(
            float(raw_um_per_px_ref),
            float(getattr(uot_config, "canonical_grid_um_per_pixel", 10.0)),
            atol=1e-6,
        )
    ):
        logger.warning(
            "Reference mask shape already matches canonical grid %s but raw_um_per_px_ref=%.6f "
            "(canonical is %.6f). This often indicates a pre-canonicalized reference passed into "
            "run_single_ot, which can shrink maps due to double canonicalization.",
            mask_ref.shape,
            raw_um_per_px_ref,
            float(getattr(uot_config, "canonical_grid_um_per_pixel", 10.0)),
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

    preprocess_meta = {}
    if isinstance(result.transform_meta, dict):
        preprocess_meta = result.transform_meta.get("preprocess", {}) or {}
    src_align = preprocess_meta.get("canonical_alignment_src", {}) or {}
    tgt_align = preprocess_meta.get("canonical_alignment_tgt", {}) or {}
    metrics = {}
    if isinstance(result.diagnostics, dict):
        metrics = result.diagnostics.get("metrics", {}) or {}

    src_mask_aligned = (
        np.asarray(result.aligned_src_mask_px, dtype=np.uint8)
        if getattr(result, "aligned_src_mask_px", None) is not None
        else None
    )
    tgt_mask_aligned = (
        np.asarray(result.aligned_tgt_mask_px, dtype=np.uint8)
        if getattr(result, "aligned_tgt_mask_px", None) is not None
        else None
    )
    overlap_iou = np.nan
    if src_mask_aligned is not None and tgt_mask_aligned is not None:
        src_bool = src_mask_aligned > 0
        tgt_bool = tgt_mask_aligned > 0
        union = float(np.logical_or(src_bool, tgt_bool).sum())
        inter = float(np.logical_and(src_bool, tgt_bool).sum())
        overlap_iou = inter / union if union > 0 else np.nan

    bbox_y0y1x0x1 = preprocess_meta.get("bbox_y0y1x0x1")
    pad_hw = preprocess_meta.get("pad_hw")
    bbox_y0 = bbox_y0y1x0x1[0] if bbox_y0y1x0x1 is not None else np.nan
    bbox_y1 = bbox_y0y1x0x1[1] if bbox_y0y1x0x1 is not None else np.nan
    bbox_x0 = bbox_y0y1x0x1[2] if bbox_y0y1x0x1 is not None else np.nan
    bbox_x1 = bbox_y0y1x0x1[3] if bbox_y0y1x0x1 is not None else np.nan
    pad_h = pad_hw[0] if pad_hw is not None else np.nan
    pad_w = pad_hw[1] if pad_hw is not None else np.nan

    alignment_debug = {
        "sample_id": sample_id,
        "source_id": source_id,
        "target_id": target_id,
        "src_rotation_deg": src_align.get("rotation_deg"),
        "src_flip": src_align.get("flip"),
        "src_retained_ratio": src_align.get("retained_ratio"),
        "src_anchor_shift_x": (
            src_align.get("anchor_shift_xy", (np.nan, np.nan))[0]
            if src_align.get("anchor_shift_xy") is not None else np.nan
        ),
        "src_anchor_shift_y": (
            src_align.get("anchor_shift_xy", (np.nan, np.nan))[1]
            if src_align.get("anchor_shift_xy") is not None else np.nan
        ),
        "src_yolk_y_final": (
            src_align.get("yolk_yx_final", (np.nan, np.nan))[0]
            if src_align.get("yolk_yx_final") is not None else np.nan
        ),
        "src_yolk_x_final": (
            src_align.get("yolk_yx_final", (np.nan, np.nan))[1]
            if src_align.get("yolk_yx_final") is not None else np.nan
        ),
        "src_back_y_final": (
            src_align.get("back_yx_final", (np.nan, np.nan))[0]
            if src_align.get("back_yx_final") is not None else np.nan
        ),
        "src_back_x_final": (
            src_align.get("back_yx_final", (np.nan, np.nan))[1]
            if src_align.get("back_yx_final") is not None else np.nan
        ),
        "tgt_rotation_deg": tgt_align.get("rotation_deg"),
        "tgt_flip": tgt_align.get("flip"),
        "tgt_retained_ratio": tgt_align.get("retained_ratio"),
        "tgt_anchor_shift_x": (
            tgt_align.get("anchor_shift_xy", (np.nan, np.nan))[0]
            if tgt_align.get("anchor_shift_xy") is not None else np.nan
        ),
        "tgt_anchor_shift_y": (
            tgt_align.get("anchor_shift_xy", (np.nan, np.nan))[1]
            if tgt_align.get("anchor_shift_xy") is not None else np.nan
        ),
        "tgt_yolk_y_final": (
            tgt_align.get("yolk_yx_final", (np.nan, np.nan))[0]
            if tgt_align.get("yolk_yx_final") is not None else np.nan
        ),
        "tgt_yolk_x_final": (
            tgt_align.get("yolk_yx_final", (np.nan, np.nan))[1]
            if tgt_align.get("yolk_yx_final") is not None else np.nan
        ),
        "tgt_back_y_final": (
            tgt_align.get("back_yx_final", (np.nan, np.nan))[0]
            if tgt_align.get("back_yx_final") is not None else np.nan
        ),
        "tgt_back_x_final": (
            tgt_align.get("back_yx_final", (np.nan, np.nan))[1]
            if tgt_align.get("back_yx_final") is not None else np.nan
        ),
        "pair_bbox_y0": bbox_y0,
        "pair_bbox_y1": bbox_y1,
        "pair_bbox_x0": bbox_x0,
        "pair_bbox_x1": bbox_x1,
        "pair_pad_h": pad_h,
        "pair_pad_w": pad_w,
        "total_cost_C": float(result.cost),
        "mass_delta_crop": metrics.get("mass_delta_crop"),
        "mass_ratio_crop": metrics.get("mass_ratio_crop"),
        "overlap_iou_src_tgt": overlap_iou,
    }

    return OTMapResult(
        sample_id=sample_id,
        cost_density=cost_density.astype(np.float32),
        displacement_yx=displacement_yx.astype(np.float32),
        delta_mass=delta_mass.astype(np.float32),
        aligned_ref_mask=src_mask_aligned,
        aligned_target_mask=tgt_mask_aligned,
        total_cost_C=float(result.cost),
        diagnostics=result.diagnostics or {},
        alignment_debug=alignment_debug,
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
    source_id: Optional[str] = None,
    target_ids: Optional[List[str]] = None,
    return_aligned_masks: bool = False,
    collect_debug: bool = True,
    strict_debug_ids: bool = False,
    return_debug_df: bool = False,
):
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
    If return_aligned_masks=False:
        X : (N, 256, 576, C) float32 — feature maps per sample on canonical grid
        total_cost_C : (N,) float32 — total OT cost per sample
    If return_aligned_masks=True:
        X, total_cost_C, aligned_ref_mask, aligned_target_masks
        aligned_ref_mask : (256, 576) uint8
        aligned_target_masks : (N, 256, 576) uint8
    If return_debug_df=True:
        returns an additional DataFrame with per-sample alignment diagnostics.
    """
    N = len(target_masks)
    assert len(sample_ids) == N
    assert len(raw_um_per_px_targets) == N
    if yolk_targets is not None:
        assert len(yolk_targets) == N
    if target_ids is not None:
        assert len(target_ids) == N

    if collect_debug and (source_id is None or target_ids is None):
        msg = (
            "Alignment debug capture is enabled, but source_id/target_ids were not fully provided. "
            "Debug rows will miss explicit source/target IDs. Pass source_id + target_ids "
            "(for example embryo_id|frame_index) to make downstream debugging traceable."
        )
        if strict_debug_ids:
            raise ValueError(msg)
        logger.warning(msg)
    
    # Canonical grid shape
    H, W = 256, 576
    channel_schemas = PHASE0_CHANNEL_SCHEMAS[feature_set]
    C = len(channel_schemas)

    X = np.zeros((N, H, W, C), dtype=np.float32)
    total_cost_C = np.zeros(N, dtype=np.float32)
    aligned_ref_mask = None
    aligned_target_masks = np.zeros((N, H, W), dtype=np.uint8) if return_aligned_masks else None
    debug_rows = []

    for i, (mask_tgt, sid) in enumerate(zip(target_masks, sample_ids)):
        logger.info(f"OT map {i+1}/{N}: {sid}")

        ot_result = run_single_ot(
            mask_ref, mask_tgt, sid,
            raw_um_per_px_ref=raw_um_per_px_ref,
            raw_um_per_px_tgt=raw_um_per_px_targets[i],
            yolk_ref=yolk_ref,
            yolk_tgt=yolk_targets[i] if yolk_targets is not None else None,
            source_id=source_id,
            target_id=target_ids[i] if target_ids is not None else None,
            uot_config=uot_config, backend=backend,
        )

        total_cost_C[i] = ot_result.total_cost_C

        if return_aligned_masks:
            if ot_result.aligned_ref_mask is None or ot_result.aligned_target_mask is None:
                raise ValueError(
                    "run_single_ot did not return aligned masks. "
                    "Expected aligned_ref_mask and aligned_target_mask for QC overlays."
                )
            if ot_result.aligned_ref_mask.shape != (H, W):
                raise ValueError(
                    f"aligned_ref_mask shape mismatch: {ot_result.aligned_ref_mask.shape} vs {(H, W)}"
                )
            if ot_result.aligned_target_mask.shape != (H, W):
                raise ValueError(
                    f"aligned_target_mask shape mismatch: {ot_result.aligned_target_mask.shape} vs {(H, W)}"
                )

            if aligned_ref_mask is None:
                aligned_ref_mask = ot_result.aligned_ref_mask.astype(np.uint8)
            else:
                if not np.array_equal(aligned_ref_mask, ot_result.aligned_ref_mask):
                    logger.warning(
                        "Aligned reference mask changed across samples for the same reference. "
                        "Using first sample's aligned reference mask."
                    )

            aligned_target_masks[i] = ot_result.aligned_target_mask.astype(np.uint8)

        if collect_debug and ot_result.alignment_debug is not None:
            debug_rows.append(ot_result.alignment_debug)

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
    debug_df = pd.DataFrame(debug_rows) if return_debug_df else None

    if return_aligned_masks and return_debug_df:
        if aligned_ref_mask is None:
            raise ValueError("Aligned reference mask was never populated.")
        return X, total_cost_C, aligned_ref_mask, aligned_target_masks, debug_df
    if return_aligned_masks:
        if aligned_ref_mask is None:
            raise ValueError("Aligned reference mask was never populated.")
        return X, total_cost_C, aligned_ref_mask, aligned_target_masks
    if return_debug_df:
        return X, total_cost_C, debug_df
    return X, total_cost_C


__all__ = [
    "OTMapResult",
    "run_single_ot",
    "generate_ot_maps",
]
