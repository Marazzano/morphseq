"""Legacy import surface for canonical grid utilities.

This module is kept for backward compatibility with older notebooks/scripts.
New code should import from `analyze.utils.coord`:
- `analyze.utils.coord.grids.canonical` for canonicalization
- `analyze.utils.coord.register` for explicit registration

IMPORTANT: Canonicalization/registration are geometry concerns, not UOT concerns.
UOT should only ingest their outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from analyze.utils.coord.grids.canonical import (  # noqa: F401
    CanonicalAligner,
    CanonicalGridConfig,
    to_canonical_grid_frame,
    to_canonical_grid_image,
    to_canonical_grid_mask,
)
from analyze.utils.coord.register import (  # noqa: F401
    RegisterConfig,
    register_to_fixed,
    _apply_pivot_rotation,
    _apply_translation,
    _iou,
    _mask_com,
)


def _warn_legacy(name: str) -> None:
    warnings.warn(
        f"Importing {name} from uot_masks.uot_grid is deprecated. Use analyze.utils.coord instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Legacy Stage 2 API wrappers (kept for compatibility)
# ---------------------------------------------------------------------------

def generic_src_tgt_register(
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    *,
    tgt_pivot_yx=None,
    src_pivot_yx=None,
    mode: str = "rotate_only",
    angle_step_deg: float = 1.0,
    min_iou_absolute: float = 0.25,
    min_iou_improvement: float = 0.02,
) -> tuple[np.ndarray, dict]:
    _warn_legacy("generic_src_tgt_register")
    cfg = RegisterConfig(
        mode=mode,
        angle_step_deg=float(angle_step_deg),
        min_iou_absolute=float(min_iou_absolute),
        min_iou_improvement=float(min_iou_improvement),
    )
    reg = register_to_fixed(
        moving=tgt_mask,
        fixed=src_mask,
        cfg=cfg,
        apply=True,
        moving_pivot_yx=tgt_pivot_yx,
        fixed_pivot_yx=src_pivot_yx,
    )
    meta = dict(reg.meta.get("register_to_fixed", {}))
    # Back-compat key names
    meta_out = {
        "applied": bool(reg.applied),
        "mode": cfg.mode,
        "best_angle_deg": meta.get("best_angle_deg", meta.get("angle_deg", 0.0)),
        "best_iou": meta.get("best_iou", meta.get("best_iou", float("nan"))),
        "angle_deg": meta.get("angle_deg", 0.0),
        "iou_before": meta.get("iou_before", float("nan")),
        "iou_after": meta.get("iou_after", float("nan")),
        "tgt_pivot_yx": meta.get("moving_pivot_yx"),
        "tgt_pivot_source": "provided" if tgt_pivot_yx is not None else "tgt_mask_com",
        "src_pivot_yx": meta.get("fixed_pivot_yx"),
        "src_pivot_source": "provided" if src_pivot_yx is not None else "src_mask_com",
        "translate_dyx": meta.get("translate_dyx", (0.0, 0.0)),
        "hit_boundary": bool(abs(float(meta.get("best_angle_deg", 0.0))) >= 179.5),
    }
    return (reg.moving_in_fixed if reg.moving_in_fixed is not None else tgt_mask).astype(np.uint8), meta_out


def embryo_src_tgt_register(
    src_canonical: np.ndarray,
    tgt_canonical: np.ndarray,
    *,
    src_yolk_com_yx=None,
    tgt_yolk_com_yx=None,
    mode: str = "rotate_only",
    angle_step_deg: float = 1.0,
    min_iou_absolute: float = 0.25,
    min_iou_improvement: float = 0.02,
) -> tuple[np.ndarray, dict]:
    _warn_legacy("embryo_src_tgt_register")
    tgt_pivot = tgt_yolk_com_yx
    src_pivot = src_yolk_com_yx
    if src_pivot is None and mode == "rotate_then_pivot_translate":
        mode = "rotate_only"
    refined, meta = generic_src_tgt_register(
        src_canonical,
        tgt_canonical,
        tgt_pivot_yx=tgt_pivot,
        src_pivot_yx=src_pivot,
        mode=mode,
        angle_step_deg=angle_step_deg,
        min_iou_absolute=min_iou_absolute,
        min_iou_improvement=min_iou_improvement,
    )
    meta["tgt_pivot_source"] = "tgt_yolk_com" if tgt_yolk_com_yx is not None else "tgt_mask_com"
    meta["src_pivot_source"] = "src_yolk_com" if src_yolk_com_yx is not None else "src_mask_com"
    return refined, meta


# ---------------------------------------------------------------------------
# Legacy canonical-grid transform record (used by UOT meta today)
# ---------------------------------------------------------------------------

@dataclass
class GridTransform:
    """Records transformation from source to canonical grid (legacy)."""

    source_um_per_pixel: float
    target_um_per_pixel: float
    scale_factor: float
    rotation_rad: float
    offset_yx_px: tuple[int, int]
    grid_shape_hw: tuple[int, int]
    downsample_factor: int
    effective_um_per_pixel: float


def rescale_velocity_to_um(velocity_yx_hw2: np.ndarray, transform: GridTransform) -> np.ndarray:
    return velocity_yx_hw2 * float(transform.effective_um_per_pixel)


def rescale_distance_to_um(distance: float, transform: GridTransform) -> float:
    return float(distance) * float(transform.effective_um_per_pixel)
