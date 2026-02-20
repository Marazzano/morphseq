"""
Canonical grid standardization for UOT mask transport.

Provides transformation to a fixed reference grid (576×256 @ 7.8 μm/px) matching
the snip export pipeline (src/build/build03A_process_images.py).

This enables:
- Resolution-independent comparisons across embryos
- Interpretable physical units (micrometers)
- Consistent orientation using yolk-based alignment
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
import warnings
from scipy import ndimage
from skimage.measure import regionprops

from data_pipeline.snip_processing.rotation import (
    get_embryo_rotation_angle,
    rotate_image,
)


@dataclass
class CanonicalGridConfig:
    """
    Configuration for canonical grid standardization.

    IMPORTANT: These defaults match the snip export pipeline in
    src/build/build03A_process_images.py (lines 1436-1467).
    Any changes here should be coordinated with snip export parameters.
    """
    reference_um_per_pixel: float = 10.0       # Updated canonical target resolution
    grid_shape_hw: tuple[int, int] = (256, 576)  # Match snip output shape (H, W)
    padding_um: float = 50.0                  # Padding in micrometers
    align_mode: str = "yolk"                  # "yolk" | "centroid" | "none"
    allow_flip: bool = True                   # Try horizontal flip to enforce stereotype
    anchor_mode: str = "yolk_anchor"          # "yolk_anchor" | "com_center"
    anchor_frac_yx: tuple[float, float] = (0.50, 0.50)  # (y_frac, x_frac)
    clipping_threshold: float = 0.98
    error_on_clip: bool = False
    downsample_factor: int = 1                # 1 = no downsampling (recommended for 576×256)


@dataclass
class GridTransform:
    """Records transformation from source to canonical grid."""
    source_um_per_pixel: float      # From CSV: Height(um) / Height(px)
    target_um_per_pixel: float      # = reference_um_per_pixel (7.8)
    scale_factor: float             # = source / target
    rotation_rad: float             # Yolk-based rotation angle
    offset_yx_px: tuple[int, int]   # Translation offset in target pixels
    grid_shape_hw: tuple[int, int]  # (256, 576)
    downsample_factor: int          # Additional downsampling after grid (1 = none)
    effective_um_per_pixel: float   # = target_um_per_pixel * downsample_factor


class CanonicalAligner:
    """
    Canonical alignment using PCA long-axis alignment, stereotyped orientation,
    optional flip, and yolk anchoring.

    Coordinate system: image coordinates (origin top-left, +y down).
    """

    def __init__(
        self,
        target_shape_hw: tuple[int, int] = (256, 576),
        target_um_per_pixel: float = 7.8,
        allow_flip: bool = True,
        anchor_mode: str = "yolk_anchor",  # "yolk_anchor" | "com_center"
        anchor_frac_yx: tuple[float, float] = (0.50, 0.50),
        clipping_threshold: float = 0.98,
        error_on_clip: bool = True,
        yolk_weight: float = 1.0,
        back_weight: float = 1.0,
        back_sample_radius_k: float = 1.5,
        yolk_pivot_angle_range_deg: float = 180.0,
        yolk_pivot_angle_step_deg: float = 1.0,
    ) -> None:
        self.H, self.W = target_shape_hw
        self.target_res = target_um_per_pixel
        self.allow_flip = allow_flip
        self.anchor_mode = anchor_mode
        self.anchor_frac_yx = anchor_frac_yx
        self.clipping_threshold = clipping_threshold
        self.error_on_clip = error_on_clip
        self.yolk_weight = yolk_weight
        self.back_weight = back_weight
        self.back_sample_radius_k = back_sample_radius_k
        self.yolk_pivot_angle_range_deg = yolk_pivot_angle_range_deg
        self.yolk_pivot_angle_step_deg = yolk_pivot_angle_step_deg
        self._last_back_debug: Optional[dict] = None

        self.is_landscape = self.W >= self.H

        y_frac, x_frac = anchor_frac_yx
        self.anchor_point_xy = (self.W * x_frac, self.H * y_frac)

    @classmethod
    def from_config(cls, config: "CanonicalGridConfig") -> "CanonicalAligner":
        return cls(
            target_shape_hw=config.grid_shape_hw,
            target_um_per_pixel=config.reference_um_per_pixel,
            allow_flip=config.allow_flip,
            anchor_mode=config.anchor_mode,
            anchor_frac_yx=config.anchor_frac_yx,
            clipping_threshold=config.clipping_threshold,
            error_on_clip=config.error_on_clip,
        )

    def _center_of_mass(self, mask: np.ndarray) -> tuple[float, float]:
        if mask is None or np.sum(mask) == 0:
            return (self.H / 2.0, self.W / 2.0)
        cy, cx = ndimage.center_of_mass(mask)
        return float(cy), float(cx)

    def _pca_angle_deg(self, mask: np.ndarray) -> tuple[float, tuple[float, float], bool]:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return 0.0, (self.W / 2.0, self.H / 2.0), False
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        mean = coords.mean(axis=0)
        coords_centered = coords - mean
        cov = np.cov(coords_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvals[0] <= 0:
            return 0.0, (float(mean[0]), float(mean[1])), False
        angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
        return angle, (float(mean[0]), float(mean[1])), True

    def _warp(self, mask: np.ndarray, M: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(mask.astype(np.float32), M, (self.W, self.H), flags=cv2.INTER_NEAREST)

    def _bbox(self, mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        ys, xs = np.where(mask > 0.5)
        if ys.size == 0:
            return None
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    def _project_point_to_mask_in_disk(
        self,
        mask: np.ndarray,
        yx: tuple[float, float],
        disk_center_yx: tuple[float, float],
        disk_radius: float,
    ) -> tuple[float, float]:
        """Project a point to the nearest mask pixel within the sampling disk.

        If the centroid lands off-mask, project to the nearest mask pixel
        *within the sampling disk*, not globally. This prevents projecting
        to a distant part of the embryo.
        """
        y, x = float(yx[0]), float(yx[1])
        iy = int(np.clip(round(y), 0, mask.shape[0] - 1))
        ix = int(np.clip(round(x), 0, mask.shape[1] - 1))
        if mask[iy, ix] > 0.5:
            return float(iy), float(ix)

        ys, xs = np.where(mask > 0.5)
        if ys.size == 0:
            return y, x

        # Restrict to pixels within the sampling disk
        dy_disk = ys.astype(np.float64) - disk_center_yx[0]
        dx_disk = xs.astype(np.float64) - disk_center_yx[1]
        in_disk = (dy_disk**2 + dx_disk**2) <= disk_radius**2
        if not in_disk.any():
            # No mask pixels in disk — fall back to nearest mask pixel globally
            d2 = (ys.astype(np.float64) - y) ** 2 + (xs.astype(np.float64) - x) ** 2
            idx = int(np.argmin(d2))
            return float(ys[idx]), float(xs[idx])

        ys_disk = ys[in_disk]
        xs_disk = xs[in_disk]
        d2 = (ys_disk.astype(np.float64) - y) ** 2 + (xs_disk.astype(np.float64) - x) ** 2
        idx = int(np.argmin(d2))
        return float(ys_disk[idx]), float(xs_disk[idx])

    def _compute_back_direction(
        self,
        mask: np.ndarray,
        yolk_mask: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """
        Compute back direction point using yolk-surrounding centroid.

        Geometry: sample all embryo-mask pixels within back_sample_radius_k × r_yolk
        of the yolk COM and compute their centroid. This centroid IS the back point.

        No fallback cascade — if yolk is available, use yolk-surrounding centroid.
        If yolk is None/empty, warn and return mask COM as a degenerate fallback.
        """
        back_debug = {}

        if yolk_mask is None or np.sum(yolk_mask) == 0:
            warnings.warn(
                "No yolk mask available for back-direction computation. "
                "Returning mask COM as degenerate back point.",
                RuntimeWarning, stacklevel=2,
            )
            fallback = self._center_of_mass(mask)
            back_debug["selected"] = "no_yolk_fallback"
            back_debug["back_yx"] = (float(fallback[0]), float(fallback[1]))
            self._last_back_debug = back_debug
            return fallback

        # Yolk COM — origin for everything
        yolk_com_y, yolk_com_x = self._center_of_mass(yolk_mask)
        yolk_area = float(yolk_mask.sum())
        r_yolk = np.sqrt(max(yolk_area, 1.0) / np.pi)
        r_sample = self.back_sample_radius_k * r_yolk

        back_debug["yolk_com_yx"] = (float(yolk_com_y), float(yolk_com_x))
        back_debug["r_yolk"] = float(r_yolk)
        back_debug["r_sample"] = float(r_sample)
        back_debug["back_sample_radius_k"] = float(self.back_sample_radius_k)

        # Find all embryo-mask pixels within the sampling disk
        ys, xs = np.nonzero(mask > 0.5)
        if ys.size == 0:
            back_debug["selected"] = "empty_mask"
            self._last_back_debug = back_debug
            return (yolk_com_y, yolk_com_x)

        dy = ys.astype(np.float64) - yolk_com_y
        dx = xs.astype(np.float64) - yolk_com_x
        dist_from_yolk = np.sqrt(dy**2 + dx**2)
        in_disk = dist_from_yolk <= r_sample

        n_pixels_in_disk = int(in_disk.sum())
        back_debug["n_pixels_in_disk"] = n_pixels_in_disk

        if n_pixels_in_disk == 0:
            warnings.warn(
                f"No embryo-mask pixels within sampling disk "
                f"(r_sample={r_sample:.1f}px around yolk COM). "
                f"Using yolk COM as back point.",
                RuntimeWarning, stacklevel=2,
            )
            back_debug["selected"] = "empty_disk"
            back_debug["back_yx"] = (float(yolk_com_y), float(yolk_com_x))
            self._last_back_debug = back_debug
            return (yolk_com_y, yolk_com_x)

        if n_pixels_in_disk < 50:
            warnings.warn(
                f"Only {n_pixels_in_disk} embryo-mask pixels in sampling disk "
                f"(r_sample={r_sample:.1f}px). Result may be noisy.",
                RuntimeWarning, stacklevel=2,
            )

        # Centroid of embryo pixels in the disk = back point
        back_centroid_y = float(ys[in_disk].mean())
        back_centroid_x = float(xs[in_disk].mean())

        back_debug["raw_back_centroid_yx"] = (back_centroid_y, back_centroid_x)

        # If centroid is off-mask, project to nearest mask pixel within disk
        back_y, back_x = self._project_point_to_mask_in_disk(
            mask, (back_centroid_y, back_centroid_x),
            disk_center_yx=(yolk_com_y, yolk_com_x),
            disk_radius=r_sample,
        )

        back_debug["selected"] = "yolk_surrounding_centroid"
        back_debug["back_yx"] = (float(back_y), float(back_x))
        self._last_back_debug = back_debug
        return (back_y, back_x)

    def _yolk_pivot_rotate(
        self,
        mask: np.ndarray,
        yolk_mask: Optional[np.ndarray],
        reference_mask: np.ndarray,
        angle_range_deg: float,
        angle_step_deg: float,
    ) -> tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Fine rotation sweep around the yolk COM to maximize IoU with reference_mask.

        The yolk COM is used as the pivot point so the yolk stays fixed during the sweep.
        Only the target mask is rotated; the reference stays fixed.

        Args:
            mask: Target binary mask on canonical grid.
            yolk_mask: Target yolk mask on canonical grid (may be None).
            reference_mask: Reference binary mask on canonical grid (fixed).
            angle_range_deg: Sweep ±angle_range_deg around 0.
            angle_step_deg: Step size in degrees.

        Returns:
            (best_mask, best_yolk_mask, best_angle_deg)
        """
        # Determine pivot point: yolk COM if yolk available, else mask COM
        if yolk_mask is not None and yolk_mask.sum() > 0:
            pivot_cy, pivot_cx = self._center_of_mass(yolk_mask)
        else:
            pivot_cy, pivot_cx = self._center_of_mass(mask)

        ref_bool = reference_mask > 0.5

        best_angle = 0.0
        best_iou = -1.0
        best_mask = mask
        best_yolk = yolk_mask

        angles = np.arange(
            -angle_range_deg,
            angle_range_deg + angle_step_deg,
            angle_step_deg,
        )
        for angle_deg in angles:
            theta = np.radians(float(angle_deg))
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            # Pivot-point rotation matrix (OpenCV convention: (cx, cy) = (x, y))
            M = np.float32([
                [cos_t, -sin_t, pivot_cx * (1 - cos_t) + pivot_cy * sin_t],
                [sin_t,  cos_t, pivot_cy * (1 - cos_t) - pivot_cx * sin_t],
            ])
            rotated = self._warp(mask, M)
            rot_bool = rotated > 0.5
            intersection = float(np.logical_and(rot_bool, ref_bool).sum())
            union = float(np.logical_or(rot_bool, ref_bool).sum())
            iou = intersection / (union + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_angle = float(angle_deg)
                best_mask = rotated
                if yolk_mask is not None:
                    best_yolk = self._warp(yolk_mask, M)
                else:
                    best_yolk = None

        # Report if the optimum landed at the sweep boundary — indicates the coarse
        # alignment step failed and the true optimum may lie outside the search range.
        # TODO: If best_angle is at the boundary, consider re-running the coarse step
        # with a wider candidate set or flagging the embryo for manual review.
        if abs(best_angle) >= angle_range_deg - angle_step_deg / 2:
            warnings.warn(
                f"Yolk-pivot sweep hit boundary at best_angle={best_angle:+.1f}° "
                f"(range=±{angle_range_deg}°). Coarse alignment may have failed.",
                RuntimeWarning, stacklevel=3,
            )

        return best_mask, best_yolk, best_angle

    # ------------------------------------------------------------------
    # Stage 1 helpers
    # ------------------------------------------------------------------

    def _coarse_candidate_select(
        self,
        mask: np.ndarray,
        yolk: Optional[np.ndarray],
        rotation_needed: float,
        scale: float,
        cx: float,
        cy: float,
        use_yolk: bool,
    ) -> tuple[float, float, bool, tuple, tuple]:
        """Evaluate 0°/180° × flip candidates and return best rotation + flip.

        Returns (best_rot_add, final_rotation, best_flip, best_yolk_yx, best_back_yx).
        Scoring is purely geometric (no reference mask).
        """
        candidates = []
        rot_options = [0, 180]
        flip_options = [False, True] if self.allow_flip else [False]

        for rot_add in rot_options:
            for do_flip in flip_options:
                M = cv2.getRotationMatrix2D((cx, cy), rotation_needed + rot_add, scale)
                M[0, 2] += (self.W / 2) - cx
                M[1, 2] += (self.H / 2) - cy

                mask_w = self._warp(mask, M)
                yolk_w = self._warp(yolk, M) if yolk is not None else None
                if do_flip:
                    mask_w = cv2.flip(mask_w, 1)
                    if yolk_w is not None:
                        yolk_w = cv2.flip(yolk_w, 1)

                yolk_feature_mask = (
                    yolk_w
                    if (use_yolk and yolk_w is not None and yolk_w.sum() > 0)
                    else mask_w
                )
                yolk_yx = self._center_of_mass(yolk_feature_mask)
                back_yx = self._compute_back_direction(
                    mask_w, yolk_mask=yolk_w if use_yolk else None
                )

                yolk_cost = yolk_yx[1] + yolk_yx[0]
                back_score = back_yx[1] + back_yx[0]
                score = (self.back_weight * back_score) - (self.yolk_weight * yolk_cost)

                candidates.append((score, rot_add, do_flip, yolk_yx, back_yx))

        best_score, best_rot, best_flip, best_yolk_yx, best_back_yx = max(
            candidates, key=lambda x: x[0]
        )
        final_rotation = rotation_needed + best_rot
        return final_rotation, best_flip, best_yolk_yx, best_back_yx

    def _apply_anchor_shift(
        self,
        aligned_mask: np.ndarray,
        aligned_yolk: Optional[np.ndarray],
        use_yolk: bool,
    ) -> tuple[np.ndarray, Optional[np.ndarray], float, float, bool, bool]:
        """Shift so that the feature COM lands at self.anchor_point_xy.

        Returns (shifted_mask, shifted_yolk, shift_x, shift_y, clamped, fit_impossible).
        """
        if self.anchor_mode == "yolk_anchor" and use_yolk and aligned_yolk is not None and aligned_yolk.sum() > 0:
            feat_mask = aligned_yolk
        else:
            feat_mask = aligned_mask

        cur_cy, cur_cx = self._center_of_mass(feat_mask)
        desired_shift_x = self.anchor_point_xy[0] - cur_cx
        desired_shift_y = self.anchor_point_xy[1] - cur_cy

        bbox = self._bbox(aligned_mask)
        shift_x = desired_shift_x
        shift_y = desired_shift_y
        clamped = False
        fit_impossible = False
        if bbox is not None:
            min_y, max_y, min_x, max_x = bbox
            min_shift_x = -min_x
            max_shift_x = (self.W - 1) - max_x
            min_shift_y = -min_y
            max_shift_y = (self.H - 1) - max_y
            if min_shift_x <= max_shift_x:
                shift_x = float(np.clip(desired_shift_x, min_shift_x, max_shift_x))
            else:
                fit_impossible = True
            if min_shift_y <= max_shift_y:
                shift_y = float(np.clip(desired_shift_y, min_shift_y, max_shift_y))
            else:
                fit_impossible = True
            clamped = (shift_x != desired_shift_x) or (shift_y != desired_shift_y)

        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_mask = self._warp(aligned_mask, M_shift)
        shifted_yolk = self._warp(aligned_yolk, M_shift) if aligned_yolk is not None else None
        return shifted_mask, shifted_yolk, shift_x, shift_y, clamped, fit_impossible

    def _validate_output_mask(
        self,
        final_mask: np.ndarray,
        original_mask: np.ndarray,
        scale: float,
        final_rotation: float,
        best_flip: bool,
        shift_x: float,
        shift_y: float,
        fit_impossible: bool,
        retained_ratio: float,
    ) -> None:
        """Raise RuntimeError if mask is empty or touches grid edges."""
        if final_mask.sum() == 0:
            raise RuntimeError(
                f"CanonicalAligner produced EMPTY mask (0 pixels). "
                f"Input mask: {original_mask.sum()} pixels, "
                f"retained_ratio={retained_ratio:.4f}, scale={scale:.3f}, "
                f"rotation={final_rotation:.1f}°, flip={best_flip}, "
                f"shift=({shift_x:.1f}, {shift_y:.1f}), fit_impossible={fit_impossible}. "
                f"This indicates a bug in the alignment transform."
            )
        h, w = final_mask.shape
        touches_top = final_mask[0, :].any()
        touches_bottom = final_mask[-1, :].any()
        touches_left = final_mask[:, 0].any()
        touches_right = final_mask[:, -1].any()
        if touches_top or touches_bottom or touches_left or touches_right:
            edges = []
            if touches_top: edges.append("top")
            if touches_bottom: edges.append("bottom")
            if touches_left: edges.append("left")
            if touches_right: edges.append("right")
            raise RuntimeError(
                f"CanonicalAligner produced mask touching grid edges: {', '.join(edges)}. "
                f"Aligned mask: {final_mask.sum()} pixels on {h}×{w} grid, "
                f"retained_ratio={retained_ratio:.4f}, scale={scale:.3f}, "
                f"rotation={final_rotation:.1f}°, flip={best_flip}, "
                f"shift=({shift_x:.1f}, {shift_y:.1f}), fit_impossible={fit_impossible}. "
                f"This likely indicates incorrect orientation detection or embryo too large for canonical grid."
            )

    # ------------------------------------------------------------------
    # Stage 1 public API
    # ------------------------------------------------------------------

    def generic_canonical_alignment(
        self,
        mask: np.ndarray,
        original_um_per_px: float,
        use_pca: bool = True,
        return_debug: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Stage 1: Single-embryo canonical alignment WITHOUT yolk.

        PCA rotation + scale + 0°/180°/flip (geometric heuristic).
        Anchor shift: mask COM → anchor_point_xy.
        No yolk; pivot = mask COM.

        Args:
            mask: Binary mask in source resolution.
            original_um_per_px: Source resolution in µm/pixel.
            use_pca: Whether to apply PCA-based rotation.
            return_debug: Whether to include debug arrays in meta.

        Returns:
            (canonical_mask, meta)
        """
        if mask is None or mask.sum() == 0:
            return np.zeros((self.H, self.W), dtype=np.uint8), {"error": "empty_mask"}

        scale = original_um_per_px / self.target_res
        angle_deg, centroid_xy, pca_used = (
            self._pca_angle_deg(mask) if use_pca else (0.0, (0.0, 0.0), False)
        )
        cx, cy = centroid_xy
        target_angle = 0.0 if self.is_landscape else 90.0
        rotation_needed = angle_deg - target_angle

        # Evaluate candidates geometrically (no yolk)
        candidates = []
        rot_options = [0, 180]
        flip_options = [False, True] if self.allow_flip else [False]
        for rot_add in rot_options:
            for do_flip in flip_options:
                M = cv2.getRotationMatrix2D((cx, cy), rotation_needed + rot_add, scale)
                M[0, 2] += (self.W / 2) - cx
                M[1, 2] += (self.H / 2) - cy
                mask_w = self._warp(mask, M)
                if do_flip:
                    mask_w = cv2.flip(mask_w, 1)
                mask_com_yx = self._center_of_mass(mask_w)
                # Prefer COM in upper-left
                score = -(mask_com_yx[0] + mask_com_yx[1])
                candidates.append((score, rot_add, do_flip, mask_com_yx))

        best_score, best_rot, best_flip, best_com_yx = max(candidates, key=lambda x: x[0])
        final_rotation = rotation_needed + best_rot

        M_final = cv2.getRotationMatrix2D((cx, cy), final_rotation, scale)
        M_final[0, 2] += (self.W / 2) - cx
        M_final[1, 2] += (self.H / 2) - cy
        aligned_mask = self._warp(mask, M_final)
        if best_flip:
            aligned_mask = cv2.flip(aligned_mask, 1)

        aligned_mask_pre_shift = aligned_mask.copy()
        aligned_angle_deg, _, _ = self._pca_angle_deg(aligned_mask_pre_shift)

        # Anchor shift: mask COM → anchor_point_xy
        aligned_mask, _, shift_x, shift_y, clamped, fit_impossible = self._apply_anchor_shift(
            aligned_mask, None, use_yolk=False
        )

        expected_area = float(mask.sum()) * (scale ** 2)
        final_area = float(aligned_mask.sum())
        retained_ratio = final_area / max(expected_area, 1e-6)
        clipped = retained_ratio < self.clipping_threshold
        if clipped:
            message = (
                f"generic_canonical_alignment clipped mask: retained_ratio={retained_ratio:.4f} "
                f"(threshold={self.clipping_threshold:.4f})"
            )
            if self.error_on_clip:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        final_mask = (aligned_mask > 0.5).astype(np.uint8)
        self._validate_output_mask(
            final_mask, mask, scale, final_rotation, best_flip, shift_x, shift_y, fit_impossible, retained_ratio
        )

        meta = {
            "pca_angle_deg": float(angle_deg),
            "rotation_needed_deg": float(rotation_needed),
            "rotation_deg": float(final_rotation),
            "flip": bool(best_flip),
            "scale": float(scale),
            "pca_used": bool(pca_used),
            "aligned_pca_angle_deg": float(aligned_angle_deg),
            "anchor_xy": (float(self.anchor_point_xy[0]), float(self.anchor_point_xy[1])),
            "anchor_shift_xy": (float(shift_x), float(shift_y)),
            "anchor_shift_clamped": bool(clamped),
            "fit_impossible": bool(fit_impossible),
            "retained_ratio": float(retained_ratio),
            "clipped": bool(clipped),
            "yolk_used": False,
        }
        if return_debug:
            meta["debug"] = {"aligned_mask_pre_shift": aligned_mask_pre_shift}
        return final_mask, meta

    def embryo_canonical_alignment(
        self,
        mask: np.ndarray,
        original_um_per_px: float,
        yolk: Optional[np.ndarray] = None,
        use_pca: bool = True,
        return_debug: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
        """Stage 1: Single-embryo canonical alignment WITH optional yolk.

        If yolk is missing or empty: warns loudly and delegates to
        generic_canonical_alignment() (canonical_yolk = None).
        If yolk is present: uses yolk-based anchor + yolk COM pivot.

        Meta includes "yolk_com_yx" (post-alignment canonical coords) when yolk exists.

        Args:
            mask: Binary mask in source resolution.
            original_um_per_px: Source resolution in µm/pixel.
            yolk: Optional binary yolk mask in source resolution.
            use_pca: Whether to apply PCA-based rotation.
            return_debug: Whether to include debug arrays in meta.

        Returns:
            (canonical_mask, canonical_yolk, meta)
            canonical_yolk is None when yolk is not available.
        """
        if mask is None or mask.sum() == 0:
            return (
                np.zeros((self.H, self.W), dtype=np.uint8),
                None,
                {"error": "empty_mask"},
            )

        if yolk is None or (hasattr(yolk, 'sum') and yolk.sum() == 0):
            warnings.warn(
                "embryo_canonical_alignment: yolk mask is None or empty. "
                "Falling back to generic_canonical_alignment() (no yolk-based anchoring). "
                "Provide a valid yolk mask for best results.",
                RuntimeWarning, stacklevel=2,
            )
            canonical_mask, meta = self.generic_canonical_alignment(
                mask, original_um_per_px, use_pca=use_pca, return_debug=return_debug
            )
            meta["yolk_used"] = False
            meta["yolk_com_yx"] = None
            return canonical_mask, None, meta

        # --- Yolk path: same logic as original align() but with stable internals ---
        scale = original_um_per_px / self.target_res
        angle_deg, centroid_xy, pca_used = (
            self._pca_angle_deg(mask) if use_pca else (0.0, (0.0, 0.0), False)
        )
        cx, cy = centroid_xy
        target_angle = 0.0 if self.is_landscape else 90.0
        rotation_needed = angle_deg - target_angle

        final_rotation, best_flip, best_yolk_yx, best_back_yx = self._coarse_candidate_select(
            mask, yolk, rotation_needed, scale, cx, cy, use_yolk=True
        )

        M_final = cv2.getRotationMatrix2D((cx, cy), final_rotation, scale)
        M_final[0, 2] += (self.W / 2) - cx
        M_final[1, 2] += (self.H / 2) - cy
        aligned_mask = self._warp(mask, M_final)
        aligned_yolk = self._warp(yolk, M_final)
        if best_flip:
            aligned_mask = cv2.flip(aligned_mask, 1)
            aligned_yolk = cv2.flip(aligned_yolk, 1)

        aligned_mask_pre_shift = aligned_mask.copy()
        aligned_yolk_pre_shift = aligned_yolk.copy() if aligned_yolk is not None else None
        aligned_angle_deg, _, _ = self._pca_angle_deg(aligned_mask_pre_shift)

        aligned_mask, aligned_yolk, shift_x, shift_y, clamped, fit_impossible = \
            self._apply_anchor_shift(aligned_mask, aligned_yolk, use_yolk=True)

        expected_area = float(mask.sum()) * (scale ** 2)
        final_area = float(aligned_mask.sum())
        retained_ratio = final_area / max(expected_area, 1e-6)
        clipped = retained_ratio < self.clipping_threshold
        if clipped:
            message = (
                f"embryo_canonical_alignment clipped mask: retained_ratio={retained_ratio:.4f} "
                f"(threshold={self.clipping_threshold:.4f})"
            )
            if self.error_on_clip:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        final_mask = (aligned_mask > 0.5).astype(np.uint8)
        self._validate_output_mask(
            final_mask, mask, scale, final_rotation, best_flip, shift_x, shift_y, fit_impossible, retained_ratio
        )
        final_yolk_mask = (aligned_yolk > 0.5).astype(np.uint8) if aligned_yolk is not None else None

        # Final positions
        yolk_feature_mask = aligned_yolk if (aligned_yolk is not None and aligned_yolk.sum() > 0) else aligned_mask
        final_yolk_yx = self._center_of_mass(yolk_feature_mask)
        final_back_yx = self._compute_back_direction(aligned_mask, yolk_mask=aligned_yolk)

        meta = {
            "pca_angle_deg": float(angle_deg),
            "rotation_needed_deg": float(rotation_needed),
            "rotation_deg": float(final_rotation),
            "flip": bool(best_flip),
            "scale": float(scale),
            "pca_used": bool(pca_used),
            "aligned_pca_angle_deg": float(aligned_angle_deg),
            "anchor_xy": (float(self.anchor_point_xy[0]), float(self.anchor_point_xy[1])),
            "anchor_shift_xy": (float(shift_x), float(shift_y)),
            "anchor_shift_clamped": bool(clamped),
            "fit_impossible": bool(fit_impossible),
            "yolk_yx_pre_shift": (
                (float(best_yolk_yx[0]), float(best_yolk_yx[1])) if best_yolk_yx is not None else None
            ),
            "back_yx_pre_shift": (
                (float(best_back_yx[0]), float(best_back_yx[1])) if best_back_yx is not None else None
            ),
            "yolk_yx_final": (float(final_yolk_yx[0]), float(final_yolk_yx[1])),
            "back_yx_final": (float(final_back_yx[0]), float(final_back_yx[1])),
            "retained_ratio": float(retained_ratio),
            "clipped": bool(clipped),
            "yolk_used": True,
            # REQUIRED: yolk COM in canonical coords (for Stage 2 use)
            "yolk_com_yx": (float(final_yolk_yx[0]), float(final_yolk_yx[1])),
        }
        if return_debug:
            meta["debug"] = {
                "aligned_mask_pre_shift": aligned_mask_pre_shift,
                "aligned_yolk_pre_shift": aligned_yolk_pre_shift,
                "back_direction": copy.deepcopy(self._last_back_debug),
            }
        return final_mask, final_yolk_mask, meta

    def align(
        self,
        mask: np.ndarray,
        yolk: Optional[np.ndarray],
        original_um_per_px: float,
        use_pca: bool = True,
        use_yolk: bool = True,
        reference_mask: Optional[np.ndarray] = None,
        return_debug: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
        """Deprecated. Use embryo_canonical_alignment() instead.

        Stage 1 canonical alignment (single-embryo, no reference).
        The ``reference_mask`` parameter is a no-op and will be ignored with a warning.
        """
        warnings.warn(
            "CanonicalAligner.align() is deprecated. "
            "Use embryo_canonical_alignment() for Stage 1 alignment instead.",
            DeprecationWarning, stacklevel=2,
        )
        if reference_mask is not None:
            warnings.warn(
                "align() reference_mask parameter is a no-op and has been removed. "
                "For Stage 2 src→tgt registration use embryo_src_tgt_register().",
                DeprecationWarning, stacklevel=2,
            )
        if not use_yolk:
            canonical_mask, meta = self.generic_canonical_alignment(
                mask, original_um_per_px, use_pca=use_pca, return_debug=return_debug
            )
            return canonical_mask, None, meta
        return self.embryo_canonical_alignment(
            mask, original_um_per_px, yolk=yolk, use_pca=use_pca, return_debug=return_debug
        )

# ---------------------------------------------------------------------------
# Stage 2: Src → Tgt Registration (module-level functions)
# ---------------------------------------------------------------------------

def _apply_pivot_rotation(mask: np.ndarray, pivot_yx: tuple, angle_deg: float) -> np.ndarray:
    """Rotate mask about pivot_yx by angle_deg (OpenCV convention, clockwise positive)."""
    cy, cx = float(pivot_yx[0]), float(pivot_yx[1])
    h, w = mask.shape[:2]
    theta = np.radians(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    M = np.float32([
        [cos_t, -sin_t, cx * (1 - cos_t) + cy * sin_t],
        [sin_t,  cos_t, cy * (1 - cos_t) - cx * sin_t],
    ])
    return cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST)


def _apply_translation(mask: np.ndarray, dyx: tuple) -> np.ndarray:
    """Translate mask by (dy, dx)."""
    dy, dx = float(dyx[0]), float(dyx[1])
    h, w = mask.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST)


def _mask_com(mask: np.ndarray) -> tuple:
    """Return (cy, cx) center of mass of a binary mask."""
    ys, xs = np.nonzero(mask > 0.5)
    if ys.size == 0:
        h, w = mask.shape[:2]
        return (h / 2.0, w / 2.0)
    return (float(ys.mean()), float(xs.mean()))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ab = a > 0.5
    bb = b > 0.5
    intersection = float(np.logical_and(ab, bb).sum())
    union = float(np.logical_or(ab, bb).sum())
    return intersection / (union + 1e-6)


def generic_src_tgt_register(
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    *,
    tgt_pivot_yx: Optional[Tuple[float, float]] = None,
    src_pivot_yx: Optional[Tuple[float, float]] = None,
    mode: str = "rotate_only",
    angle_step_deg: float = 1.0,
    min_iou_absolute: float = 0.25,
    min_iou_improvement: float = 0.02,
) -> tuple:
    """Stage 2: Src → Tgt registration. Pure geometry engine.

    Sweeps tgt_mask rotation over [-180°, 180°] about tgt_pivot_yx to maximise
    IoU with src_mask.  Optionally translates so tgt_pivot aligns to src_pivot.

    Gating: applies rotation only when
        best_iou >= min_iou_absolute  AND  best_iou >= iou_at_0deg + min_iou_improvement

    Args:
        src_mask: Source (fixed) binary mask on canonical grid.
        tgt_mask: Target binary mask on canonical grid to be registered.
        tgt_pivot_yx: Pivot for rotation; defaults to COM(tgt_mask).
        src_pivot_yx: Pivot for translation (rotate_then_pivot_translate mode);
            defaults to COM(src_mask).
        mode: "rotate_only" | "rotate_then_pivot_translate".
        angle_step_deg: Step size in degrees (default 1.0 → 361 evaluations).
        min_iou_absolute: Minimum absolute IoU for the sweep winner to be applied.
        min_iou_improvement: Minimum IoU improvement over 0° for the sweep winner to be applied.

    Returns:
        (refined_tgt_mask, meta)  where meta follows the documented schema.

    # TODO: Add unimodality / peak-sharpness check — skip pivot on flat/multi-modal IoU curves
    # (degenerate early-stage embryos with no clear A-P axis).
    """
    if mode not in ("rotate_only", "rotate_then_pivot_translate"):
        raise ValueError(f"Unknown mode={mode!r}. Expected 'rotate_only' or 'rotate_then_pivot_translate'.")

    # Resolve pivots
    if tgt_pivot_yx is None:
        tgt_pivot_yx = _mask_com(tgt_mask)
        tgt_pivot_source = "tgt_mask_com"
    else:
        tgt_pivot_yx = (float(tgt_pivot_yx[0]), float(tgt_pivot_yx[1]))
        tgt_pivot_source = "provided"

    if src_pivot_yx is None:
        src_pivot_yx = _mask_com(src_mask)
        src_pivot_source = "src_mask_com"
    else:
        src_pivot_yx = (float(src_pivot_yx[0]), float(src_pivot_yx[1]))
        src_pivot_source = "provided"

    iou_before = _iou(tgt_mask, src_mask)

    # Sweep ±180°
    angles = np.arange(-180.0, 180.0 + angle_step_deg, angle_step_deg)
    best_angle = 0.0
    best_iou = -1.0
    iou_at_0 = iou_before  # will be overwritten from sweep

    for angle_deg_f in angles:
        rotated = _apply_pivot_rotation(tgt_mask, tgt_pivot_yx, float(angle_deg_f))
        iou_val = _iou(rotated, src_mask)
        if abs(float(angle_deg_f)) < 0.5:
            iou_at_0 = iou_val
        if iou_val > best_iou:
            best_iou = iou_val
            best_angle = float(angle_deg_f)

    hit_boundary = abs(best_angle) >= 179.5

    # Gating
    apply = (best_iou >= min_iou_absolute) and (best_iou >= iou_at_0 + min_iou_improvement)

    if apply:
        refined = _apply_pivot_rotation(tgt_mask, tgt_pivot_yx, best_angle)
        angle_out = best_angle
    else:
        refined = tgt_mask
        angle_out = 0.0

    iou_after = _iou(refined, src_mask)

    meta: dict = {
        "applied": bool(apply),
        "mode": mode,
        "best_angle_deg": float(best_angle),
        "best_iou": float(best_iou),
        "hit_boundary": bool(hit_boundary),
        "angle_deg": float(angle_out),
        "iou_before": float(iou_before),
        "iou_after": float(iou_after),
        "tgt_pivot_yx": (float(tgt_pivot_yx[0]), float(tgt_pivot_yx[1])),
        "tgt_pivot_source": tgt_pivot_source,
        "src_pivot_yx": (float(src_pivot_yx[0]), float(src_pivot_yx[1])),
        "src_pivot_source": src_pivot_source,
    }

    if mode == "rotate_then_pivot_translate":
        if apply:
            # After rotation about tgt_pivot, the tgt_pivot itself doesn't move.
            # Translate so tgt_pivot → src_pivot.
            dy = src_pivot_yx[0] - tgt_pivot_yx[0]
            dx = src_pivot_yx[1] - tgt_pivot_yx[1]
            dyx = (dy, dx)
            refined = _apply_translation(refined, dyx)
            iou_after = _iou(refined, src_mask)
            meta["iou_after"] = float(iou_after)
        else:
            dyx = (0.0, 0.0)
        meta["translate_dyx"] = (float(dyx[0]), float(dyx[1]))

    return (refined > 0.5).astype(np.uint8), meta


def embryo_src_tgt_register(
    src_canonical: np.ndarray,
    tgt_canonical: np.ndarray,
    *,
    src_yolk_com_yx: Optional[Tuple[float, float]] = None,
    tgt_yolk_com_yx: Optional[Tuple[float, float]] = None,
    mode: str = "rotate_only",
    angle_step_deg: float = 1.0,
    min_iou_absolute: float = 0.25,
    min_iou_improvement: float = 0.02,
) -> tuple:
    """Stage 2: Embryo-aware src→tgt registration (thin biological dispatch wrapper).

    Uses yolk COMs as pivots where available; falls back to mask COMs with warnings.
    Translation mode is silently downgraded to rotate_only when both yolk COMs are absent.

    Args:
        src_canonical: Source canonical mask (fixed).
        tgt_canonical: Target canonical mask to be registered.
        src_yolk_com_yx: Yolk COM (y, x) of the source embryo in canonical coords.
        tgt_yolk_com_yx: Yolk COM (y, x) of the target embryo in canonical coords.
        mode: "rotate_only" | "rotate_then_pivot_translate".
        angle_step_deg: Rotation sweep step size in degrees.
        min_iou_absolute: Gating threshold — minimum absolute IoU.
        min_iou_improvement: Gating threshold — minimum improvement over 0°.

    Returns:
        (refined_tgt_canonical, meta)  meta follows the generic_src_tgt_register schema with
        relabeled pivot source fields.
    """
    # Resolve tgt pivot
    if tgt_yolk_com_yx is not None:
        tgt_pivot_yx = (float(tgt_yolk_com_yx[0]), float(tgt_yolk_com_yx[1]))
        tgt_pivot_source_label = "tgt_yolk_com"
    else:
        warnings.warn(
            "embryo_src_tgt_register: tgt_yolk_com_yx not provided. "
            "Falling back to tgt_mask COM as pivot. "
            "Supply tgt_yolk_com_yx from Stage 1 meta for best results.",
            RuntimeWarning, stacklevel=2,
        )
        tgt_pivot_yx = None  # generic_src_tgt_register will resolve
        tgt_pivot_source_label = "tgt_mask_com"

    # Resolve src pivot and mode
    if src_yolk_com_yx is not None:
        src_pivot_yx = (float(src_yolk_com_yx[0]), float(src_yolk_com_yx[1]))
        src_pivot_source_label = "src_yolk_com"
        effective_mode = mode
    else:
        src_pivot_yx = None  # generic_src_tgt_register will resolve
        src_pivot_source_label = "src_mask_com"
        if mode == "rotate_then_pivot_translate":
            # Silently downgrade — translation without both yolks is unreliable
            effective_mode = "rotate_only"
        else:
            effective_mode = mode

    refined, meta = generic_src_tgt_register(
        src_canonical, tgt_canonical,
        tgt_pivot_yx=tgt_pivot_yx,
        src_pivot_yx=src_pivot_yx,
        mode=effective_mode,
        angle_step_deg=angle_step_deg,
        min_iou_absolute=min_iou_absolute,
        min_iou_improvement=min_iou_improvement,
    )

    # Relabel pivot sources with biological names
    meta["tgt_pivot_source"] = tgt_pivot_source_label
    meta["src_pivot_source"] = src_pivot_source_label

    return refined, meta


def compute_grid_transform(
    mask: np.ndarray,
    source_um_per_pixel: float,
    yolk_mask: Optional[np.ndarray],
    config: CanonicalGridConfig,
) -> GridTransform:
    """
    Compute transform to place mask on canonical grid.

    Steps:
    1. Compute rotation angle (yolk-based or centroid-based)
    2. Compute scale factor to reach target resolution
    3. Compute crop offsets to center on mask centroid

    Args:
        mask: Binary mask in source resolution
        source_um_per_pixel: Physical resolution of source mask
        yolk_mask: Binary yolk mask (None if unavailable)
        config: Canonical grid configuration

    Returns:
        GridTransform capturing the full transformation
    """
    # 1. Compute rotation angle
    rotation_rad = 0.0
    if config.align_mode == "yolk":
        yolk_for_rotation = yolk_mask if yolk_mask is not None else np.zeros_like(mask)
        rotation_rad = get_embryo_rotation_angle(
            mask.astype(np.uint8),
            yolk_for_rotation.astype(np.uint8)
        )
    elif config.align_mode == "centroid":
        # Use PCA orientation without yolk guidance
        rp = regionprops(mask.astype(int))
        if rp:
            rotation_rad = -rp[0].orientation
    # else: align_mode == "none", rotation_rad = 0.0

    # 2. Compute scale factor
    scale_factor = source_um_per_pixel / config.reference_um_per_pixel

    # 3. Apply rotation and scaling to compute centroid in target space
    # (We'll compute offset when actually applying the transform)
    # For now, record the parameters
    effective_um_per_pixel = config.reference_um_per_pixel * config.downsample_factor

    return GridTransform(
        source_um_per_pixel=source_um_per_pixel,
        target_um_per_pixel=config.reference_um_per_pixel,
        scale_factor=scale_factor,
        rotation_rad=rotation_rad,
        offset_yx_px=(0, 0),  # Will be computed during apply
        grid_shape_hw=config.grid_shape_hw,
        downsample_factor=config.downsample_factor,
        effective_um_per_pixel=effective_um_per_pixel,
    )


def _rotate_and_scale_mask(
    mask: np.ndarray,
    transform: GridTransform,
) -> np.ndarray:
    """Rotate + scale mask into canonical pixel size, without centering/padding."""
    # 1. Rotate
    if abs(transform.rotation_rad) > 1e-6:
        angle_deg = np.rad2deg(transform.rotation_rad)
        mask_rotated = rotate_image(mask.astype(float), angle_deg)
        mask_rotated = (mask_rotated > 0.5).astype(np.uint8)
    else:
        mask_rotated = mask.astype(np.uint8)

    # 2. Rescale to target resolution
    if abs(transform.scale_factor - 1.0) > 1e-6:
        # scale_factor = source_um_per_pixel / target_um_per_pixel
        # Target pixel size is larger when scale_factor > 1, so we upscale by scale_factor.
        new_shape = (
            max(1, int(round(mask_rotated.shape[0] * transform.scale_factor))),
            max(1, int(round(mask_rotated.shape[1] * transform.scale_factor))),
        )
        mask_scaled = cv2.resize(
            mask_rotated.astype(float),
            (new_shape[1], new_shape[0]),  # cv2 uses (width, height)
            interpolation=cv2.INTER_LINEAR
        )
        mask_scaled = (mask_scaled > 0.5).astype(np.uint8)
    else:
        mask_scaled = mask_rotated

    return mask_scaled


def apply_grid_transform(
    mask: np.ndarray,
    transform: GridTransform,
    *,
    center_mode: str = "align_centroids",
    center_yx: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Apply transform to resample mask onto canonical grid.

    Steps:
    1. Rotate by transform.rotation_rad
    2. Rescale by transform.scale_factor
    3. Crop/pad to transform.grid_shape_hw centered on centroid

    Args:
        mask: Binary mask in source resolution
        transform: Grid transform computed by compute_grid_transform

    Returns:
        Resampled mask on canonical grid (shape = transform.grid_shape_hw)
    """
    mask_scaled = _rotate_and_scale_mask(mask, transform)

    # 3. Crop/pad to canonical grid centered on centroid
    target_h, target_w = transform.grid_shape_hw

    # Compute crop window origin (y0, x0) based on centering mode
    if center_mode == "align_centroids":
        if mask_scaled.sum() > 0:
            cy, cx = ndimage.center_of_mass(mask_scaled)
            cy, cx = int(cy), int(cx)
        else:
            cy, cx = mask_scaled.shape[0] // 2, mask_scaled.shape[1] // 2
        y0 = cy - target_h // 2
        x0 = cx - target_w // 2
    elif center_mode == "joint_centering":
        if center_yx is None:
            raise ValueError("center_yx is required for center_mode='joint_centering'")
        cy, cx = center_yx
        y0 = int(cy) - target_h // 2
        x0 = int(cx) - target_w // 2
    elif center_mode == "off":
        y0 = 0
        x0 = 0
    else:
        raise ValueError(
            f"Unknown center_mode='{center_mode}'. "
            "Expected 'align_centroids', 'joint_centering', or 'off'."
        )
    y1 = y0 + target_h
    x1 = x0 + target_w

    # Create output canvas
    canonical = np.zeros((target_h, target_w), dtype=np.uint8)

    # Compute overlap region
    src_y0 = max(0, y0)
    src_y1 = min(mask_scaled.shape[0], y1)
    src_x0 = max(0, x0)
    src_x1 = min(mask_scaled.shape[1], x1)

    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    # Copy data
    if src_y1 > src_y0 and src_x1 > src_x0:
        canonical[dst_y0:dst_y1, dst_x0:dst_x1] = mask_scaled[src_y0:src_y1, src_x0:src_x1]

    # Apply final downsampling if needed
    if transform.downsample_factor > 1:
        d = transform.downsample_factor
        downsampled_h = target_h // d
        downsampled_w = target_w // d
        canonical = cv2.resize(
            canonical.astype(float),
            (downsampled_w, downsampled_h),
            interpolation=cv2.INTER_LINEAR
        )
        canonical = (canonical > 0.5).astype(np.uint8)

    return canonical


def transform_coords_to_um(
    coords_yx: np.ndarray,
    transform: GridTransform,
    origin: str = "grid_center",
) -> np.ndarray:
    """
    Convert pixel coordinates to micrometers in canonical space.

    Args:
        coords_yx: (N, 2) array of (y, x) coordinates in canonical grid pixels
        transform: Grid transform
        origin: "grid_center" or "top_left"

    Returns:
        (N, 2) array of (y, x) coordinates in micrometers
    """
    coords_um = coords_yx * transform.effective_um_per_pixel

    if origin == "grid_center":
        # Shift origin to center of grid
        h, w = transform.grid_shape_hw
        if transform.downsample_factor > 1:
            h = h // transform.downsample_factor
            w = w // transform.downsample_factor
        center_um = np.array([h / 2, w / 2]) * transform.effective_um_per_pixel
        coords_um -= center_um

    return coords_um


def rescale_velocity_to_um(
    velocity_yx_hw2: np.ndarray,
    transform: GridTransform,
) -> np.ndarray:
    """
    Rescale velocity field from pixels to micrometers.

    Args:
        velocity_yx_hw2: (H, W, 2) velocity field in pixels
        transform: Grid transform

    Returns:
        (H, W, 2) velocity field in micrometers
    """
    return velocity_yx_hw2 * transform.effective_um_per_pixel


def rescale_distance_to_um(
    distance: float,
    transform: GridTransform,
) -> float:
    """
    Rescale a distance from pixels to micrometers.

    Args:
        distance: Distance in canonical grid pixels
        transform: Grid transform

    Returns:
        Distance in micrometers
    """
    return distance * transform.effective_um_per_pixel
