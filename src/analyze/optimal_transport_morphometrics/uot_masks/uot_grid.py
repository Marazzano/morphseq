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
        yolk_pivot_angle_range_deg: float = 45.0,
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
        if mask is None or mask.sum() == 0:
            return (
                np.zeros((self.H, self.W), dtype=np.uint8),
                None,
                {"error": "empty_mask"},
            )

        if use_yolk and (yolk is None or yolk.sum() == 0):
            raise ValueError(
                "use_yolk=True but yolk mask is None or empty. "
                "Ensure yolk masks are loaded (pass data_root to load_mask_from_csv). "
                "Set align_mode='centroid' if yolk alignment is not needed."
            )

        scale = original_um_per_px / self.target_res

        angle_deg, centroid_xy, pca_used = self._pca_angle_deg(mask) if use_pca else (0.0, (0.0, 0.0), False)
        cx, cy = centroid_xy
        # Note: OpenCV positive angles rotate clockwise in image coordinates (+y down).
        # To align the PCA axis to the target axis, we invert the usual sign.
        target_angle = 0.0 if self.is_landscape else 90.0
        rotation_needed = angle_deg - target_angle

        candidates = []
        best_yolk_yx = None
        best_back_yx = None
        rot_options = [0, 180]
        flip_options = [False, True] if self.allow_flip else [False]

        for rot_add in rot_options:
            for do_flip in flip_options:
                M = cv2.getRotationMatrix2D((cx, cy), rotation_needed + rot_add, scale)
                M[0, 2] += (self.W / 2) - cx
                M[1, 2] += (self.H / 2) - cy

                mask_w = self._warp(mask, M)
                yolk_w = self._warp(yolk, M) if (yolk is not None) else None
                if do_flip:
                    mask_w = cv2.flip(mask_w, 1)
                    if yolk_w is not None:
                        yolk_w = cv2.flip(yolk_w, 1)

                # Yolk COM = anchor point (we want it top-left)
                yolk_feature_mask = yolk_w if (use_yolk and yolk_w is not None and yolk_w.sum() > 0) else mask_w
                yolk_yx = self._center_of_mass(yolk_feature_mask)

                # Back direction = centroid of embryo tissue surrounding yolk
                back_yx = self._compute_back_direction(
                    mask_w,
                    yolk_mask=yolk_w if use_yolk else None,
                )

                if reference_mask is not None:
                    intersection = np.logical_and(mask_w > 0.5, reference_mask > 0.5).sum()
                    union = np.logical_or(mask_w > 0.5, reference_mask > 0.5).sum()
                    score = intersection / (union + 1e-6)
                else:
                    # Scoring: yolk near top-left (low y+x) = low cost;
                    # back near bottom-right (high y+x) = high score.
                    yolk_cost = yolk_yx[1] + yolk_yx[0]
                    back_score = back_yx[1] + back_yx[0]
                    score = (self.back_weight * back_score) - (self.yolk_weight * yolk_cost)

                candidates.append((score, rot_add, do_flip, yolk_yx, back_yx))

        best_score, best_rot, best_flip, best_yolk_yx, best_back_yx = max(
            candidates, key=lambda x: x[0]
        )
        final_rotation = rotation_needed + best_rot

        M_final = cv2.getRotationMatrix2D((cx, cy), final_rotation, scale)
        M_final[0, 2] += (self.W / 2) - cx
        M_final[1, 2] += (self.H / 2) - cy

        aligned_mask = self._warp(mask, M_final)
        aligned_yolk = self._warp(yolk, M_final) if yolk is not None else None
        if best_flip:
            aligned_mask = cv2.flip(aligned_mask, 1)
            if aligned_yolk is not None:
                aligned_yolk = cv2.flip(aligned_yolk, 1)

        aligned_mask_pre_shift = aligned_mask.copy()
        aligned_yolk_pre_shift = aligned_yolk.copy() if aligned_yolk is not None else None
        aligned_angle_deg, _, _ = self._pca_angle_deg(aligned_mask_pre_shift)

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
        aligned_mask = self._warp(aligned_mask, M_shift)
        if aligned_yolk is not None:
            aligned_yolk = self._warp(aligned_yolk, M_shift)

        # Yolk-pivot fine rotation (only when reference is available)
        yolk_pivot_angle_deg = 0.0
        yolk_pivot_hit_boundary = False
        if reference_mask is not None and use_yolk and aligned_yolk is not None and aligned_yolk.sum() > 0:
            aligned_mask, aligned_yolk, yolk_pivot_angle_deg = self._yolk_pivot_rotate(
                aligned_mask, aligned_yolk, reference_mask,
                self.yolk_pivot_angle_range_deg,
                self.yolk_pivot_angle_step_deg,
            )
            yolk_pivot_hit_boundary = (
                abs(yolk_pivot_angle_deg) >= self.yolk_pivot_angle_range_deg - self.yolk_pivot_angle_step_deg / 2
            )

        # Final yolk/back positions (post shift)
        final_yolk_mask = aligned_yolk if (use_yolk and aligned_yolk is not None and aligned_yolk.sum() > 0) else aligned_mask
        final_yolk_yx = self._center_of_mass(final_yolk_mask)
        final_back_yx = self._compute_back_direction(
            aligned_mask,
            yolk_mask=aligned_yolk if use_yolk else None,
        )

        expected_area = float(mask.sum()) * (scale ** 2)
        final_area = float(aligned_mask.sum())
        retained_ratio = final_area / max(expected_area, 1e-6)
        clipped = retained_ratio < self.clipping_threshold
        if clipped:
            message = (
                f"Canonical alignment clipped mask: retained_ratio={retained_ratio:.4f} "
                f"(threshold={self.clipping_threshold:.4f})"
            )
            if self.error_on_clip:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

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
            "yolk_pivot_angle_deg": float(yolk_pivot_angle_deg),
            "yolk_pivot_hit_boundary": bool(yolk_pivot_hit_boundary),
        }

        # CRITICAL VALIDATION: Never return empty masks
        final_mask = (aligned_mask > 0.5).astype(np.uint8)
        if final_mask.sum() == 0:
            error_msg = (
                f"CanonicalAligner produced EMPTY mask (0 pixels). "
                f"Input mask: {mask.sum()} pixels, "
                f"retained_ratio={retained_ratio:.4f}, "
                f"scale={scale:.3f}, "
                f"rotation={final_rotation:.1f}°, "
                f"flip={best_flip}, "
                f"shift=({shift_x:.1f}, {shift_y:.1f}), "
                f"fit_impossible={fit_impossible}. "
                f"This indicates a bug in the alignment transform."
            )
            raise RuntimeError(error_msg)

        # CRITICAL VALIDATION: Mask should not touch grid edges
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

            error_msg = (
                f"CanonicalAligner produced mask touching grid edges: {', '.join(edges)}. "
                f"Aligned mask: {final_mask.sum()} pixels on {h}×{w} grid, "
                f"retained_ratio={retained_ratio:.4f}, "
                f"scale={scale:.3f}, "
                f"rotation={final_rotation:.1f}°, "
                f"flip={best_flip}, "
                f"shift=({shift_x:.1f}, {shift_y:.1f}), "
                f"fit_impossible={fit_impossible}. "
                f"This likely indicates incorrect orientation detection or embryo too large for canonical grid."
            )
            raise RuntimeError(error_msg)

        if return_debug:
            meta["debug"] = {
                "aligned_mask_pre_shift": aligned_mask_pre_shift,
                "aligned_yolk_pre_shift": aligned_yolk_pre_shift,
                "back_direction": copy.deepcopy(self._last_back_debug),
            }

        return final_mask, (
            (aligned_yolk > 0.5).astype(np.uint8) if aligned_yolk is not None else None
        ), meta

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
