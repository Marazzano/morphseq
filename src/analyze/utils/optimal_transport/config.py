"""Core configuration and dataclasses for UOT mask transport."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np

try:
    import scipy.sparse as sp
    Coupling = Union[np.ndarray, sp.coo_matrix]
except Exception:  # pragma: no cover - scipy optional in some envs
    Coupling = np.ndarray


class SamplingMode(str, Enum):
    AUTO = "auto"
    RAISE = "raise"


class MassMode(str, Enum):
    UNIFORM = "uniform"
    BOUNDARY_BAND = "boundary_band"
    DISTANCE_TRANSFORM = "distance_transform"


@dataclass
class UOTFrame:
    frame: Optional[np.ndarray] = None
    embryo_mask: Optional[np.ndarray] = None
    meta: Optional[dict] = None


@dataclass
class UOTFramePair:
    src: UOTFrame
    tgt: UOTFrame
    pair_meta: Optional[dict] = None


@dataclass
class UOTSupport:
    coords_yx: np.ndarray
    weights: np.ndarray


@dataclass
class UOTProblem:
    src: UOTSupport
    tgt: UOTSupport
    work_shape_hw: tuple[int, int]
    transform_meta: dict


@dataclass
class UOTResult:
    cost: float
    coupling: Optional[Coupling]

    mass_created_hw: np.ndarray
    mass_destroyed_hw: np.ndarray
    velocity_field_yx_hw2: np.ndarray

    support_src_yx: np.ndarray
    support_tgt_yx: np.ndarray
    weights_src: np.ndarray
    weights_tgt: np.ndarray

    transform_meta: dict
    diagnostics: Optional[dict] = None


@dataclass
class UOTConfig:
    downsample_factor: int = 4
    max_support_points: int = 5000
    sampling_mode: SamplingMode = SamplingMode.AUTO
    sampling_strategy: str = "stratified_boundary_interior"

    epsilon: float = 1e-2
    marginal_relaxation: float = 10.0
    metric: str = "sqeuclidean"
    coord_scale: float = 1.0

    mass_mode: MassMode = MassMode.UNIFORM
    align_mode: str = "centroid"

    store_coupling: bool = True
    random_seed: int = 0
    padding_px: int = 8
    downsample_divisor: int = 16
