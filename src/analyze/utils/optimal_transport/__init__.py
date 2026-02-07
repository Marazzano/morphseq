"""Reusable optimal transport utilities for mask-based analyses."""

from analyze.utils.optimal_transport.config import (
    UOTConfig,
    UOTFrame,
    UOTFramePair,
    UOTSupport,
    UOTProblem,
    UOTResult,
    SamplingMode,
    MassMode,
    Coupling,
)
from analyze.utils.optimal_transport.backends.base import UOTBackend, BackendResult
from analyze.utils.optimal_transport.backends.pot_backend import POTBackend

try:
    from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend
except ImportError:
    OTTBackend = None
from analyze.utils.optimal_transport.density_transforms import (
    mask_to_density,
    mask_to_density_uniform,
    mask_to_density_boundary_band,
    mask_to_density_distance_transform,
    enforce_min_mass,
)
from analyze.utils.optimal_transport.multiscale_sampling import (
    pad_to_divisible,
    downsample_density,
    build_support,
)
from analyze.utils.optimal_transport.transport_maps import compute_transport_maps, compute_cost_maps
from analyze.utils.optimal_transport.metrics import summarize_metrics, compute_transport_metrics

__all__ = [
    # Config and data structures
    "UOTConfig",
    "UOTFrame",
    "UOTFramePair",
    "UOTSupport",
    "UOTProblem",
    "UOTResult",
    "SamplingMode",
    "MassMode",
    "Coupling",
    # Backends
    "UOTBackend",
    "BackendResult",
    "POTBackend",
    "OTTBackend",
    # Density transforms
    "mask_to_density",
    "mask_to_density_uniform",
    "mask_to_density_boundary_band",
    "mask_to_density_distance_transform",
    "enforce_min_mass",
    # Multiscale and sampling
    "pad_to_divisible",
    "downsample_density",
    "build_support",
    # Transport maps
    "compute_transport_maps",
    "compute_cost_maps",
    # Metrics
    "summarize_metrics",
    "compute_transport_metrics",
]
