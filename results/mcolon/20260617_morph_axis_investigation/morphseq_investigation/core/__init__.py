"""Core geometry and density utilities for modal organization analysis."""

from .density_composition import (  # noqa: F401
    CanonicalGrid,
    BridgeRegionTruth,
    ComponentTruth,
    ComposedDensityTruth,
    ComposedDensityValidation,
    DensityComponentSpec,
    DensityGrid,
    DensityRealization,
    DensitySpec,
    build_density_spec,
    compose_density_truth,
    infer_canonical_grid,
    validate_composed_density_truth,
    validate_density_spec,
    realize_from_truth,
    realize_density,
)
from .peak_counting import (  # noqa: F401
    PeakCountDetail,
    count_mass_significant_modes,
    peak_count_detail,
)
