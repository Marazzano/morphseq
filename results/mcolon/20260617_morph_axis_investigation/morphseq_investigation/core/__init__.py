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
from .bandwidth_tuning import (  # noqa: F401
    BandwidthCandidate,
    BridgeRegionMeasurement,
    DEFAULT_CONNECTIVITY_MASS,
    DEFAULT_KNN_K,
    DEFAULT_MULTIPLIERS,
    DEFAULT_RULE_NAMES,
    DensityGeometryMeasurement,
    RULE_NAME_ALIASES,
    bandwidth_geometry_scales,
    evaluate_isotropic_gaussian_kde_from_dist2,
    measure_density_geometry,
    precompute_squared_distances,
    propose_bandwidth_candidates,
)
from .peak_counting import (  # noqa: F401
    PeakCountDetail,
    count_mass_significant_modes,
    peak_count_detail,
)
