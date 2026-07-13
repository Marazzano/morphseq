"""Public value objects, identifiers, facets, and invariant guards."""

from .objects import (
    UNASSIGNED_LABEL,
    Distribution,
    Grid,
    DensityGrid,
    SampleSet,
    SampleSetGeometry,
    LabelGroup,
    LabelingProvenance,
    DensityEstimate,
    DensityEstimateSpec,
)
from ..core.peak_stability import (
    PeakVotingSpec,
    PeakCountRobustnessPolicy,
    PeakResolutionSummary,
)
from .facets import (
    CoordinateFacet,
    LabelGroupFacet,
    FacetKey,
)
from .identifiers import (
    make_distribution_id,
    make_sample_set_id,
    make_grid_id,
)
from .invariants import (
    validate_label_group,
    validate_sample_sets,
    InvariantError,
)

__all__ = [
    "UNASSIGNED_LABEL",
    "Distribution",
    "Grid",
    "DensityGrid",
    "SampleSet",
    "SampleSetGeometry",
    "LabelGroup",
    "LabelingProvenance",
    "DensityEstimate",
    "DensityEstimateSpec",
    "PeakVotingSpec",
    "PeakCountRobustnessPolicy",
    "PeakResolutionSummary",
    "CoordinateFacet",
    "LabelGroupFacet",
    "FacetKey",
    "make_distribution_id",
    "make_sample_set_id",
    "make_grid_id",
    "validate_label_group",
    "validate_sample_sets",
    "InvariantError",
]
