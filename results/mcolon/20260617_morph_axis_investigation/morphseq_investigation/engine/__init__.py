"""Distribution catalog engine — the frozen ontology + id/invariant machinery.

The typed-column ``Distribution`` (samples + features + label groups +
coordinates), its derived views (``SampleSet`` via ``sample_sets``,
``DistributionLabelGroup`` via ``label_group``), the label-column types, the
typed ``FacetKey``s, the structured id constructors, and the central invariant
guards. See ``docs/DISTRIBUTION_CATALOG_API.md`` — the LOCKED spec.

TASK_0 (this module set): the frozen contract A/B/C key off. Catalog / compare /
pool_by / from_dataframe (TASK_A), column-writer labelers + discover_modes body
(TASK_B), and plotting (TASK_C/D) live elsewhere.

``DistributionGrouping`` / ``MaterializedDistributionGrouping`` are RETIRED from
the public API (spec §"Removed vocabulary") — they are not exported here.
"""

from .objects import (
    UNASSIGNED_LABEL,
    Distribution,
    DistributionLabelGroup,
    LabelColumn,
    LabelProvenance,
    Grid,
    DensityGrid,
    SampleSet,
    SampleSetGeometry,
    HDR,
    FeatureProfile,
    LabelGroup,
    LabelGroupArtifacts,
    LabelGroups,
    derive_label_groups,
    resolve_label_group,
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
    "DistributionLabelGroup",
    "LabelColumn",
    "LabelProvenance",
    "Grid",
    "DensityGrid",
    "SampleSet",
    "SampleSetGeometry",
    "HDR",
    "FeatureProfile",
    "LabelGroup",
    "LabelGroupArtifacts",
    "LabelGroups",
    "derive_label_groups",
    "resolve_label_group",
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
