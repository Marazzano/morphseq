"""Distribution comparison + plotting engine.

The four dumb domain objects (`Distribution`, `Grid`, `LabelGroup`, `SampleSet`),
the derived `label_groups` view, the structured id constructors, and the central
invariant guards. See ``docs/PRIMITIVE_ONTOLOGY.md`` — the LOCKED spec.

TASK_0 (this module set): shapes + id/invariant machinery ONLY. Grid construction
(TASK_A), labelers (TASK_B), comparison (TASK_C), plotting (TASK_D) live elsewhere.
"""

from .objects import (
    Distribution,
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
from .identifiers import (
    make_distribution_id,
    make_sample_set_id,
    make_grid_id,
)
from .invariants import (
    validate_label_group,
    InvariantError,
)

__all__ = [
    "Distribution",
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
    "make_distribution_id",
    "make_sample_set_id",
    "make_grid_id",
    "validate_label_group",
    "InvariantError",
]
