"""The Distribution ontology — typed-column sample table + derived views.

LOCKED shapes — from ``docs/DISTRIBUTION_CATALOG_API.md`` ("The ontology",
"Objects", "Typed facet keys"). This is the TASK_0 reshape of the older
PRIMITIVE_ONTOLOGY objects onto the catalog model:

    samples        (sample_ids — the join key)
    features       (feature_names + feature_values)
    label groups   (sample-aligned partitions: genotype / phenotype / resolved_peak)
    coordinates    (distribution-level constants: time_bin, scope_id, ...)

Two metadata types, two jobs (spec §"The ontology"):
    coordinates  →  which distribution you have  →  selection / faceting / compare()
    label groups →  how samples inside it split   →  within-cell grouping / overlay

Design rule above all else: keep the objects dumb, don't leak between layers.
Validity / grain / time / binning are the *caller's* job. ``role`` is a
RELATIONSHIP assigned by a comparison, NEVER a field here (spec §"role is a
RELATIONSHIP").

TASK_0 builds the frozen shapes + id/invariant machinery. Grid construction and
``from_dataframe`` (TASK_A), the ``discover_modes`` body (TASK_B), and all
plotting (TASK_C/D) live elsewhere — the impl-later method is a documented stub
that raises ``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Hashable, Mapping

import numpy as np

from .identifiers import make_sample_set_id

if TYPE_CHECKING:  # avoid a hard import cycle; facets only needed for typing.
    from .facets import FacetKey


# The canonical "no call" value used by ``with_label`` and pooling. A sample that
# a label group does not name is UNASSIGNED — never silently dropped, never a
# SampleSet of its own (residual support is not a coherent group).
UNASSIGNED_LABEL = "__unassigned__"


# --------------------------------------------------------------------------- #
# Shared idioms (mirror core/distribution_records.py — do not reinvent).
# --------------------------------------------------------------------------- #
def _readonly_array(
    values: np.ndarray | list[Any] | tuple[Any, ...],
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    """Return an immutable copy of ``values`` (write flag cleared)."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _readonly_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable view of ``values``."""
    return MappingProxyType(dict(values))


# --------------------------------------------------------------------------- #
# Label columns — sample-aligned partitions (the "how samples split" axis)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelProvenance:
    """How a label column was produced (the SPEC + any eager geometry).

    ``discover_modes`` (TASK_B) fills ``method`` / ``features`` / ``spec`` and,
    per-category, the eager peak geometry consumed by SampleSet rings. Carried-
    through columns (genotype / phenotype from the source frame) leave the
    geometry-bearing fields empty; ``method`` records their origin.
    """

    method: str
    features: tuple[str, ...] = ()
    spec: Mapping[str, Any] = field(default_factory=dict)
    # {category -> geometry payload}; opaque here, typed/consumed downstream.
    geometry: Mapping[Hashable, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "spec", _readonly_mapping(self.spec))
        object.__setattr__(self, "geometry", _readonly_mapping(self.geometry))


@dataclass(frozen=True)
class LabelColumn:
    """A sample-aligned partition attached to a Distribution.

    ``values`` maps ``sample_id -> call``. A sample_id absent from the map is
    treated as :data:`UNASSIGNED_LABEL` by consumers; ``with_label`` normalizes
    coverage so every Distribution sample has an explicit entry.
    """

    name: str
    values: Mapping[str, Hashable]
    provenance: LabelProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly_mapping(self.values))

    def categories(self) -> tuple[Hashable, ...]:
        """Distinct calls in first-appearance order, EXCLUDING unassigned.

        Residual support is not a category (spec §PATH A one-distribution rule):
        unassigned samples do not become a SampleSet and do not stretch a grid.
        """
        seen: list[Hashable] = []
        for call in self.values.values():
            if call == UNASSIGNED_LABEL:
                continue
            if call not in seen:
                seen.append(call)
        return tuple(seen)


# --------------------------------------------------------------------------- #
# Distribution — the typed-column sample table (ONE population)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Distribution:
    """One population: samples + features + label groups + coordinates.

    NOT a generic ``columns: Mapping[str, Column]`` framework — that was
    explicitly rejected as recreating AnnData. Features are a dense
    ``(n_samples, n_features)`` block (col ``j`` ↔ ``feature_names[j]``); label
    groups are sparse categorical partitions; coordinates are distribution-level
    constants.

    ``distribution_id`` is DERIVED from ``coordinates`` (implementation identity
    for cache / provenance / equality — never a plotting concept). Changing
    coordinates yields a new id. Plotting reads ``coordinate(name)``, not the id.
    """

    # --- identity (DERIVED from coordinates; never parsed back) ---
    distribution_id: str
    # --- payload (NATIVE feature values; features ARE the coordinates) ---
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_values: np.ndarray  # (n_samples, n_features); col j <-> feature_names[j]
    # --- metadata ---
    labels: Mapping[str, LabelColumn] = field(default_factory=dict)
    coordinates: Mapping[str, Hashable] = field(default_factory=dict)
    # Names of coordinate(s) collapsed by a pool_by (spec §pool_by "Provenance is
    # in the samples"): the ONE distribution-grain note kept, so compare() won't
    # re-treat a pooled-away coordinate as a coordinate. Empty for un-pooled
    # distributions. All other lineage is recoverable at the SAMPLE grain (labels).
    pooled_coordinates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        values = _readonly_array(self.feature_values, dtype=float)
        if values.ndim != 2:
            raise ValueError(
                f"feature_values must be 2-D (n_samples, n_features); got shape {values.shape}"
            )
        # Hard invariant: feature_values[:, j] <-> feature_names[j].
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                "feature_values second axis must match feature_names length: "
                f"{values.shape[1]} columns vs {len(self.feature_names)} feature_names"
            )
        if values.shape[0] != len(self.sample_ids):
            raise ValueError(
                "feature_values first axis must match sample_ids length: "
                f"{values.shape[0]} rows vs {len(self.sample_ids)} sample_ids"
            )
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "labels", _readonly_mapping(self.labels))
        object.__setattr__(self, "coordinates", _readonly_mapping(self.coordinates))
        object.__setattr__(self, "pooled_coordinates", tuple(self.pooled_coordinates))
        # A sample_id is the unique join key.
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique (they are the join key)")

    # --- coordinate access -------------------------------------------------- #
    def coordinate(self, name: str) -> Hashable:
        """Read one distribution-level coordinate (used by faceting)."""
        try:
            return self.coordinates[name]
        except KeyError as exc:
            raise KeyError(
                f"coordinate {name!r} not on distribution {self.distribution_id!r}; "
                f"have {sorted(self.coordinates)}"
            ) from exc

    def feature_column(self, feature: str) -> np.ndarray:
        """Return the 1-D column of ``feature`` (read-only view helper)."""
        try:
            j = self.feature_names.index(feature)
        except ValueError as exc:
            raise KeyError(
                f"feature {feature!r} not in {list(self.feature_names)}"
            ) from exc
        return self.feature_values[:, j]

    # --- label writers / views (pure; return NEW objects) ------------------- #
    def with_label(
        self,
        name: str,
        assignments: Mapping[str, Hashable],
        *,
        provenance: LabelProvenance | None = None,
    ) -> "Distribution":
        """Attach a label column; return a NEW Distribution (original unchanged).

        Every Distribution sample gets an explicit entry: samples missing from
        ``assignments`` are set to :data:`UNASSIGNED_LABEL`. Assignments naming a
        sample_id not in this Distribution raise (a label can only describe
        samples it actually has).
        """
        dist_samples = set(self.sample_ids)
        stray = set(assignments) - dist_samples
        if stray:
            raise ValueError(
                f"with_label({name!r}): assignments name samples not in this "
                f"distribution: {sorted(stray)}"
            )
        full = {
            sid: assignments.get(sid, UNASSIGNED_LABEL) for sid in self.sample_ids
        }
        column = LabelColumn(name=name, values=full, provenance=provenance)
        new_labels = dict(self.labels)
        new_labels[name] = column
        return replace(self, labels=new_labels)

    def label_column(self, label_name: str) -> LabelColumn:
        """Fetch one attached label column (raises if absent)."""
        try:
            return self.labels[label_name]
        except KeyError as exc:
            raise KeyError(
                f"label {label_name!r} not on distribution {self.distribution_id!r}; "
                f"have {sorted(self.labels)}"
            ) from exc

    def sample_sets(self, label_name: str) -> tuple["SampleSet", ...]:
        """DERIVE one :class:`SampleSet` per category of a label column.

        A DERIVED VIEW, never stored on the object (spec §"What must stay in
        sync": SampleSets are derived from labels, never durable glue). Categories
        appear in first-appearance order; unassigned samples yield NO set. Peak
        geometry, when the column's provenance carries it, is read back onto the
        set so rings survive.
        """
        column = self.label_column(label_name)
        provenance = column.provenance
        geometry_by_cat: Mapping[Hashable, Any] = (
            provenance.geometry if provenance is not None else {}
        )
        members: dict[Hashable, list[str]] = {}
        for sid in self.sample_ids:  # deterministic: Distribution sample order
            call = column.values.get(sid, UNASSIGNED_LABEL)
            if call == UNASSIGNED_LABEL:
                continue
            members.setdefault(call, []).append(sid)

        sets: list[SampleSet] = []
        for category in column.categories():
            name = str(category)
            sets.append(
                SampleSet(
                    sample_set_id=make_sample_set_id(self.distribution_id, name),
                    sample_set_name=name,
                    distribution_id=self.distribution_id,
                    sample_ids=tuple(members.get(category, ())),
                    geometry=geometry_by_cat.get(category),
                )
            )
        return tuple(sets)

    def label_group(
        self, label_name: str, *, display_name: str | None = None
    ) -> "DistributionLabelGroup":
        """Bind "use THIS label group from THIS distribution" (PATH A input)."""
        # Fail fast if the label is absent.
        self.label_column(label_name)
        return DistributionLabelGroup(
            distribution=self,
            label_name=label_name,
            display_name=display_name if display_name is not None else label_name,
        )

    # --- impl-later methods (bodies land in A / B) -------------------------- #
    def detect_peaks(
        self,
        *,
        features: tuple[str, ...],
        output_label: str = "resolved_peak",
        spec: Mapping[str, Any] | None = None,
    ) -> "DistributionLabelGroup":
        """Fit peaks on THIS distribution's own points and write a label column.

        Writes ``output_label`` (e.g. ``resolved_peak``) AND its EAGER peak
        geometry into the column's provenance, then returns the bound
        DistributionLabelGroup (``.distribution`` carries the new column). Body
        implemented in TASK_B (labelers); ``spec`` defaults to
        ``DEFAULT_ANALYSIS_SPEC`` there. Peak ids are LOCAL to this distribution
        (matching ≠ discovery) — no cross-distribution peak_0↔peak_0 claim.
        """
        raise NotImplementedError(
            "Distribution.detect_peaks is implemented in TASK_B (engine/labelers.py)"
        )


# --------------------------------------------------------------------------- #
# Feature -> Grid -> DensityGrid  (features are primitive; unchanged shapes)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    """A chosen discretization over some features. Axes IN FEATURE UNITS.

    ``grid_id`` hashes the ACTUAL produced ``axis_values`` so that
    ``same grid_id <=> same evaluation coordinates`` (raster comparability — a
    PRESERVED invariant). ``build_grid`` (TASK_C reuse) populates it.
    """

    grid_id: str
    feature_names: tuple[str, ...]
    axis_values: tuple[np.ndarray, ...]  # per-axis cell coordinates, feature units
    construction_method: str
    construction_params: Mapping[str, Any] = field(default_factory=dict)
    fit_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(
            self,
            "axis_values",
            tuple(_readonly_array(a, dtype=float) for a in self.axis_values),
        )
        object.__setattr__(self, "construction_params", _readonly_mapping(self.construction_params))
        object.__setattr__(self, "fit_sample_ids", tuple(self.fit_sample_ids))
        if len(self.axis_values) != len(self.feature_names):
            raise ValueError(
                "axis_values must have one axis per feature: "
                f"{len(self.axis_values)} axes vs {len(self.feature_names)} feature_names"
            )


@dataclass(frozen=True)
class DensityGrid:
    """A KDE field evaluated on a :class:`Grid` (carries that grid's ``grid_id``).

    ``density.shape == tuple(len(a) for a in grid.axis_values)`` (checked by the
    central invariants against the owning grid). A KDE strip is the 1-D case.

    PRESERVED invariant: KDE is never fit on a Distribution — densities are
    materialized per facet cell downstream (shared-grid rule).
    """

    grid_id: str
    feature_names: tuple[str, ...]
    density: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "density", _readonly_array(self.density, dtype=float))


# --------------------------------------------------------------------------- #
# SampleSet — a DERIVED view of one label-column category (+ measured shape)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureProfile:
    """Per-feature summary stats, in FEATURE units. Self-describing."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "mean", _readonly_array(self.mean, dtype=float))
        object.__setattr__(self, "std", _readonly_array(self.std, dtype=float))


@dataclass(frozen=True)
class SampleSetGeometry:
    """INTRINSIC geometry only (feature units). Carries ``grid_id`` + ``feature_names``.

    Run-relative scalars (support_fraction / prominence_rank / ...) do NOT live
    here — they belong to the label group's metrics. One grain each.
    """

    grid_id: str
    feature_names: tuple[str, ...]
    center: np.ndarray  # intrinsic only — feature units
    radius: float
    r80: float
    cv_radius_from_center: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "center", _readonly_array(self.center, dtype=float))


@dataclass(frozen=True)
class HDR:
    """Highest-density-region mask. Carries ``grid_id`` explicitly (comparability
    without tracing)."""

    grid_id: str
    feature_names: tuple[str, ...]
    level: float
    mask: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "mask", _readonly_array(self.mask))


@dataclass(frozen=True)
class SampleSet:
    """A DERIVED category of one label group + its measured shape.

    Produced by :meth:`Distribution.sample_sets`, NOT hand-constructed by callers
    (spec §"What must stay in sync"). A genotype subset, a DTW cluster, and a peak
    are the SAME type — they differ only in which optional slots are filled;
    consumers check ``geometry is not None``, never spelunk a dict for a center.

    ``sample_set_id`` is durable + composed (unambiguous across distributions);
    ``sample_set_name`` is the short readable, group-local label.
    """

    sample_set_id: str
    sample_set_name: str
    distribution_id: str
    sample_ids: tuple[str, ...]
    feature_profile: FeatureProfile | None = None
    hdr: HDR | None = None
    geometry: SampleSetGeometry | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(self, "provenance", _readonly_mapping(self.provenance))


# --------------------------------------------------------------------------- #
# DistributionLabelGroup — the plotting bridge ("this label group, this dist")
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistributionLabelGroup:
    """Bind ONE label group to ONE distribution (the PATH A curve source).

    The label group is chosen by the INPUT object — the density-grid builder gets
    NO second grouping parameter. ``coordinate(FacetKey)`` resolves either the
    label-group display axis or a distribution coordinate, so faceting keys off a
    typed key without the plotter knowing the difference.
    """

    distribution: Distribution
    label_name: str
    display_name: str

    def sample_sets(self) -> tuple[SampleSet, ...]:
        return self.distribution.sample_sets(self.label_name)

    def coordinate(self, key: "FacetKey") -> Hashable:
        """Resolve a facet key: label-group axis → display_name, else a coordinate."""
        # Local import to avoid the objects<->facets import cycle at module load.
        from .facets import CoordinateFacet, LabelGroupFacet

        if isinstance(key, LabelGroupFacet):
            return self.display_name
        if isinstance(key, CoordinateFacet):
            return self.distribution.coordinate(key.name)
        raise TypeError(f"unsupported FacetKey: {key!r}")


# --------------------------------------------------------------------------- #
# LabelGroup run-result + label_groups view — KEPT for the labeler/plotting
# layers (TASK_B/C reuse; unchanged from the peer skeleton).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelGroupArtifacts:
    """Shared, heavy field products of a labeler run. Grid is an artifact, not
    provenance."""

    grid_id: str | None = None
    grid: Grid | None = None
    density_grid: DensityGrid | None = None
    basin_labels: np.ndarray | None = None
    detection_result: Any | None = None  # PeakDetectionResult (TASK_B types it)

    def __post_init__(self) -> None:
        if self.basin_labels is not None:
            object.__setattr__(self, "basin_labels", _readonly_array(self.basin_labels))


@dataclass(frozen=True)
class LabelGroup:
    """One labeler RUN over a Distribution (assignments + evidence).

    ``sample_set_ids`` are REAL groups only — ``unassigned_sample_ids`` is a
    FIELD, never a SampleSet: residual support is not a coherent group and
    counting it would inflate the mode count.
    """

    label_group_name: str
    distribution_id: str
    sample_set_ids: tuple[str, ...]
    sample_id_to_sample_set_id: Mapping[str, str]
    unassigned_sample_ids: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    artifacts: LabelGroupArtifacts | None = None
    per_sample_set_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    across_sample_set_metrics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_set_ids", tuple(self.sample_set_ids))
        object.__setattr__(
            self,
            "sample_id_to_sample_set_id",
            _readonly_mapping(self.sample_id_to_sample_set_id),
        )
        object.__setattr__(self, "unassigned_sample_ids", tuple(self.unassigned_sample_ids))
        object.__setattr__(self, "provenance", _readonly_mapping(self.provenance))
        object.__setattr__(
            self,
            "per_sample_set_metrics",
            _readonly_mapping(
                {k: _readonly_mapping(v) for k, v in dict(self.per_sample_set_metrics).items()}
            ),
        )


# {label_group_name: (sample_set_ids,)}
LabelGroups = Mapping[str, tuple[str, ...]]


def derive_label_groups(label_groups_list: list[LabelGroup]) -> LabelGroups:
    """Derive the ``{name: (sample_set_ids,)}`` view from a list of LabelGroups."""
    view: dict[str, tuple[str, ...]] = {}
    for group in label_groups_list:
        if group.label_group_name in view:
            raise ValueError(
                f"duplicate label_group_name {group.label_group_name!r} — cannot "
                "derive an unambiguous label_groups view"
            )
        view[group.label_group_name] = tuple(group.sample_set_ids)
    return MappingProxyType(view)


def resolve_label_group(label_groups: LabelGroups, alias: str) -> tuple[str, ...]:
    """Strict alias resolution: exact name, else a UNIQUE prefix, else raise."""
    if alias in label_groups:
        return label_groups[alias]
    matches = [name for name in label_groups if name.startswith(alias)]
    if len(matches) == 1:
        return label_groups[matches[0]]
    if not matches:
        raise KeyError(f"alias {alias!r} matches no label_group")
    raise ValueError(
        f"alias {alias!r} is ambiguous — matches {sorted(matches)}; "
        "use an exact label_group_name"
    )
