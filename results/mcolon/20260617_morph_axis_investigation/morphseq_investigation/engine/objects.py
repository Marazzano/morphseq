"""Immutable value objects for distributions, densities, and label groups.

The authoritative ontology has one durable label representation:
``Distribution.label_groups`` maps names to unified :class:`LabelGroup` values.
Density calculation, density registration, and labeling are separate effects.

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

"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil
from types import MappingProxyType
from typing import Any, Hashable, Literal, Mapping

import numpy as np

from .identifiers import make_sample_set_id
from ..core.peak_stability import (
    PeakCountRobustnessPolicy,
    PeakResolutionSummary,
    PeakVotingSpec,
)
from ..core.distribution_records import PeakResolutionConfig

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
    label_groups: Mapping[str, "LabelGroup"] = field(default_factory=dict)
    densities: tuple["DensityEstimate", ...] = ()
    shared_density_index: int | None = None
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
        object.__setattr__(self, "label_groups", _readonly_mapping(self.label_groups))
        object.__setattr__(self, "densities", tuple(self.densities))
        object.__setattr__(self, "coordinates", _readonly_mapping(self.coordinates))
        object.__setattr__(self, "pooled_coordinates", tuple(self.pooled_coordinates))
        # A sample_id is the unique join key.
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique (they are the join key)")
        if self.shared_density_index is not None and not (
            0 <= self.shared_density_index < len(self.densities)
        ):
            raise ValueError("shared_density_index must index Distribution.densities")
        for density in self.densities:
            self._validate_density_compatibility(density)
        for name, group in self.label_groups.items():
            if name != group.name:
                raise ValueError("label_groups keys must equal LabelGroup.name")
            if group.distribution_id != self.distribution_id:
                raise ValueError("label group distribution identity must match")
            if set(group.assignments) != set(self.sample_ids):
                raise ValueError("label group assignments must cover every sample exactly once")
            if group.density is not None:
                self._validate_density_compatibility(group.density)
            for geometry in group.sample_set_geometries.values():
                if geometry.feature_names != self.feature_names:
                    raise ValueError("geometry ordered feature names must match")

    def _validate_density_compatibility(self, density: "DensityEstimate") -> None:
        if density.distribution_id != self.distribution_id:
            raise ValueError("density distribution identity must match")
        if density.feature_names != self.feature_names:
            raise ValueError("density ordered feature names must match")

    @property
    def shared_density(self) -> "DensityEstimate | None":
        if self.shared_density_index is not None:
            return self.densities[self.shared_density_index]
        return self.densities[0] if self.densities else None

    def calc_density(self, spec: "DensityEstimateSpec | None" = None) -> "DensityEstimate":
        from .density import calculate_density
        return calculate_density(self, spec=spec)

    def with_density(
        self, density: "DensityEstimate", *, select_as_shared: bool = False
    ) -> "Distribution":
        self._validate_density_compatibility(density)
        if any(
            retained.grid.grid_id == density.grid.grid_id and retained.spec == density.spec
            for retained in self.densities
        ):
            raise ValueError("an equivalent density estimate is already registered")
        densities = self.densities + (density,)
        index = len(densities) - 1 if select_as_shared else self.shared_density_index
        return replace(self, densities=densities, shared_density_index=index)

    def select_shared_density(self, index_or_density: int | "DensityEstimate") -> "Distribution":
        """Select one already-retained density without changing any evidence.

        A density value is matched by object identity, deliberately avoiding an
        ambiguous equality comparison over NumPy raster arrays.  Callers that
        persist only the registry position may instead supply its integer index.
        """
        if isinstance(index_or_density, (int, np.integer)):
            index = int(index_or_density)
            if not 0 <= index < len(self.densities):
                raise IndexError("shared density index must index Distribution.densities")
        else:
            matches = [
                index
                for index, retained in enumerate(self.densities)
                if retained is index_or_density
            ]
            if not matches:
                raise ValueError("density estimate is not retained by this distribution")
            index = matches[0]
        return replace(self, shared_density_index=index)

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
        labeling_provenance: "LabelingProvenance | None" = None,
    ) -> "Distribution":
        """Attach a provided-label group; return a new Distribution.

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
        if name in self.label_groups:
            raise ValueError(f"label group {name!r} already exists")
        group = LabelGroup(
            name=name,
            distribution_id=self.distribution_id,
            assignments=full,
            labeling_provenance=labeling_provenance or LabelingProvenance(method="provided"),
        )
        groups = dict(self.label_groups)
        groups[name] = group
        return replace(self, label_groups=groups)

    def get_label_group(self, label_name: str) -> "LabelGroup":
        """Fetch one authoritative unified label group."""
        try:
            return self.label_groups[label_name]
        except KeyError as exc:
            raise KeyError(
                f"label {label_name!r} not on distribution {self.distribution_id!r}; "
                f"have {sorted(self.label_groups)}"
            ) from exc

    def sample_sets(self, label_name: str) -> tuple["SampleSet", ...]:
        """Derive one :class:`SampleSet` per non-unassigned group category.

        A DERIVED VIEW, never stored on the object (spec §"What must stay in
        sync": SampleSets are derived from labels, never durable glue). Categories
        appear in first-appearance order; unassigned samples yield NO set. Peak
        Geometry is transferred directly from the authoritative label group.
        """
        group = self.get_label_group(label_name)
        geometry_by_cat = group.sample_set_geometries
        members: dict[Hashable, list[str]] = {}
        for sid in self.sample_ids:  # deterministic: Distribution sample order
            call = group.assignments[sid]
            if call == UNASSIGNED_LABEL:
                continue
            members.setdefault(call, []).append(sid)

        sets: list[SampleSet] = []
        for category in group.categories():
            name = str(category)
            shape = geometry_by_cat.get(category)
            sets.append(
                SampleSet(
                    sample_set_id=make_sample_set_id(
                        self.distribution_id, f"{group.name}__{name}"
                    ),
                    sample_set_name=name,
                    distribution_id=self.distribution_id,
                    sample_ids=tuple(members.get(category, ())),
                    geometry=shape,
                )
            )
        return tuple(sets)

    def effective_density(self, label_name: str) -> "DensityEstimate | None":
        group = self.get_label_group(label_name)
        return group.density if group.density is not None else self.shared_density

    def detect_peaks(
        self,
        *,
        output_label: str = "resolved_peaks",
        density: "DensityEstimate | None" = None,
        density_spec: "DensityEstimateSpec | None" = None,
        n_draws: int = 80,
        sample_fraction: float = 0.80,
        min_valid_draws: int | None = None,
        min_mode_frequency: float = 0.80,
    ) -> "Distribution":
        """Resolve peaks and immutably attach the resulting unified label group.

        Density selection is explicit and side-effect free: a supplied estimate
        is used directly, a specification creates an analysis-local estimate,
        and otherwise the selected shared estimate is required. Neither density
        calculation nor peak resolution changes the density registry.
        """
        if density is not None and density_spec is not None:
            raise ValueError("density and density_spec are mutually exclusive")
        if output_label in self.label_groups:
            raise ValueError(f"label group {output_label!r} already exists")

        resolved_min_valid_draws = (
            ceil(0.80 * n_draws) if min_valid_draws is None else min_valid_draws
        )
        voting_spec = PeakVotingSpec(
            n_draws=n_draws,
            sample_fraction=sample_fraction,
            min_valid_draws=resolved_min_valid_draws,
        )
        robustness_policy = PeakCountRobustnessPolicy(
            min_mode_frequency=min_mode_frequency
        )
        resolution_config = PeakResolutionConfig(
            n_bootstrap_draws=voting_spec.n_draws,
            bootstrap_sample_fraction=voting_spec.sample_fraction,
            min_valid_draws=voting_spec.min_valid_draws,
            robustness_policy=robustness_policy,
            voting_spec=voting_spec,
        )

        if density is not None:
            selected_density = density
            self._validate_density_compatibility(selected_density)
        elif density_spec is not None:
            selected_density = self.calc_density(density_spec)
        else:
            selected_density = self.shared_density
            if selected_density is None:
                raise ValueError(
                    "peak detection requires density, density_spec, or a shared density"
                )

        from .labelers import detect_peaks as _detect_peaks

        group = _detect_peaks(
            self,
            output_label=output_label,
            density=selected_density,
            resolution_config=resolution_config,
        )
        if group.name != output_label:
            raise ValueError("peak labeler returned a different label-group name")
        if group.distribution_id != self.distribution_id:
            raise ValueError("peak labeler returned a group for another distribution")
        if group.density is not selected_density:
            raise ValueError("peak labeler must retain the exact generating density")

        groups = dict(self.label_groups)
        groups[output_label] = group
        result = replace(self, label_groups=groups)
        if result.densities != self.densities or result.shared_density_index != self.shared_density_index:
            raise RuntimeError("peak detection must not mutate the density registry")
        return result

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
# SampleSet — a derived view of one label-group category
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SampleSetGeometry:
    """Intrinsic per-category geometry in ordered feature units."""

    feature_names: tuple[str, ...]
    center: np.ndarray  # intrinsic only — feature units
    radius: float
    support_fraction: float
    r80_radial_concentration: float
    cv_radius_from_center: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "center", _readonly_array(self.center, dtype=float))
        if self.center.shape != (len(self.feature_names),):
            raise ValueError("geometry center must have one value per feature")
        if self.radius < 0 or not 0 <= self.support_fraction <= 1:
            raise ValueError("geometry radius must be nonnegative and support_fraction in [0, 1]")


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
    geometry: SampleSetGeometry | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))


@dataclass(frozen=True)
class DensityEstimateSpec:
    """Complete, immutable recipe for one retained density estimate."""

    method: str = "isotropic_gaussian_kde"
    bandwidth_rule: str = "longest_non_outlier_MST_edge"
    bandwidth_multiplier: float = 0.75
    grid_method: str = "pooled_min_max"
    grid_params: Mapping[str, Any] = field(default_factory=lambda: {"resolution": 100})

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("density method must be non-empty")
        if not np.isfinite(self.bandwidth_multiplier) or self.bandwidth_multiplier <= 0:
            raise ValueError("bandwidth_multiplier must be finite and positive")
        object.__setattr__(self, "grid_params", _readonly_mapping(self.grid_params))


@dataclass(frozen=True)
class DensityEstimate:
    distribution_id: str
    feature_names: tuple[str, ...]
    spec: DensityEstimateSpec
    grid: Grid
    density_grid: DensityGrid

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        if self.grid.feature_names != self.feature_names:
            raise ValueError("density grid ordered feature names must match estimate")
        if self.density_grid.feature_names != self.feature_names:
            raise ValueError("density field ordered feature names must match estimate")
        if self.grid.grid_id != self.density_grid.grid_id:
            raise ValueError("density grid identity must match its evaluation grid")
        expected = tuple(len(axis) for axis in self.grid.axis_values)
        if self.density_grid.density.shape != expected:
            raise ValueError("density field shape must match evaluation grid")


@dataclass(frozen=True)
class LabelingProvenance:
    method: Literal["provided", "resolved_peaks"]
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", _readonly_mapping(self.detail))


@dataclass(frozen=True)
class LabelGroup:
    """The sole durable representation of one labeling run."""

    name: str
    distribution_id: str
    assignments: Mapping[str, Hashable]
    density: DensityEstimate | None = None
    sample_set_geometries: Mapping[Hashable, SampleSetGeometry] = field(default_factory=dict)
    peak_resolution_summary: PeakResolutionSummary | None = None
    labeling_provenance: LabelingProvenance | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("label group name must be non-empty")
        object.__setattr__(self, "assignments", _readonly_mapping(self.assignments))
        object.__setattr__(self, "sample_set_geometries", _readonly_mapping(self.sample_set_geometries))
        categories = set(self.categories())
        if set(self.sample_set_geometries) - categories:
            raise ValueError("geometry must correspond to an assigned sample set")
        if self.peak_resolution_summary is not None:
            if self.density is None:
                raise ValueError("resolved peaks must retain their generating density")
            if self.peak_resolution_summary.resolved_peak_count != len(categories):
                raise ValueError("resolved peak count must equal materialized sample-set count")
            if set(self.sample_set_geometries) != categories:
                raise ValueError("every resolved peak assignment requires geometry")
        if self.labeling_provenance is not None and self.labeling_provenance.method == "resolved_peaks":
            if self.peak_resolution_summary is None or self.density is None:
                raise ValueError("resolved-peak provenance requires summary and generating density")
            expected_ids = tuple(f"peak_{i}" for i in range(len(categories)))
            if self.categories() != expected_ids:
                raise ValueError("resolved peak IDs must be deterministic local peak_0..peak_n")

    def categories(self) -> tuple[Hashable, ...]:
        seen: list[Hashable] = []
        for call in self.assignments.values():
            if call != UNASSIGNED_LABEL and call not in seen:
                seen.append(call)
        return tuple(seen)

    @property
    def sample_set_count(self) -> int:
        return len(self.categories())

    @property
    def peak_count(self) -> int | None:
        return None if self.peak_resolution_summary is None else self.peak_resolution_summary.resolved_peak_count

    @property
    def is_robust(self) -> bool | None:
        return None if self.peak_resolution_summary is None else self.peak_resolution_summary.is_robust
