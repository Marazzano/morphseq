"""The four ontology nouns + the ``label_groups`` view.

LOCKED shapes — copied field-for-field from ``docs/PRIMITIVE_ONTOLOGY.md``
(§1 Distribution, §1b Grid/DensityGrid, §2 SampleSet, §3 LabelGroup, §4 label_groups).

Design rule above all else (ontology line 7): keep the objects dumb, don't leak
between layers. Validity / grain / time / binning are the *caller's* job. These
objects only hold what they are.

TASK_0 builds the shapes + the id/invariant machinery only. No grid construction,
no labelers, no comparison, no plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


# --------------------------------------------------------------------------- #
# Shared idioms (mirror core/distribution_records.py:72 — do not reinvent).
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
# §1  Distribution — the dumb substrate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Distribution:
    """A dumb bag: named samples + named features, caller-composed.

    NO ``coordinate_frame`` (ontology §1) — features ARE the primitive
    coordinates, stored in feature units. Comparability is by ``feature_names``
    equality; scientific validity (a shared representation before splitting into
    target/reference) is the *caller's* responsibility, not the object's to infer.

    ``time_bin`` / ``role`` are frozen *identity parts* (which bin/role this bag
    IS), never a live time column to slice. ``sample_ids`` is the caller's grain,
    opaque to the object. ``scope_tag`` is a free human label, never parsed.
    """

    # --- structured identity (composed from typed parts; NEVER parsed back) ---
    distribution_id: str
    scope_id: str
    time_bin: Any
    role: str
    # --- payload (NATIVE feature values; no owned coordinate frame) ---
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_values: np.ndarray  # (n_samples, n_features); col j <-> feature_names[j]
    scope_tag: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        values = _readonly_array(self.feature_values, dtype=float)
        if values.ndim != 2:
            raise ValueError(
                f"feature_values must be 2-D (n_samples, n_features); got shape {values.shape}"
            )
        # Hard invariant (#3): feature_values[:, j] <-> feature_names[j].
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


# --------------------------------------------------------------------------- #
# §1b  Feature -> Grid -> DensityGrid  (features are primitive)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    """A chosen discretization over some features. Axes IN FEATURE UNITS.

    There is NO "canonical" basis and nothing to invert (ontology §1b): whitening
    / quantile only picks bounds and cell spacing, never a change of what the
    numbers mean. ``grid_id`` hashes the ACTUAL produced ``axis_values`` so that
    ``same grid_id <=> same evaluation coordinates`` (raster comparability).

    TASK_0 freezes the shape only; ``build_grid`` (TASK_A) populates it.
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
    """

    grid_id: str
    feature_names: tuple[str, ...]
    density: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "density", _readonly_array(self.density, dtype=float))


# --------------------------------------------------------------------------- #
# §2  SampleSet — the durable atom (+ its typed measured-shape fields)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureProfile:
    """Per-feature summary stats, in FEATURE units.

    Self-describing: carries ``feature_names`` so a profile never needs to trace
    back through provenance to know what its columns mean. Any feature in the
    Distribution's menu can be profiled on any set.
    """

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

    NOTE (ontology §2, #4/#6): run-relative scalars (support_fraction /
    prominence_rank / height_relative_to_max / is_dominant) do NOT live here —
    they belong to ``LabelGroup.per_sample_set_metrics``. One grain each.
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
    """The durable atom: one named subset + its measured shape.

    A genotype subset, a DTW cluster, and a peak are the SAME type — they differ
    only in which optional slots are filled (ontology §2). Consumers check
    ``geometry is not None``, never spelunk a dict for a center.

    ``sample_set_id`` is durable + composed (unambiguous across distributions);
    ``sample_set_name`` is the short readable, LabelGroup-local label.
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
# §3  LabelGroup — one labeler run (thin run-result, NOT a dict)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelGroupArtifacts:
    """Shared, heavy field products of a run (§3). Grid is an artifact, not
    provenance — provenance records the *spec* (bandwidth/resolution), this holds
    the concrete data products."""

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
    FIELD, never a SampleSet (ontology §3): residual support is not a coherent
    group and counting it would inflate the mode count.

    ``per_sample_set_metrics`` — sibling-relative scalars (one value per
    ``sample_set_id``). ``across_sample_set_metrics`` — WITHIN-one-run pairwise +
    summary (needs >=2 sets). Cross-run relations live at the comparison layer.
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


# --------------------------------------------------------------------------- #
# §4  label_groups — a lightweight index (a VIEW, not an object)
# --------------------------------------------------------------------------- #
# {label_group_name: (sample_set_ids,)}
LabelGroups = Mapping[str, tuple[str, ...]]


def derive_label_groups(label_groups_list: list[LabelGroup]) -> LabelGroups:
    """Derive the ``{name: (sample_set_ids,)}`` view from a list of LabelGroups.

    Multiple views coexist for free (two peak bandwidths = two keys). Raises if
    two runs share a ``label_group_name`` — the view maps a name to exactly one
    set of ids (strict alias resolution is done by :func:`resolve_label_group`).
    """
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
    """Strict alias resolution (#8).

    Exact ``label_group_name``s are canonical and returned directly. A non-exact
    alias (e.g. ``"peak"``) resolves ONLY if it identifies exactly one group by
    prefix; an ambiguous alias (``peak_bwA`` + ``peak_bwB`` both present) raises.
    No silent "first peak-ish thing wins."
    """
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
