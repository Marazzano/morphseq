"""``DistributionCatalog`` — the control tower (spec §"Construction and labeling
are SEPARATE steps", §"ID helpers", §"pool_by").

The catalog answers "what populations exist". It imports the FROZEN TASK_0
shapes from :mod:`engine.objects` / :mod:`engine.identifiers` and never
redefines them.

Two metadata types, two catalog jobs (mirrors the ontology split):
    coordinates  ->  which distribution you have  ->  split / find_ids / compare
    label groups ->  how samples inside it split   ->  label_groups (PATH A)

This slice adds ``pool_by`` — the second half of the construction idiom (spec
§"pool_by — the second half of the construction idiom (BUILD NOW)"): the honest
split keeps a nuisance coordinate (e.g. ``experiment``) SEPARATE so it is never
silently confounded, then ``pool_by`` collapses it for the actual comparison.
``compare`` / catalog-wide labeling conveniences / ``label_groups`` land in a
later commit on this same branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from .identifiers import make_distribution_id
from .objects import UNASSIGNED_LABEL, Distribution, LabelProvenance


def _readonly_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Same idiom as ``engine.objects`` — immutable view, not a naked dict."""
    return MappingProxyType(dict(values))


# --------------------------------------------------------------------------- #
# DistributionCatalog — coordinate-indexed collection of populations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistributionCatalog:
    """A coordinate-indexed collection of :class:`~engine.objects.Distribution`.

    ``coordinate_names`` is the catalog's OWN record of which coordinate keys it
    was split on (spec §"What must stay in sync": "the catalog must know its
    coordinate names") — it is what ``compare()``'s default ``match_on`` and the
    "is X a coordinate or a label" guard read, independent of any single
    Distribution's ``coordinates`` mapping (which could theoretically drift).
    """

    distributions: tuple[Distribution, ...]
    coordinate_names: tuple[str, ...] = ()
    # Softened pool_by provenance note (spec §"Provenance is in the samples"):
    # {distribution_id: (collapsed_coordinate_names,)}. NOT a Distribution field
    # (that shape is frozen by TASK_0) — kept here so callers can still see
    # "this distribution came from a pool_by" without re-deriving it, while the
    # REAL lineage (which experiment each sample came from) stays in the
    # samples' own labels, never duplicated here.
    pooled_coordinates: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "distributions", tuple(self.distributions))
        object.__setattr__(self, "coordinate_names", tuple(self.coordinate_names))
        object.__setattr__(
            self, "pooled_coordinates", _readonly_mapping(self.pooled_coordinates)
        )

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        sample_id_column: str,
        feature_columns: Sequence[str],
        label_columns: Sequence[str] = (),
        split_columns: Sequence[str] = (),
    ) -> "DistributionCatalog":
        """Build one :class:`Distribution` per unique ``split_columns`` combo.

        ``df`` is already tidy (spec §"Scope": "Binning/cleaning is UPSTREAM ...
        the catalog never bins"). ``label_columns`` are attached best-effort via
        ``Distribution.with_label`` — a sample lacking a value for a label
        becomes :data:`UNASSIGNED_LABEL`, never silently dropped.

        With ``split_columns=()`` the whole frame becomes ONE Distribution with
        empty ``coordinates`` (a valid degenerate case — ``compare()`` handles a
        catalog with zero split coordinates as the "one global comparison"
        case, §"PATH B" algorithm step 3).
        """
        feature_columns = tuple(feature_columns)
        label_columns = tuple(label_columns)
        split_columns = tuple(split_columns)

        distributions: list[Distribution] = []
        if split_columns:
            grouped = df.groupby(list(split_columns), sort=False, dropna=False)
            groups = list(grouped)
        else:
            groups = [((), df)]

        for group_key, group_df in groups:
            if not split_columns:
                coordinates: dict[str, Any] = {}
            elif len(split_columns) == 1:
                coordinates = {split_columns[0]: group_key}
            else:
                coordinates = dict(zip(split_columns, group_key))

            sample_ids = tuple(group_df[sample_id_column].tolist())
            feature_values = group_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
            distribution = Distribution(
                distribution_id=make_distribution_id(coordinates),
                sample_ids=sample_ids,
                feature_names=feature_columns,
                feature_values=feature_values,
                coordinates=coordinates,
            )

            for label_name in label_columns:
                assignments = {
                    sid: value
                    for sid, value in zip(sample_ids, group_df[label_name].tolist())
                    if pd.notna(value)
                }
                distribution = distribution.with_label(
                    label_name,
                    assignments,
                    provenance=LabelProvenance(method="from_dataframe", features=()),
                )

            distributions.append(distribution)

        return cls(distributions=tuple(distributions), coordinate_names=split_columns)

    # ----------------------------------------------------------------- #
    # Index / discovery (secondary surface — below compare())
    # ----------------------------------------------------------------- #
    def to_index_dataframe(self) -> pd.DataFrame:
        """One row per distribution, one column per catalog coordinate.

        The key debugging affordance (spec §"ID helpers") — filterable /
        queryable without touching feature data.
        """
        rows = []
        for distribution in self.distributions:
            row = {"distribution_id": distribution.distribution_id}
            for name in self.coordinate_names:
                row[name] = distribution.coordinates.get(name)
            rows.append(row)
        columns = ["distribution_id", *self.coordinate_names]
        return pd.DataFrame(rows, columns=columns)

    def _label_names(self) -> set[str]:
        names: set[str] = set()
        for distribution in self.distributions:
            names.update(distribution.labels)
        return names

    def _check_coords_not_labels(self, coords: Mapping[str, Any]) -> None:
        label_names = self._label_names()
        for key in coords:
            if key in label_names and key not in self.coordinate_names:
                raise ValueError(
                    f"{key!r} is a label group, not a coordinate — find_ids/"
                    "resolve_id search COORDINATES only. Use label_groups"
                    f"({key!r}) / label_group({key!r}) to select inside a "
                    "distribution instead."
                )

    def find_ids(self, **coords: Any) -> tuple[str, ...]:
        """Search COORDINATES only; raise a helpful message for a label kwarg."""
        self._check_coords_not_labels(coords)
        matches = []
        for distribution in self.distributions:
            if all(
                key in distribution.coordinates and distribution.coordinates[key] == value
                for key, value in coords.items()
            ):
                matches.append(distribution.distribution_id)
        return tuple(matches)

    def resolve_id(self, **coords: Any) -> str:
        """Exactly one match or raise (ambiguous / none)."""
        matches = self.find_ids(**coords)
        if not matches:
            raise ValueError(f"resolve_id({coords!r}): no distribution matches")
        if len(matches) > 1:
            raise ValueError(
                f"resolve_id({coords!r}): ambiguous — {len(matches)} distributions "
                f"match: {matches}"
            )
        return matches[0]

    # ----------------------------------------------------------------- #
    # pool_by — sample pooling (BUILD NOW, the 2nd half of the construction idiom)
    # ----------------------------------------------------------------- #
    def pool_by(self, coordinate: str) -> "DistributionCatalog":
        """Collapse ``coordinate`` by concatenating sample-aligned rows.

        For distributions differing ONLY in ``coordinate``: union their samples
        into one Distribution and drop ``coordinate`` from the coordinate map
        (new ``distribution_id`` re-derives from the reduced coordinates).
        Density is not stored, so nothing to re-fit (spec §"pool_by").

        Guards:
          - duplicate ``sample_id`` across pooled sources RAISES (no auto-dedup;
            join-key integrity, spec §"Provenance is in the samples").
          - labels ride along per-sample (concatenated; ``UNASSIGNED_LABEL``
            where a source lacks that label group) — never merged by category
            name.
        """
        if coordinate not in self.coordinate_names:
            raise ValueError(
                f"pool_by({coordinate!r}): not a catalog coordinate; have "
                f"{list(self.coordinate_names)}"
            )
        remaining = tuple(name for name in self.coordinate_names if name != coordinate)

        groups: dict[tuple[Any, ...], list[Distribution]] = {}
        group_order: list[tuple[Any, ...]] = []
        for distribution in self.distributions:
            key = tuple(distribution.coordinates.get(name) for name in remaining)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(distribution)

        pooled: list[Distribution] = []
        new_pooled_notes: dict[str, tuple[str, ...]] = {}
        for key in group_order:
            members = groups[key]
            new_distribution = _pool_distributions(members, remaining, key)
            pooled.append(new_distribution)
            # Carry forward any UPSTREAM pooled coordinates (pooling twice) plus
            # this one — the note is additive, never overwritten.
            prior: tuple[str, ...] = ()
            for member in members:
                prior = tuple(
                    dict.fromkeys(
                        prior + self.pooled_coordinates.get(member.distribution_id, ())
                    )
                )
            new_pooled_notes[new_distribution.distribution_id] = tuple(
                dict.fromkeys(prior + (coordinate,))
            )

        return DistributionCatalog(
            distributions=tuple(pooled),
            coordinate_names=remaining,
            pooled_coordinates=new_pooled_notes,
        )


def _pool_distributions(
    members: Sequence[Distribution],
    remaining_coordinate_names: tuple[str, ...],
    remaining_key: tuple[Any, ...],
) -> Distribution:
    """Concatenate ``members``' sample-aligned rows into ONE Distribution.

    Implements spec §"pool_by" / §"Provenance is in the samples": duplicate
    ``sample_id`` across sources raises; labels ride along per-sample
    (``UNASSIGNED_LABEL`` where a source lacks a given label group); only a
    ``pooled_coordinates`` note survives (kept on the CATALOG, not this object).
    """
    if not members:
        raise ValueError("pool_by: empty member group (nothing to pool)")

    feature_names = members[0].feature_names
    for member in members:
        if tuple(member.feature_names) != tuple(feature_names):
            raise ValueError(
                "pool_by: cannot pool distributions with different feature_names: "
                f"{member.feature_names!r} != {feature_names!r}"
            )

    seen_ids: dict[str, str] = {}
    for member in members:
        for sid in member.sample_ids:
            if sid in seen_ids:
                raise ValueError(
                    f"pool_by: duplicate sample_id {sid!r} across pooled sources "
                    f"({seen_ids[sid]!r} and {member.distribution_id!r}) — "
                    "join-key integrity would be ambiguous; no auto-dedup."
                )
            seen_ids[sid] = member.distribution_id

    all_sample_ids = tuple(sid for member in members for sid in member.sample_ids)
    all_feature_values = np.concatenate(
        [member.feature_values for member in members], axis=0
    )

    new_coordinates = dict(zip(remaining_coordinate_names, remaining_key))
    pooled = Distribution(
        distribution_id=make_distribution_id(new_coordinates),
        sample_ids=all_sample_ids,
        feature_names=feature_names,
        feature_values=all_feature_values,
        coordinates=new_coordinates,
    )

    # Labels ride along per-sample: union of label names across sources;
    # concatenate assignments; UNASSIGNED_LABEL where a source lacks the group.
    label_names: list[str] = []
    for member in members:
        for name in member.labels:
            if name not in label_names:
                label_names.append(name)

    for label_name in label_names:
        assignments: dict[str, Hashable] = {}
        for member in members:
            if label_name in member.labels:
                column = member.label_column(label_name)
                for sid in member.sample_ids:
                    value = column.values.get(sid, UNASSIGNED_LABEL)
                    if value != UNASSIGNED_LABEL:
                        assignments[sid] = value
            # else: this source has no such label group -> its samples stay
            # unassigned for label_name (with_label's default fill handles it).
        pooled = pooled.with_label(
            label_name,
            assignments,
            provenance=LabelProvenance(method="pool_by", features=()),
        )

    return pooled
