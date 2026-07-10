"""``DistributionCatalog`` — the control tower (spec §"Construction and labeling
are SEPARATE steps", §"PATH B — cross-population", §"pool_by", §"ID helpers").

The catalog answers "what populations exist" and turns them into matched
comparisons. It imports the FROZEN TASK_0 shapes from :mod:`engine.objects` /
:mod:`engine.identifiers` / :mod:`engine.facets` and never redefines them.

Two metadata types, two catalog jobs (mirrors the ontology split):
    coordinates  ->  which distribution you have  ->  split / find_ids / compare
    label groups ->  how samples inside it split   ->  label_groups (PATH A)

Construction (``from_dataframe``) and labeling (``with_labels`` / ``detect_peaks``
/ ``map_distributions``) are separate steps — a caller builds the catalog, THEN
attaches label columns, mirroring the spec's two-line example.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from .facets import CoordinateFacet, LabelGroupFacet  # noqa: F401  (re-exported for callers)
from .identifiers import make_distribution_id
from .objects import (
    UNASSIGNED_LABEL,
    Distribution,
    DistributionLabelGroup,
    LabelProvenance,
)

logger = logging.getLogger(__name__)


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "distributions", tuple(self.distributions))
        object.__setattr__(self, "coordinate_names", tuple(self.coordinate_names))

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

    def _pooled_away_names(self) -> set[str]:
        """Coordinate names any distribution collapsed via pool_by (read off the
        distributions' own ``pooled_coordinates``, so the fact travels with the
        objects). compare() must not offer these as across/match_on axes."""
        names: set[str] = set()
        for distribution in self.distributions:
            names.update(distribution.pooled_coordinates)
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
        for key in group_order:
            members = groups[key]
            pooled.append(_pool_distributions(members, coordinate, remaining, key))

        return DistributionCatalog(
            distributions=tuple(pooled), coordinate_names=remaining
        )

    # ----------------------------------------------------------------- #
    # Catalog-wide labeling conveniences
    # ----------------------------------------------------------------- #
    def map_distributions(
        self, fn: Callable[[Distribution], Distribution]
    ) -> "DistributionCatalog":
        """Generic ESCAPE HATCH: apply a ``Distribution -> Distribution`` fn to
        every distribution. The only place a lambda is appropriate (spec
        §"Catalog-wide labeling conveniences") — ``with_labels`` / ``detect_peaks``
        are the named conveniences callers should reach for first.
        """
        new_distributions = tuple(fn(distribution) for distribution in self.distributions)
        return DistributionCatalog(
            distributions=new_distributions, coordinate_names=self.coordinate_names
        )

    def with_labels(
        self, df: pd.DataFrame, label_columns: Sequence[str]
    ) -> "DistributionCatalog":
        """Attach label columns catalog-wide, keyed by sample_id (thin wrapper
        over :meth:`map_distributions` — the caller never writes a lambda).

        ``df`` must carry a ``sample_id`` index or column that matches each
        distribution's ``sample_ids``; only rows for that distribution's
        samples are consulted (best-effort — missing rows -> unassigned).
        """
        label_columns = tuple(label_columns)
        if "sample_id" in df.columns:
            lookup = df.set_index("sample_id")
        else:
            lookup = df

        def _attach(distribution: Distribution) -> Distribution:
            new = distribution
            for label_name in label_columns:
                assignments = {
                    sid: lookup.loc[sid, label_name]
                    for sid in distribution.sample_ids
                    if sid in lookup.index and pd.notna(lookup.loc[sid, label_name])
                }
                new = new.with_label(
                    label_name,
                    assignments,
                    provenance=LabelProvenance(method="with_labels", features=()),
                )
            return new

        return self.map_distributions(_attach)

    def detect_peaks(
        self,
        *,
        features: Sequence[str],
        output_label: str = "resolved_peak",
        spec: Mapping[str, Any] | None = None,
    ) -> "DistributionCatalog":
        """Thin wrapper over :meth:`map_distributions` calling
        ``Distribution.detect_peaks`` (TASK_B fills that stub's body) on every
        distribution with NATIVE arguments — the caller never writes a lambda.
        """
        features = tuple(features)

        def _detect(distribution: Distribution) -> Distribution:
            label_group = distribution.detect_peaks(
                features=features, output_label=output_label, spec=spec
            )
            return label_group.distribution

        return self.map_distributions(_detect)

    # ----------------------------------------------------------------- #
    # label_groups — PATH A convenience
    # ----------------------------------------------------------------- #
    def label_groups(
        self, label_name: str, *, display_name: str | None = None
    ) -> tuple[DistributionLabelGroup, ...]:
        """One :class:`DistributionLabelGroup` per distribution IN the catalog
        (each wraps that distribution + this label). Distributions missing the
        label are skipped (best-effort — a label attached to only some
        distributions is a valid, if partial, catalog state)."""
        groups = []
        for distribution in self.distributions:
            if label_name not in distribution.labels:
                continue
            groups.append(
                distribution.label_group(label_name, display_name=display_name)
            )
        return tuple(groups)

    # ----------------------------------------------------------------- #
    # compare() — the matched-comparison engine (PATH B)
    # ----------------------------------------------------------------- #
    def compare(
        self,
        across: str,
        *,
        values: Sequence[Hashable] | None = None,
        match_on: Sequence[str] | None = None,
    ) -> "DistributionComparisons":
        """Group by everything-but-``across``, vary ``across`` (spec §"PATH B").

        See :mod:`engine.catalog` module docstring / ``docs/tasks_catalog/
        TASK_A_catalog.md`` for the full 7-step algorithm this implements.
        """
        # 1. across must be a catalog COORDINATE. A name that a pool_by
        #    collapsed away (recorded on each distribution's pooled_coordinates)
        #    is NOT a coordinate anymore — reject it with a targeted message so
        #    the caller doesn't try to resolve across an axis they pooled out.
        pooled_away = self._pooled_away_names()
        if across not in self.coordinate_names:
            label_names = self._label_names()
            if across in pooled_away:
                raise ValueError(
                    f"{across!r} was collapsed by pool_by — it is no longer a "
                    "coordinate to resolve across. Rebuild the catalog without "
                    f"pooling {across!r} (keep it in split_columns) if you need "
                    "to compare across it."
                )
            if across in label_names:
                raise ValueError(
                    f"{across!r} is a label group, not a coordinate — split on "
                    f"it (add {across!r} to split_columns at construction), or "
                    f"use build_1d_density_grid with label_group({across!r})."
                )
            raise ValueError(
                f"{across!r} is not a catalog coordinate; have "
                f"{list(self.coordinate_names)}"
            )

        # 2. match_on default = ALL catalog coordinates except across.
        default_match_on = tuple(name for name in self.coordinate_names if name != across)
        if match_on is None:
            resolved_match_on = default_match_on
        else:
            resolved_match_on = tuple(match_on)
            # An explicit match_on may not name a coordinate that pool_by
            # collapsed away — it is no longer available to hold constant.
            bad_match = tuple(name for name in resolved_match_on if name in pooled_away)
            if bad_match:
                raise ValueError(
                    f"match_on names pooled-away coordinate(s) {list(bad_match)} "
                    "— they were collapsed by pool_by and are no longer "
                    "coordinates. Remove them from match_on."
                )

        # 6. Omitting a coordinate from an EXPLICIT match_on asserts it is
        #    constant in the selection (checked per-group below, not pooling).
        omitted = tuple(
            name for name in default_match_on if name not in resolved_match_on
        )

        logger.info(
            "compare(): matching on %s; resolving across %s",
            resolved_match_on,
            across,
        )

        # 3. Group distributions by match_on values. Zero match_on columns ->
        #    ONE global comparison keyed by {} (valid).
        groups: dict[tuple[Any, ...], list[Distribution]] = {}
        group_order: list[tuple[Any, ...]] = []
        for distribution in self.distributions:
            key = tuple(distribution.coordinates.get(name) for name in resolved_match_on)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(distribution)

        # Determine the values order: explicit `values` wins (preserve order,
        # step 5); otherwise infer from first-appearance across the catalog.
        if values is not None:
            resolved_values = tuple(values)
        else:
            seen: list[Hashable] = []
            for distribution in self.distributions:
                value = distribution.coordinates.get(across)
                if value not in seen:
                    seen.append(value)
            resolved_values = tuple(seen)

        comparisons: list[DistributionComparison] = []
        for key in group_order:
            members_in_group = groups[key]

            # 6. Omitted-but-varying coordinate check: for every omitted
            #    coordinate, all distributions in this match_on group must
            #    agree on its value — otherwise this group would silently
            #    produce duplicate `across` members.
            for omitted_name in omitted:
                observed = {d.coordinates.get(omitted_name) for d in members_in_group}
                if len(observed) > 1:
                    raise ValueError(
                        f"Cannot omit coordinate {omitted_name!r}: multiple "
                        f"values {sorted(observed, key=str)} would produce "
                        f"duplicate {across!r} members. Filter the catalog "
                        f"first or include {omitted_name!r} in match_on."
                    )

            # 4. Index this group's distributions by across value.
            by_across: dict[Hashable, list[Distribution]] = {}
            for distribution in members_in_group:
                by_across.setdefault(distribution.coordinates.get(across), []).append(
                    distribution
                )

            # 5. Require exactly one distribution per requested value.
            members: dict[Hashable, Distribution] = {}
            for value in resolved_values:
                candidates = by_across.get(value, [])
                if not candidates:
                    raise ValueError(
                        f"compare(across={across!r}): missing member for "
                        f"{across}={value!r} in group "
                        f"{dict(zip(resolved_match_on, key))}"
                    )
                if len(candidates) > 1:
                    raise ValueError(
                        f"compare(across={across!r}): duplicate members for "
                        f"{across}={value!r} in group "
                        f"{dict(zip(resolved_match_on, key))} "
                        f"({len(candidates)} distributions match)"
                    )
                members[value] = candidates[0]

            comparisons.append(
                DistributionComparison(
                    coordinates=dict(zip(resolved_match_on, key)),
                    members=members,
                )
            )

        return DistributionComparisons(
            comparisons=tuple(comparisons),
            across=across,
            values=resolved_values,
            match_on=resolved_match_on,
        )


def _pool_distributions(
    members: Sequence[Distribution],
    pooled_coordinate: str,
    remaining_coordinate_names: tuple[str, ...],
    remaining_key: tuple[Any, ...],
) -> Distribution:
    """Concatenate ``members``' sample-aligned rows into ONE Distribution.

    Implements spec §"pool_by" / §"Provenance is in the samples": duplicate
    ``sample_id`` across sources raises; labels ride along per-sample
    (``UNASSIGNED_LABEL`` where a source lacks a given label group); only a
    ``pooled_coordinates`` note survives — kept ON the pooled Distribution (so
    the note travels with the object outside its catalog), accumulating
    ``pooled_coordinate`` onto whatever the sources already collapsed.
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

    # Accumulate the collapsed-coordinate note: any coordinate(s) the SOURCES
    # already pooled away (repeated pool_by) + this one. distribution_id is NOT
    # affected — it derives from `coordinates` only (§"What must stay in sync").
    prior_pooled: list[str] = []
    for member in members:
        for name in member.pooled_coordinates:
            if name not in prior_pooled:
                prior_pooled.append(name)
    accumulated_pooled = tuple(
        dict.fromkeys([*prior_pooled, pooled_coordinate])
    )

    new_coordinates = dict(zip(remaining_coordinate_names, remaining_key))
    pooled = Distribution(
        distribution_id=make_distribution_id(new_coordinates),
        sample_ids=all_sample_ids,
        feature_names=feature_names,
        feature_values=all_feature_values,
        coordinates=new_coordinates,
        pooled_coordinates=accumulated_pooled,
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


# --------------------------------------------------------------------------- #
# DistributionComparison(s) — PATH B analysis structures (NOT plotting objects)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistributionComparison:
    """ONE matched group: held-constant coordinates + members indexed by the
    ``across`` value (ORDERED — spec §"Minimal shapes"). No ``comparison_id``
    (deferred, spec §"Deferred")."""

    coordinates: Mapping[str, Hashable]
    members: Mapping[Hashable, Distribution]

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", _readonly_mapping(self.coordinates))
        object.__setattr__(self, "members", _readonly_mapping(self.members))


@dataclass(frozen=True)
class DistributionComparisons:
    """The family :meth:`DistributionCatalog.compare` returns.

    Invariant (checked): ``tuple(c.members) == values`` for every comparison
    ``c`` — order-preserving, so downstream plots/tables never silently
    misalign (spec §"What must stay in sync").
    """

    comparisons: tuple[DistributionComparison, ...]
    across: str
    values: tuple[Hashable, ...]
    match_on: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparisons", tuple(self.comparisons))
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "match_on", tuple(self.match_on))
        for comparison in self.comparisons:
            if tuple(comparison.members) != self.values:
                raise ValueError(
                    "DistributionComparisons invariant violated: "
                    f"tuple(comparison.members)={tuple(comparison.members)!r} != "
                    f"values={self.values!r}"
                )
