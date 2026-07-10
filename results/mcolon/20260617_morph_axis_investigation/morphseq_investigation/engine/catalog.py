"""``DistributionCatalog`` — the control tower (spec §"Construction and labeling
are SEPARATE steps", §"ID helpers").

The catalog answers "what populations exist". It imports the FROZEN TASK_0
shapes from :mod:`engine.objects` / :mod:`engine.identifiers` and never
redefines them.

Two metadata types, two catalog jobs (mirrors the ontology split):
    coordinates  ->  which distribution you have  ->  split / find_ids / compare
    label groups ->  how samples inside it split   ->  label_groups (PATH A)

This first slice builds construction (``from_dataframe``) + the coordinate
index/lookup surface (``to_index_dataframe`` / ``find_ids`` / ``resolve_id``).
``pool_by`` / ``compare`` / catalog-wide labeling conveniences / ``label_groups``
land in later commits on this same branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from .identifiers import make_distribution_id
from .objects import Distribution, LabelProvenance


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
