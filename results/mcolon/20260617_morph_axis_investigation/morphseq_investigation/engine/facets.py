"""Typed facet keys — NOT an enum (spec §"Typed facet keys").

Coordinates are open-ended (``time_bin``, ``batch``, ``plate_id``, ``stage``, …),
so an enum rots. A typed key says WHERE the facet value comes from; the plotter
resolves it from the paired ``Distribution``/``LabelGroup`` without knowing the
difference between a coordinate and the label-group display axis.

``ComparisonMemberFacet`` is DEFERRED (spec §Deferred) — ship ``CoordinateFacet``
+ ``LabelGroupFacet`` only. Member identity lives in ``CurveKey`` until a plot
actually needs to facet WT-vs-b9d2 across cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class CoordinateFacet:
    """Facet on a distribution coordinate (e.g. ``CoordinateFacet("time_bin")``)."""

    name: str


@dataclass(frozen=True)
class LabelGroupFacet:
    """Facet on the label-group display axis (resolves to the group's display_name)."""


# ComparisonMemberFacet is DEFERRED — do not add it here.
FacetKey = Union[CoordinateFacet, LabelGroupFacet]
