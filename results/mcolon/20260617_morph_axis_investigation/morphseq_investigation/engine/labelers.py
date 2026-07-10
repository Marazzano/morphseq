"""Labelers — the peer contract (ontology "Labelers are peers", §2, §3).

A labeler is ``label(distribution, method, params) -> (LabelGroup, [SampleSet])``.
``genotype`` (provided / column) is the first peer; ``peak_finding`` (unsupervised)
lands next. Both run the SAME skeleton (:func:`_finalize`), differing only in which
optional SampleSet/LabelGroup slots fill — there is NO ``if genotype ... else if
peak ...`` anywhere: one return path, one ``validate_label_group`` call.

  genotype  : geometry=None, provenance.labeler.feature_names=(), artifacts=None,
              usually unassigned=(). Three distinct beasts kept apart (§2):
                * ``unlabeled`` = a REAL SampleSet (literal category, is_missing_value=False)
                * NA source value -> a REAL SampleSet, is_missing_value=True
                * labeler abstention -> LabelGroup.unassigned_sample_ids (no SampleSet)
              Unknown genotype = a REAL SampleSet named "unknown" with its own HDR.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .identifiers import make_sample_set_id
from .invariants import validate_label_group
from .objects import (
    Distribution,
    LabelGroup,
    LabelGroupArtifacts,
    SampleSet,
)
# A source value counts as "missing" (NA) when it is None or a float NaN. Everything
# else — including the literal string "unlabeled" and "unknown" — is a real category
# and becomes a real SampleSet (§2: unlabeled != NA != abstention).
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))  # float NaN
    except (TypeError, ValueError):
        return False


def _feature_profile_for(distribution: Distribution, member_indices: Sequence[int]):
    """Per-feature mean/std over the members, in FEATURE units (self-describing).

    Any feature in the Distribution's menu can be profiled on any set (§2).
    Returns ``None`` for an empty set (no meaningful stats).
    """
    from .objects import FeatureProfile

    if len(member_indices) == 0:
        return None
    values = distribution.feature_values[np.asarray(member_indices, dtype=int), :]
    return FeatureProfile(
        feature_names=distribution.feature_names,
        mean=values.mean(axis=0),
        std=values.std(axis=0),
    )


# --------------------------------------------------------------------------- #
# The ONE shared skeleton both labelers return through.
# --------------------------------------------------------------------------- #
def _finalize(
    distribution: Distribution,
    label_group_name: str,
    sample_sets: Sequence[SampleSet],
    assignment: Mapping[str, str],
    *,
    unassigned_sample_ids: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
    artifacts: LabelGroupArtifacts | None = None,
    per_sample_set_metrics: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[LabelGroup, list[SampleSet]]:
    """Assemble the LabelGroup, validate centrally, and return the peer tuple.

    Both labelers funnel here — one return path, one ``validate_label_group``
    call (the proof the ontology holds: no per-labeler drift check).
    """
    sample_sets = list(sample_sets)
    label_group = LabelGroup(
        label_group_name=label_group_name,
        distribution_id=distribution.distribution_id,
        sample_set_ids=tuple(s.sample_set_id for s in sample_sets),
        sample_id_to_sample_set_id=dict(assignment),
        unassigned_sample_ids=tuple(unassigned_sample_ids),
        provenance=dict(provenance or {}),
        artifacts=artifacts,
        per_sample_set_metrics=dict(per_sample_set_metrics or {}),
    )
    validate_label_group(distribution, label_group, sample_sets)
    return label_group, sample_sets


# --------------------------------------------------------------------------- #
# genotype (provided / column) labeler
# --------------------------------------------------------------------------- #
def label_genotype(
    distribution: Distribution,
    method: str = "column",
    params: Mapping[str, Any] | None = None,
) -> tuple[LabelGroup, list[SampleSet]]:
    """Provided/column labeler — one SampleSet per category value.

    The column is handed in via ``params`` (the Distribution NEVER parses ids, §1):

      params["labels"]  : a sequence aligned 1:1 with ``distribution.sample_ids``
                          (the resolved label per sample). REQUIRED.
      params["column"]  : the human name of the source column (recorded in
                          provenance; documentation only). Optional.

    NA source values (None / NaN) collapse into a SINGLE real SampleSet whose name
    is ``params.get("missing_name", "unknown")`` with ``evidence.is_missing_value=
    True``. Every other distinct value — INCLUDING the literal strings "unlabeled"
    and "unknown" — is its own real SampleSet (``is_missing_value=False``). A
    genotype labeler does not abstain, so ``unassigned_sample_ids`` is typically
    ``()`` (a caller may pre-place samples in ``params["unassigned"]`` — those are
    the only abstentions).
    """
    if method != "column":
        raise ValueError(f"genotype labeler only implements method='column'; got {method!r}")
    params = dict(params or {})
    if "labels" not in params:
        raise ValueError(
            "genotype labeler requires params['labels'] aligned to distribution.sample_ids"
        )
    labels = list(params["labels"])
    if len(labels) != len(distribution.sample_ids):
        raise ValueError(
            "params['labels'] must align 1:1 with distribution.sample_ids: "
            f"{len(labels)} labels vs {len(distribution.sample_ids)} samples"
        )
    missing_name = str(params.get("missing_name", "unknown"))
    forced_unassigned = set(params.get("unassigned", ()))

    # Group sample indices by their resolved category (NA -> the missing bucket),
    # preserving first-seen category order for stable SampleSet ids.
    category_order: list[str] = []
    category_indices: dict[str, list[int]] = {}
    category_is_missing: dict[str, bool] = {}
    assignment: dict[str, str] = {}
    unassigned: list[str] = []

    for idx, sample_id in enumerate(distribution.sample_ids):
        if sample_id in forced_unassigned:
            unassigned.append(sample_id)
            continue
        raw = labels[idx]
        if _is_missing(raw):
            category = missing_name
            is_missing = True
        else:
            category = str(raw)
            is_missing = False
        if category not in category_indices:
            category_order.append(category)
            category_indices[category] = []
            category_is_missing[category] = is_missing
        # A literal "unknown" string and NA both map to the same bucket name only
        # if the caller uses the default missing_name; if a real "unknown" category
        # coexists with NA, the is_missing flag reflects whichever was seen first —
        # callers wanting them distinct should pass a non-colliding missing_name.
        category_indices[category].append(idx)
        assignment[sample_id] = ""  # filled below once ids are minted

    sample_sets: list[SampleSet] = []
    for category in category_order:
        member_indices = category_indices[category]
        member_ids = tuple(distribution.sample_ids[i] for i in member_indices)
        set_id = make_sample_set_id(distribution.distribution_id, category)
        for sid in member_ids:
            assignment[sid] = set_id
        sample_sets.append(
            SampleSet(
                sample_set_id=set_id,
                sample_set_name=category,
                distribution_id=distribution.distribution_id,
                sample_ids=member_ids,
                feature_profile=_feature_profile_for(distribution, member_indices),
                hdr=None,  # a caller may attach an HDR later; not this labeler's job
                geometry=None,  # provided labels carry no measured center (§2)
                provenance={
                    "labeler": {
                        "method": "column",
                        "params": {"column": params.get("column")},
                        "feature_names": (),  # genotype uses no features to FORM groups
                    },
                    "evidence": {
                        "source_value": category,
                        "is_missing_value": category_is_missing[category],
                    },
                },
            )
        )

    return _finalize(
        distribution,
        label_group_name=str(params.get("label_group_name", "genotype")),
        sample_sets=sample_sets,
        assignment=assignment,
        unassigned_sample_ids=tuple(unassigned),
        provenance={
            "labeler": {
                "method": "column",
                "params": {"column": params.get("column")},
                "feature_names": (),
            }
        },
        artifacts=None,
    )

