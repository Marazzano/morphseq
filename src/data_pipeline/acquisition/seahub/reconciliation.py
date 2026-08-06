"""Audited SeaHub image-to-collection metadata reconciliation.

The source workbook is never modified.  Existing successful matches are retained;
only ``unmatched_stage`` and ``unmatched_condition`` rows are remediated, as
required by ``SEAHUB_RECONCILIATION_ADDENDUM.md``.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .stages import normalize_stage_label, stage_label_to_hpf
from .xlsx_reader import read_xlsx_sheet

REMEDIABLE_MATCH_STATUSES: frozenset[str] = frozenset(
    {"unmatched_stage", "unmatched_condition"}
)
PASS_THROUGH_FAILURE_STATUSES: frozenset[str] = REMEDIABLE_MATCH_STATUSES
INCLUDABLE_MATCH_STATUSES: frozenset[str] = frozenset(
    {"exact", "fuzzy", *PASS_THROUGH_FAILURE_STATUSES}
)

_EXPERIMENT_PREFIX_RE = re.compile(
    r"^\s*((?:GENE|CHEM)\d+)(?=$|[^A-Za-z0-9])", re.IGNORECASE
)
_EXPLICIT_NOT_COLLECTED_RE = re.compile(
    r"not[\s._-]*collected", re.IGNORECASE
)
_EXPLICIT_NOT_SEQUENCED_RE = re.compile(
    r"not[\s._-]*sequenced", re.IGNORECASE
)
_ABANDONED_RE = re.compile(r"abandon(?:ed)?", re.IGNORECASE)
_NOT_USED_RE = re.compile(r"not[\s._-]*used", re.IGNORECASE)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold() in {"", "nan", "none", "#no match"}


def _is_true(value: Any) -> bool:
    """Interpret source booleans without treating NaN as true."""
    if _is_missing(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _snake_case(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip())
    return re.sub(r"_+", "_", text).strip("_").lower()


def _expand_paralog_shorthand(text: str) -> str:
    """Expand paralog shorthand where a bare numeric suffix inherits the letter
    prefix of the preceding gene, so image-side forms match the workbook's full
    names: ``meis1a,1b`` -> ``meis1a;meis1b``, ``pax3a;3b;7a;7b`` ->
    ``pax3a;pax3b;pax7a;pax7b``, ``twist1a;1b;2`` -> ``twist1a;twist1b;twist2``.

    Only strings with a ``;``/``,`` separator are touched, and a suffix is only
    prefixed when a preceding gene supplies one, so single tokens and chemicals
    (e.g. ``2-dg``, ``treatment_28c_14hpf_wash``) are returned unchanged.
    """
    parts = re.split(r"[;,]", text)
    if len(parts) < 2:
        return text
    prefix = ""
    expanded: list[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        head = re.match(r"^([a-z]+)", token)
        if head:  # a full gene token — becomes the prefix subsequent suffixes inherit
            prefix = head.group(1)
            expanded.append(token)
        elif prefix:  # a bare numeric suffix — inherit the current gene prefix
            expanded.append(prefix + token)
        else:  # no gene prefix seen yet — leave as-is (e.g. a numeric-led chemical)
            expanded.append(token)
    return ";".join(expanded)


def canonical_condition(value: Any) -> str | None:
    """Build the existing separator- and order-tolerant condition key."""
    if _is_missing(value):
        return None
    text = str(value).casefold().replace("\n", " ")
    text = _expand_paralog_shorthand(text)
    tokens = re.findall(r"[a-z]+[a-z0-9]*|\d+", text)
    ignored = {
        "cropped",
        "uncropped",
        "closeup",
        "image",
        "images",
        "hpf",
        "pos",
        "neg",
    }
    filtered = [token for token in tokens if token not in ignored]
    return "|".join(sorted(filtered)) or None


def load_collection_metadata(path: str | Path) -> pd.DataFrame:
    """Read and normalize the authoritative SeaHub collection workbook."""
    path = Path(path)
    metadata = read_xlsx_sheet(path, sheet_name="collection_metadata")
    metadata.columns = [_snake_case(column) for column in metadata.columns]
    metadata = metadata.dropna(how="all").reset_index(drop=True)
    metadata.insert(0, "metadata_row_number", metadata.index + 2)
    if "collection_date" in metadata:
        metadata["collection_date_excel_raw"] = metadata["collection_date"]
        raw_dates = metadata["collection_date"].copy()
        numeric = pd.to_numeric(raw_dates, errors="coerce")
        numeric_dates = pd.to_datetime(
            numeric,
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
        text_dates = pd.to_datetime(
            raw_dates.where(numeric.isna()), errors="coerce"
        )
        metadata["collection_date"] = numeric_dates.fillna(text_dates).dt.date
    return metadata


def _collection_name_tokens(collection_name: Any) -> list[str]:
    if _is_missing(collection_name):
        return []
    return [
        token.strip()
        for token in re.sub(r"[\r\n]+", "", str(collection_name)).split("_")
        if token.strip()
    ]


def _collection_name_experiment(collection_name: Any) -> str | None:
    text = "" if _is_missing(collection_name) else str(collection_name)
    match = _EXPERIMENT_PREFIX_RE.match(text)
    return match.group(1).upper() if match else None


def _stage_from_treatment_token(token: str) -> str | None:
    match = re.search(
        r"(\d+(?:\.\d+)?\s*hpf|\d+\s*s|shield|bud)\s*[- ]*treatment",
        token,
        flags=re.IGNORECASE,
    )
    return normalize_stage_label(match.group(1)) if match else None


def _parse_collection_name(
    collection_name: Any,
    effective_experiment: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(condition, collected_stage, addition_stage)`` fallbacks."""
    tokens = _collection_name_tokens(collection_name)
    if tokens and effective_experiment and tokens[0].casefold() == effective_experiment.casefold():
        tokens = tokens[1:]

    condition_tokens: list[str] = []
    stage_tokens: list[str] = []
    treatment_stage: str | None = None
    for token in tokens:
        normalized = normalize_stage_label(token)
        hpf, method, _ = stage_label_to_hpf(token)
        if hpf is not None and method != "unresolved":
            stage_tokens.append(normalized or token)
            continue
        treatment = _stage_from_treatment_token(token)
        if treatment is not None:
            treatment_stage = treatment
            continue
        condition_tokens.append(token)

    collected = stage_tokens[-1] if stage_tokens else None
    addition = (
        treatment_stage
        if treatment_stage is not None
        else (stage_tokens[-2] if len(stage_tokens) > 1 else None)
    )
    condition = "_".join(condition_tokens).strip() or None
    return condition, collected, addition


def _prepare_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    prepared = metadata.copy()
    if "expt" not in prepared.columns:
        raise KeyError("SeaHub collection metadata must contain an 'expt' column.")

    records: list[dict[str, Any]] = []
    for _, row in prepared.iterrows():
        record = row.to_dict()
        original_experiment = (
            None if _is_missing(row.get("expt")) else str(row.get("expt")).strip().upper()
        )
        prefix_experiment = _collection_name_experiment(row.get("collection_name"))
        effective_experiment = prefix_experiment or original_experiment
        corrected = bool(
            prefix_experiment
            and original_experiment
            and prefix_experiment != original_experiment
        )
        fallback_condition, fallback_collected, fallback_addition = _parse_collection_name(
            row.get("collection_name"), effective_experiment
        )

        collected_raw = (
            row.get("stage_collected")
            if not _is_missing(row.get("stage_collected"))
            else fallback_collected
        )
        addition_raw = (
            row.get("stage_addition")
            if not _is_missing(row.get("stage_addition"))
            else fallback_addition
        )
        collected_hpf, collected_method, collected_normalized = stage_label_to_hpf(
            collected_raw
        )
        addition_hpf, addition_method, addition_normalized = stage_label_to_hpf(
            addition_raw
        )

        structured_condition = row.get("perturbation")
        condition_values = [
            value
            for value in (structured_condition, fallback_condition)
            if not _is_missing(value)
        ]
        condition_keys = tuple(
            dict.fromkeys(
                key
                for key in (canonical_condition(value) for value in condition_values)
                if key
            )
        )

        record.update(
            {
                "_metadata_experiment_original": original_experiment,
                "_metadata_experiment_effective": effective_experiment,
                "_metadata_experiment_corrected": corrected,
                "_metadata_experiment_correction_source": (
                    "collection_name_prefix" if corrected else None
                ),
                "_metadata_stage_collected_raw": collected_raw,
                "_metadata_stage_collected_normalized": collected_normalized,
                "_metadata_stage_collected_hpf": collected_hpf,
                "_metadata_stage_collected_method": collected_method,
                "_metadata_stage_addition_raw": addition_raw,
                "_metadata_stage_addition_normalized": addition_normalized,
                "_metadata_stage_addition_hpf": addition_hpf,
                "_metadata_stage_addition_method": addition_method,
                "_metadata_condition_keys": condition_keys,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _condition_similarity(image_key: str | None, candidate_keys: Iterable[str]) -> float:
    if not image_key:
        return 0.0
    keys = tuple(candidate_keys)
    return max(
        (SequenceMatcher(None, image_key, key).ratio() for key in keys),
        default=0.0,
    )


def _copy_metadata_payload(
    output: dict[str, Any],
    selected: pd.Series,
    metadata_columns: list[str],
) -> None:
    output["metadata_row_number"] = selected.get("metadata_row_number")
    for column in metadata_columns:
        output[f"metadata_{column}"] = selected.get(column)
    output["metadata_experiment_original"] = selected.get(
        "_metadata_experiment_original"
    )
    output["metadata_experiment_effective"] = selected.get(
        "_metadata_experiment_effective"
    )
    output["metadata_experiment_corrected"] = bool(
        selected.get("_metadata_experiment_corrected")
    )
    output["metadata_experiment_correction_source"] = selected.get(
        "_metadata_experiment_correction_source"
    )
    for suffix in (
        "stage_collected_raw",
        "stage_collected_normalized",
        "stage_collected_hpf",
        "stage_collected_method",
        "stage_addition_raw",
        "stage_addition_normalized",
        "stage_addition_hpf",
        "stage_addition_method",
    ):
        output[f"metadata_{suffix}"] = selected.get(f"_metadata_{suffix}")


def _audit_source_stage(record: dict[str, Any]) -> None:
    stage_hpf, method, normalized = stage_label_to_hpf(
        record.get("stage_source_label")
    )
    addition_hpf, addition_method, addition_normalized = stage_label_to_hpf(
        record.get("stage_addition_source_label")
    )
    record["stage_hpf"] = stage_hpf
    record["stage_normalized_label"] = normalized
    record["stage_match_method"] = method
    record["stage_addition_hpf"] = addition_hpf
    record["stage_addition_normalized_label"] = addition_normalized
    record["stage_addition_match_method"] = addition_method
    record["stage_match_delta_hpf"] = np.nan


def _audit_existing_metadata_payload(output: dict[str, Any]) -> None:
    """Backfill the addendum audit block for rows matched by the earlier workflow."""
    collection_name = output.get("metadata_collection_name")
    original = (
        None
        if _is_missing(output.get("metadata_expt"))
        else str(output["metadata_expt"]).strip().upper()
    )
    prefix = _collection_name_experiment(collection_name)
    effective = prefix or original
    corrected = bool(prefix and original and prefix != original)
    output.setdefault("metadata_experiment_original", original)
    output.setdefault("metadata_experiment_effective", effective)
    output.setdefault("metadata_experiment_corrected", corrected)
    output.setdefault(
        "metadata_experiment_correction_source",
        "collection_name_prefix" if corrected else None,
    )

    fallback_condition, fallback_collected, fallback_addition = _parse_collection_name(
        collection_name, effective
    )
    del fallback_condition
    collected_raw = (
        output.get("metadata_stage_collected")
        if not _is_missing(output.get("metadata_stage_collected"))
        else fallback_collected
    )
    addition_raw = (
        output.get("metadata_stage_addition")
        if not _is_missing(output.get("metadata_stage_addition"))
        else fallback_addition
    )
    for prefix_name, raw in (
        ("metadata_stage_collected", collected_raw),
        ("metadata_stage_addition", addition_raw),
    ):
        hpf, method, normalized = stage_label_to_hpf(raw)
        output.setdefault(f"{prefix_name}_raw", raw)
        output.setdefault(f"{prefix_name}_normalized", normalized)
        output.setdefault(f"{prefix_name}_hpf", hpf)
        output.setdefault(f"{prefix_name}_method", method)


def reconcile_seahub_metadata(
    image_rows: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    fuzzy_threshold: float = 0.90,
) -> pd.DataFrame:
    """Remediate failed matches while preserving successful existing matches."""
    prepared = _prepare_metadata(metadata)
    metadata_columns = [
        column
        for column in metadata.columns
        if column != "metadata_row_number"
    ]
    by_experiment = {
        str(key): group
        for key, group in prepared.groupby(
            "_metadata_experiment_effective", dropna=True
        )
    }

    outputs: list[dict[str, Any]] = []
    for _, image_row in image_rows.iterrows():
        output = image_row.to_dict()
        if "source_fov_id" not in output and "image_id" in output:
            output["source_fov_id"] = output["image_id"]
        _audit_source_stage(output)
        _audit_existing_metadata_payload(output)

        prior_status = (
            "unmatched"
            if _is_missing(output.get("metadata_match_status"))
            else str(output["metadata_match_status"])
        )
        if prior_status not in REMEDIABLE_MATCH_STATUSES:
            outputs.append(output)
            continue

        experiment = (
            None
            if _is_missing(output.get("experiment_id"))
            else str(output.get("experiment_id")).strip().upper()
        )
        candidates = by_experiment.get(experiment or "", prepared.iloc[0:0])
        stage_hpf = output.get("stage_hpf")
        addition_hpf = output.get("stage_addition_hpf")
        # Derive the condition key from the RAW parsed perturbation so paralog
        # shorthand is expanded consistently with the workbook (the precomputed
        # perturbation_key predates expansion and would not match). For non-shorthand
        # rows this reproduces the precomputed key exactly. Fall back to the
        # precomputed key only when the raw form is missing; a NaN key (pandas reads a
        # blank cell as float('nan'), which is truthy) is coerced to None so it can
        # never reach the fuzzy matcher (SequenceMatcher on a float would crash).
        condition_key = canonical_condition(output.get("perturbation_parsed"))
        if _is_missing(condition_key):
            fallback = output.get("perturbation_key")
            condition_key = None if _is_missing(fallback) else fallback

        if stage_hpf is None or pd.isna(stage_hpf):
            output["metadata_match_status"] = "unmatched_stage"
            output["metadata_candidate_count"] = 0
            outputs.append(output)
            continue

        same_stage = candidates[
            pd.to_numeric(
                candidates["_metadata_stage_collected_hpf"], errors="coerce"
            ).eq(float(stage_hpf))
        ]
        if addition_hpf is not None and not pd.isna(addition_hpf):
            same_stage = same_stage[
                pd.to_numeric(
                    same_stage["_metadata_stage_addition_hpf"], errors="coerce"
                ).eq(float(addition_hpf))
            ]
        if same_stage.empty:
            output["metadata_match_status"] = "unmatched_stage"
            output["metadata_candidate_count"] = 0
            outputs.append(output)
            continue

        exact = same_stage[
            same_stage["_metadata_condition_keys"].map(
                lambda keys: bool(condition_key and condition_key in keys)
            )
        ]
        selected: pd.Series | None = None
        score = np.nan
        if len(exact) == 1:
            selected = exact.iloc[0]
            status = "exact"
            score = 1.0
            candidate_count = 1
        elif len(exact) > 1:
            status = "exact_duplicate"
            candidate_count = len(exact)
        elif condition_key:
            scored = same_stage.copy()
            scored["_similarity"] = scored["_metadata_condition_keys"].map(
                lambda keys: _condition_similarity(condition_key, keys)
            )
            scored = scored.sort_values(
                ["_similarity", "metadata_row_number"],
                ascending=[False, True],
            )
            best_score = float(scored.iloc[0]["_similarity"])
            close = scored[scored["_similarity"] >= best_score - 0.02]
            candidate_count = len(close)
            if best_score >= fuzzy_threshold and len(close) == 1:
                selected = scored.iloc[0]
                status = "fuzzy"
                score = best_score
            elif best_score >= fuzzy_threshold:
                status = "fuzzy_ambiguous"
                score = best_score
            else:
                status = "unmatched_condition"
                score = best_score
        else:
            status = "unmatched_condition"
            candidate_count = len(same_stage)

        output["metadata_match_status"] = status
        output["metadata_match_score"] = score
        output["metadata_candidate_count"] = candidate_count
        if selected is not None:
            _copy_metadata_payload(output, selected, metadata_columns)
            metadata_hpf = selected.get("_metadata_stage_collected_hpf")
            if metadata_hpf is not None and not pd.isna(metadata_hpf):
                output["stage_match_delta_hpf"] = float(stage_hpf) - float(metadata_hpf)
        outputs.append(output)

    return pd.DataFrame.from_records(outputs)


def apply_inclusion_policy(reconciled: pd.DataFrame) -> pd.DataFrame:
    """Annotate every FOV with an explicit include decision and exclusion reason."""
    source_fov_ids = pd.Series(pd.NA, index=reconciled.index, dtype=object)
    for source_column in ("source_fov_id", "image_id"):
        if source_column not in reconciled.columns:
            continue
        fallback = reconciled[source_column]
        missing = source_fov_ids.map(_is_missing)
        source_fov_ids.loc[missing] = fallback.loc[missing]
    source_fov_ids = source_fov_ids.map(
        lambda value: pd.NA if _is_missing(value) else str(value)
    )
    duplicate_source_ids = source_fov_ids.notna() & source_fov_ids.duplicated(
        keep=False
    )

    outputs: list[dict[str, Any]] = []
    for row_position, (_, row) in enumerate(reconciled.iterrows()):
        record = row.to_dict()
        source_fov_id = source_fov_ids.iloc[row_position]
        record["source_fov_id"] = source_fov_id
        path = str(record.get("relative_path") or record.get("image_path") or "")
        status = (
            ""
            if _is_missing(record.get("metadata_match_status"))
            else str(record.get("metadata_match_status"))
        )
        reason: str | None = None

        if str(record.get("image_role")) != "eight_embryo_fov":
            reason = "wrong_image_role"
        elif _ABANDONED_RE.search(path):
            reason = "abandoned"
        elif _NOT_USED_RE.search(path):
            reason = "not_used"
        elif _EXPLICIT_NOT_COLLECTED_RE.search(path):
            reason = "not_collected"
        elif _EXPLICIT_NOT_SEQUENCED_RE.search(path):
            reason = "not_sequenced"
        elif _is_true(record.get("excluded_path", False)):
            reason = "excluded_source_path"
        elif not _is_missing(record.get("read_error")):
            reason = "unreadable"
        elif _is_missing(record.get("experiment_id")):
            reason = "missing_experiment"
        elif _is_missing(source_fov_id):
            reason = "contract_invalid_missing_source_fov_id"
        elif bool(duplicate_source_ids.iloc[row_position]):
            reason = "contract_invalid_duplicate_source_fov_id"
        elif _is_missing(record.get("image_path")):
            reason = "contract_invalid_missing_image_path"
        elif "duplicate" in status or "ambiguous" in status:
            reason = "duplicate_or_ambiguous_metadata"
        elif status == "unmatched_experiment":
            reason = "unmatched_experiment"
        elif status not in INCLUDABLE_MATCH_STATUSES:
            reason = "contract_invalid_metadata_match_status"

        record["include_for_seahub"] = reason is None
        record["exclusion_reason"] = reason
        outputs.append(record)
    return pd.DataFrame.from_records(outputs)


__all__ = [
    "INCLUDABLE_MATCH_STATUSES",
    "PASS_THROUGH_FAILURE_STATUSES",
    "REMEDIABLE_MATCH_STATUSES",
    "apply_inclusion_policy",
    "canonical_condition",
    "load_collection_metadata",
    "reconcile_seahub_metadata",
]
