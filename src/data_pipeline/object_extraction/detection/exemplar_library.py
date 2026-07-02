"""Model-neutral detection exemplar library resolver.

An exemplar set defines visual prompt examples for a concept. It is intentionally separate from
seed selection, mask propagation, tracking direction, or any backend-specific request payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_EXEMPLAR_COLUMNS = (
    "exemplar_id",
    "concept_label",
    "prompt_role",
    "prompt_type",
    "reference_image_path",
    "bbox_x_min_px",
    "bbox_y_min_px",
    "bbox_x_max_px",
    "bbox_y_max_px",
)

OPTIONAL_EXEMPLAR_COLUMNS = (
    "notes",
)

ALLOWED_PROMPT_ROLES = frozenset({"positive", "negative"})
ALLOWED_PROMPT_TYPES = frozenset({"box"})


@dataclass(frozen=True)
class Exemplar:
    exemplar_id: str
    concept_label: str
    prompt_role: str
    prompt_type: str
    reference_image_path: Path
    box_xyxy: tuple[float, float, float, float]
    notes: str | None = None


@dataclass(frozen=True)
class ExemplarSet:
    name: str
    root: Path
    manifest_path: Path
    exemplars: tuple[Exemplar, ...]


def resolve_exemplar_set(name: str, *, library_root: str | Path) -> ExemplarSet:
    """Load and validate ``library_root/name/manifest.csv``."""

    root = Path(library_root).expanduser().resolve() / name
    manifest_path = root / "manifest.csv"
    exemplars = load_exemplar_manifest(manifest_path)
    return ExemplarSet(name=name, root=root, manifest_path=manifest_path, exemplars=exemplars)


def load_exemplar_manifest(manifest_path: str | Path) -> tuple[Exemplar, ...]:
    """Load one exemplar manifest and return validated model-neutral exemplars."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Exemplar manifest not found: {path}")

    df = pd.read_csv(path)
    validate_exemplar_manifest(df, manifest_path=path)
    return tuple(_row_to_exemplar(row, manifest_path=path) for _, row in df.iterrows())


def validate_exemplar_manifest(df: pd.DataFrame, *, manifest_path: str | Path | None = None) -> None:
    """Validate the v1 exemplar manifest schema."""

    context = f"{manifest_path}: " if manifest_path is not None else ""
    missing = [col for col in REQUIRED_EXEMPLAR_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{context}missing required exemplar columns: {missing}")
    if df.empty:
        raise ValueError(f"{context}exemplar manifest has no rows")
    if df["exemplar_id"].isna().any() or df["exemplar_id"].duplicated().any():
        raise ValueError(f"{context}exemplar_id values must be present and unique")

    for idx, row in df.iterrows():
        row_context = f"{context}row {idx} exemplar_id={row['exemplar_id']!r}"
        _validate_choice(row["prompt_role"], ALLOWED_PROMPT_ROLES, field="prompt_role", context=row_context)
        _validate_choice(row["prompt_type"], ALLOWED_PROMPT_TYPES, field="prompt_type", context=row_context)
        if not str(row["concept_label"]).strip():
            raise ValueError(f"{row_context}: concept_label is required")
        if not str(row["reference_image_path"]).strip():
            raise ValueError(f"{row_context}: reference_image_path is required")

        x0, y0, x1, y1 = _coerce_box(row)
        if x0 < 0 or y0 < 0:
            raise ValueError(f"{row_context}: bbox minimum coordinates must be non-negative")
        if not x0 < x1:
            raise ValueError(f"{row_context}: bbox_x_min_px must be < bbox_x_max_px")
        if not y0 < y1:
            raise ValueError(f"{row_context}: bbox_y_min_px must be < bbox_y_max_px")


def exemplars_for_sam3_prompt(exemplars: Iterable[Exemplar]) -> list[dict[str, object]]:
    """Return a backend-friendly neutral payload shape for SAM3 request construction."""

    return [
        {
            "exemplar_id": ex.exemplar_id,
            "role": ex.prompt_role,
            "type": ex.prompt_type,
            "image_path": str(ex.reference_image_path),
            "box_xyxy": list(ex.box_xyxy),
        }
        for ex in exemplars
    ]


def _row_to_exemplar(row: pd.Series, *, manifest_path: Path) -> Exemplar:
    image_path = Path(str(row["reference_image_path"]))
    if not image_path.is_absolute():
        image_path = manifest_path.parent / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Exemplar reference image not found: {image_path}")
    notes = row.get("notes")
    return Exemplar(
        exemplar_id=str(row["exemplar_id"]),
        concept_label=str(row["concept_label"]),
        prompt_role=str(row["prompt_role"]),
        prompt_type=str(row["prompt_type"]),
        reference_image_path=image_path,
        box_xyxy=_coerce_box(row),
        notes=None if pd.isna(notes) else str(notes),
    )


def _validate_choice(value: object, allowed: frozenset[str], *, field: str, context: str) -> None:
    if str(value) not in allowed:
        raise ValueError(f"{context}: {field} must be one of {sorted(allowed)}, got {value!r}")


def _coerce_box(row: pd.Series) -> tuple[float, float, float, float]:
    values = (
        row["bbox_x_min_px"],
        row["bbox_y_min_px"],
        row["bbox_x_max_px"],
        row["bbox_y_max_px"],
    )
    try:
        return tuple(float(v) for v in values)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox columns must be numeric xyxy pixel values: {values!r}") from exc
