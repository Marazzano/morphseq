"""Canonical SeaHub morphological-stage normalization."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

STAGE_HPF_CROSSWALK: dict[str, float] = {
    "1-cell": 0.00,
    "2-cell": 0.75,
    "4-cell": 1.00,
    "8-cell": 1.25,
    "16-cell": 1.50,
    "32-cell": 1.75,
    "64-cell": 2.00,
    "128-cell": 2.25,
    "256-cell": 2.50,
    "512-cell": 2.75,
    "1k-cell": 3.00,
    "high": 3.33,
    "oblong": 3.67,
    "sphere": 4.00,
    "dome": 4.33,
    "30%-epiboly": 4.67,
    "50%-epiboly": 5.25,
    "germ-ring": 5.67,
    "shield": 6.00,
    "75%-epiboly": 8.00,
    "90%-epiboly": 9.00,
    "bud": 10.00,
    "1-somite": 10.33,
    "5-somite": 11.67,
    "8-somite": 13.00,
    "10-somite": 14.00,
    "12-somite": 15.00,
    "14-somite": 16.00,
    "18-somite": 18.00,
    "20-somite": 19.00,
    "26-somite": 22.00,
    "prim-5": 24.00,
}

_ALIASES: dict[str, str] = {
    "germ ring": "germ-ring",
    "germring": "germ-ring",
    "prim5": "prim-5",
    "prim 5": "prim-5",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_stage_label(value: Any) -> str | None:
    """Return a canonical hpf or morphological label without losing the raw source."""
    if _is_missing(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip()).casefold()
    if not text or text in {"#no match", "nan", "none"}:
        return None
    text = text.replace("hrs", "hpf").replace("hr", "hpf")
    text = text.replace("48hpd", "48hpf")
    text = re.sub(r"\s*-\s*", "-", text)
    text = _ALIASES.get(text, text)

    somite = re.fullmatch(r"(\d+)\s*s", text)
    if somite:
        return f"{int(somite.group(1))}-somite"
    somite = re.fullmatch(r"(\d+)[-_ ]*somites?", text)
    if somite:
        return f"{int(somite.group(1))}-somite"

    epiboly = re.fullmatch(r"(\d+)\s*%\s*[-_ ]*epiboly", text)
    if epiboly:
        return f"{int(epiboly.group(1))}%-epiboly"

    cell = re.fullmatch(r"(\d+|1k)\s*[-_ ]*cells?", text)
    if cell:
        return f"{cell.group(1)}-cell"

    hpf = re.fullmatch(r"(\d+(?:\.\d+)?)\s*hpf", text)
    if hpf:
        number = float(hpf.group(1))
        return f"{number:g}hpf"
    return text


def stage_label_to_hpf(value: Any) -> tuple[float | None, str, str | None]:
    """Return ``(hpf, method, normalized_label)`` for one raw stage value."""
    normalized = normalize_stage_label(value)
    if normalized is None:
        return None, "unresolved", None
    if normalized.endswith("hpf"):
        try:
            return float(normalized.removesuffix("hpf")), "stage_exact", normalized
        except ValueError:
            return None, "unresolved", normalized
    if normalized in STAGE_HPF_CROSSWALK:
        return (
            float(STAGE_HPF_CROSSWALK[normalized]),
            "stage_crosswalk",
            normalized,
        )
    return None, "unresolved", normalized


__all__ = [
    "STAGE_HPF_CROSSWALK",
    "normalize_stage_label",
    "stage_label_to_hpf",
]
