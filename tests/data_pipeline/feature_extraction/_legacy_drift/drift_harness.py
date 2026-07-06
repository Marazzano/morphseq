"""Shared drift-benchmark harness reused by every product's ``test_<product>_legacy_drift.py``.

One harness so the three things every product comparison must do identically — resolve the new
side, normalize+join on ``snip_id``, and emit an auditable report — live in exactly one place and
cannot drift between products. A product test supplies only what is product-specific: which step to
load, and a per-column comparison map (legacy column + tolerance class).

The new side is resolved through the orchestration registry (``artifact_path`` / ``per_well_step_dir``)
so the benchmark reads the same files the pipeline writes. It prefers the merged table; if only the
per-well shards exist (a partial pilot run), it concatenates them; if neither exists it skips with a
clear message (a checkout/partial-build without the product must deselect, not error).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    artifact_path,
    per_well_step_dir,
)

from .load_legacy_qc_staged import legacy_key_from_new_snip_id, load_legacy_pilot

EXPERIMENT_ID = "20250912"

# New-pipeline output root (merged/per-well tables land under <root>/features|quality_control/...).
NEW_PIPELINE_ROOT = Path(__file__).resolve().parents[4] / "data_pipeline_output"

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@dataclass(frozen=True)
class NumericColumn:
    """One asserted numeric comparison: new column vs a legacy column, to a relative tolerance.

    ``tolerance_rel`` encodes the tolerance class: ~1e-3 is numeric-tight (same object + formula),
    ~1e-2..1e-1 is numeric-loose (a known method/resolution change — assert a generous bound, report
    the residual distribution). A miss above the bound is gross drift, not the expected method shift.
    """

    new_col: str
    legacy_col: str
    tolerance_rel: float
    note: str = ""


def load_new_side(step: str, *, artifact: str | None = None) -> pd.DataFrame:
    """Load a new-pipeline product table for the pilot, preferring merged then per-well shards.

    ``artifact`` defaults to ``step`` (most products name their single artifact after the step).
    Skips the benchmark (``pytest.skip``) when neither a merged file nor any per-well shard exists.
    """
    artifact = artifact or step
    merged = artifact_path(
        NEW_PIPELINE_ROOT, step, artifact, EXPERIMENT_ID, path_mode="merged"
    )
    if merged.exists():
        return pd.read_csv(merged)

    # Fall back to concatenating per-well shards (a partial pilot run with no merge yet).
    shard_root = per_well_step_dir(NEW_PIPELINE_ROOT, step, EXPERIMENT_ID)
    shards = sorted(shard_root.glob(f"*/*_{artifact}.csv")) if shard_root.exists() else []
    if shards:
        return pd.concat((pd.read_csv(p) for p in shards), ignore_index=True)

    pytest.skip(
        f"new-side {step!r} output not found (no merged file at {merged} and no per-well shards "
        f"under {shard_root}). Run the {step} target for {EXPERIMENT_ID} first "
        f"(feature_world.md → 'How to produce the new side'). Skipping."
    )


def _relative_residual(new: np.ndarray, legacy: np.ndarray) -> np.ndarray:
    """Symmetric relative residual ``|new - legacy| / max(|legacy|, eps)``."""
    denom = np.maximum(np.abs(legacy), 1e-9)
    return np.abs(new - legacy) / denom


def normalize_and_join(new: pd.DataFrame, legacy: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map new ``snip_id`` → legacy grammar and inner-join on it.

    Returns the joined frame (columns suffixed ``_new`` / ``_legacy`` where they overlap, plus a
    ``legacy_snip_id`` join key) and the list of new keys absent from the legacy universe (a
    coverage gap is itself drift). Fails loud if any new ``snip_id`` cannot be normalized.
    """
    new = new.copy()
    new["legacy_snip_id"] = new["snip_id"].map(legacy_key_from_new_snip_id)
    unparsed = new.loc[new["legacy_snip_id"].isna(), "snip_id"].tolist()
    assert not unparsed, (
        f"{len(unparsed)} new snip_id(s) did not match the expected new-pipeline grammar and "
        f"could not be normalized to the legacy key (e.g. {unparsed[:5]}). The join key normalizer "
        f"is out of date with the snip_id grammar — fix legacy_key_from_new_snip_id."
    )

    legacy_keys = set(legacy["snip_id"])
    matched = new["legacy_snip_id"].isin(legacy_keys)
    missing_in_legacy = new.loc[~matched, "legacy_snip_id"].tolist()
    joined = new.loc[matched].merge(
        legacy.reset_index(drop=True),
        left_on="legacy_snip_id",
        right_on="snip_id",
        suffixes=("_new", "_legacy"),
    )
    return joined, missing_in_legacy


def run_numeric_drift_benchmark(
    *,
    product: str,
    step: str,
    columns: list[NumericColumn],
    artifact: str | None = None,
    reported_only: tuple[str, ...] = (),
) -> None:
    """Full numeric drift benchmark for one product: load → join → assert tolerances → report.

    ``product`` names the report file; ``step``/``artifact`` resolve the new side; ``columns`` are
    the asserted numeric comparisons; ``reported_only`` are new columns recorded without a legacy
    ground-truth column (e.g. a quantity the legacy file doesn't carry comparably).
    """
    new = load_new_side(step, artifact=artifact)
    legacy = load_legacy_pilot(EXPERIMENT_ID)
    joined, missing_in_legacy = normalize_and_join(new, legacy)

    def _col(base: str, side: str) -> str:
        """Resolve a merged column name: overlapping names get a suffix, unique ones do not."""
        suffixed = f"{base}_{side}"
        if suffixed in joined.columns:
            return suffixed
        if base in joined.columns:
            return base
        raise KeyError(
            f"neither {suffixed!r} nor {base!r} present in joined columns: {list(joined.columns)}"
        )

    report: dict[str, object] = {
        "product": product,
        "experiment_id": EXPERIMENT_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "new_rows": int(len(new)),
        "legacy_rows": int(len(legacy)),
        "joined_rows": int(len(joined)),
        "new_snips_absent_from_legacy": missing_in_legacy,
        "columns": {},
    }
    failures: list[str] = []

    for spec in columns:
        new_vals = pd.to_numeric(joined[_col(spec.new_col, "new")], errors="coerce").to_numpy(float)
        leg_vals = pd.to_numeric(
            joined[_col(spec.legacy_col, "legacy")], errors="coerce"
        ).to_numpy(float)
        resid = _relative_residual(new_vals, leg_vals)
        worst_idx = int(np.argmax(resid)) if resid.size else -1
        col_report: dict[str, object] = {
            "legacy_column": spec.legacy_col,
            "tolerance_rel": spec.tolerance_rel,
            "note": spec.note,
            "max_rel_residual": float(np.max(resid)) if resid.size else None,
            "median_rel_residual": float(np.median(resid)) if resid.size else None,
            "worst_snip_id": (
                str(joined["legacy_snip_id"].iloc[worst_idx]) if worst_idx >= 0 else None
            ),
        }
        if resid.size and np.max(resid) > spec.tolerance_rel:
            over = joined.loc[resid > spec.tolerance_rel, "legacy_snip_id"].tolist()
            col_report["snips_over_tolerance"] = over
            failures.append(
                f"{spec.new_col}: max rel residual {np.max(resid):.4g} > {spec.tolerance_rel:g} "
                f"({len(over)} snip(s) over tolerance, e.g. {over[:5]})"
            )
        report["columns"][spec.new_col] = col_report

    for col in reported_only:
        if f"{col}_new" in joined.columns or col in joined.columns:
            report["columns"][col] = {
                "legacy_column": None,
                "note": "reported only — no comparable legacy ground-truth column in qc_staged.",
            }

    # Persist the auditable report before asserting.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{product}_legacy_drift_{EXPERIMENT_ID}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    assert not missing_in_legacy, (
        f"{len(missing_in_legacy)} new {product} snip(s) have no legacy ground-truth row "
        f"(coverage gap is itself drift): {missing_in_legacy[:10]}. Report: {report_path}"
    )
    assert len(joined) > 0, (
        f"no snip_id overlap between new {product} and legacy pilot — the join produced 0 rows. "
        f"Report: {report_path}"
    )
    assert not failures, (
        f"{product} drifted from the legacy pilot beyond its tolerance class on column(s):\n  "
        + "\n  ".join(failures)
        + f"\nFull report: {report_path}"
    )
