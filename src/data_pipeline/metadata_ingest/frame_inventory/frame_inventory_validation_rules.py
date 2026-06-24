"""Product-specific frame_inventory contract rules (L3 grain + L4 sources).

These are the strict, frame_inventory-specific rules that ``frame_inventory_validation.py``
sequences. They IMPORT the column vocabulary from
``image_materialization/frame_inventory_contract.py`` — they never duplicate the contract's column
names or the REQUIRED_CHANNEL.

Each rule raises ``ValueError`` with a fix-named message on violation (errors are the UX). The public
gate catches the failure, writes the errors report, and re-raises.

Naming doctrine: ``grain`` names the row; ``validation_scope`` names the gate. The L3 rules take a
``validation_scope`` of ``"per_well"`` (one well per table) or ``"merged"`` (many wells; per-well
checks applied grouped by ``well_id``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.image_materialization.frame_inventory_contract import (
    ALLOWED_IMAGE_SUFFIXES,
    REQUIRED_CHANNEL,
)
from data_pipeline.shared.identifiers import build_well_id

VALIDATION_SCOPES = ("per_well", "merged")


# ---------------------------------------------------------------------------
# L3 — grain rules (scope-aware)
# ---------------------------------------------------------------------------

def validate_grain(df: pd.DataFrame, *, validation_scope: str, scope_label: str) -> None:
    """Validate the per-well temporal grain, at the requested validation scope.

    Both scopes require exactly one ``experiment_id``. ``per_well`` additionally requires exactly one
    well; ``merged`` allows many wells and applies the per-well temporal checks grouped by ``well_id``.
    """
    if validation_scope not in VALIDATION_SCOPES:
        raise ValueError(
            f"[{scope_label}] unknown validation_scope {validation_scope!r}; "
            f"expected one of {VALIDATION_SCOPES}."
        )

    _assert_single_experiment(df, scope_label=scope_label)

    if validation_scope == "per_well":
        _assert_single_well(df, scope_label=scope_label)
        _validate_well_temporal_grain(df, scope_label=scope_label)
        return

    # merged: many wells allowed — apply the per-well temporal checks within each well_id group.
    well_ids = _derive_well_ids(df)
    for well_id, group in df.groupby(well_ids, sort=False):
        _validate_well_temporal_grain(group, scope_label=f"{scope_label}:{well_id}")


def _assert_single_experiment(df: pd.DataFrame, *, scope_label: str) -> None:
    experiments = sorted(df["experiment_id"].dropna().astype(str).unique())
    if len(experiments) != 1:
        raise ValueError(
            f"[{scope_label}] frame_inventory must carry exactly one experiment_id; "
            f"found {experiments}. Split the table by experiment before validating."
        )


def _assert_single_well(df: pd.DataFrame, *, scope_label: str) -> None:
    well_indices = sorted(df["well_index"].dropna().astype(str).unique())
    if len(well_indices) != 1:
        raise ValueError(
            f"[{scope_label}] a per-well frame_inventory shard must carry exactly one well_index; "
            f"found {well_indices}. (For the merged experiment view, use validation_scope='merged'.)"
        )


def _derive_well_ids(df: pd.DataFrame) -> pd.Series:
    # Route the atoms through the identifier grammar — never f-string the well_id inline.
    return df.apply(
        lambda r: build_well_id(str(r["experiment_id"]), str(r["well_index"])), axis=1
    )


def _validate_well_temporal_grain(df: pd.DataFrame, *, scope_label: str) -> None:
    """Within ONE well: BF present + contiguous; channels rectangular; multi-timepoint ⇒ elapsed_time_s."""
    channels = sorted(df["channel_id"].dropna().astype(str).unique())

    # BF is the required reference channel and defines the segmentation timeline.
    if REQUIRED_CHANNEL not in channels:
        raise ValueError(
            f"[{scope_label}] required channel {REQUIRED_CHANNEL!r} is absent (present: {channels}). "
            f"Every well must have a {REQUIRED_CHANNEL} channel — it anchors the segmentation timeline."
        )

    # BF time_index must be contiguous 0..N-1.
    bf_times = sorted(
        int(t) for t in df.loc[df["channel_id"].astype(str) == REQUIRED_CHANNEL, "time_index"].unique()
    )
    expected = list(range(len(bf_times)))
    if bf_times != expected:
        raise ValueError(
            f"[{scope_label}] {REQUIRED_CHANNEL} time_index must be contiguous 0..N-1; "
            f"found {bf_times} (expected {expected}). Renumber or fill the missing frames."
        )

    # All present channels must share the SAME time_index set (rectangular).
    bf_time_set = set(bf_times)
    for channel in channels:
        ch_times = {
            int(t) for t in df.loc[df["channel_id"].astype(str) == channel, "time_index"].unique()
        }
        if ch_times != bf_time_set:
            raise ValueError(
                f"[{scope_label}] channel {channel!r} has time_index set {sorted(ch_times)} "
                f"but {REQUIRED_CHANNEL} has {sorted(bf_time_set)}. All present channels must share the "
                "same time_index set (rectangular) — a ragged channel is almost always a dropped frame."
            )

    # Temporal rule: a well with >1 distinct time_index requires elapsed_time_s on every row.
    distinct_times = df["time_index"].dropna().astype(int).nunique()
    if distinct_times > 1:
        if "elapsed_time_s" not in df.columns or df["elapsed_time_s"].isna().any():
            raise ValueError(
                f"[{scope_label}] this well has {distinct_times} distinct time_index values "
                "(multi-timepoint) so elapsed_time_s is REQUIRED and non-null on every row. "
                "A single-timepoint well may omit it; this one may not."
            )


# ---------------------------------------------------------------------------
# L4 — source / image contract (gated by check_sources)
# ---------------------------------------------------------------------------

def validate_sources(
    df: pd.DataFrame, *, image_root: Path | None, scope_label: str
) -> None:
    """For every row: resolve source_image_path, open the image, self-check dims, require µm/px > 0.

    The CSV is never rewritten — paths are resolved only during validation.
    """
    from PIL import Image  # local import: only the strict source mode needs pillow.

    for idx, row in df.iterrows():
        resolved = _resolve_source_path(
            str(row["source_image_path"]), image_root=image_root, scope_label=scope_label
        )
        if not resolved.exists():
            raise ValueError(
                f"[{scope_label}] source_image_path does not exist: {resolved} "
                f"(row {idx}). Fix the path or place the image where it resolves."
            )
        if resolved.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(
                f"[{scope_label}] unsupported image suffix {resolved.suffix!r} for {resolved}; "
                f"allowed: {ALLOWED_IMAGE_SUFFIXES}."
            )
        try:
            with Image.open(resolved) as im:
                real_w, real_h = im.size
        except Exception as exc:  # noqa: BLE001 — surface the underlying open error by name.
            raise ValueError(
                f"[{scope_label}] image at {resolved} could not be opened: {exc}"
            ) from exc

        declared_w = int(row["image_width_px"])
        declared_h = int(row["image_height_px"])
        if (real_w, real_h) != (declared_w, declared_h):
            raise ValueError(
                f"[{scope_label}] image dims mismatch for {resolved}: header says "
                f"{real_w}x{real_h} but the manifest declares {declared_w}x{declared_h}. "
                "Correct image_width_px / image_height_px (they are a self-check, not new info)."
            )

        um_per_px = float(row["source_micrometers_per_pixel"])
        if not um_per_px > 0:
            raise ValueError(
                f"[{scope_label}] source_micrometers_per_pixel must be > 0; "
                f"row {idx} has {um_per_px}."
            )


def _resolve_source_path(
    raw_path: str, *, image_root: Path | None, scope_label: str
) -> Path:
    """Absolute paths validate directly; relative paths resolve under image_root (no `..` escape)."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if image_root is None:
        raise ValueError(
            f"[{scope_label}] relative source_image_path {raw_path!r} requires image_root when "
            "check_sources=True. Pass --image-root or author absolute paths."
        )
    root = Path(image_root).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"[{scope_label}] relative source_image_path {raw_path!r} escapes image_root "
            f"({root}) via '..'. Paths may not climb out of the image root."
        )
    return resolved
