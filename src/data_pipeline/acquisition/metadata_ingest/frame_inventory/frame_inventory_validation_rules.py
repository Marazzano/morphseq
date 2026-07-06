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

import logging
from pathlib import Path

import pandas as pd

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    ALLOWED_IMAGE_SUFFIXES,
    REQUIRED_CHANNEL,
    frame_inventory_product_keys,
)
from data_pipeline.acquisition.image_materialization.materialized_image_write_policy import (
    expected_downsampled_dims,
)
from data_pipeline.shared.identifiers import build_well_id

log = logging.getLogger(__name__)

VALIDATION_SCOPES = ("per_well", "merged")

# Policy for a ragged secondary channel (a channel whose time set differs from BF's). "fail" is the
# default fail-loud contract behavior; "warn" accepts it (a secondary-channel gap does not corrupt
# the BF segmentation timeline). BF contiguity + per-product-stream contiguity are ALWAYS hard fails.
RAGGED_CHANNEL_FAIL = "fail"
RAGGED_CHANNEL_WARN = "warn"
RAGGED_CHANNEL_POLICIES = (RAGGED_CHANNEL_FAIL, RAGGED_CHANNEL_WARN)


# ---------------------------------------------------------------------------
# L3 — grain rules (scope-aware)
# ---------------------------------------------------------------------------

def validate_grain(
    df: pd.DataFrame,
    *,
    validation_scope: str,
    scope_label: str,
    ragged_channel_policy: str = RAGGED_CHANNEL_FAIL,
) -> None:
    """Validate the per-well temporal grain, at the requested validation scope.

    Both scopes require exactly one ``experiment_id``. ``per_well`` additionally requires exactly one
    well; ``merged`` allows many wells and applies the per-well temporal checks grouped by ``well_id``.
    ``ragged_channel_policy`` ("fail" default | "warn") is forwarded to the channel-rectangular check.
    """
    if validation_scope not in VALIDATION_SCOPES:
        raise ValueError(
            f"[{scope_label}] unknown validation_scope {validation_scope!r}; "
            f"expected one of {VALIDATION_SCOPES}."
        )

    _assert_single_experiment(df, scope_label=scope_label)

    if validation_scope == "per_well":
        _assert_single_well(df, scope_label=scope_label)
        _validate_well_temporal_grain(
            df, scope_label=scope_label, ragged_channel_policy=ragged_channel_policy
        )
        return

    # merged: many wells allowed — apply the per-well temporal checks within each well_id group.
    well_ids = _derive_well_ids(df)
    for well_id, group in df.groupby(well_ids, sort=False):
        _validate_well_temporal_grain(
            group,
            scope_label=f"{scope_label}:{well_id}",
            ragged_channel_policy=ragged_channel_policy,
        )


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


def _validate_well_temporal_grain(
    df: pd.DataFrame, *, scope_label: str, ragged_channel_policy: str = RAGGED_CHANNEL_FAIL
) -> None:
    """Within ONE well: each PRODUCT STREAM is time-contiguous; BF channel present; elapsed_time_s.

    A frame's identity is ``well + channel + product_key + time + z``; a materialized well is just a
    set of product streams. The temporal guarantee is therefore PER STREAM — each
    ``(channel_id, product_key)`` stream must have contiguous time_index 0..N-1 on its OWN axis
    (a z_stack stream collapses its planes to distinct timepoints first — N planes at t0 is one
    timepoint, not N). Streams are NOT cross-checked against each other: a well may legitimately
    carry a projection across t0..t9 but a z_stack only at t0; that is a product choice, not a
    dropped frame, so there is no cross-stream "rectangular" requirement.

    Well-level invariants (all predate products; channel/row level, NOT product level):
      - the BF channel must be present (it anchors the segmentation timeline);
      - all CHANNELS are rectangular — every channel shares the BF channel's time set (a ragged
        channel, e.g. GFP missing a BF timepoint, is a dropped frame). This is across CHANNELS, not
        across products: products within a channel are independent (z_stack only at t0 while the
        projection spans t0..t9 is a product choice, not a defect);
      - a multi-timepoint well requires non-null elapsed_time_s on every row.
    """
    df = df.copy()
    df["product_key"] = frame_inventory_product_keys(df, scope_label=scope_label)
    df["channel_id"] = df["channel_id"].astype(str)

    _assert_each_product_stream_contiguous(df, scope_label=scope_label)
    _assert_required_channel_present(df, scope_label=scope_label)
    _assert_channels_rectangular(
        df, scope_label=scope_label, ragged_channel_policy=ragged_channel_policy
    )
    _assert_multitimepoint_has_elapsed_time(df, scope_label=scope_label)


def _distinct_times(group: pd.DataFrame) -> list[int]:
    """Distinct timepoints in a row group — a z_stack's N planes at one time_index count once."""
    return sorted(int(t) for t in group["time_index"].dropna().unique())


def _assert_each_product_stream_contiguous(df: pd.DataFrame, *, scope_label: str) -> None:
    """Each ``(channel, product_key)`` stream is contiguous 0..N-1 on its OWN axis.

    Per stream, never across products — a well may carry a projection across t0..t9 but a z_stack
    only at t0; that is a product choice, not a dropped frame.
    """
    for product_key, stream in df.groupby("product_key", sort=True):
        stream_times = _distinct_times(stream)
        expected = list(range(len(stream_times)))
        if stream_times != expected:
            raise ValueError(
                f"[{scope_label}] product stream {product_key!r} time_index must be contiguous "
                f"0..N-1; found {stream_times} (expected {expected}). Renumber or fill the missing "
                "frames for that product."
            )


def _assert_required_channel_present(df: pd.DataFrame, *, scope_label: str) -> None:
    """The BF channel must be present — it anchors the segmentation timeline."""
    channels = sorted(df["channel_id"].unique())
    if REQUIRED_CHANNEL not in channels:
        raise ValueError(
            f"[{scope_label}] required channel {REQUIRED_CHANNEL!r} is absent (present: {channels}). "
            f"Every well must have a {REQUIRED_CHANNEL} channel — it anchors the segmentation timeline."
        )


def _assert_channels_rectangular(
    df: pd.DataFrame, *, scope_label: str, ragged_channel_policy: str = RAGGED_CHANNEL_FAIL
) -> None:
    """Every CHANNEL shares the BF channel's time set (a ragged channel is a dropped frame).

    Across channels, NOT across products — channels of one well are one acquisition (GFP missing a BF
    timepoint is a real defect); products within a channel are independent choices.

    ``ragged_channel_policy`` controls the response to a ragged channel:
      - ``"fail"`` (default): raise — fail-loud contract boundary;
      - ``"warn"``: log a warning and accept (a secondary-channel gap does not corrupt the BF
        segmentation timeline). BF contiguity and per-product-stream contiguity are ALWAYS hard
        failures regardless of this policy.
    """
    if ragged_channel_policy not in RAGGED_CHANNEL_POLICIES:
        raise ValueError(
            f"[{scope_label}] unknown ragged_channel_policy {ragged_channel_policy!r}; "
            f"expected one of {RAGGED_CHANNEL_POLICIES}."
        )
    bf_times = _distinct_times(df[df["channel_id"] == REQUIRED_CHANNEL])
    for channel in sorted(df["channel_id"].unique()):
        ch_times = _distinct_times(df[df["channel_id"] == channel])
        if ch_times != bf_times:
            message = (
                f"[{scope_label}] channel {channel!r} has time_index set {ch_times} but "
                f"{REQUIRED_CHANNEL} has {bf_times}. All channels must share the same time_index set "
                "(rectangular) — a ragged channel is almost always a dropped frame."
            )
            if ragged_channel_policy == RAGGED_CHANNEL_WARN:
                log.warning(message)
            else:
                raise ValueError(message)


def _assert_multitimepoint_has_elapsed_time(df: pd.DataFrame, *, scope_label: str) -> None:
    """A well with >1 distinct time_index requires non-null elapsed_time_s on every row."""
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
        declared_format = str(row["image_file_format"]).lower().lstrip(".")
        if declared_format == "jpeg":
            declared_format = "jpg"
        if declared_format == "tiff":
            declared_format = "tif"
        suffix = resolved.suffix.lower().lstrip(".")
        if suffix == "jpeg":
            suffix = "jpg"
        if suffix == "tiff":
            suffix = "tif"
        if suffix != declared_format:
            raise ValueError(
                f"[{scope_label}] image_file_format mismatch for {resolved}: path suffix "
                f"{resolved.suffix!r} implies {suffix!r}, row declares {declared_format!r}."
            )
        downsample_factor = int(row["downsample_factor"])
        if downsample_factor < 1:
            raise ValueError(
                f"[{scope_label}] downsample_factor must be >= 1; row {idx} has "
                f"{downsample_factor}."
            )
        if declared_format == "jpg" and _is_nullish(row["jpeg_quality"]):
            raise ValueError(
                f"[{scope_label}] jpg row {idx} requires non-null jpeg_quality."
            )
        if declared_format != "jpg" and not _is_nullish(row["jpeg_quality"]):
            raise ValueError(
                f"[{scope_label}] non-jpg row {idx} must leave jpeg_quality null; "
                f"got {row['jpeg_quality']!r}."
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
        source_w = int(row["source_image_width_px"])
        source_h = int(row["source_image_height_px"])
        expected_w, expected_h = expected_downsampled_dims(
            source_w, source_h, downsample_factor, str(row["downsample_method"])
        )
        if (declared_w, declared_h) != (expected_w, expected_h):
            raise ValueError(
                f"[{scope_label}] image dims disagree with write policy for {resolved}: "
                f"source dims {source_w}x{source_h} with downsample_factor={downsample_factor} "
                f"and downsample_method={row['downsample_method']!r} imply "
                f"{expected_w}x{expected_h}, but the manifest declares {declared_w}x{declared_h}."
            )
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

    # L4b — construction-provenance: the focus_index_map .npz (focus_stack projection only).
    _validate_focus_index_map_provenance(df, image_root=image_root, scope_label=scope_label)


def _is_nullish(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""


def _row_is_focus_stack_projection(row: pd.Series) -> bool:
    """Legacy-tolerant: a row is focus_stack projection if it is projection/focus_stack.

    If ``image_product_type`` is absent, treat the row as legacy projection; if
    ``projection_method`` is absent on a projection row, treat it as focus_stack.
    """
    ipt = str(row["image_product_type"]) if "image_product_type" in row.index and not pd.isna(
        row.get("image_product_type")
    ) else "projection"
    if ipt != "projection":
        return False
    pm = row.get("projection_method") if "projection_method" in row.index else None
    if pm is None or pd.isna(pm):
        return True  # legacy projection rows are focus_stack
    return str(pm) == "focus_stack"


def _validate_focus_index_map_provenance(
    df: pd.DataFrame, *, image_root: Path | None, scope_label: str
) -> None:
    """L4b — validate the focus_index_map construction-provenance .npz.

    Context available here is the shard rows + the .npz files (NOT the acquisition inventory), so the
    inventory-aware ``z_indices == ordered acquisition labels`` check is intentionally NOT performed
    here — see ``validate_focus_index_map_against_inventory`` for that (called only where the
    acquisition inventory is in hand). Here we check, per row:

      - projection/focus_stack rows: ``focus_index_map_path`` present; the ``.npz`` resolves + loads;
        it carries ``focus_index_map`` + ``z_indices``; ``focus_index_map`` is an integer-valued
        2D array with shape ``(image_height_px, image_width_px)``; ``focus_index_map.min() >= 0``
        and ``focus_index_map.max() < len(z_indices)``.
      - z_stack and non-focus_stack projection rows: ``focus_index_map_path`` must be NA (provenance
        is not an image and rides only with the focus_stack projection it explains).

    If the column is entirely absent (legacy shards), the check is skipped (back-compat).
    """
    import numpy as np

    if "focus_index_map_path" not in df.columns:
        return

    for idx, row in df.iterrows():
        raw = row.get("focus_index_map_path")
        has_path = not (raw is None or pd.isna(raw) or str(raw).strip() == "")
        if not _row_is_focus_stack_projection(row):
            if has_path:
                raise ValueError(
                    f"[{scope_label}] row {idx} is not projection/focus_stack but carries a "
                    f"focus_index_map_path ({raw!r}). Provenance rides only with the focus_stack "
                    "projection it explains; z_stack / non-focus_stack rows must leave it NA."
                )
            continue

        if not has_path:
            raise ValueError(
                f"[{scope_label}] projection/focus_stack row {idx} is missing focus_index_map_path. "
                "Every focus_stack projection must carry its focus_index_map provenance .npz."
            )
        resolved = _resolve_source_path(str(raw), image_root=image_root, scope_label=scope_label)
        if resolved.suffix.lower() != ".npz":
            raise ValueError(
                f"[{scope_label}] focus_index_map_path must be a .npz; row {idx} has {resolved}."
            )
        if not resolved.exists():
            raise ValueError(
                f"[{scope_label}] focus_index_map_path does not exist: {resolved} (row {idx})."
            )
        try:
            with np.load(resolved) as data:
                if "focus_index_map" not in data or "z_indices" not in data:
                    raise ValueError(
                        f"[{scope_label}] focus_index_map .npz {resolved} must contain both "
                        f"'focus_index_map' and 'z_indices'; has {list(data.keys())}."
                    )
                fim = data["focus_index_map"]
                z_indices = data["z_indices"]
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface the underlying load error by name.
            raise ValueError(
                f"[{scope_label}] focus_index_map .npz {resolved} could not be loaded: {exc}"
            ) from exc

        if len(z_indices) == 0:
            raise ValueError(
                f"[{scope_label}] focus_index_map .npz {resolved} has empty z_indices (row {idx})."
            )
        expected_shape = (int(row["image_height_px"]), int(row["image_width_px"]))
        if fim.ndim != 2:
            raise ValueError(
                f"[{scope_label}] focus_index_map must be a 2D array with shape "
                f"{expected_shape}; row {idx} has ndim={fim.ndim} at {resolved}."
            )
        if tuple(fim.shape) != expected_shape:
            raise ValueError(
                f"[{scope_label}] focus_index_map shape mismatch for row {idx}: "
                f"expected {expected_shape} from image_height_px/image_width_px, "
                f"got {tuple(fim.shape)} at {resolved}."
            )
        if not np.issubdtype(fim.dtype, np.integer):
            raise ValueError(
                f"[{scope_label}] focus_index_map values must be integer stack-axis offsets; "
                f"row {idx} has dtype {fim.dtype} at {resolved}."
            )
        if int(fim.min()) < 0 or int(fim.max()) >= len(z_indices):
            raise ValueError(
                f"[{scope_label}] focus_index_map values must be stack-axis offsets in "
                f"[0, {len(z_indices)}); row {idx} has range "
                f"[{int(fim.min())}, {int(fim.max())}] at {resolved}."
            )


def validate_focus_index_map_against_inventory(
    df: pd.DataFrame,
    *,
    well_acquisition_inventory_df: pd.DataFrame,
    image_root: Path | None = None,
    scope_label: str = "frame_inventory",
) -> None:
    """OPTIONAL inventory-aware check: z_indices == ordered acquisition z_index labels.

    This is the context-dependent half of focus_index_map validation, intentionally SEPARATE from
    the shard-only ``_validate_focus_index_map_provenance`` so the shard validator never requires
    context (the acquisition inventory) it does not receive. Call this only where the acquisition
    inventory is in hand. For each projection/focus_stack row it asserts the .npz ``z_indices``
    equals the ordered (sorted) acquisition ``z_index`` labels for that (channel_id, time_index).
    """
    import numpy as np

    if "focus_index_map_path" not in df.columns:
        return
    for idx, row in df.iterrows():
        if not _row_is_focus_stack_projection(row):
            continue
        raw = row.get("focus_index_map_path")
        if raw is None or pd.isna(raw):
            continue
        resolved = _resolve_source_path(str(raw), image_root=image_root, scope_label=scope_label)
        with np.load(resolved) as data:
            z_indices = list(int(z) for z in data["z_indices"])
        sel = well_acquisition_inventory_df[
            (well_acquisition_inventory_df["channel_id"].astype(str) == str(row["channel_id"]))
            & (well_acquisition_inventory_df["time_index"].astype(int) == int(row["time_index"]))
        ]
        expected = sorted(int(z) for z in sel["z_index"].dropna().unique())
        if z_indices != expected:
            raise ValueError(
                f"[{scope_label}] focus_index_map z_indices {z_indices} for row {idx} do not match "
                f"the ordered acquisition z_index labels {expected} for "
                f"(channel={row['channel_id']}, time_index={row['time_index']})."
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
