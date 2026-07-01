"""Materialized image readers — the read side of the materialized image tree.

Read-only mirror of ``materialized_image_paths.py`` (the write grammar). ``frame_inventory`` is the
validated read index for materialized image products: the materializer writes files using the
write grammar and emits ``frame_inventory`` rows recording each product's path as
``source_image_path``; the ``frame_inventory`` validator proves those rows agree with the product
identity/path contract. After that point, downstream consumers must not reconstruct paths from
identity atoms — they resolve the validated row by product identity and read its recorded
``source_image_path``.

Paths writes. Inventory remembers. Validation makes memory trustworthy. Readers read the memory.

Two verbs:
  - ``resolve_*`` returns ``frame_inventory`` row(s) — facts, no pixels.
  - ``load_*`` returns ndarray pixels, reading the row's RECORDED ``source_image_path`` — never
    recomputed via ``materialized_image_paths.py``.
``*_from_image_id`` performs an inventory lookup; ``*_from_row(s)`` consumes already-resolved rows.

Image-id deconstruction is never done by hand in this module — it is imported from
``shared/identifiers/parsers`` (``parse_image_id_with_z_index``). Hand-parsing would fork the id
grammar.

Import direction: this module imports ``frame_inventory_contract`` (column names / id semantics),
``image_product_keys`` (product-key parsing), and ``shared/identifiers/parsers`` (id parsing). It
must NOT import ``materialized_image_paths``, orchestration, tasks, rules, or QC.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data_pipeline.acquisition.image_materialization.image_product_keys import parse_image_product_key
from data_pipeline.shared.identifiers.parsers import parse_image_id_with_z_index


# ─────────────────────────────────────────────────────────────────────────────
# resolve: image identity -> validated frame_inventory row(s)
# ─────────────────────────────────────────────────────────────────────────────


def resolve_projection_row_from_image_id(
    frame_inventory_df: pd.DataFrame,
    *,
    image_id: str,
    product_key: str,
) -> pd.Series:
    """Return the single validated projection row for ``(image_id, product_key)``.

    Fails loud if ``product_key`` is not a projection product, if no row matches, or if more
    than one row matches.
    """
    _, image_product_type, _ = parse_image_product_key(product_key)
    if image_product_type != "projection":
        raise ValueError(
            f"resolve_projection_row_from_image_id: product_key {product_key!r} is not a "
            "projection product (got image_product_type="
            f"{image_product_type!r}). Use resolve_z_stack_rows_from_image_id for z_stack products."
        )

    well_id, channel_id, time_index, z_index = parse_image_id_with_z_index(image_id)
    if z_index is not None:
        raise ValueError(
            f"resolve_projection_row_from_image_id: image_id {image_id!r} carries a z_index "
            f"({z_index}) — a z-plane id cannot name a projection. A projection image_id has no "
            "z atom."
        )

    matches = frame_inventory_df[
        (frame_inventory_df["image_id"].astype(str) == str(image_id))
        & (_row_product_keys(frame_inventory_df) == product_key)
    ]
    if len(matches) == 0:
        raise ValueError(
            f"resolve_projection_row_from_image_id: no row found for image_id={image_id!r}, "
            f"product_key={product_key!r}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"resolve_projection_row_from_image_id: {len(matches)} rows found for "
            f"image_id={image_id!r}, product_key={product_key!r}; expected exactly one. "
            "Missing or ambiguous rows are contract failures."
        )
    row = matches.iloc[0]
    if str(row["image_product_type"]) != "projection":
        raise ValueError(
            f"resolve_projection_row_from_image_id: matched row's image_product_type "
            f"{row['image_product_type']!r} disagrees with product_key {product_key!r} "
            "(expected 'projection')."
        )
    return row


def resolve_z_stack_rows_from_image_id(
    frame_inventory_df: pd.DataFrame,
    *,
    image_id: str,
    product_key: str,
) -> pd.DataFrame:
    """Return all validated z_stack rows for the stack containing ``image_id``, sorted by z_index.

    The realistic caller holds a PROJECTION-grain ``image_id`` (snip provenance is always
    projection-grain) and wants the z-stack for that same timepoint — that is the normal path. A
    z-plane image_id is also accepted; its plane atom is used only to locate the stack, then
    discarded for selection (auto-resolve: any member of the stack returns the whole stack).

    Fails loud if ``product_key`` is not a z_stack product, if no rows match, or if any matched
    z_index is duplicated.
    """
    _, image_product_type, _ = parse_image_product_key(product_key)
    if image_product_type != "z_stack":
        raise ValueError(
            f"resolve_z_stack_rows_from_image_id: product_key {product_key!r} is not a z_stack "
            f"product (got image_product_type={image_product_type!r}). Use "
            "resolve_projection_row_from_image_id for projection products."
        )

    # image_id may be projection-grain (normal) or z-plane-grain — both accepted. The parsed
    # z_index, if any, is discarded here: only the stack key (well, channel, time) is used.
    well_id, channel_id, time_index, _z_index = parse_image_id_with_z_index(image_id)

    matches = frame_inventory_df[
        (frame_inventory_df["well_id"].astype(str) == str(well_id))
        & (frame_inventory_df["channel_id"].astype(str) == str(channel_id))
        & (frame_inventory_df["time_index"].astype(int) == int(time_index))
        & (_row_product_keys(frame_inventory_df) == product_key)
    ]
    if len(matches) == 0:
        raise ValueError(
            f"resolve_z_stack_rows_from_image_id: no z_stack rows found for image_id={image_id!r} "
            f"(well_id={well_id!r}, channel_id={channel_id!r}, time_index={time_index!r}), "
            f"product_key={product_key!r}."
        )
    bad_types = matches[matches["image_product_type"].astype(str) != "z_stack"]
    if len(bad_types) > 0:
        raise ValueError(
            f"resolve_z_stack_rows_from_image_id: matched row(s) have image_product_type != "
            f"'z_stack' (product_key={product_key!r} disagrees with the data)."
        )

    z_values = matches["z_index"].tolist()
    dupes = {z for z in z_values if z_values.count(z) > 1}
    if dupes:
        raise ValueError(
            f"resolve_z_stack_rows_from_image_id: duplicate z_index value(s) {sorted(dupes)} for "
            f"well_id={well_id!r}, channel_id={channel_id!r}, time_index={time_index!r}, "
            f"product_key={product_key!r}. Every z-plane must be unique."
        )

    return matches.sort_values("z_index").reset_index(drop=True)


def _row_product_keys(frame_inventory_df: pd.DataFrame) -> pd.Series:
    """Compose the product_key for every row from its product columns (channel/type/method)."""
    from data_pipeline.acquisition.image_materialization.image_product_keys import (
        image_product_key_for_frame_row,
    )

    return frame_inventory_df.apply(
        lambda row: image_product_key_for_frame_row(
            channel_id=row["channel_id"],
            image_product_type=row["image_product_type"],
            projection_method=row.get("projection_method"),
        ),
        axis=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# load: row(s) -> pixels (reads RECORDED source_image_path, never recomputes it)
# ─────────────────────────────────────────────────────────────────────────────


def load_materialized_image_from_row(
    frame_inventory_row: pd.Series,
    *,
    image_root: Path | None = None,
) -> np.ndarray:
    """Read the pixels named by ``frame_inventory_row['source_image_path']``.

    Reads the RECORDED path; never reconstructs it via ``materialized_image_paths.py``.
    Encoding-agnostic (jpg/png/tif) — the caller does not need to know the suffix.
    """
    path = _resolve_source_image_path(frame_inventory_row["source_image_path"], image_root)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(
            f"load_materialized_image_from_row: could not read image at {path} "
            f"(source_image_path={frame_inventory_row['source_image_path']!r})."
        )
    return image


def load_materialized_images_from_rows(
    frame_inventory_rows: pd.DataFrame,
    *,
    image_root: Path | None = None,
) -> list[np.ndarray]:
    """Read pixels for each row, preserving input row order."""
    return [
        load_materialized_image_from_row(row, image_root=image_root)
        for _, row in frame_inventory_rows.iterrows()
    ]


def _resolve_source_image_path(value: object, image_root: Path | None) -> Path:
    path = Path(str(value))
    if not path.is_absolute() and image_root is not None:
        path = image_root / path
    return path


# ─────────────────────────────────────────────────────────────────────────────
# convenience wrappers: resolve + load in one call
# ─────────────────────────────────────────────────────────────────────────────


def load_projection_image(
    frame_inventory_df: pd.DataFrame,
    *,
    image_id: str,
    product_key: str,
    image_root: Path | None = None,
) -> np.ndarray:
    """Resolve the projection row for ``(image_id, product_key)`` and load its pixels."""
    row = resolve_projection_row_from_image_id(
        frame_inventory_df, image_id=image_id, product_key=product_key
    )
    return load_materialized_image_from_row(row, image_root=image_root)


def load_z_stack_images_from_image_id(
    frame_inventory_df: pd.DataFrame,
    *,
    image_id: str,
    product_key: str,
    image_root: Path | None = None,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Resolve the z-stack rows for ``image_id`` and load their pixels, z-ordered."""
    rows = resolve_z_stack_rows_from_image_id(
        frame_inventory_df, image_id=image_id, product_key=product_key
    )
    return load_materialized_images_from_rows(rows, image_root=image_root), rows
