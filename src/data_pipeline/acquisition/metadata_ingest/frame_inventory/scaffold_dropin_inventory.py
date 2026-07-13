"""Scaffold a starter drop-in frame_inventory manifest from an image directory.

For the "researcher + AI assistant" user, a blank CSV is a worse start than a filled-in draft. This
helper reads an image directory, reads each image header for the real dimensions, infers the frame
atoms it can from the filename, and emits a starter ``dropin_frame_inventory.csv`` the user edits
(filling ``image_micrometers_per_pixel`` and correcting any inferred atoms).

It is the inverse read direction of the validator, expressed as **free functions, not a class**, and
it **does not move or reorganize the user's images** (manifest-is-truth makes reorganization never
required). Age does **not** live here — ``start_age_hpf`` is biology metadata, not a frame_inventory
column; the scaffold only touches the manifest.

Filename convention it recognizes (the recommended self-describing layout):
``{well_id}_{channel_id}_t{time_index:04d}.{ext}`` — e.g. ``my_experiment_B01_BF_t0000.png``. The
``experiment_id`` / ``well_index`` atoms and ``image_micrometers_per_pixel`` are left blank for the
user to fill (a well_id embeds them, but the scaffold does not guess the experiment/well split — that
is the one place the user must declare intent). Unparseable names get blank channel/time the user fills.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from PIL import Image

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    ALLOWED_IMAGE_SUFFIXES,
    REQUIRED_FRAME_INVENTORY_COLUMNS,
)

# {anything}_{channel}_t{NNNN}.{ext} — channel is a token, time is the zero-padded T index.
_FRAME_NAME_RE = re.compile(r"^(?P<rest>.+)_(?P<channel>[A-Za-z0-9]+)_t(?P<time>\d+)$")


def discover_image_files(image_dir: Path) -> list[Path]:
    """Return the sorted list of supported image files under ``image_dir`` (recursive)."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"[scaffold] image_dir is not a directory: {image_dir}")
    files = [
        p for p in sorted(image_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
    ]
    if not files:
        raise ValueError(
            f"[scaffold] no supported images found under {image_dir} "
            f"(allowed suffixes: {ALLOWED_IMAGE_SUFFIXES})."
        )
    return files


def scaffold_row_for_image(image_path: Path, *, image_root: Path | None = None) -> dict:
    """Build one starter manifest row for ``image_path`` (dims from the header; atoms inferred)."""
    image_path = Path(image_path)
    with Image.open(image_path) as im:
        width, height = im.size

    channel_id, time_index = _infer_channel_and_time(image_path.stem)

    if image_root is not None:
        try:
            materialized_image_path = str(image_path.resolve().relative_to(Path(image_root).resolve()))
        except ValueError:
            materialized_image_path = str(image_path.resolve())  # outside root → absolute
    else:
        materialized_image_path = str(image_path.resolve())

    # Author the atoms the scaffold cannot guess as blanks; the user fills experiment_id / well_index
    # / image_micrometers_per_pixel before validation. well_id / image_id are NEVER authored.
    row = {col: "" for col in REQUIRED_FRAME_INVENTORY_COLUMNS}
    row["channel_id"] = channel_id
    row["time_index"] = time_index
    row["image_path"] = materialized_image_path
    row["image_micrometers_per_pixel"] = ""
    row["image_width_px"] = width
    row["image_height_px"] = height
    fmt = image_path.suffix.lower().lstrip(".")
    row["image_file_format"] = {"jpeg": "jpg", "tiff": "tif"}.get(fmt, fmt)
    row["pixel_dtype"] = "uint8"
    row["downsample_factor"] = 1
    row["downsample_method"] = "none"
    row["jpeg_quality"] = pd.NA
    # A dropped-in image is a single materialized frame, i.e. a projection product (NOT a z-plane).
    # z_index stays NA; projection_method is the dropin default. The user may retarget if needed.
    row["image_product_type"] = "projection"
    row["projection_method"] = "focus_stack"
    row["z_index"] = pd.NA
    return row


def scaffold_dropin_inventory(
    image_dir: Path, output_csv: Path, *, image_root: Path | None = None
) -> pd.DataFrame:
    """Emit a starter ``dropin_frame_inventory.csv`` for every image under ``image_dir``.

    ``image_root`` (default = ``image_dir``) controls whether ``image_path`` is written
    relative (portable) or absolute. Does not move or reorganize any images.
    """
    image_dir = Path(image_dir)
    root = Path(image_root) if image_root is not None else image_dir
    rows = [scaffold_row_for_image(p, image_root=root) for p in discover_image_files(image_dir)]
    df = pd.DataFrame(rows, columns=list(REQUIRED_FRAME_INVENTORY_COLUMNS))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(
        f"[scaffold] wrote {len(df)} starter rows → {output_csv}. "
        "Fill experiment_id / well_index / image_micrometers_per_pixel, then validate."
    )
    return df


def _infer_channel_and_time(stem: str) -> tuple[str, object]:
    """Infer (channel_id, time_index) from a recommended-layout filename stem; blanks if unparseable."""
    match = _FRAME_NAME_RE.match(stem)
    if match is None:
        return "", ""
    return match.group("channel"), int(match.group("time"))
