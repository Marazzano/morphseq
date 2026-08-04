"""Run box-prompted SAM2 over a SeaHub embryo detection manifest.

The default visualization mode preserves the original validation products:
full-frame masks, masked snips, overlays, contact sheets, and a mask manifest.

``--areas-only`` is the full-corpus production mode. It makes one batched,
box-prompted SAM2 prediction call per source FOV and writes an atomically
checkpointed ``sam2_mask_areas.csv`` plus one cleaned, full-FOV binary mask per
embryo. Overlays, masked snips, and contact sheets are omitted in this mode.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage


HERE = Path(__file__).resolve().parent
SAM2_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/models/sam2"
)
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = SAM2_ROOT / "checkpoints/sam2.1_hiera_large.pt"
CROP_BOX_COLUMNS = ("crop_x1_px", "crop_y1_px", "crop_x2_px", "crop_y2_px")
RAW_BOX_COLUMNS = ("box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm")
AREA_CARRY_COLUMNS = (
    "experiment_id",
    "source_experiment_id",
    "stage_hpf",
    "stage_source_label",
    "perturbation_parsed",
    "perturbation_key",
    "perturbation_domain",
    "fov_label",
    "metadata_match_status",
    "metadata_collection_name",
    "image_path",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            HERE
            / "outputs/segmentation_validation_3_experiments/embryo_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "outputs/sam2_masks_3_experiments",
    )
    parser.add_argument(
        "--prompt-box-source",
        choices=("raw", "crop"),
        default="raw",
        help=(
            "Use unpadded normalized detector boxes ('raw', recommended) or "
            "padded presentation crop boxes ('crop') as SAM2 prompts."
        ),
    )
    parser.add_argument(
        "--areas-only",
        action="store_true",
        help=(
            "Full-corpus production mode: write a checkpointed CSV plus cleaned "
            "full-FOV binary masks; omit overlays, masked snips, and contact sheets."
        ),
    )
    return parser


def _prepare_output(output: Path, *, areas_only: bool) -> Path:
    output = output.expanduser().resolve()
    try:
        output.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside {HERE}: {output}") from exc

    if areas_only and output.exists():
        if not output.is_dir():
            raise FileExistsError(f"SAM2 area output is not a directory: {output}")
        existing = sorted(output.iterdir())
        if existing:
            preview = ", ".join(path.name for path in existing[:5])
            raise FileExistsError(
                "SAM2 area output must be fresh and empty; refusing to mix runs "
                f"in {output}. Existing entries include: {preview}."
            )

    output.mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(parents=True, exist_ok=True)
    if not areas_only:
        for subdirectory in (
            "mask_snips",
            "overlays",
            "mask_contact_sheets",
        ):
            (output / subdirectory).mkdir(parents=True, exist_ok=True)
    return output


def _validate_manifest(
    manifest: pd.DataFrame, *, prompt_box_source: str, areas_only: bool
) -> pd.DataFrame:
    required = {"image_id", "image_path", "embryo_position"}
    required.update(
        RAW_BOX_COLUMNS if prompt_box_source == "raw" else CROP_BOX_COLUMNS
    )
    if not areas_only:
        required.update({"experiment_id", "stem", *CROP_BOX_COLUMNS})
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"SAM2 embryo manifest is missing columns: {missing}")
    if manifest.empty:
        raise ValueError("SAM2 embryo manifest is empty.")
    if manifest[list(required)].isna().any().any():
        null_columns = sorted(
            column for column in required if manifest[column].isna().any()
        )
        raise ValueError(
            f"SAM2 embryo manifest has null values in required columns: {null_columns}"
        )

    validated = manifest.copy()
    if "source_fov_id" in validated.columns:
        source_ids = validated["source_fov_id"].where(
            validated["source_fov_id"].notna()
            & validated["source_fov_id"].astype(str).str.strip().ne("")
        )
        validated["_source_fov_id"] = source_ids.fillna(validated["image_id"])
    else:
        validated["_source_fov_id"] = validated["image_id"]
    validated["_source_fov_id"] = validated["_source_fov_id"].astype(str)

    for source_fov_id, group in validated.groupby("_source_fov_id", sort=False):
        if group["image_path"].astype(str).nunique() != 1:
            raise ValueError(
                f"Source FOV {source_fov_id!r} maps to multiple image paths."
            )
        if group["image_id"].astype(str).nunique() != 1:
            raise ValueError(
                f"Source FOV {source_fov_id!r} maps to multiple image_id values."
            )
        positions = pd.to_numeric(group["embryo_position"], errors="coerce")
        if positions.isna().any() or positions.duplicated().any():
            raise ValueError(
                f"Source FOV {source_fov_id!r} has invalid/duplicate embryo positions."
            )
        source_path = Path(str(group["image_path"].iloc[0]))
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Source image for FOV {source_fov_id!r} does not exist: {source_path}"
            )
    return validated


def _prompt_boxes(
    group: pd.DataFrame,
    *,
    image_width: int,
    image_height: int,
    prompt_box_source: str,
) -> np.ndarray:
    if prompt_box_source == "raw":
        boxes = group[list(RAW_BOX_COLUMNS)].to_numpy(dtype=np.float32)
        boxes[:, [0, 2]] *= int(image_width)
        boxes[:, [1, 3]] *= int(image_height)
        return boxes
    return group[list(CROP_BOX_COLUMNS)].to_numpy(dtype=np.float32)


def _normalize_predictions(
    masks: Any,
    scores: Any,
    *,
    expected_count: int,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    masks_array = np.asarray(masks)
    if masks_array.ndim == 4 and masks_array.shape[1] == 1:
        masks_array = masks_array[:, 0]
    scores_array = np.asarray(scores).reshape(-1)
    if masks_array.shape != (expected_count, *image_shape):
        raise ValueError(
            "SAM2 returned an unexpected mask shape: "
            f"{masks_array.shape}; expected {(expected_count, *image_shape)}."
        )
    if scores_array.shape != (expected_count,):
        raise ValueError(
            "SAM2 returned an unexpected score shape: "
            f"{scores_array.shape}; expected {(expected_count,)}."
        )
    return masks_array > 0.0, scores_array


def _rounded_prompt_box(
    prompt_box: Sequence[float],
    *,
    image_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Return a validated integer ``xyxy`` prompt box.

    Rounding deliberately matches the historical area-manifest behavior so the
    existing prompt coordinate and area columns retain their meaning.
    """
    values = np.asarray(prompt_box, dtype=float).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid SAM2 prompt box: {prompt_box!r}")
    x1, y1, x2, y2 = (int(round(value)) for value in values)
    if image_shape is not None:
        height, width = image_shape
        x1 = min(max(x1, 0), int(width))
        x2 = min(max(x2, 0), int(width))
        y1 = min(max(y1, 0), int(height))
        y2 = min(max(y2, 0), int(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "SAM2 prompt box must have positive area after rounding/clipping; "
            f"got {(x1, y1, x2, y2)}."
        )
    return x1, y1, x2, y2


def _prompt_area(prompt_box: Sequence[float]) -> int:
    x1, y1, x2, y2 = _rounded_prompt_box(prompt_box)
    return int((x2 - x1) * (y2 - y1))


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return a nonempty binary mask's half-open ``xyxy`` bounding box."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Mask bbox requires a 2-D mask; got {binary.shape}.")
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("Mask bbox requires a nonempty mask.")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _validate_cleaned_mask(
    mask: np.ndarray,
    *,
    prompt_box: Sequence[float],
    context: str,
) -> None:
    """Fail loud unless a cleaned embryo mask satisfies the production contract."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"{context}: cleaned mask must be 2-D; got {binary.shape}.")
    if not binary.any():
        raise ValueError(f"{context}: cleaned mask is empty.")

    _, component_count = ndimage.label(binary)
    if int(component_count) != 1:
        raise ValueError(
            f"{context}: cleaned mask must contain exactly one connected component; "
            f"found {component_count}."
        )
    if not np.array_equal(ndimage.binary_fill_holes(binary), binary):
        raise ValueError(f"{context}: cleaned mask still contains holes.")

    ratio = float(binary.sum() / _prompt_area(prompt_box))
    if ratio >= 0.95:
        raise ValueError(
            f"{context}: cleaned mask-to-prompt area ratio {ratio:.6f} is >= 0.95; "
            "rejecting a likely prompt-box/full-inset blowout."
        )


def _clean_prompt_associated_component(
    raw_mask: np.ndarray,
    *,
    prompt_box: Sequence[float],
    context: str = "SAM2 mask",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the prompt-associated component, fill holes, and return audit fields.

    Selection is deterministic: use the component containing the prompt center
    when one exists; otherwise maximize prompt-box overlap, then component area,
    then prefer the lowest component label as a stable final tie-break.
    """
    binary = np.asarray(raw_mask) > 0
    if binary.ndim != 2:
        raise ValueError(f"{context}: raw mask must be 2-D; got {binary.shape}.")

    labels, component_count = ndimage.label(binary)
    component_count = int(component_count)
    if component_count == 0:
        raise ValueError(f"{context}: raw SAM2 mask is empty.")

    component_areas = np.bincount(labels.ravel(), minlength=component_count + 1)
    component_areas[0] = 0
    raw_area = int(binary.sum())

    height, width = binary.shape
    prompt_values = np.asarray(prompt_box, dtype=float).reshape(4)
    center_x = min(
        max(int(np.floor((prompt_values[0] + prompt_values[2]) / 2)), 0),
        width - 1,
    )
    center_y = min(
        max(int(np.floor((prompt_values[1] + prompt_values[3]) / 2)), 0),
        height - 1,
    )
    center_label = int(labels[center_y, center_x])

    if center_label > 0:
        selected_label = center_label
        selection_method = "prompt_center"
    else:
        x1, y1, x2, y2 = _rounded_prompt_box(
            prompt_box,
            image_shape=(height, width),
        )
        overlaps = np.bincount(
            labels[y1:y2, x1:x2].ravel(),
            minlength=component_count + 1,
        )
        overlaps[0] = 0
        selected_label = max(
            range(1, component_count + 1),
            key=lambda label_id: (
                int(overlaps[label_id]),
                int(component_areas[label_id]),
                -int(label_id),
            ),
        )
        selection_method = "max_prompt_box_overlap"

    selected = labels == selected_label
    selected_area = int(selected.sum())
    cleaned = np.asarray(ndimage.binary_fill_holes(selected), dtype=bool)
    holes_filled = int(cleaned.sum()) - selected_area
    removed_area = raw_area - selected_area

    _validate_cleaned_mask(cleaned, prompt_box=prompt_box, context=context)
    audit = {
        "raw_mask_area_px": raw_area,
        "component_count_raw": component_count,
        # Compatibility alias for the original audit-field vocabulary.
        "raw_component_count": component_count,
        "removed_component_area_px": removed_area,
        "holes_filled_px": holes_filled,
        # Compatibility alias for the original audit-field vocabulary.
        "holes_filled_area_px": holes_filled,
        "component_selection_method": selection_method,
    }
    return cleaned, audit


def _binary_mask_path(output: Path, source_fov_id: str, embryo_position: int) -> Path:
    token = str(source_fov_id).strip()
    if not token or Path(token).name != token or token in {".", ".."}:
        raise ValueError(f"Unsafe source_fov_id for mask filename: {source_fov_id!r}")
    return output / "masks" / f"{token}__embryo_{int(embryo_position):02d}_mask.png"


def _write_binary_mask(mask: np.ndarray, output_path: Path) -> Path:
    """Atomically persist a full-FOV mask as an 8-bit binary PNG."""
    binary = np.asarray(mask, dtype=bool)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    Image.fromarray(binary.astype(np.uint8) * 255).save(temporary, format="PNG")
    temporary.replace(output_path)
    return output_path


def _area_record(
    row: pd.Series,
    *,
    source_fov_id: str,
    mask: np.ndarray,
    mask_score: float,
    prompt_box: Sequence[float],
    prompt_box_source: str,
    mask_path: str | Path,
    mask_audit: dict[str, Any],
) -> dict[str, Any]:
    x1, y1, x2, y2 = _rounded_prompt_box(prompt_box)
    prompt_area = _prompt_area(prompt_box)
    mask_area = int(mask.sum())
    mask_x1, mask_y1, mask_x2, mask_y2 = _mask_bbox(mask)
    cleaned_ratio = float(mask_area / prompt_area)
    record: dict[str, Any] = {
        "image_id": str(row["image_id"]),
        "source_fov_id": str(source_fov_id),
        "embryo_position": int(row["embryo_position"]),
        "mask_path": str(Path(mask_path).resolve()),
        "mask_score": float(mask_score),
        "mask_area_px": mask_area,
        "mask_bbox_x1_px": mask_x1,
        "mask_bbox_y1_px": mask_y1,
        "mask_bbox_x2_px": mask_x2,
        "mask_bbox_y2_px": mask_y2,
        "prompt_box_source": prompt_box_source,
        "prompt_x1_px": x1,
        "prompt_y1_px": y1,
        "prompt_x2_px": x2,
        "prompt_y2_px": y2,
        "prompt_box_area_px": int(prompt_area),
        # Historical name retained for calibration consumers.
        "mask_to_prompt_area_ratio": cleaned_ratio,
        "cleaned_to_prompt_area_ratio": cleaned_ratio,
        **mask_audit,
    }
    for column in AREA_CARRY_COLUMNS:
        if column in row.index:
            record[column] = row[column]
    return record


def _write_checkpoint(
    rows: list[dict[str, Any]],
    output_csv: Path,
    *,
    processed_fov_count: int,
    total_fov_count: int,
    complete: bool,
) -> None:
    checkpoint = pd.DataFrame.from_records(rows)
    checkpoint["processed_fov_count"] = int(processed_fov_count)
    checkpoint["total_fov_count"] = int(total_fov_count)
    checkpoint["checkpoint_complete"] = bool(complete)
    temporary = output_csv.with_name(f".{output_csv.name}.tmp")
    checkpoint.to_csv(temporary, index=False)
    temporary.replace(output_csv)


def _validate_production_coverage(
    rows: list[dict[str, Any]],
    manifest: pd.DataFrame,
    *,
    output: Path,
) -> None:
    """Require exact 1:1 manifest and PNG coverage for production mask output."""
    required_columns = {
        "source_fov_id",
        "embryo_position",
        "mask_path",
        "mask_area_px",
        "mask_bbox_x1_px",
        "mask_bbox_y1_px",
        "mask_bbox_x2_px",
        "mask_bbox_y2_px",
        "raw_mask_area_px",
        "component_count_raw",
        "component_selection_method",
        "removed_component_area_px",
        "holes_filled_px",
        "cleaned_to_prompt_area_ratio",
        "mask_score",
    }
    records = pd.DataFrame.from_records(rows)
    missing = sorted(required_columns.difference(records.columns))
    if missing:
        raise ValueError(f"Production SAM2 manifest is missing columns: {missing}")

    expected_keys = [
        (str(row["_source_fov_id"]), int(row["embryo_position"]))
        for _, row in manifest.iterrows()
    ]
    actual_keys = [
        (str(row["source_fov_id"]), int(row["embryo_position"]))
        for _, row in records.iterrows()
    ]
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("Input manifest has duplicate source-FOV/embryo-position keys.")
    if len(set(actual_keys)) != len(actual_keys):
        raise ValueError("Production SAM2 manifest has duplicate mask keys.")
    if set(actual_keys) != set(expected_keys):
        missing_keys = sorted(set(expected_keys) - set(actual_keys))[:5]
        extra_keys = sorted(set(actual_keys) - set(expected_keys))[:5]
        raise ValueError(
            "Production SAM2 mask coverage does not match the input manifest; "
            f"missing={missing_keys}, extra={extra_keys}."
        )

    recorded_paths = [Path(value) for value in records["mask_path"]]
    if not all(path.is_absolute() for path in recorded_paths):
        raise ValueError("Production SAM2 mask_path values must all be absolute.")
    if len(set(recorded_paths)) != len(recorded_paths):
        raise ValueError("Production SAM2 mask_path values must be unique.")
    if not all(path.is_file() for path in recorded_paths):
        missing_paths = [str(path) for path in recorded_paths if not path.is_file()][:5]
        raise FileNotFoundError(
            f"Production SAM2 manifest references missing masks: {missing_paths}"
        )

    disk_paths = set((output / "masks").glob("*.png"))
    if disk_paths != set(recorded_paths):
        raise ValueError(
            "Production SAM2 masks directory is not an exact 1:1 realization of "
            "the manifest mask_path column."
        )


def _install_iopath_shim() -> None:
    # The local checkout imports iopath only for an optional Hiera backbone
    # checkpoint. The configured model loads its SAM2 checkpoint separately.
    if importlib.util.find_spec("iopath") is not None:
        return
    iopath_module = types.ModuleType("iopath")
    common_module = types.ModuleType("iopath.common")
    file_io_module = types.ModuleType("iopath.common.file_io")

    class LocalPathManager:
        @staticmethod
        def open(path, mode="r", **kwargs):
            return builtins.open(path, mode, **kwargs)

    file_io_module.g_pathmgr = LocalPathManager()
    common_module.file_io = file_io_module
    iopath_module.common = common_module
    sys.modules["iopath"] = iopath_module
    sys.modules["iopath.common"] = common_module
    sys.modules["iopath.common.file_io"] = file_io_module


def _load_predictor():
    _install_iopath_shim()
    sys.path.insert(0, str(SAM2_ROOT))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; loading SAM2 hiera_large ...", flush=True)
    sam2 = build_sam2(SAM2_CFG, str(SAM2_CKPT), device=device)
    print("SAM2 loaded.", flush=True)
    return SAM2ImagePredictor(sam2), torch


def _write_visualization_products(
    *,
    output: Path,
    image_id: str,
    group: pd.DataFrame,
    image: np.ndarray,
    crop_boxes: np.ndarray,
    prompt_boxes: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    mask_audits: Sequence[dict[str, Any]],
    rows_out: list[dict[str, Any]],
    prompt_box_source: str,
) -> None:
    overlay = Image.fromarray(image).convert("RGBA")
    tint = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(tint)
    palette = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 120, 255),
        (255, 220, 60),
        (255, 80, 255),
        (60, 230, 230),
        (255, 150, 40),
        (160, 120, 255),
    ]
    crops: list[Image.Image] = []

    for index, (_, row) in enumerate(group.iterrows()):
        position = int(row["embryo_position"])
        mask = masks[index]
        crop_x1, crop_y1, crop_x2, crop_y2 = (
            int(value) for value in crop_boxes[index]
        )
        mask_path = output / "masks" / (
            f"{image_id}__embryo_{position:02d}_mask.png"
        )
        snip_path = output / "mask_snips" / (
            f"{image_id}__embryo_{position:02d}.png"
        )
        mask_path = _write_binary_mask(mask, mask_path)

        sub_image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        sub_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        sub_image[~sub_mask] = 255
        masked_image = Image.fromarray(sub_image)
        masked_image.save(snip_path)
        display_crop = masked_image.copy()
        ImageDraw.Draw(display_crop).text(
            (6, 4),
            f"{position}: SAM {scores[index]:.3f}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        crops.append(display_crop)

        color = palette[(position - 1) % len(palette)]
        ys, xs = np.where(mask)
        if len(xs):
            tint_array = np.asarray(tint).copy()
            tint_array[ys, xs] = (*color, 110)
            tint = Image.fromarray(tint_array)
            draw = ImageDraw.Draw(tint)
        prompt_coords = [int(round(value)) for value in prompt_boxes[index]]
        draw.rectangle(prompt_coords, outline=(*color, 255), width=3)

        record = _area_record(
            row,
            source_fov_id=str(row["_source_fov_id"]),
            mask=mask,
            mask_score=float(scores[index]),
            prompt_box=prompt_boxes[index],
            prompt_box_source=prompt_box_source,
            mask_path=mask_path,
            mask_audit=mask_audits[index],
        )
        record.update(
            {
                "stem": row["stem"],
                "crop_x1_px": crop_x1,
                "crop_y1_px": crop_y1,
                "crop_x2_px": crop_x2,
                "crop_y2_px": crop_y2,
                "mask_snip_path": str(snip_path),
            }
        )
        rows_out.append(record)

    Image.alpha_composite(overlay, tint).convert("RGB").save(
        output / "overlays" / f"{image_id}__overlay.jpg", quality=88
    )
    crop_width = max(crop.width for crop in crops)
    crop_height = max(crop.height for crop in crops)
    sheet = Image.new(
        "RGB", (crop_width * 4 + 10, crop_height * 2 + 6), (255, 255, 255)
    )
    for index, crop in enumerate(crops):
        row_index, column_index = divmod(index, 4)
        sheet.paste(
            crop,
            (column_index * (crop_width + 2), row_index * (crop_height + 2)),
        )
    experiment_id = str(group["experiment_id"].iloc[0])
    stem = str(group["stem"].iloc[0])
    sheet.save(
        output / "mask_contact_sheets" / f"{experiment_id}_{stem}.jpg",
        quality=88,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_manifest = args.manifest.expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"SAM2 embryo manifest does not exist: {source_manifest}")
    output = _prepare_output(args.output, areas_only=args.areas_only)
    manifest = _validate_manifest(
        pd.read_csv(source_manifest),
        prompt_box_source=args.prompt_box_source,
        areas_only=args.areas_only,
    )

    predictor, torch = _load_predictor()
    rows_out: list[dict[str, Any]] = []
    grouped = manifest.groupby("_source_fov_id", sort=False)
    total_fov_count = grouped.ngroups
    output_csv = output / (
        "sam2_mask_areas.csv" if args.areas_only else "sam2_mask_manifest.csv"
    )

    for fov_index, (source_fov_id, group) in enumerate(grouped, start=1):
        group = group.sort_values("embryo_position")
        image_id = str(group["image_id"].iloc[0])
        image_path = Path(str(group["image_path"].iloc[0]))
        with Image.open(image_path) as source_image:
            image = np.asarray(source_image.convert("RGB"))
        image_height, image_width = image.shape[:2]
        prompt_boxes = _prompt_boxes(
            group,
            image_width=image_width,
            image_height=image_height,
            prompt_box_source=args.prompt_box_source,
        )

        predictor.set_image(image)
        with torch.inference_mode():
            predicted_masks, predicted_scores, _ = predictor.predict(
                box=prompt_boxes,
                multimask_output=False,
            )
        masks, scores = _normalize_predictions(
            predicted_masks,
            predicted_scores,
            expected_count=len(group),
            image_shape=(image_height, image_width),
        )

        cleaned_masks: list[np.ndarray] = []
        mask_audits: list[dict[str, Any]] = []
        for local_index, (_, row) in enumerate(group.iterrows()):
            context = (
                f"source_fov_id={source_fov_id!r}, "
                f"embryo_position={int(row['embryo_position'])}"
            )
            cleaned, audit = _clean_prompt_associated_component(
                masks[local_index],
                prompt_box=prompt_boxes[local_index],
                context=context,
            )
            cleaned_masks.append(cleaned)
            mask_audits.append(audit)
        masks = np.stack(cleaned_masks, axis=0)

        if args.areas_only:
            for local_index, (_, row) in enumerate(group.iterrows()):
                mask_path = _write_binary_mask(
                    masks[local_index],
                    _binary_mask_path(
                        output,
                        str(source_fov_id),
                        int(row["embryo_position"]),
                    ),
                )
                rows_out.append(
                    _area_record(
                        row,
                        source_fov_id=str(source_fov_id),
                        mask=masks[local_index],
                        mask_score=float(scores[local_index]),
                        prompt_box=prompt_boxes[local_index],
                        prompt_box_source=args.prompt_box_source,
                        mask_path=mask_path,
                        mask_audit=mask_audits[local_index],
                    )
                )
            _write_checkpoint(
                rows_out,
                output_csv,
                processed_fov_count=fov_index,
                total_fov_count=total_fov_count,
                complete=False,
            )
        else:
            crop_boxes = group[list(CROP_BOX_COLUMNS)].to_numpy(dtype=np.int32)
            _write_visualization_products(
                output=output,
                image_id=image_id,
                group=group,
                image=image,
                crop_boxes=crop_boxes,
                prompt_boxes=prompt_boxes,
                masks=masks,
                scores=scores,
                mask_audits=mask_audits,
                rows_out=rows_out,
                prompt_box_source=args.prompt_box_source,
            )
            pd.DataFrame.from_records(rows_out).to_csv(output_csv, index=False)

        print(
            f"  [{fov_index}/{total_fov_count}] FOV {source_fov_id}: "
            f"{len(group)} masks, scores {scores.min():.2f}-{scores.max():.2f}",
            flush=True,
        )

    if args.areas_only:
        _validate_production_coverage(rows_out, manifest, output=output)
        _write_checkpoint(
            rows_out,
            output_csv,
            processed_fov_count=total_fov_count,
            total_fov_count=total_fov_count,
            complete=True,
        )
    print(
        f"\nDONE. {len(rows_out)} embryo masks across {total_fov_count} FOVs "
        f"-> {output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
