"""Run box-prompted SAM2 over a SeaHub embryo detection manifest.

The default visualization mode preserves the original validation products:
full-frame masks, masked snips, overlays, contact sheets, and a mask manifest.

``--areas-only`` is the full-corpus calibration mode. It makes one batched,
box-prompted SAM2 prediction call per source FOV and writes only an atomically
checkpointed ``sam2_mask_areas.csv``. No masks, overlays, or snips are retained.
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
            "Full-corpus calibration mode: write only a checkpointed CSV of "
            "mask areas and scores; do not retain masks or visualizations."
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
    if not areas_only:
        for subdirectory in (
            "masks",
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


def _area_record(
    row: pd.Series,
    *,
    source_fov_id: str,
    mask: np.ndarray,
    mask_score: float,
    prompt_box: Sequence[float],
    prompt_box_source: str,
) -> dict[str, Any]:
    x1, y1, x2, y2 = (int(round(value)) for value in prompt_box)
    prompt_area = max(1, (x2 - x1) * (y2 - y1))
    mask_area = int(mask.sum())
    record: dict[str, Any] = {
        "image_id": str(row["image_id"]),
        "source_fov_id": str(source_fov_id),
        "embryo_position": int(row["embryo_position"]),
        "mask_score": float(mask_score),
        "mask_area_px": mask_area,
        "prompt_box_source": prompt_box_source,
        "prompt_x1_px": x1,
        "prompt_y1_px": y1,
        "prompt_x2_px": x2,
        "prompt_y2_px": y2,
        "prompt_box_area_px": int(prompt_area),
        "mask_to_prompt_area_ratio": float(mask_area / prompt_area),
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
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

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
        )
        record.update(
            {
                "stem": row["stem"],
                "crop_x1_px": crop_x1,
                "crop_y1_px": crop_y1,
                "crop_x2_px": crop_x2,
                "crop_y2_px": crop_y2,
                "mask_path": str(mask_path),
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

        if args.areas_only:
            for local_index, (_, row) in enumerate(group.iterrows()):
                rows_out.append(
                    _area_record(
                        row,
                        source_fov_id=str(source_fov_id),
                        mask=masks[local_index],
                        mask_score=float(scores[local_index]),
                        prompt_box=prompt_boxes[local_index],
                        prompt_box_source=args.prompt_box_source,
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
