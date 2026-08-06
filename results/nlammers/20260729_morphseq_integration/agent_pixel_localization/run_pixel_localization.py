"""Stagewise localization of the 20250612 snip saturation regression.

This is a diagnostic only. It reads production artifacts and writes tables/figures
under this directory; it does not mutate any pipeline products.
"""

from __future__ import annotations

import io
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.io as skio
from PIL import Image
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity
from skimage.morphology import binary_closing, disk, remove_small_objects

from data_pipeline.object_extraction.segmentation.masks.mask_rle import (
    decode_binary_mask_rle,
)
from data_pipeline.object_extraction.snip_processing.augmentation import (
    apply_clahe,
    blend_with_background_noise,
)
from data_pipeline.object_extraction.snip_processing.extraction import (
    crop_to_embryo_bounds,
    extract_embryo_crop,
)
from data_pipeline.object_extraction.snip_processing.rotation import (
    apply_rotation_to_snip,
)


REPO_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq"
)
DATA_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
OUTPUT_ROOT = DATA_ROOT / "pipeline" / "output"
RESULT_DIR = (
    REPO_ROOT
    / "results"
    / "nlammers"
    / "20260729_morphseq_integration"
    / "agent_pixel_localization"
)
EXPERIMENT = "20250612_30hpf_ctrl_atf6"
OUTPUT_SHAPE = (576, 256)
SOURCE_PIXEL_SIZE_UM = 2717.581 / 1440.0
RADIUS_GRID_UM = (
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    65.0,
    70.0,
    75.0,
    100.0,
)
MONTAGE_WELLS = ("A01", "A02", "B01", "D01", "E07", "H07")


class HistogramAccumulator:
    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], np.ndarray] = defaultdict(
            lambda: np.zeros(256, dtype=np.int64)
        )
        self.image_counts: dict[tuple[str, str], int] = defaultdict(int)

    def add(
        self,
        stage: str,
        region: str,
        image: np.ndarray,
        select: np.ndarray,
    ) -> None:
        values = np.asarray(image, dtype=np.uint8)[select]
        if values.size == 0:
            return
        self.counts[(stage, region)] += np.bincount(
            values, minlength=256
        ).astype(np.int64)
        self.image_counts[(stage, region)] += 1

    @staticmethod
    def _quantile(counts: np.ndarray, q: float) -> int:
        total = int(counts.sum())
        if total == 0:
            return -1
        target = q * (total - 1)
        return int(np.searchsorted(np.cumsum(counts), target, side="right"))

    def to_frame(self) -> pd.DataFrame:
        rows = []
        levels = np.arange(256)
        for (stage, region), counts in sorted(self.counts.items()):
            n = int(counts.sum())
            rows.append(
                {
                    "stage": stage,
                    "region": region,
                    "n_images": self.image_counts[(stage, region)],
                    "n_pixels": n,
                    "mean": float(np.dot(levels, counts) / n),
                    "p50": self._quantile(counts, 0.50),
                    "p95": self._quantile(counts, 0.95),
                    "p99": self._quantile(counts, 0.99),
                    "fraction_ge_245": float(counts[245:].sum() / n),
                    "fraction_ge_250": float(counts[250:].sum() / n),
                    "fraction_eq_255": float(counts[255] / n),
                }
            )
        return pd.DataFrame(rows)


def resolve_current(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else OUTPUT_ROOT / path


def jpeg_roundtrip(array: np.ndarray, quality: int = 75) -> np.ndarray:
    handle = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(
        handle, format="JPEG", quality=quality
    )
    handle.seek(0)
    return np.asarray(Image.open(handle).convert("L"))


def clean_legacy_masks(
    embryo_raw: np.ndarray, yolk_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the exact process_masks implementation present when these snips were made."""
    # Historical code used round(raw / 255 * 2 - 1), whose positive-class
    # boundary is ~191 rather than 127. Keep this oddity: it materially changes
    # the crop mask and therefore the Gaussian blend.
    embryo_binary = np.round(
        np.asarray(embryo_raw, dtype=float) / 255.0 * 2.0 - 1.0
    ).astype(np.uint8)
    embryo_labels = label(embryo_binary)
    embryo = embryo_labels == 1
    if embryo.any():
        embryo = binary_closing(embryo, disk(15))

    yolk = np.round(
        np.asarray(yolk_raw, dtype=float) / 255.0 * 2.0 - 1.0
    ).astype(np.uint8)
    if np.any(yolk == 1):
        yolk = remove_small_objects(yolk.astype(bool), min_size=75)
    else:
        yolk = np.zeros_like(embryo)
    if embryo.any() and yolk.any():
        intersect = yolk & embryo
        if int(intersect.sum()) < 10:
            yolk = np.zeros_like(embryo)
        else:
            yolk_labels = label(yolk)
            contact_labels = label(intersect)
            if int(contact_labels.max()) <= 1:
                values = np.unique(yolk_labels[intersect])
                keep = int(values[-1]) if len(values) else 0
            else:
                props = regionprops(contact_labels)
                largest = int(np.argmax([p.area for p in props])) + 1
                values = np.unique(yolk_labels[contact_labels == largest])
                keep = int(values[-1]) if len(values) else 0
            yolk = yolk_labels == keep
    return embryo.astype(np.uint8), yolk.astype(np.uint8)


def transform_crop(
    image: np.ndarray,
    embryo_mask: np.ndarray,
    yolk_mask: np.ndarray | None,
    target_pixel_size_um: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if yolk_mask is None:
        yolk_mask = np.zeros_like(embryo_mask)
    image_rs, mask_rs, yolk_rs = extract_embryo_crop(
        image,
        embryo_mask,
        yolk_mask,
        OUTPUT_SHAPE,
        SOURCE_PIXEL_SIZE_UM,
        target_pixel_size_um,
    )
    image_rot, mask_rot, yolk_rot, angle = apply_rotation_to_snip(
        image_rs, mask_rs, yolk_rs
    )
    image_crop, mask_crop, _ = crop_to_embryo_bounds(
        image_rot, mask_rot, yolk_rot, OUTPUT_SHAPE
    )
    return image_crop.astype(np.uint8), mask_crop, float(angle)


def regions(mask: np.ndarray, image: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(mask) > 0.5
    distance = scipy.ndimage.distance_transform_edt(binary)
    return {
        "mask": binary,
        "core_ge_8px": distance >= 8.0,
        # This reproduces the fixed foreground definition used in the handoff.
        "fixed_gt30": np.asarray(image) > 30,
        "all": np.ones_like(binary, dtype=bool),
    }


def image_metrics(
    well: str,
    stage: str,
    image: np.ndarray,
    mask: np.ndarray,
    accumulator: HistogramAccumulator,
) -> list[dict[str, object]]:
    rows = []
    for region_name, select in regions(mask, image).items():
        values = np.asarray(image, dtype=np.uint8)[select]
        if values.size == 0:
            continue
        accumulator.add(stage, region_name, image, select)
        rows.append(
            {
                "well": well,
                "stage": stage,
                "region": region_name,
                "n_pixels": int(values.size),
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "fraction_ge_245": float(np.mean(values >= 245)),
                "fraction_ge_250": float(np.mean(values >= 250)),
                "fraction_eq_255": float(np.mean(values == 255)),
            }
        )
    return rows


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values)
    return float(np.percentile(values, q)) if values.size else float("nan")


def current_background_stats(
    current_image: np.ndarray, frame_masks: pd.DataFrame
) -> tuple[float, float]:
    """Exactly mirror the active per-well entrypoint's 0.1-scaled estimator."""
    valid = frame_masks[frame_masks["is_valid_mask"].astype(bool)]
    np.random.seed(309)
    indices = valid.index.tolist()
    sampled = np.random.choice(
        indices, size=min(50, len(indices)), replace=False
    )
    chunks = []
    for index in sampled:
        mask = decode_binary_mask_rle(
            json.loads(str(valid.loc[index, "mask_rle"]))
        ).astype(bool)
        chunks.append(current_image[~mask].astype(float)[:5000])
    pixels = np.concatenate(chunks)
    return 0.1 * float(pixels.mean()), 0.1 * float(pixels.std())


def load_pairs() -> tuple[pd.DataFrame, dict[str, Path], list[str]]:
    inventory_path = (
        OUTPUT_ROOT
        / "object_extraction"
        / EXPERIMENT
        / "snips"
        / f"{EXPERIMENT}_snip_inventory.csv"
    )
    inventory = pd.read_csv(inventory_path)
    inventory = inventory[inventory["is_valid_snip"].astype(bool)].copy()
    inventory["well"] = inventory["well_id"].str.rsplit("_", n=1).str[-1]
    # Legacy has one embryo per well. Use the primary e01 current embryo.
    primary = inventory[
        inventory["physical_embryo_id"].astype(str).str.endswith("_e01")
    ].copy()
    primary = primary.sort_values("snip_id").drop_duplicates("well")

    legacy_dir = (
        DATA_ROOT / "training_data" / "bf_embryo_snips" / EXPERIMENT
    )
    pattern = re.compile(
        rf"^{re.escape(EXPERIMENT)}_([A-H]\d{{2}})_e00_t0000\.jpg$"
    )
    legacy: dict[str, Path] = {}
    for path in sorted(legacy_dir.glob("*.jpg")):
        match = pattern.match(path.name)
        if match:
            legacy[match.group(1)] = path

    current_wells = set(primary["well"])
    paired_wells = sorted(set(legacy) & current_wells)
    missing = sorted((set(legacy) - current_wells) | (current_wells - set(legacy)))
    return (
        primary[primary["well"].isin(paired_wells)].set_index("well"),
        legacy,
        missing,
    )


def plot_montage(montage: dict[str, dict[str, np.ndarray]]) -> None:
    rows = (
        ("legacy_clahe_stored", "legacy CLAHE-only"),
        ("legacy_final_stored", "legacy final"),
        ("current_clahe_7p8", "current CLAHE-only @7.8"),
        ("current_rerender_7p8_r20", "current blend radius 20"),
        ("current_rerender_7p8_r50", "counterfactual radius 50"),
        ("current_rerender_7p8_r75", "counterfactual radius 75"),
        ("current_production", "current production"),
    )
    wells = [w for w in MONTAGE_WELLS if w in montage]
    fig, axes = plt.subplots(
        len(rows),
        len(wells),
        figsize=(2.0 * len(wells), 2.55 * len(rows)),
        squeeze=False,
    )
    for col, well in enumerate(wells):
        for row_index, (key, label_text) in enumerate(rows):
            axes[row_index, col].imshow(
                montage[well][key], cmap="gray", vmin=0, vmax=255
            )
            axes[row_index, col].axis("off")
            if row_index == 0:
                axes[row_index, col].set_title(well)
            if col == 0:
                axes[row_index, col].text(
                    -0.12,
                    0.5,
                    label_text,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=axes[row_index, col].transAxes,
                    fontsize=9,
                )
    fig.suptitle(
        f"{EXPERIMENT}: the saturation difference enters at Gaussian mask blending",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "radius_comparison_montage.png", dpi=180)
    plt.close(fig)


def plot_histograms(
    accumulator: HistogramAccumulator, summary: pd.DataFrame
) -> None:
    stages = (
        ("legacy_clahe_stored", "legacy CLAHE-only", "#4c78a8"),
        ("legacy_final_stored", "legacy final", "#1f3b5d"),
        ("current_clahe_7p8", "current CLAHE-only", "#f58518"),
        ("current_rerender_7p8_r20", "current radius 20", "#e45756"),
        ("current_rerender_7p8_r50", "counterfactual radius 50", "#54a24b"),
        ("current_production", "current production", "#b279a2"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(256)
    for stage, label_text, color in stages:
        counts = accumulator.counts.get((stage, "mask"))
        if counts is None or counts.sum() == 0:
            continue
        density = counts / counts.sum()
        axes[0].plot(x, density, label=label_text, color=color, lw=1.5)
        axes[1].plot(x, density, label=label_text, color=color, lw=1.5)
    axes[0].set_xlim(0, 255)
    axes[1].set_xlim(220, 255)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("intensity")
        ax.set_ylabel("fraction of mask pixels")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Whole embryo-mask histogram")
    axes[1].set_title("Highlight tail")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "stage_histograms.png", dpi=180)
    plt.close(fig)

    radius = summary[
        summary["stage"].str.match(r"current_rerender_6p5_r")
        & summary["region"].eq("fixed_gt30")
    ].copy()
    radius["radius_um"] = radius["stage"].str.extract(
        r"_r(\d+)$"
    ).astype(float)
    legacy = summary[
        summary["stage"].eq("legacy_final_stored")
        & summary["region"].eq("fixed_gt30")
    ].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(radius["radius_um"], radius["fraction_ge_250"], "o-")
    axes[0].axhline(
        legacy["fraction_ge_250"],
        color="black",
        ls="--",
        label="legacy final",
    )
    axes[0].set_ylabel("fraction >=250 among pixels >30")
    axes[1].plot(radius["radius_um"], radius["p95"], "o-")
    axes[1].axhline(
        legacy["p95"], color="black", ls="--", label="legacy final"
    )
    axes[1].set_ylabel("p95 among pixels >30")
    for ax in axes:
        ax.set_xlabel("blend radius (µm), current chain @6.5 µm/px")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "radius_sweep.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    current_rows, legacy_paths, missing = load_pairs()
    wells = sorted(current_rows.index.tolist())
    print(
        f"legacy={len(legacy_paths)} current_primary={len(current_rows)} "
        f"paired={len(wells)} missing_or_unpaired={missing}",
        flush=True,
    )

    accumulator = HistogramAccumulator()
    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    montage: dict[str, dict[str, np.ndarray]] = {}

    for well_index, well in enumerate(wells, start=1):
        current_row = current_rows.loc[well]
        current_path = resolve_current(str(current_row["processed_snip_path"]))
        current_mask_path = resolve_current(
            str(current_row["embryo_mask_snip_path"])
        )
        current_production = skio.imread(current_path).astype(np.uint8)
        current_production_mask = skio.imread(current_mask_path) > 0
        current_source = skio.imread(str(current_row["image_path"]))
        if current_source.ndim == 3:
            current_source = current_source[:, :, 0]

        frame_masks_path = (
            OUTPUT_ROOT
            / "object_extraction"
            / EXPERIMENT
            / "frame_masks"
            / "per_well"
            / f"{EXPERIMENT}_{well}"
            / f"{EXPERIMENT}_{well}_frame_masks.csv"
        )
        frame_masks = pd.read_csv(frame_masks_path)
        selected = frame_masks[
            frame_masks["mask_id"].astype(str)
            == str(current_row["mask_id"])
        ].iloc[0]
        current_full_mask = decode_binary_mask_rle(
            json.loads(str(selected["mask_rle"]))
        ).astype(np.uint8)
        bg_mean, bg_std = current_background_stats(
            current_source, frame_masks
        )

        legacy_final = skio.imread(legacy_paths[well]).astype(np.uint8)
        legacy_uncropped_path = (
            DATA_ROOT
            / "training_data"
            / "bf_embryo_snips_uncropped"
            / EXPERIMENT
            / legacy_paths[well].name
        )
        legacy_clahe_stored = skio.imread(legacy_uncropped_path).astype(
            np.uint8
        )
        legacy_saved_mask = (
            skio.imread(
                DATA_ROOT
                / "training_data"
                / "bf_embryo_masks"
                / f"emb_{EXPERIMENT}_{well}_e00_t0000.jpg"
            )
            > 0
        )
        legacy_source_path = (
            DATA_ROOT
            / "built_image_data"
            / "stitched_FF_images"
            / EXPERIMENT
            / f"{well}_t0000_stitch.jpg"
        )
        legacy_source = skio.imread(legacy_source_path)
        legacy_mask_raw = skio.imread(
            DATA_ROOT
            / "segmentation"
            / "mask_v0_0100_predictions"
            / EXPERIMENT
            / f"{well}_t0000_stitch.jpg.jpg"
        )
        legacy_yolk_raw = skio.imread(
            DATA_ROOT
            / "segmentation"
            / "yolk_v1_0050_predictions"
            / EXPERIMENT
            / f"{well}_t0000_stitch.jpg.jpg"
        )
        legacy_full_mask, legacy_full_yolk = clean_legacy_masks(
            legacy_mask_raw, legacy_yolk_raw
        )

        # Rebuild the exact legacy crop/CLAHE boundary from its source and masks.
        legacy_raw_rebuilt, legacy_mask_crop, legacy_angle = transform_crop(
            legacy_source, legacy_full_mask, legacy_full_yolk, 6.5
        )
        legacy_clahe_rebuilt = apply_clahe(legacy_raw_rebuilt)

        # Current geometry at the legacy scale, swapping only the full-frame source.
        current_raw_from_current_65, current_mask_65, current_angle_65 = (
            transform_crop(current_source, current_full_mask, None, 6.5)
        )
        current_raw_from_legacy_65, current_mask_oldsource_65, _ = (
            transform_crop(legacy_source, current_full_mask, None, 6.5)
        )
        if not np.array_equal(
            current_mask_65, current_mask_oldsource_65
        ):
            raise AssertionError("source swap changed the transformed mask")
        current_clahe_65 = apply_clahe(current_raw_from_current_65)
        oldsource_clahe_current_geometry_65 = apply_clahe(
            current_raw_from_legacy_65
        )

        # Current production geometry and the direct radius counterfactuals.
        current_raw_78, current_mask_78, current_angle_78 = transform_crop(
            current_source, current_full_mask, None, 7.8
        )
        current_clahe_78 = apply_clahe(current_raw_78)
        current_rerenders_78 = {
            radius: blend_with_background_noise(
                current_clahe_78,
                current_mask_78,
                bg_mean,
                bg_std,
                radius,
                7.8,
            )
            for radius in (20.0, 50.0, 75.0)
        }
        current_rerenders_65 = {
            radius: blend_with_background_noise(
                current_clahe_65,
                current_mask_65,
                bg_mean,
                bg_std,
                radius,
                6.5,
            )
            for radius in RADIUS_GRID_UM
        }
        current_r20_65_bgzero = blend_with_background_noise(
            current_clahe_65,
            current_mask_65,
            0.0,
            0.0,
            20.0,
            6.5,
        )
        current_r75_65_bgzero = blend_with_background_noise(
            current_clahe_65,
            current_mask_65,
            0.0,
            0.0,
            75.0,
            6.5,
        )
        current_r20_65_legacy_bg = blend_with_background_noise(
            current_clahe_65,
            current_mask_65,
            10.0,
            5.0,
            20.0,
            6.5,
        )

        # Re-blend the stored legacy CLAHE intermediate across candidate radii.
        legacy_binary = legacy_mask_crop > 0.5
        far_exterior = (
            scipy.ndimage.distance_transform_edt(~legacy_binary) >= 25
        )
        exterior_values = legacy_final[far_exterior]
        legacy_bg_mean = float(exterior_values.mean())
        legacy_bg_std = float(exterior_values.std())
        legacy_reblends = {
            radius: jpeg_roundtrip(
                blend_with_background_noise(
                    legacy_clahe_stored,
                    legacy_mask_crop,
                    legacy_bg_mean,
                    legacy_bg_std,
                    radius,
                    6.5,
                ),
                75,
            )
            for radius in RADIUS_GRID_UM
        }

        stages: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "legacy_clahe_stored": (
                legacy_clahe_stored,
                legacy_mask_crop,
            ),
            "legacy_clahe_rebuilt": (
                legacy_clahe_rebuilt,
                legacy_mask_crop,
            ),
            "legacy_final_stored": (legacy_final, legacy_mask_crop),
            "old_ff_raw_current_geometry_6p5": (
                current_raw_from_legacy_65,
                current_mask_65,
            ),
            "current_ff_raw_current_geometry_6p5": (
                current_raw_from_current_65,
                current_mask_65,
            ),
            "old_ff_clahe_current_geometry_6p5": (
                oldsource_clahe_current_geometry_65,
                current_mask_65,
            ),
            "current_ff_clahe_current_geometry_6p5": (
                current_clahe_65,
                current_mask_65,
            ),
            "current_clahe_7p8": (current_clahe_78, current_mask_78),
            "current_production": (
                current_production,
                current_production_mask,
            ),
            "current_production_jpeg75": (
                jpeg_roundtrip(current_production, 75),
                current_production_mask,
            ),
            "current_rerender_6p5_r20_bgzero": (
                current_r20_65_bgzero,
                current_mask_65,
            ),
            "current_rerender_6p5_r75_bgzero": (
                current_r75_65_bgzero,
                current_mask_65,
            ),
            "current_rerender_6p5_r20_legacy_bg10_5": (
                current_r20_65_legacy_bg,
                current_mask_65,
            ),
        }
        for radius, image in current_rerenders_65.items():
            stages[
                f"current_rerender_6p5_r{int(radius)}"
            ] = (image, current_mask_65)
        for radius, image in current_rerenders_78.items():
            stages[
                f"current_rerender_7p8_r{int(radius)}"
            ] = (image, current_mask_78)
        for radius, image in legacy_reblends.items():
            stages[
                f"legacy_reblend_r{int(radius)}"
            ] = (image, legacy_mask_crop)

        for stage, (image, mask) in stages.items():
            metric_rows.extend(
                image_metrics(well, stage, image, mask, accumulator)
            )

        full_select = current_full_mask.astype(bool)
        source_old = legacy_source[full_select].astype(float)
        source_new = current_source[full_select].astype(float)
        crop_select = current_mask_65 > 0.5
        raw_old = current_raw_from_legacy_65[crop_select].astype(float)
        raw_new = current_raw_from_current_65[crop_select].astype(float)
        clahe_old = oldsource_clahe_current_geometry_65[
            crop_select
        ].astype(float)
        clahe_new = current_clahe_65[crop_select].astype(float)
        legacy_distance = scipy.ndimage.distance_transform_edt(
            legacy_mask_crop > 0.5
        )
        current_distance = scipy.ndimage.distance_transform_edt(crop_select)
        legacy_saturated = (
            (legacy_clahe_stored >= 250) & (legacy_mask_crop > 0.5)
        )
        current_saturated = (current_clahe_65 >= 250) & crop_select
        legacy_generated_binary = legacy_mask_crop > 0.5
        mask_union = legacy_generated_binary | legacy_saved_mask
        mask_intersection = legacy_generated_binary & legacy_saved_mask
        generated20 = current_rerenders_78[20.0]
        comparison_rows.append(
            {
                "well": well,
                "background_mean_current": bg_mean,
                "background_std_current": bg_std,
                "background_mean_legacy_exterior": legacy_bg_mean,
                "background_std_legacy_exterior": legacy_bg_std,
                "legacy_angle_deg": float(np.rad2deg(legacy_angle)),
                "current_angle_6p5_deg": float(
                    np.rad2deg(current_angle_65)
                ),
                "current_angle_7p8_deg": float(
                    np.rad2deg(current_angle_78)
                ),
                "source_corr_same_coordinates": safe_corr(
                    source_old, source_new
                ),
                "source_corr_if_current_inverted": safe_corr(
                    source_old, 255.0 - source_new
                ),
                "source_mae_same_coordinates": float(
                    np.mean(np.abs(source_old - source_new))
                ),
                "source_old_mean_mask": float(source_old.mean()),
                "source_current_mean_mask": float(source_new.mean()),
                "source_old_p95_mask": float(
                    np.percentile(source_old, 95)
                ),
                "source_current_p95_mask": float(
                    np.percentile(source_new, 95)
                ),
                "raw_crop_source_swap_corr": safe_corr(raw_old, raw_new),
                "raw_crop_source_swap_mae": float(
                    np.mean(np.abs(raw_old - raw_new))
                ),
                "clahe_source_swap_corr": safe_corr(
                    clahe_old, clahe_new
                ),
                "clahe_source_swap_mae": float(
                    np.mean(np.abs(clahe_old - clahe_new))
                ),
                "legacy_saved_mask_iou_with_rebuild": float(
                    mask_intersection.sum() / max(mask_union.sum(), 1)
                ),
                "legacy_saved_mask_area": int(legacy_saved_mask.sum()),
                "legacy_rebuilt_mask_area": int(
                    legacy_generated_binary.sum()
                ),
                "legacy_saturated_distance_p50": float(
                    safe_percentile(legacy_distance[legacy_saturated], 50)
                ),
                "legacy_saturated_distance_p90": float(
                    safe_percentile(legacy_distance[legacy_saturated], 90)
                ),
                "current_saturated_distance_p50": float(
                    safe_percentile(current_distance[current_saturated], 50)
                ),
                "current_saturated_distance_p90": float(
                    safe_percentile(current_distance[current_saturated], 90)
                ),
                "legacy_clahe_rebuild_mae_all": float(
                    np.mean(
                        np.abs(
                            legacy_clahe_rebuilt.astype(float)
                            - legacy_clahe_stored.astype(float)
                        )
                    )
                ),
                "legacy_clahe_rebuild_corr_all": safe_corr(
                    legacy_clahe_rebuilt, legacy_clahe_stored
                ),
                "legacy_clahe_rebuild_ssim_all": float(
                    structural_similarity(
                        legacy_clahe_rebuilt,
                        legacy_clahe_stored,
                        data_range=255,
                    )
                ),
                "current_r20_reproduction_mae_all": float(
                    np.mean(
                        np.abs(
                            generated20.astype(float)
                            - current_production.astype(float)
                        )
                    )
                ),
                "current_r20_reproduction_corr_all": safe_corr(
                    generated20, current_production
                ),
                "current_r20_reproduction_exact_fraction": float(
                    np.mean(generated20 == current_production)
                ),
            }
        )

        if well in MONTAGE_WELLS:
            montage[well] = {
                "legacy_clahe_stored": legacy_clahe_stored,
                "legacy_final_stored": legacy_final,
                "current_clahe_7p8": current_clahe_78,
                "current_rerender_7p8_r20": current_rerenders_78[20.0],
                "current_rerender_7p8_r50": current_rerenders_78[50.0],
                "current_rerender_7p8_r75": current_rerenders_78[75.0],
                "current_production": current_production,
            }

        if well_index % 10 == 0 or well_index == len(wells):
            print(f"processed {well_index}/{len(wells)} wells", flush=True)

    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    summary = accumulator.to_frame()
    metrics.to_csv(RESULT_DIR / "per_pair_stage_metrics.csv", index=False)
    comparisons.to_csv(
        RESULT_DIR / "per_pair_source_and_reproduction_metrics.csv",
        index=False,
    )
    summary.to_csv(RESULT_DIR / "pooled_stage_metrics.csv", index=False)

    plot_montage(montage)
    plot_histograms(accumulator, summary)

    focus_stages = [
        "legacy_clahe_stored",
        "legacy_final_stored",
        "current_ff_clahe_current_geometry_6p5",
        "current_rerender_6p5_r20",
        "current_rerender_6p5_r50",
        "current_rerender_6p5_r75",
        "current_clahe_7p8",
        "current_rerender_7p8_r20",
        "current_rerender_7p8_r50",
        "current_rerender_7p8_r75",
        "current_production",
        "current_production_jpeg75",
    ]
    focus = summary[
        summary["stage"].isin(focus_stages)
        & summary["region"].isin(["mask", "core_ge_8px", "fixed_gt30"])
    ]
    print("\nPOOLED STAGE METRICS", flush=True)
    print(
        focus[
            [
                "stage",
                "region",
                "n_images",
                "n_pixels",
                "mean",
                "p95",
                "p99",
                "fraction_ge_250",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nPAIRWISE VALIDATION MEDIANS", flush=True)
    print(
        comparisons.median(numeric_only=True).to_string(), flush=True
    )


if __name__ == "__main__":
    main()
