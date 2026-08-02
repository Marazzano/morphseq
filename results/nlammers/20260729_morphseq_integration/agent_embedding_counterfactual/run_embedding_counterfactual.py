"""Encode a controlled scale x blend-radius snip factorial with the production VAE.

Diagnostic only: reads pipeline/legacy products and writes only below this file's
directory.  It deliberately holds source image, current full-frame mask, crop logic,
CLAHE, background estimate, and encoder fixed, varying only:

    target pixel size: 6.5 vs 7.8 um/px
    Gaussian blend radius: 75 vs 20 um

Stored legacy and current production images are encoded as positive controls.
"""

from __future__ import annotations

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
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from data_pipeline.feature_extraction.legacy_embeddings.legacy_vae_inference_loader import (
    load_legacy_vae_encoder,
)
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


REPO_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq")
DATA_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
OUTPUT_ROOT = DATA_ROOT / "pipeline" / "output"
HERE = Path(__file__).resolve().parent
EXPERIMENT = "20250612_30hpf_ctrl_atf6"
MODEL_NAME = "20241107_ds_sweep01_optimum"
MODEL_DIR = DATA_ROOT / "models" / "legacy" / MODEL_NAME
LEGACY_LATENTS = (
    DATA_ROOT / "legacy" / MODEL_NAME / f"morph_latents_{EXPERIMENT}.csv"
)
CURRENT_ANALYSIS = (
    OUTPUT_ROOT
    / "analysis_ready"
    / EXPERIMENT
    / "analysis_ready"
    / f"{EXPERIMENT}_analysis_ready.parquet"
)
INVENTORY = (
    OUTPUT_ROOT
    / "object_extraction"
    / EXPERIMENT
    / "snips"
    / f"{EXPERIMENT}_snip_inventory.csv"
)
LEGACY_SNIPS = DATA_ROOT / "training_data" / "bf_embryo_snips" / EXPERIMENT

OUTPUT_SHAPE = (576, 256)
MODEL_INPUT_SHAPE = (288, 128)
SOURCE_PIXEL_SIZE_UM = 2717.581 / 1440.0
VARIANTS = (
    ("render_6p5_r75", 6.5, 75.0),
    ("render_7p8_r75", 7.8, 75.0),
    ("render_6p5_r20", 6.5, 20.0),
    ("render_7p8_r20", 7.8, 20.0),
)
WELL_RE = re.compile(r"_([A-H]\d{2})_e\d+_t\d+$")


def resolve_current(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else OUTPUT_ROOT / path


def transform_crop(
    image: np.ndarray,
    embryo_mask: np.ndarray,
    target_pixel_size_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    yolk_mask = np.zeros_like(embryo_mask)
    image_rs, mask_rs, yolk_rs = extract_embryo_crop(
        image,
        embryo_mask,
        yolk_mask,
        OUTPUT_SHAPE,
        SOURCE_PIXEL_SIZE_UM,
        target_pixel_size_um,
    )
    image_rot, mask_rot, yolk_rot, _ = apply_rotation_to_snip(
        image_rs, mask_rs, yolk_rs
    )
    image_crop, mask_crop, _ = crop_to_embryo_bounds(
        image_rot, mask_rot, yolk_rot, OUTPUT_SHAPE
    )
    return image_crop.astype(np.uint8), mask_crop


def current_background_stats(
    image: np.ndarray, frame_masks: pd.DataFrame
) -> tuple[float, float]:
    """Mirror the active entrypoint, including its 0.1 intensity multiplier."""
    valid = frame_masks[frame_masks["is_valid_mask"].astype(bool)]
    np.random.seed(309)
    sampled = np.random.choice(
        valid.index.tolist(), size=min(50, len(valid)), replace=False
    )
    pieces: list[np.ndarray] = []
    for index in sampled:
        mask = decode_binary_mask_rle(
            json.loads(str(valid.loc[index, "mask_rle"]))
        ).astype(bool)
        pieces.append(image[~mask].astype(float)[:5000])
    pixels = np.concatenate(pieces)
    return 0.1 * float(pixels.mean()), 0.1 * float(pixels.std())


def load_pairs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(INVENTORY)
    inventory = inventory[inventory["is_valid_snip"].astype(bool)].copy()
    inventory["well"] = inventory["well_id"].str.rsplit("_", n=1).str[-1]
    inventory = (
        inventory[
            inventory["physical_embryo_id"].astype(str).str.endswith("_e01")
        ]
        .sort_values("snip_id")
        .drop_duplicates("well")
        .set_index("well")
    )

    legacy = pd.read_csv(LEGACY_LATENTS)
    legacy["well"] = legacy["snip_id"].astype(str).map(
        lambda x: WELL_RE.search(x).group(1) if WELL_RE.search(x) else None
    )
    legacy = legacy.drop_duplicates("well").set_index("well")

    current_columns = [
        "well_id",
        "physical_embryo_id",
        "temperature",
        "genotype",
        "embedding_model_name",
    ] + [f"z_mu_{i:02d}" for i in range(100)]
    current = pd.read_parquet(CURRENT_ANALYSIS, columns=current_columns)
    current["well"] = current["well_id"].astype(str).str.rsplit("_", n=1).str[-1]
    current = (
        current[
            current["physical_embryo_id"].astype(str).str.endswith("_e01")
        ]
        .drop_duplicates("well")
        .set_index("well")
    )

    wells = sorted(set(inventory.index) & set(legacy.index) & set(current.index))
    return inventory.loc[wells], legacy.loc[wells], current.loc[wells]


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L")
    pil = TF.resize(pil, list(MODEL_INPUT_SHAPE))
    return TF.to_tensor(pil)


def encode_image_sets(
    images: dict[str, list[np.ndarray]], batch_size: int = 32
) -> dict[str, np.ndarray]:
    encoder = load_legacy_vae_encoder(MODEL_DIR, device="cpu")
    output: dict[str, np.ndarray] = {}
    for name, arrays in images.items():
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(arrays), batch_size):
                tensor = torch.stack(
                    [image_to_tensor(x) for x in arrays[start : start + batch_size]]
                )
                batches.append(
                    encoder.encode_batch(tensor)["mu"].numpy().astype(np.float64)
                )
        output[name] = np.concatenate(batches, axis=0)
        print(f"encoded {name}: {output[name].shape}", flush=True)
    return output


def legacy_columns() -> list[str]:
    return [
        f"z_mu_{'n' if i < 20 else 'b'}_{i:02d}" for i in range(100)
    ]


def correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=0)
    b0 = b - b.mean(axis=0)
    an = np.linalg.norm(a0, axis=0)
    bn = np.linalg.norm(b0, axis=0)
    an[an == 0] = 1.0
    bn[bn == 0] = 1.0
    return (a0 / an).T @ (b0 / bn)


def standardize(
    reference: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    center = reference.mean(axis=0)
    scale = reference.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    return (reference - center) / scale, (other - center) / scale


def nearest_self_fraction(reference: np.ndarray, other: np.ndarray) -> float:
    distances = np.sqrt(
        np.sum((other[:, None, :] - reference[None, :, :]) ** 2, axis=2)
    )
    return float(np.mean(np.argmin(distances, axis=1) == np.arange(len(reference))))


def group_r2(x: np.ndarray, groups: np.ndarray) -> float:
    overall = x.mean(axis=0)
    total = float(np.sum((x - overall) ** 2))
    between = sum(
        float(
            np.sum(groups == group)
            * np.sum((x[groups == group].mean(axis=0) - overall) ** 2)
        )
        for group in pd.unique(groups)
    )
    return between / total if total else float("nan")


def space_metrics(
    reference: np.ndarray,
    other: np.ndarray,
    *,
    temperature: np.ndarray,
    dimensions: slice,
) -> dict[str, object]:
    ref, candidate = standardize(reference[:, dimensions], other[:, dimensions])
    delta = candidate - ref
    direct = np.diag(correlation_matrix(ref, candidate))
    ref_dist = pdist(ref)
    candidate_dist = pdist(candidate)
    _, _, vt = np.linalg.svd(ref, full_matrices=False)
    n_pc = min(10, vt.shape[0])
    ref_pc = ref @ vt[:n_pc].T
    candidate_pc = candidate @ vt[:n_pc].T
    pc_corr = [
        float(pearsonr(ref_pc[:, i], candidate_pc[:, i]).statistic)
        for i in range(n_pc)
    ]
    ref_norm = np.linalg.norm(ref, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    denominator = ref_norm * candidate_norm
    cosine = np.divide(
        np.sum(ref * candidate, axis=1),
        denominator,
        out=np.full(len(ref), np.nan),
        where=denominator > 0,
    )
    return {
        "standardized_rmse": float(np.sqrt(np.mean(delta**2))),
        "median_per_dimension_correlation": float(np.median(direct)),
        "per_dimension_correlation_q05_q95": [
            float(x) for x in np.quantile(direct, [0.05, 0.95])
        ],
        "n_dimensions_correlation_ge_0p9": int(np.sum(direct >= 0.9)),
        "pairwise_distance_pearson": float(
            pearsonr(ref_dist, candidate_dist).statistic
        ),
        "pairwise_distance_spearman": float(
            spearmanr(ref_dist, candidate_dist).statistic
        ),
        "nearest_reference_self_match_fraction": nearest_self_fraction(
            ref, candidate
        ),
        "median_sample_cosine_similarity": float(np.nanmedian(cosine)),
        "legacy_pc_score_correlations_first_10": pc_corr,
        "temperature_r2": group_r2(candidate, temperature),
    }


def control_metrics(
    expected: np.ndarray, observed: np.ndarray
) -> dict[str, object]:
    direct = np.diag(correlation_matrix(expected, observed))
    delta = observed - expected
    return {
        "rmse_raw_latent": float(np.sqrt(np.mean(delta**2))),
        "mae_raw_latent": float(np.mean(np.abs(delta))),
        "max_abs_error_raw_latent": float(np.max(np.abs(delta))),
        "median_per_dimension_correlation": float(np.median(direct)),
        "min_per_dimension_correlation": float(np.min(direct)),
        "identity_is_argmax_count": int(
            np.sum(
                np.argmax(
                    np.abs(correlation_matrix(expected, observed)), axis=1
                )
                == np.arange(expected.shape[1])
            )
        ),
    }


def contrast_metrics(
    reference: np.ndarray, a: np.ndarray, b: np.ndarray
) -> dict[str, float]:
    _, az = standardize(reference, a)
    _, bz = standardize(reference, b)
    delta = bz - az
    return {
        "standardized_rms_change": float(np.sqrt(np.mean(delta**2))),
        "median_sample_rms_change": float(
            np.median(np.sqrt(np.mean(delta**2, axis=1)))
        ),
        "pairwise_distance_geometry_correlation": float(
            pearsonr(pdist(az), pdist(bz)).statistic
        ),
        "median_per_dimension_correlation": float(
            np.median(np.diag(correlation_matrix(az, bz)))
        ),
    }


def factorial_metrics(
    reference: np.ndarray, embeddings: dict[str, np.ndarray]
) -> dict[str, object]:
    center = reference.mean(axis=0)
    scale = reference.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    z = {name: (value - center) / scale for name, value in embeddings.items()}
    base = z["render_6p5_r75"]
    scale_effect = z["render_7p8_r75"] - base
    radius_effect = z["render_6p5_r20"] - base
    total = z["render_7p8_r20"] - base
    interaction = total - scale_effect - radius_effect

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x**2)))

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(
            np.sum(a * b)
            / max(np.linalg.norm(a) * np.linalg.norm(b), np.finfo(float).eps)
        )

    return {
        "standardized_rms_effect": {
            "scale_6p5_to_7p8_at_r75": rms(scale_effect),
            "radius_75_to_20_at_6p5": rms(radius_effect),
            "total_legacy_knobs_to_current_knobs": rms(total),
            "factorial_interaction": rms(interaction),
        },
        "effect_norm_ratio_to_total": {
            "scale": rms(scale_effect) / rms(total),
            "radius": rms(radius_effect) / rms(total),
            "interaction": rms(interaction) / rms(total),
        },
        "effect_vector_cosine": {
            "scale_vs_total": cosine(scale_effect, total),
            "radius_vs_total": cosine(radius_effect, total),
            "scale_vs_radius": cosine(scale_effect, radius_effect),
        },
    }


def save_latents(
    wells: list[str],
    embeddings: dict[str, np.ndarray],
) -> None:
    rows: list[pd.DataFrame] = []
    columns = [f"z_mu_{i:02d}" for i in range(100)]
    for name, matrix in embeddings.items():
        frame = pd.DataFrame(matrix, columns=columns)
        frame.insert(0, "well", wells)
        frame.insert(0, "variant", name)
        rows.append(frame)
    pd.concat(rows, ignore_index=True).to_csv(
        HERE / "counterfactual_latents.csv", index=False
    )


def save_montage(
    wells: list[str], images: dict[str, list[np.ndarray]]
) -> None:
    selected = [well for well in ("A01", "A02", "B01", "D01", "E07", "H07") if well in wells]
    rows = (
        ("legacy_stored", "stored legacy"),
        ("render_6p5_r75", "6.5 / 75"),
        ("render_7p8_r75", "7.8 / 75"),
        ("render_6p5_r20", "6.5 / 20"),
        ("render_7p8_r20", "7.8 / 20"),
        ("current_stored", "stored current"),
    )
    fig, axes = plt.subplots(
        len(rows), len(selected), figsize=(2 * len(selected), 2.5 * len(rows))
    )
    for col, well in enumerate(selected):
        index = wells.index(well)
        for row, (name, label) in enumerate(rows):
            axes[row, col].imshow(images[name][index], cmap="gray", vmin=0, vmax=255)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(well)
            if col == 0:
                axes[row, col].text(
                    -0.1,
                    0.5,
                    label,
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=axes[row, col].transAxes,
                )
    fig.tight_layout()
    fig.savefig(HERE / "counterfactual_montage.png", dpi=180)
    plt.close(fig)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    inventory, legacy_frame, current_frame = load_pairs()
    wells = inventory.index.tolist()
    print(f"shared wells: {len(wells)}", flush=True)

    images: dict[str, list[np.ndarray]] = defaultdict(list)
    render_manifest: list[dict[str, object]] = []
    for position, well in enumerate(wells, start=1):
        row = inventory.loc[well]
        source = skio.imread(str(row["image_path"]))
        if source.ndim == 3:
            source = source[:, :, 0]
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
        selected_mask = frame_masks[
            frame_masks["mask_id"].astype(str) == str(row["mask_id"])
        ].iloc[0]
        full_mask = decode_binary_mask_rle(
            json.loads(str(selected_mask["mask_rle"]))
        ).astype(np.uint8)
        bg_mean, bg_std = current_background_stats(source, frame_masks)

        transformed: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for scale in (6.5, 7.8):
            raw, mask = transform_crop(source, full_mask, scale)
            transformed[scale] = (apply_clahe(raw), mask)
        for name, scale, radius in VARIANTS:
            clahe, mask = transformed[scale]
            rendered = blend_with_background_noise(
                clahe, mask, bg_mean, bg_std, radius, scale
            )
            images[name].append(rendered)
            render_manifest.append(
                {
                    "well": well,
                    "variant": name,
                    "target_pixel_size_um": scale,
                    "blend_radius_um": radius,
                    "background_mean": bg_mean,
                    "background_std": bg_std,
                }
            )

        legacy_path = LEGACY_SNIPS / f"{EXPERIMENT}_{well}_e00_t0000.jpg"
        images["legacy_stored"].append(skio.imread(legacy_path).astype(np.uint8))
        images["current_stored"].append(
            skio.imread(resolve_current(str(row["processed_snip_path"]))).astype(np.uint8)
        )

        if position % 10 == 0 or position == len(wells):
            print(f"rendered {position}/{len(wells)}", flush=True)

    pd.DataFrame(render_manifest).to_csv(HERE / "render_manifest.csv", index=False)
    save_montage(wells, images)
    encoded = encode_image_sets(images)
    save_latents(wells, encoded)

    legacy_expected = legacy_frame[legacy_columns()].to_numpy(dtype=float)
    current_expected = current_frame[
        [f"z_mu_{i:02d}" for i in range(100)]
    ].to_numpy(dtype=float)
    temperature = current_frame["temperature"].astype(str).to_numpy()

    controls = {
        "legacy_stored_reencode_vs_legacy_csv": control_metrics(
            legacy_expected, encoded["legacy_stored"]
        ),
        "current_stored_reencode_vs_analysis_ready": control_metrics(
            current_expected, encoded["current_stored"]
        ),
        "render_7p8_r20_vs_current_stored_reencode": control_metrics(
            encoded["current_stored"], encoded["render_7p8_r20"]
        ),
    }

    variant_metrics: dict[str, object] = {}
    compare_names = [
        "legacy_stored",
        "render_6p5_r75",
        "render_7p8_r75",
        "render_6p5_r20",
        "render_7p8_r20",
        "current_stored",
    ]
    for name in compare_names:
        variant_metrics[name] = {
            "all_100": space_metrics(
                legacy_expected,
                encoded[name],
                temperature=temperature,
                dimensions=slice(None),
            ),
            "biological_20_99": space_metrics(
                legacy_expected,
                encoded[name],
                temperature=temperature,
                dimensions=slice(20, 100),
            ),
            "nuisance_0_19": space_metrics(
                legacy_expected,
                encoded[name],
                temperature=temperature,
                dimensions=slice(0, 20),
            ),
        }

    contrasts = {
        "scale_at_legacy_radius__6p5r75_to_7p8r75": contrast_metrics(
            legacy_expected,
            encoded["render_6p5_r75"],
            encoded["render_7p8_r75"],
        ),
        "radius_at_legacy_scale__6p5r75_to_6p5r20": contrast_metrics(
            legacy_expected,
            encoded["render_6p5_r75"],
            encoded["render_6p5_r20"],
        ),
        "scale_at_current_radius__6p5r20_to_7p8r20": contrast_metrics(
            legacy_expected,
            encoded["render_6p5_r20"],
            encoded["render_7p8_r20"],
        ),
        "radius_at_current_scale__7p8r75_to_7p8r20": contrast_metrics(
            legacy_expected,
            encoded["render_7p8_r75"],
            encoded["render_7p8_r20"],
        ),
        "both_knobs__6p5r75_to_7p8r20": contrast_metrics(
            legacy_expected,
            encoded["render_6p5_r75"],
            encoded["render_7p8_r20"],
        ),
    }

    render_embeddings = {
        name: encoded[name] for name, _, _ in VARIANTS
    }
    result = {
        "experiment_id": EXPERIMENT,
        "checkpoint": MODEL_NAME,
        "n_shared_wells": len(wells),
        "variant_definition": {
            name: {
                "target_pixel_size_um": scale,
                "blend_radius_um": radius,
                "all_other_inputs": "current source/current mask/current background/current code",
            }
            for name, scale, radius in VARIANTS
        },
        "column_mapping": {
            "legacy_0_19": "z_mu_n_00..z_mu_n_19 -> z_mu_00..z_mu_19",
            "legacy_20_99": "z_mu_b_20..z_mu_b_99 -> z_mu_20..z_mu_99",
            "mapping_rule": "numeric suffix is the direct encoder-output index",
        },
        "controls": controls,
        "distance_to_legacy": variant_metrics,
        "controlled_contrasts": contrasts,
        "factorial_decomposition": factorial_metrics(
            legacy_expected, render_embeddings
        ),
    }
    (HERE / "embedding_counterfactual_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    mapping_corr = correlation_matrix(
        legacy_expected, encoded["legacy_stored"]
    )
    pd.DataFrame(
        {
            "latent_index": np.arange(100),
            "legacy_column": legacy_columns(),
            "current_column": [f"z_mu_{i:02d}" for i in range(100)],
            "direct_correlation_reencoded_legacy": np.diag(mapping_corr),
            "argmax_current_index": np.argmax(np.abs(mapping_corr), axis=1),
            "argmax_abs_correlation": np.max(np.abs(mapping_corr), axis=1),
        }
    ).to_csv(HERE / "latent_index_mapping_validation.csv", index=False)

    concise_rows = []
    for name in compare_names:
        all_metrics = variant_metrics[name]["all_100"]
        bio_metrics = variant_metrics[name]["biological_20_99"]
        concise_rows.append(
            {
                "variant": name,
                "all_standardized_rmse_to_legacy": all_metrics[
                    "standardized_rmse"
                ],
                "all_median_dimension_r_to_legacy": all_metrics[
                    "median_per_dimension_correlation"
                ],
                "all_pairwise_distance_r_to_legacy": all_metrics[
                    "pairwise_distance_pearson"
                ],
                "all_nearest_self_match_fraction": all_metrics[
                    "nearest_reference_self_match_fraction"
                ],
                "bio_standardized_rmse_to_legacy": bio_metrics[
                    "standardized_rmse"
                ],
                "bio_median_dimension_r_to_legacy": bio_metrics[
                    "median_per_dimension_correlation"
                ],
                "bio_pairwise_distance_r_to_legacy": bio_metrics[
                    "pairwise_distance_pearson"
                ],
            }
        )
    concise = pd.DataFrame(concise_rows)
    concise.to_csv(HERE / "variant_summary.csv", index=False)
    print("\nVARIANT SUMMARY", flush=True)
    print(concise.to_string(index=False), flush=True)
    print("\nCONTROLS", flush=True)
    print(json.dumps(controls, indent=2), flush=True)
    print("\nFACTORIAL", flush=True)
    print(json.dumps(result["factorial_decomposition"], indent=2), flush=True)


if __name__ == "__main__":
    main()
