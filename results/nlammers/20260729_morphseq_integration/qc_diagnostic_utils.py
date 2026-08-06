"""Helpers for diagnosing analysis_ready QC failures on the 20250612 (GENE7) plates.

Loads the pipeline's own artifacts — nothing is recomputed from images except the mask outlines
drawn for the montage. The surface-area band is re-derived using the pipeline's packaged reference
and its real thresholds (imported, not copied) so the numbers here cannot drift from what actually
ran.

The montage is the main work product: for every failed embryo, the extracted snip beside the raw
focus-stacked frame, with the segmentation outline drawn on both and nested colored rings encoding
which QC flag(s) fired.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.quality_control.surface_area_qc.config import resolve_config
from data_pipeline.quality_control.surface_area_qc.reference import (
    interpolate_reference_band,
    load_packaged_surface_area_reference,
)

PIPELINE_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline"
)
OUTPUT_ROOT = PIPELINE_ROOT / "output"

# The six GENE7-paired imaging plates: {24,30,36} hpf x {ctrl_atf6, wfs1_ctcf}.
EXPERIMENTS: tuple[str, ...] = (
    "20250612_24hpf_ctrl_atf6",
    "20250612_24hpf_wfs1_ctcf",
    "20250612_30hpf_ctrl_atf6",
    "20250612_30hpf_wfs1_ctcf",
    "20250612_36hpf_ctrl_atf6",
    "20250612_36hpf_wfs1_ctcf",
)

# Rearing temperatures on every plate (each plate is a 4-temperature block design).
TEMPERATURES: tuple[float, ...] = (24.0, 28.5, 34.0, 35.0)

# The lab's Arrhenius developmental-rate relation, used for the counterfactual stage below.
# Matches results/nlammers/20260805/README.md: 6 + (t - 6) * (0.055*T - 0.57).
ARRHENIUS_ANCHOR_HPF = 6.0


@dataclass(frozen=True)
class FlagSpec:
    """One QC flag: its column, a short label, and the color used for its montage ring."""

    column: str
    label: str
    color: str


# Ring order is drawn outermost-first, so the FIRST spec ends up as the OUTERMOST ring.
# sa_outlier leads because it dominates; colors are chosen to stay distinguishable on gray images.
FLAG_SPECS: tuple[FlagSpec, ...] = (
    FlagSpec("sa_outlier_flag", "surface area outlier", "#e6194b"),
    FlagSpec("focus_flag", "out of focus", "#4363d8"),
    FlagSpec("overlapping_mask_flag", "overlapping mask", "#f58231"),
    FlagSpec("motion_blur_flag", "motion blur", "#911eb4"),
    FlagSpec("edge_flag", "touching frame edge", "#ffe119"),
    FlagSpec("discontinuous_mask_flag", "discontinuous mask", "#3cb44b"),
    FlagSpec("viability_dead_flag", "dead (viability)", "#000000"),
    FlagSpec("persistence_dead_flag", "dead (persistence)", "#808080"),
)

FLAG_COLUMNS: tuple[str, ...] = tuple(spec.column for spec in FLAG_SPECS)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def analysis_ready_path(experiment_id: str) -> Path:
    return (
        OUTPUT_ROOT
        / "analysis_ready"
        / experiment_id
        / "analysis_ready"
        / f"{experiment_id}_analysis_ready.parquet"
    )


def load_analysis_ready(experiment_ids: "tuple[str, ...] | list[str]" = EXPERIMENTS) -> pd.DataFrame:
    """Concatenate the analysis_ready tables, dropping the wide latent block.

    The latent columns (200 of the 262) are irrelevant here and make every operation slower, so
    they are excluded at read time rather than after.
    """
    frames = []
    for experiment_id in experiment_ids:
        path = analysis_ready_path(experiment_id)
        if not path.is_file():
            raise FileNotFoundError(f"analysis_ready parquet not found: {path}")
        df = pd.read_parquet(path)
        keep = [c for c in df.columns if not c.startswith(("z_mu_", "z_sigma_"))]
        frames.append(df.loc[:, keep])
    out = pd.concat(frames, ignore_index=True)

    for column in FLAG_COLUMNS:
        out[column] = out[column].fillna(False).astype(bool)
    out["start_age_hpf"] = out["start_age_hpf"].astype(float)
    out["temperature"] = out["temperature"].astype(float)
    out["stage_label"] = out["start_age_hpf"].astype(int).astype(str) + " hpf"
    out["n_flags"] = out[list(FLAG_COLUMNS)].sum(axis=1)
    return out


def add_surface_area_band(df: pd.DataFrame, reference_version: str = "v1") -> pd.DataFrame:
    """Attach the surface-area tolerance band and which side (if any) each snip fell outside.

    Uses the packaged reference curve and the product's own ``k_upper``/``k_lower``, so the
    reconstructed verdict is comparable to the flag the pipeline actually wrote. ``band_agrees``
    exposes any mismatch rather than assuming the reconstruction is faithful.
    """
    config = resolve_config()
    reference = load_packaged_surface_area_reference(reference_version)

    out = df.copy()
    bands = [interpolate_reference_band(float(s), reference) for s in out["predicted_stage_hpf"]]
    out["sa_p5"] = [b[0] for b in bands]
    out["sa_p95"] = [b[1] for b in bands]
    out["sa_lower"] = config.k_lower * out["sa_p5"]
    out["sa_upper"] = config.k_upper * out["sa_p95"]
    out["sa_too_small"] = out["area_um2"] < out["sa_lower"]
    out["sa_too_large"] = out["area_um2"] > out["sa_upper"]
    out["sa_direction"] = np.select(
        [out["sa_too_small"], out["sa_too_large"]], ["too_small", "too_large"], default="in_band"
    )
    # How far into / out of the band the area sits: 0 = at the lower bound, 1 = at the upper bound.
    span = (out["sa_upper"] - out["sa_lower"]).replace(0, np.nan)
    out["sa_band_position"] = (out["area_um2"] - out["sa_lower"]) / span
    out["band_agrees"] = (out["sa_too_small"] | out["sa_too_large"]) == out["sa_outlier_flag"]
    return out


def arrhenius_stage_hpf(start_age_hpf: float, temperature_c: float) -> float:
    """Temperature-corrected developmental stage for an embryo reared at ``temperature_c``.

    ``start_age_hpf`` is nominal CLOCK time since fertilization. The lab's Arrhenius relation
    converts it to the 28.5 C-equivalent developmental stage the reference curve is indexed on:

        stage = 6 + (clock_hours - 6) * (0.055 * T - 0.57)

    At 28.5 C the rate factor is ~0.998, so the correction is a no-op for the control cohort and
    grows with |T - 28.5|.
    """
    rate = 0.055 * float(temperature_c) - 0.57
    return ARRHENIUS_ANCHOR_HPF + (float(start_age_hpf) - ARRHENIUS_ANCHOR_HPF) * rate


def add_counterfactual_band(df: pd.DataFrame, reference_version: str = "v1") -> pd.DataFrame:
    """Re-derive the surface-area verdict against a temperature-corrected stage.

    This answers "would these embryos still fail if the band were evaluated at the developmental
    stage they were actually at?" It is a diagnostic, not a proposed fix.
    """
    config = resolve_config()
    reference = load_packaged_surface_area_reference(reference_version)

    out = df.copy()
    out["arrhenius_stage_hpf"] = [
        arrhenius_stage_hpf(a, t) for a, t in zip(out["start_age_hpf"], out["temperature"])
    ]
    bands = [interpolate_reference_band(float(s), reference) for s in out["arrhenius_stage_hpf"]]
    out["cf_lower"] = config.k_lower * np.array([b[0] for b in bands])
    out["cf_upper"] = config.k_upper * np.array([b[1] for b in bands])
    out["cf_sa_outlier_flag"] = (out["area_um2"] < out["cf_lower"]) | (
        out["area_um2"] > out["cf_upper"]
    )
    # Recompute the overall verdict swapping ONLY the surface-area flag.
    other = [c for c in FLAG_COLUMNS if c != "sa_outlier_flag"]
    out["cf_use_snip"] = ~(out[other].any(axis=1) | out["cf_sa_outlier_flag"])
    return out


def load_snip_inventory(experiment_id: str) -> pd.DataFrame:
    """Per-snip image paths for one experiment, concatenated across per-well inventories."""
    root = OUTPUT_ROOT / "object_extraction" / experiment_id / "snips" / "per_well"
    files = sorted(root.glob("*/*_snip_inventory.csv"))
    if not files:
        raise FileNotFoundError(f"no snip inventories under {root}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_frame_masks(experiment_id: str) -> pd.DataFrame:
    """Per-mask geometry + RLE for one experiment, concatenated across per-well files."""
    root = OUTPUT_ROOT / "object_extraction" / experiment_id / "frame_masks" / "per_well"
    files = sorted(root.glob("*/*_frame_masks.csv"))
    if not files:
        raise FileNotFoundError(f"no frame_masks under {root}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _resolve(path_value: object) -> Path | None:
    """Resolve an inventory path, which may be absolute or relative to the pipeline output root."""
    if path_value is None or (isinstance(path_value, float) and pd.isna(path_value)):
        return None
    text = str(path_value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = OUTPUT_ROOT / path
    return path if path.is_file() else None


# ---------------------------------------------------------------------------
# Image assembly
# ---------------------------------------------------------------------------

def _to_rgb(gray: np.ndarray) -> np.ndarray:
    """Stretch a grayscale array to full range and return it as float RGB in [0, 1]."""
    arr = gray.astype(np.float32)
    lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max()) or 1.0
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.dstack([arr] * 3)


def mask_outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Boolean outline of ``mask``, computed with numpy shifts only (no scipy/skimage needed)."""
    binary = mask.astype(bool)
    if not binary.any():
        return np.zeros_like(binary)
    eroded = binary.copy()
    for _ in range(max(int(thickness), 1)):
        shifted = eroded.copy()
        shifted[1:, :] &= eroded[:-1, :]
        shifted[:-1, :] &= eroded[1:, :]
        shifted[:, 1:] &= eroded[:, :-1]
        shifted[:, :-1] &= eroded[:, 1:]
        eroded = shifted
    return binary & ~eroded


def _paint(rgb: np.ndarray, outline: np.ndarray, color: str) -> np.ndarray:
    """Paint ``outline`` onto ``rgb`` in ``color``."""
    from matplotlib.colors import to_rgb

    painted = rgb.copy()
    painted[outline] = np.asarray(to_rgb(color), dtype=np.float32)
    return painted


def _pad_to_height(rgb: np.ndarray, height: int) -> np.ndarray:
    """Center-pad an RGB array vertically to ``height`` with mid-gray."""
    if rgb.shape[0] >= height:
        return rgb
    deficit = height - rgb.shape[0]
    top = deficit // 2
    pad_top = np.full((top, rgb.shape[1], 3), 0.5, dtype=np.float32)
    pad_bottom = np.full((deficit - top, rgb.shape[1], 3), 0.5, dtype=np.float32)
    return np.vstack([pad_top, rgb, pad_bottom])


def build_pair_image(
    snip_row: pd.Series,
    mask_row: pd.Series | None,
    *,
    outline_color: str = "#00d5ff",
    raw_pad_px: int = 60,
    raw_target_height: int = 576,
) -> np.ndarray:
    """Return one embryo's [snip | raw crop] RGB panel, segmentation outlined on both.

    The snip outline comes from the exported ``*_embryo.png`` snip mask; the raw outline is decoded
    from the ``frame_masks`` RLE and cropped around the mask bbox. If either is unavailable the
    corresponding panel is still shown, just without an outline — a missing mask is itself a finding
    and should not silently drop the embryo from the montage.
    """
    snip_path = _resolve(snip_row.get("processed_snip_path"))
    snip_rgb = (
        _to_rgb(np.asarray(Image.open(snip_path).convert("L")))
        if snip_path is not None
        else np.full((raw_target_height, 256, 3), 0.5, dtype=np.float32)
    )
    snip_mask_path = _resolve(snip_row.get("embryo_mask_snip_path"))
    if snip_mask_path is not None:
        snip_mask = np.asarray(Image.open(snip_mask_path).convert("L")) > 127
        if snip_mask.shape == snip_rgb.shape[:2]:
            snip_rgb = _paint(snip_rgb, mask_outline(snip_mask, 2), outline_color)

    raw_path = _resolve(snip_row.get("image_path"))
    if raw_path is None:
        raw_rgb = np.full_like(snip_rgb, 0.5)
    else:
        raw_full = np.asarray(Image.open(raw_path).convert("L"))
        full_mask = None
        if mask_row is not None and isinstance(mask_row.get("mask_rle"), str):
            try:
                decoded = decode_binary_mask_rle(json.loads(mask_row["mask_rle"]))
                if decoded.shape == raw_full.shape:
                    full_mask = decoded
            except (ValueError, KeyError, json.JSONDecodeError):
                full_mask = None

        if full_mask is not None and full_mask.any():
            ys, xs = np.where(full_mask)
            y0 = max(int(ys.min()) - raw_pad_px, 0)
            y1 = min(int(ys.max()) + raw_pad_px, raw_full.shape[0])
            x0 = max(int(xs.min()) - raw_pad_px, 0)
            x1 = min(int(xs.max()) + raw_pad_px, raw_full.shape[1])
            crop = raw_full[y0:y1, x0:x1]
            raw_rgb = _paint(_to_rgb(crop), mask_outline(full_mask[y0:y1, x0:x1], 2), outline_color)
        else:
            raw_rgb = _to_rgb(raw_full)

        scale = raw_target_height / raw_rgb.shape[0]
        new_size = (max(int(round(raw_rgb.shape[1] * scale)), 1), raw_target_height)
        raw_rgb = np.asarray(
            Image.fromarray((raw_rgb * 255).astype(np.uint8)).resize(new_size, Image.LANCZOS)
        ).astype(np.float32) / 255.0

    height = max(snip_rgb.shape[0], raw_rgb.shape[0])
    divider = np.full((height, 4, 3), 1.0, dtype=np.float32)
    return np.hstack([_pad_to_height(snip_rgb, height), divider, _pad_to_height(raw_rgb, height)])


def active_flags(row: pd.Series) -> list[FlagSpec]:
    """The FlagSpecs whose flag is set on ``row``, in canonical (ring) order."""
    return [spec for spec in FLAG_SPECS if bool(row.get(spec.column, False))]


def _draw_flag_rings(ax, specs: "list[FlagSpec]", *, ring_width: float = 0.022) -> None:
    """Draw one nested rectangle per active flag, outermost = first spec.

    Rings are drawn in axes coordinates so their thickness is independent of image size, which keeps
    a 2-flag embryo visually comparable across plates and crop sizes.
    """
    for depth, spec in enumerate(specs):
        inset = depth * ring_width
        ax.add_patch(
            Rectangle(
                (inset, inset),
                1 - 2 * inset,
                1 - 2 * inset,
                transform=ax.transAxes,
                facecolor="none",
                edgecolor=spec.color,
                linewidth=3.0,
                zorder=10 + depth,
                clip_on=False,
            )
        )


def build_failure_montage(
    experiment_id: str,
    df: pd.DataFrame,
    *,
    n_columns: int = 6,
    cell_width: float = 2.5,
    cell_height: float = 3.1,
    sort_by: str = "temperature",
):
    """Montage every failed embryo of one plate: [snip | raw] with nested flag rings.

    Each cell is one failed snip. The cyan contour is the segmentation mask (drawn on both panels,
    so a bad mask is visible against the raw image). The nested rectangles encode which QC flags
    fired — count the rings, read the colors off the legend.

    Returns the Figure, or ``None`` when the plate has no failures.
    """
    plate = df.loc[df["experiment_id"] == experiment_id].copy()
    failures = plate.loc[~plate["use_snip"]].copy()
    if failures.empty:
        return None

    if sort_by in failures.columns:
        failures = failures.sort_values([sort_by, "well_index"])

    inventory = load_snip_inventory(experiment_id).set_index("snip_id")
    masks = load_frame_masks(experiment_id)
    masks_by_id = masks.set_index("mask_id") if "mask_id" in masks.columns else None

    n = len(failures)
    n_rows = int(np.ceil(n / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(cell_width * n_columns, cell_height * n_rows), squeeze=False
    )

    for index, (_, row) in enumerate(failures.iterrows()):
        ax = axes[index // n_columns][index % n_columns]
        snip_id = row["snip_id"]
        if snip_id not in inventory.index:
            ax.text(0.5, 0.5, f"no inventory row\n{snip_id}", ha="center", va="center", fontsize=6)
            ax.set_axis_off()
            continue
        snip_row = inventory.loc[snip_id]
        if isinstance(snip_row, pd.DataFrame):
            snip_row = snip_row.iloc[0]

        mask_row = None
        mask_id = snip_row.get("mask_id")
        if masks_by_id is not None and isinstance(mask_id, str) and mask_id in masks_by_id.index:
            candidate = masks_by_id.loc[mask_id]
            mask_row = candidate.iloc[0] if isinstance(candidate, pd.DataFrame) else candidate

        ax.imshow(build_pair_image(snip_row, mask_row))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        _draw_flag_rings(ax, active_flags(row))

        area_k = row["area_um2"] / 1000.0
        position = row.get("sa_band_position", np.nan)
        ax.set_title(
            f"{row['well_index']}  {row['genotype']}\n"
            f"{row['temperature']:.1f}°C  stage {row['predicted_stage_hpf']:.0f} hpf\n"
            f"area {area_k:,.0f}k µm²"
            + (f"  (band pos {position:+.2f})" if pd.notna(position) else ""),
            fontsize=7,
        )

    for index in range(n, n_rows * n_columns):
        axes[index // n_columns][index % n_columns].set_axis_off()

    present = [spec for spec in FLAG_SPECS if failures[spec.column].any()]
    handles = [
        Line2D([0], [0], color=spec.color, lw=3, label=f"{spec.label}  ({int(failures[spec.column].sum())})")
        for spec in present
    ]
    handles.append(Line2D([0], [0], color="#00d5ff", lw=2, label="segmentation mask outline"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 4),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.012),
    )
    fig.suptitle(
        f"{experiment_id} — {n} of {len(plate)} snips failed QC   "
        f"(left: extracted snip · right: raw focus-stacked frame · rings = QC flags, outermost first)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# Quantification plots
# ---------------------------------------------------------------------------

def plot_flag_counts(df: pd.DataFrame):
    """Two panels: per-plate pass/fail, and flag incidence broken out by flag and plate."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    per_plate = (
        df.groupby("experiment_id")
        .agg(total=("snip_id", "size"), passed=("use_snip", "sum"))
        .reindex(list(EXPERIMENTS))
    )
    per_plate["failed"] = per_plate["total"] - per_plate["passed"]
    positions = np.arange(len(per_plate))
    axes[0].bar(positions, per_plate["passed"], color="#4c9f70", label="pass")
    axes[0].bar(
        positions, per_plate["failed"], bottom=per_plate["passed"], color="#e6194b", label="fail"
    )
    for x, (_, row) in zip(positions, per_plate.iterrows()):
        axes[0].text(
            x, row["total"] + 1.5, f"{100 * row['passed'] / row['total']:.0f}%", ha="center", fontsize=9
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([e.replace("20250612_", "") for e in per_plate.index], rotation=30, ha="right")
    axes[0].set_ylabel("snips")
    axes[0].set_title("QC outcome per plate")
    axes[0].legend(frameon=False)

    present = [spec for spec in FLAG_SPECS if df[spec.column].any()]
    width = 0.8 / len(EXPERIMENTS)
    base = np.arange(len(present))
    for offset, experiment_id in enumerate(EXPERIMENTS):
        plate = df.loc[df["experiment_id"] == experiment_id]
        counts = [int(plate[spec.column].sum()) for spec in present]
        axes[1].bar(
            base + offset * width,
            counts,
            width=width,
            label=experiment_id.replace("20250612_", ""),
        )
    axes[1].set_xticks(base + 0.4 - width / 2)
    axes[1].set_xticklabels([spec.label for spec in present], rotation=25, ha="right")
    axes[1].set_ylabel("snips flagged")
    axes[1].set_title("Flag incidence by plate (flags are not mutually exclusive)")
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def plot_flag_rate_by_temperature(df: pd.DataFrame):
    """sa_outlier rate as a function of rearing temperature, one line per nominal stage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for stage, group in df.groupby("start_age_hpf"):
        rates = group.groupby("temperature")["sa_outlier_flag"].mean() * 100
        axes[0].plot(rates.index, rates.values, marker="o", label=f"{stage:.0f} hpf (nominal)")
    axes[0].axvline(28.5, color="gray", ls=":", lw=1)
    axes[0].text(28.6, axes[0].get_ylim()[1] * 0.92, "28.5 °C\n(reference)", fontsize=8, color="gray")
    axes[0].set_xlabel("rearing temperature (°C)")
    axes[0].set_ylabel("sa_outlier_flag rate (%)")
    axes[0].set_title("Surface-area failures are a temperature effect,\nnot a stage effect")
    axes[0].legend(frameon=False)

    fail_direction = df.loc[df["sa_outlier_flag"]]
    grouped = (
        fail_direction.groupby(["temperature", "sa_direction"]).size().unstack(fill_value=0)
    )
    bottom = np.zeros(len(grouped))
    for column, color in (("too_small", "#4363d8"), ("too_large", "#e6194b")):
        if column in grouped.columns:
            axes[1].bar(
                np.arange(len(grouped)), grouped[column], bottom=bottom, label=column, color=color
            )
            bottom += grouped[column].to_numpy()
    axes[1].set_xticks(np.arange(len(grouped)))
    axes[1].set_xticklabels([f"{t:.1f}" for t in grouped.index])
    axes[1].set_xlabel("rearing temperature (°C)")
    axes[1].set_ylabel("snips flagged")
    axes[1].set_title("Direction of surface-area failure")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_area_against_band(df: pd.DataFrame, reference_version: str = "v1"):
    """Measured area vs the reference band, colored by temperature, per nominal stage."""
    reference = load_packaged_surface_area_reference(reference_version)
    config = resolve_config()
    stages = df["start_age_hpf"].dropna().unique()
    stages.sort()

    fig, axes = plt.subplots(1, len(stages), figsize=(5.2 * len(stages), 4.6), sharey=True, squeeze=False)
    colors = {24.0: "#4363d8", 28.5: "#4c9f70", 34.0: "#f58231", 35.0: "#e6194b"}
    grid = np.linspace(float(reference["stage_hpf"].min()), float(reference["stage_hpf"].max()), 300)
    bands = [interpolate_reference_band(float(s), reference) for s in grid]
    lower = config.k_lower * np.array([b[0] for b in bands])
    upper = config.k_upper * np.array([b[1] for b in bands])

    # The 9 whole-frame mask blowouts sit near 8e6 and would compress everything else to a sliver,
    # so the axis is clipped to the biologically plausible range and off-scale points are counted
    # in the panel title instead of silently cropped.
    y_max = 2.0e6

    for index, stage in enumerate(stages):
        ax = axes[0][index]
        ax.fill_between(grid, lower, upper, color="gray", alpha=0.16, label="passing band")
        ax.plot(grid, lower, color="#555555", lw=1.4, label="lower cut (0.9 × p5)")
        ax.plot(grid, upper, color="#555555", lw=1.4, ls="--", label="upper cut (1.4 × p95)")
        plate = df.loc[df["start_age_hpf"] == stage]
        off_scale = 0
        for temperature, group in plate.groupby("temperature"):
            jitter = np.random.default_rng(0).normal(0, 0.22, len(group))
            ax.scatter(
                group["predicted_stage_hpf"] + jitter,
                group["area_um2"],
                s=18,
                alpha=0.85,
                color=colors.get(temperature, "black"),
                label=f"{temperature:.1f} °C",
                edgecolor="none",
            )
            off_scale += int((group["area_um2"] > y_max).sum())
            corrected = arrhenius_stage_hpf(stage, temperature)
            ax.scatter(
                [corrected],
                [group["area_um2"].median()],
                marker="X",
                s=130,
                color=colors.get(temperature, "black"),
                edgecolor="black",
                linewidth=0.8,
                zorder=6,
            )
        ax.set_xlim(12, 48)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("stage_hpf")
        title = f"nominal {stage:.0f} hpf"
        if off_scale:
            title += f"   ({off_scale} mask blowouts off scale)"
        ax.set_title(title)
        if index == 0:
            ax.set_ylabel("area_um2")
            ax.legend(frameon=False, fontsize=7, loc="upper left")

    fig.suptitle(
        "Dots: area plotted at the stage QC used (= nominal clock time).  "
        "X: median area at the Arrhenius-corrected stage — where the cohort actually belongs.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_marginality(df: pd.DataFrame):
    """How far below the lower cut the 'too small' failures actually fall.

    This is the plot that separates "broken mask" from "tight threshold": a broken mask lands far
    below the cut, a healthy-but-small embryo lands just below it.
    """
    small = df.loc[df["sa_too_small"]].copy()
    small["frac_of_cut"] = small["area_um2"] / small["sa_lower"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].hist(small["frac_of_cut"], bins=np.linspace(0.2, 1.0, 33), color="#4363d8", edgecolor="white")
    axes[0].axvline(1.0, color="#e6194b", lw=2, label="the cut (0.9 × p5)")
    axes[0].axvline(0.8, color="gray", ls=":", lw=1.5, label="80% of the cut")
    axes[0].set_xlabel("area ÷ lower cut")
    axes[0].set_ylabel("snips flagged too small")
    n_marginal = int((small["frac_of_cut"] >= 0.8).sum())
    axes[0].set_title(
        f"{n_marginal} of {len(small)} 'too small' failures are within 20% of the cut\n"
        "(marginal — consistent with a tight threshold, not a bad mask)"
    )
    axes[0].legend(frameon=False, fontsize=8)

    ks = np.arange(0.50, 0.95, 0.01)
    reference_other = [c for c in FLAG_COLUMNS if c != "sa_outlier_flag"]
    other_fail = df[reference_other].any(axis=1)
    total = len(df)
    rates = []
    for k in ks:
        lower = (k / resolve_config().k_lower) * df["sa_lower"]
        sa = (df["area_um2"] < lower) | (df["area_um2"] > df["sa_upper"])
        rates.append(100 * (~(sa | other_fail)).sum() / total)
    axes[1].plot(ks, rates, marker="o", ms=3, color="#4c9f70")
    axes[1].axvline(0.9, color="#e6194b", lw=2, label="k_lower in force (0.9)")
    axes[1].set_xlabel("k_lower (multiplier on p5)")
    axes[1].set_ylabel("% of all snips passing QC")
    axes[1].set_title("Sensitivity: pass rate vs the lower-cut multiplier\n(upper cut held at 1.4 × p95)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    return fig


def plot_counterfactual(df: pd.DataFrame):
    """Pass rate as-run vs. with the surface-area band evaluated at a corrected stage."""
    counterfactual = add_counterfactual_band(df)
    summary = (
        counterfactual.groupby("experiment_id")
        .agg(
            total=("snip_id", "size"),
            passed=("use_snip", "sum"),
            counterfactual_passed=("cf_use_snip", "sum"),
        )
        .reindex(list(EXPERIMENTS))
    )
    summary["as_run_%"] = 100 * summary["passed"] / summary["total"]
    summary["corrected_%"] = 100 * summary["counterfactual_passed"] / summary["total"]

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(summary))
    ax.bar(positions - 0.2, summary["as_run_%"], width=0.4, label="as run", color="#e6194b")
    ax.bar(
        positions + 0.2,
        summary["corrected_%"],
        width=0.4,
        label="temperature-corrected stage",
        color="#4c9f70",
    )
    for x, (_, row) in zip(positions, summary.iterrows()):
        ax.text(x - 0.2, row["as_run_%"] + 1, f"{row['as_run_%']:.0f}", ha="center", fontsize=8)
        ax.text(x + 0.2, row["corrected_%"] + 1, f"{row['corrected_%']:.0f}", ha="center", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels([e.replace("20250612_", "") for e in summary.index], rotation=30, ha="right")
    ax.set_ylabel("% snips passing QC")
    ax.set_ylim(0, 108)
    ax.set_title("Counterfactual: pass rate if surface-area QC used the Arrhenius-corrected stage")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, summary
