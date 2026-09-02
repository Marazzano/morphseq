"""Render representative current-pipeline snips for the GENE7 morphology audit.

The model-input image is taken directly from ``processed_snip_path`` in each
experiment's merged snip inventory.  In particular, this script never displays
the legacy ``*_embryo.png`` mask files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq")
RESULT_ROOT = REPO_ROOT / "results/nlammers/20261002_GENE7"
PIPELINE_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output"
)

EXPERIMENTS = [
    "20250612_24hpf_ctrl_atf6",
    "20250612_24hpf_wfs1_ctcf",
    "20250612_30hpf_ctrl_atf6",
    "20250612_30hpf_wfs1_ctcf",
    "20250612_36hpf_ctrl_atf6",
    "20250612_36hpf_wfs1_ctcf",
    "20240626",
    "20240812",
    "20250215",
    "20240813_24hpf",
    "20240813_30hpf",
    "20240813_36hpf",
]

DISPLAY_NAMES = {
    "20250612_24hpf_ctrl_atf6": "GENE7 24 hpf | ctrl-atf6",
    "20250612_24hpf_wfs1_ctcf": "GENE7 24 hpf | wfs1-ctcf",
    "20250612_30hpf_ctrl_atf6": "GENE7 30 hpf | ctrl-atf6",
    "20250612_30hpf_wfs1_ctcf": "GENE7 30 hpf | wfs1-ctcf",
    "20250612_36hpf_ctrl_atf6": "GENE7 36 hpf | ctrl-atf6",
    "20250612_36hpf_wfs1_ctcf": "GENE7 36 hpf | wfs1-ctcf",
    "20240626": "WT reference | 20240626",
    "20240812": "WT reference | 20240812",
    "20250215": "WT reference | 20250215",
    "20240813_24hpf": "WT Hotfish | 24 hpf",
    "20240813_30hpf": "WT Hotfish | 30 hpf",
    "20240813_36hpf": "WT Hotfish | 36 hpf",
}

N_SAMPLES = 5


def inventory_path(experiment_id: str) -> Path:
    return (
        PIPELINE_ROOT
        / "object_extraction"
        / experiment_id
        / "snips"
        / f"{experiment_id}_snip_inventory.csv"
    )


def resolve_pipeline_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PIPELINE_ROOT / path


def select_representative_rows(frame: pd.DataFrame, n: int = N_SAMPLES) -> pd.DataFrame:
    """Select deterministic acquisition-spanning rows.

    Sorting time first makes longitudinal-reference samples span development.
    Snapshot experiments have only t0000, so the same rule spreads the samples
    over wells instead.
    """

    frame = frame.copy()
    if "is_valid_snip" in frame:
        valid = frame["is_valid_snip"].astype(str).str.lower().isin({"true", "1"})
        frame = frame.loc[valid]
    frame = frame.loc[frame["processed_snip_path"].notna()].copy()
    frame["resolved_image_path"] = frame["processed_snip_path"].map(resolve_pipeline_path)
    frame = frame.loc[frame["resolved_image_path"].map(Path.is_file)].copy()
    frame = frame.sort_values(["time_index", "well_id", "snip_id"], kind="stable")
    if len(frame) < n:
        raise RuntimeError(f"Only {len(frame)} usable snips found; requested {n}")

    indices = np.linspace(0, len(frame) - 1, n, dtype=int)
    return frame.iloc[indices].reset_index(drop=True)


def mask_path_for_row(row: pd.Series) -> Path | None:
    """Locate a mask only for calculating outside-embryo intensity metrics."""

    legacy_value = row.get("legacy_flat_snip_path")
    if not isinstance(legacy_value, str) or not legacy_value:
        return None
    base = resolve_pipeline_path(legacy_value)
    candidate = base.with_name(f"{base.stem}_embryo.png")
    return candidate if candidate.is_file() else None


def load_sample(row: pd.Series) -> tuple[np.ndarray, dict[str, object]]:
    image_path = Path(row["resolved_image_path"])
    array = np.asarray(Image.open(image_path).convert("L"))

    # Prefer the segmentation-defined background. The mask is used only to
    # calculate metrics; the displayed image is always the unmasked model input.
    mask_path = mask_path_for_row(row)
    if mask_path is not None:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        background = array[~mask] if mask.shape == array.shape else np.array([])
    else:
        mask = None
        background = np.array([])

    if background.size == 0:
        border = max(8, min(array.shape) // 12)
        border_mask = np.zeros(array.shape, dtype=bool)
        border_mask[:border] = True
        border_mask[-border:] = True
        border_mask[:, :border] = True
        border_mask[:, -border:] = True
        background = array[border_mask]
        metric_region = "image_border"
    else:
        metric_region = "outside_embryo_mask"

    metrics = {
        "experiment_id": row["experiment_id"],
        "snip_id": row["snip_id"],
        "well_id": row["well_id"],
        "time_index": int(row["time_index"]),
        "processed_snip_path": str(image_path),
        "background_metric_region": metric_region,
        "background_mean": float(np.mean(background)),
        "background_median": float(np.median(background)),
        "background_sd": float(np.std(background)),
        "background_nonzero_fraction": float(np.mean(background != 0)),
        "image_mean": float(np.mean(array)),
        "image_median": float(np.median(array)),
        "image_sd": float(np.std(array)),
        "image_min": int(np.min(array)),
        "image_max": int(np.max(array)),
    }
    return array, metrics


def short_label(row: pd.Series) -> str:
    well = str(row["well_id"]).removeprefix(f"{row['experiment_id']}_")
    return f"{well}  |  t{int(row['time_index']):04d}"


def draw_panel(
    axes: list[plt.Axes] | np.ndarray,
    rows: pd.DataFrame,
    arrays: list[np.ndarray],
    *,
    title: str,
    show_sample_labels: bool,
    vmax: float = 255,
) -> None:
    flat_axes = np.asarray(axes, dtype=object).ravel()
    for axis, (_, row), array in zip(flat_axes, rows.iterrows(), arrays):
        axis.imshow(array, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        axis.set_axis_off()
        if show_sample_labels:
            axis.set_title(short_label(row), fontsize=8, pad=3)
    flat_axes[0].text(
        0,
        1.15 if show_sample_labels else 1.04,
        title,
        transform=flat_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="semibold",
    )


def main() -> None:
    figure_root = RESULT_ROOT / "figures/image_processing_qc"
    panel_root = figure_root / "panels"
    data_root = RESULT_ROOT / "data"
    panel_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    selected: dict[str, pd.DataFrame] = {}
    images: dict[str, list[np.ndarray]] = {}
    metric_rows: list[dict[str, object]] = []

    for experiment_id in EXPERIMENTS:
        frame = pd.read_csv(inventory_path(experiment_id))
        rows = select_representative_rows(frame)
        arrays = []
        for _, row in rows.iterrows():
            array, metrics = load_sample(row)
            arrays.append(array)
            metric_rows.append(metrics)
        selected[experiment_id] = rows
        images[experiment_id] = arrays

        fig, axes = plt.subplots(1, N_SAMPLES, figsize=(10, 5.2), constrained_layout=True)
        draw_panel(
            axes,
            rows,
            arrays,
            title=f"{DISPLAY_NAMES[experiment_id]} — current non-_embryo model inputs",
            show_sample_labels=True,
        )
        fig.savefig(panel_root / f"{experiment_id}_current_snips.png", dpi=200)
        plt.close(fig)

    # One page for rapid cross-dataset inspection: each mini-panel is one dataset.
    fig = plt.figure(figsize=(24, 20), constrained_layout=True)
    outer = fig.add_gridspec(4, 3, hspace=0.10, wspace=0.06)
    for index, experiment_id in enumerate(EXPERIMENTS):
        sub = outer[index // 3, index % 3].subgridspec(1, N_SAMPLES, wspace=0.02)
        axes = [fig.add_subplot(sub[0, col]) for col in range(N_SAMPLES)]
        draw_panel(
            axes,
            selected[experiment_id],
            images[experiment_id],
            title=DISPLAY_NAMES[experiment_id],
            show_sample_labels=False,
        )
    fig.suptitle(
        "Current legacy-VAE model inputs (non-_embryo PNGs; fixed 0–255 display scale)",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(figure_root / "current_snips_overview.png", dpi=150)
    fig.savefig(figure_root / "current_snips_overview.pdf")
    plt.close(fig)

    # A low-intensity rendering of the same unmodified arrays makes residual
    # background structure visible. Embryo pixels are intentionally saturated.
    fig = plt.figure(figsize=(24, 20), constrained_layout=True)
    outer = fig.add_gridspec(4, 3, hspace=0.10, wspace=0.06)
    for index, experiment_id in enumerate(EXPERIMENTS):
        sub = outer[index // 3, index % 3].subgridspec(1, N_SAMPLES, wspace=0.02)
        axes = [fig.add_subplot(sub[0, col]) for col in range(N_SAMPLES)]
        draw_panel(
            axes,
            selected[experiment_id],
            images[experiment_id],
            title=DISPLAY_NAMES[experiment_id],
            show_sample_labels=False,
            vmax=40,
        )
    fig.suptitle(
        "Current model inputs — background-enhanced display (0–40; pixels unmodified)",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(figure_root / "current_snips_background_enhanced.png", dpi=150)
    plt.close(fig)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(data_root / "image_processing_qc_samples.csv", index=False)
    summary = (
        metrics.groupby("experiment_id", sort=False)
        .agg(
            n_samples=("snip_id", "size"),
            median_background_intensity=("background_median", "median"),
            mean_background_intensity=("background_mean", "mean"),
            mean_background_sd=("background_sd", "mean"),
            mean_background_nonzero_fraction=("background_nonzero_fraction", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(data_root / "image_processing_qc_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
