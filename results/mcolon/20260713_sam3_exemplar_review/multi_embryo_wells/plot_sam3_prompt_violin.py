"""Violin plots of SAM3 detection score distributions per prompt.

Reads benchmark_results.json (written by run_sam3_prompt_benchmark.py) and
makes two violin plots:
  1. All raw detection scores per prompt (everything SAM3 fired on, junk included)
  2. GT-matched scores per prompt (only detections that best-match a real
     reference box at IoU>=0.5) -- this is the "conditional on an embryo
     really being there" distribution that should calibrate the threshold.

Run with: conda run -n segmentation_grounded_sam --no-capture-output python \
    tmp/sam3_exemplar_review/multi_embryo_wells/plot_sam3_prompt_violin.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_PATH = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/multi_embryo_wells/"
    "sam3_prompt_benchmark/benchmark_results.json"
)
OUT_PATH = RESULTS_PATH.parent / "violin_score_distributions.png"


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    prompts = data["prompts"]  # name -> text
    prompt_order = list(prompts.keys())
    display_threshold = data["display_threshold"]

    raw_by_prompt = {name: [] for name in prompt_order}
    for row in data["raw_detections"]:
        raw_by_prompt[row["prompt"]].extend(row["scores"])

    matched_by_prompt = {name: [] for name in prompt_order}
    for row in data["calibration_rows"]:
        if row["best_iou"] >= 0.5 and row["matched_score"] is not None:
            matched_by_prompt[row["prompt"]].append(row["matched_score"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, dist_by_prompt, title in [
        (axes[0], raw_by_prompt, "All raw detections (incl. junk boxes)"),
        (axes[1], matched_by_prompt, "GT-matched detections only (IoU≥0.5)"),
    ]:
        plot_data = [dist_by_prompt[name] if dist_by_prompt[name] else [0] for name in prompt_order]
        parts = ax.violinplot(plot_data, showmeans=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.6)
        ax.axhline(display_threshold, color="red", linestyle="--", linewidth=1.5, label=f"display threshold ({display_threshold})")
        ax.set_xticks(range(1, len(prompt_order) + 1))
        ax.set_xticklabels([prompts[n] for n in prompt_order], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("SAM3 confidence score")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=9)

        for i, name in enumerate(prompt_order):
            n = len(dist_by_prompt[name])
            ax.annotate(f"n={n}", (i + 1, -0.03), ha="center", fontsize=8, color="gray")

    fig.suptitle("SAM3 text-prompt score distributions (zero-shot, no boxes/exemplars)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
