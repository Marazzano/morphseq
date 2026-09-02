"""Side-by-side contact sheets for legacy vs regenerated snip pairs.

Companion to ``snip_legacy_diff.py``. That script says how much each pair differs; this one shows
what the difference looks like, because a number cannot distinguish a uniform brightness shift from
a structural failure confined to the embryo.

Sheets are selected rather than sampled: the worst pairs, the median band, the best pairs, and any
orientation-flipped pair each get their own sheet. Contact sheets that show only medians hide the
failures they exist to surface.

Each row is: legacy | regenerated | |difference|, with the difference on its own colour scale and a
shared grayscale for the two images so they are visually comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as skio


def load_gray(path: Path) -> np.ndarray:
    image = skio.imread(str(path))
    return image[..., 0] if image.ndim == 3 else image


def resolve(legacy_dir: Path, snips_root: Path, row) -> tuple[np.ndarray, np.ndarray] | None:
    legacy_candidates = [legacy_dir / f"{row.snip_id}{ext}" for ext in (".jpg", ".jpeg", ".png")]
    legacy_path = next((p for p in legacy_candidates if p.exists()), None)
    regenerated_path = next(snips_root.rglob(f"{row.regenerated_snip_id}.png"), None)
    if legacy_path is None or regenerated_path is None:
        return None
    return load_gray(legacy_path), load_gray(regenerated_path)


def build_sheet(rows, legacy_dir: Path, snips_root: Path, title: str, out_path: Path) -> bool:
    loaded = []
    for row in rows:
        pair = resolve(legacy_dir, snips_root, row)
        if pair is not None:
            loaded.append((row, *pair))
    if not loaded:
        print(f"  {title}: no resolvable pairs, skipped")
        return False

    n = len(loaded)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.7 * n), squeeze=False)
    for i, (row, legacy, regenerated) in enumerate(loaded):
        shown = np.rot90(regenerated, 2) if bool(row.orientation_flipped) else regenerated
        diff = np.abs(legacy.astype(np.int16) - shown.astype(np.int16))

        for j, (image, label) in enumerate(
            ((legacy, "legacy"), (shown, "regenerated"), (diff, "|difference|"))
        ):
            ax = axes[i][j]
            if j < 2:
                ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            else:
                im = ax.imshow(image, cmap="magma", vmin=0, vmax=max(10, int(diff.max())))
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(label, fontsize=9)
        flag = "  [FLIPPED, shown de-rotated]" if bool(row.orientation_flipped) else ""
        axes[i][0].set_ylabel(
            f"{row.snip_id.replace('20250612_30hpf_ctrl_atf6_', '')}\n"
            f"mean {row.mean_abs_diff:.2f}  max {int(row.max_abs_diff)}{flag}",
            fontsize=7, rotation=0, ha="right", va="center", labelpad=48,
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {title}: {n} pairs -> {out_path.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--snips-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-per-sheet", type=int, default=6)
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv)
    comparable = results[results["comparable"]] if "comparable" in results else results
    aligned = comparable[~comparable["orientation_flipped"]].sort_values("mean_abs_diff")
    flipped = comparable[comparable["orientation_flipped"]]

    n = args.n_per_sheet
    midpoint = len(aligned) // 2
    groups = [
        (list(aligned.tail(n)[::-1].itertuples()), f"Worst {n} pairs by mean|difference|", "worst"),
        (list(aligned.iloc[max(0, midpoint - n // 2): midpoint + (n + 1) // 2].itertuples()),
         "Median band", "median"),
        (list(aligned.head(n).itertuples()), f"Best {n} pairs", "best"),
        (list(flipped.itertuples()), "Orientation-flipped (shown de-rotated)", "flipped"),
    ]
    print(f"pairs available: {len(aligned)} aligned, {len(flipped)} flipped")
    for rows, title, slug in groups:
        if rows:
            build_sheet(rows, args.legacy_dir, args.snips_root, title,
                        args.out_dir / f"panels_{slug}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
