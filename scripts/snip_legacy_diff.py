"""One-to-one pixel comparison of legacy vs regenerated embryo snips.

The acceptance gate for the 6.5 um/px + 75 um blend-radius restoration. Pairs snips by
``snip_id`` and reports a per-snip pixel difference. Aggregates are printed as a summary only;
the artifact is the per-snip CSV, because a median over a skewed distribution is exactly what
would hide a localized rendering failure.

Two confounds are handled explicitly rather than averaged away:

ORIENTATION. A 180-degree flip produces an enormous pixel difference that means "orientation
disagrees", not "rendering disagrees". Flipped pairs are detected and bucketed separately; letting
them into the diff distribution would swamp the signal being measured.

JPEG. Legacy snips are JPEG (lossy), regenerated snips are PNG (lossless), so a raw diff carries a
compression noise floor. Measured at quality 95 on this corpus: mean|d| 0.123, p99 1, max 3 grey
levels. ``--noise-floor`` defaults to 3 accordingly: differences at or below it are not evidence.

Self-contained by design -- imports no ``data_pipeline`` module, so it runs without the package
being installed and without sys.path manipulation (AGENTS.md).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio

LEGACY_NAME_RE = re.compile(r"^(?P<snip_id>.+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

# Legacy and regenerated snip_ids do not match literally. Legacy is
# ``{experiment}_{well}_e00_t0000``; the regenerated id is 1-based and carries a channel token,
# ``{experiment}_{well}_e01_BF_t0000``. Both parse with one pattern, anchored from the right so an
# experiment name containing underscores stays intact.
SNIP_ID_RE = re.compile(
    r"^(?P<experiment>.+)_(?P<well>[A-Z]\d{2})_e(?P<embryo>\d+)"
    r"(?:_(?P<channel>[A-Za-z0-9]+))?_t(?P<time>\d+)$"
)


def pair_key(snip_id: str) -> tuple[str, str, str] | None:
    """Canonical join key: (experiment, well, timepoint).

    Deliberately excludes the embryo index. The two sides number embryos differently, so pairing on
    it would silently mis-join. Wells holding more than one embryo per timepoint are reported as
    ambiguous rather than guessed at -- in this corpus the legacy side kept only one embryo for such
    wells, so any forced pairing would compare different animals.
    """
    match = SNIP_ID_RE.match(snip_id)
    if not match:
        return None
    return (match.group("experiment"), match.group("well"), match.group("time"))


def pair_by_key(
    legacy_index: dict, regenerated_index: dict
) -> tuple[list[tuple[str, str]], list[str], list[str], list[tuple]]:
    """Return (pairs, legacy_unpaired, regenerated_unpaired, ambiguous)."""
    from collections import defaultdict

    legacy_by_key, regenerated_by_key = defaultdict(list), defaultdict(list)
    unparsed_legacy, unparsed_regenerated = [], []
    for snip_id in legacy_index:
        key = pair_key(snip_id)
        (legacy_by_key[key].append(snip_id) if key else unparsed_legacy.append(snip_id))
    for snip_id in regenerated_index:
        key = pair_key(snip_id)
        (regenerated_by_key[key].append(snip_id) if key else unparsed_regenerated.append(snip_id))

    pairs, ambiguous = [], []
    for key in sorted(set(legacy_by_key) & set(regenerated_by_key)):
        left, right = legacy_by_key[key], regenerated_by_key[key]
        if len(left) == 1 and len(right) == 1:
            pairs.append((left[0], right[0]))
        else:
            ambiguous.append((key, len(left), len(right)))

    paired_legacy = {a for a, _ in pairs}
    paired_regenerated = {b for _, b in pairs}
    legacy_unpaired = sorted(set(legacy_index) - paired_legacy)
    regenerated_unpaired = sorted(set(regenerated_index) - paired_regenerated)
    return pairs, legacy_unpaired, regenerated_unpaired, ambiguous


def load_gray(path: Path) -> np.ndarray:
    """Read a snip as a 2-D uint8 array, dropping a redundant channel axis if present."""
    image = skio.imread(str(path))
    if image.ndim == 3:
        if image.shape[2] not in (1, 3, 4):
            raise ValueError(f"{path}: unexpected channel count {image.shape[2]}")
        image = image[..., 0]
    if image.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8, got {image.dtype}")
    return image


def index_legacy(legacy_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(legacy_dir.iterdir()):
        match = LEGACY_NAME_RE.match(path.name)
        if match:
            out[match.group("snip_id")] = path
    return out


def index_regenerated(snips_root: Path, inventory_csv: Path | None) -> dict[str, Path]:
    """Resolve regenerated snips by snip_id.

    Prefers the inventory's ``processed_snip_path`` -- that is the authority the pipeline itself
    writes and reads. Falls back to a recursive filename scan so a pilot run without a merged
    inventory can still be compared.
    """
    if inventory_csv is not None and inventory_csv.exists():
        frame = pd.read_csv(inventory_csv)
        missing = {"snip_id", "processed_snip_path"} - set(frame.columns)
        if missing:
            raise ValueError(f"{inventory_csv}: inventory missing column(s) {sorted(missing)}")
        frame = frame[frame["processed_snip_path"].notna()]
        return {
            str(row.snip_id): (snips_root / str(row.processed_snip_path)).resolve()
            if not Path(str(row.processed_snip_path)).is_absolute()
            else Path(str(row.processed_snip_path))
            for row in frame.itertuples()
        }
    return {path.stem: path for path in sorted(snips_root.rglob("*.png"))}


def compare_pair(legacy: np.ndarray, regenerated: np.ndarray, noise_floor: int) -> dict:
    """Per-snip difference. Reports the flipped-orientation case rather than absorbing it."""
    direct = np.abs(legacy.astype(np.int16) - regenerated.astype(np.int16))
    flipped_candidate = np.abs(legacy.astype(np.int16) - np.rot90(regenerated, 2).astype(np.int16))

    # A genuine 180-degree flip makes the rotated comparison dramatically better. Requiring a
    # clear margin keeps near-symmetric embryos from being mislabeled.
    is_flipped = bool(flipped_candidate.mean() < 0.5 * direct.mean())
    diff = flipped_candidate if is_flipped else direct

    above = diff > noise_floor
    return {
        "orientation_flipped": is_flipped,
        "mean_abs_diff": float(diff.mean()),
        "rmse": float(np.sqrt((diff.astype(np.float64) ** 2).mean())),
        "max_abs_diff": int(diff.max()),
        "p99_abs_diff": float(np.percentile(diff, 99)),
        "frac_pixels_above_floor": float(above.mean()),
        "n_pixels_above_floor": int(above.sum()),
        "legacy_mean": float(legacy.mean()),
        "regenerated_mean": float(regenerated.mean()),
        "legacy_frac_saturated": float((legacy >= 250).mean()),
        "regenerated_frac_saturated": float((regenerated >= 250).mean()),
        "legacy_foreground_px": int((legacy > 0).sum()),
        "regenerated_foreground_px": int((regenerated > 0).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-dir", type=Path, required=True,
                        help="Directory of legacy snips for one experiment.")
    parser.add_argument("--snips-root", type=Path, required=True,
                        help="Root of the regenerated snip output.")
    parser.add_argument("--inventory-csv", type=Path, default=None,
                        help="Regenerated snip inventory; resolves processed_snip_path.")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Per-snip results. Always written, one row per pair.")
    parser.add_argument("--noise-floor", type=int, default=3,
                        help="Grey levels at or below which a difference is not evidence.")
    args = parser.parse_args()

    legacy_index = index_legacy(args.legacy_dir)
    regenerated_index = index_regenerated(args.snips_root, args.inventory_csv)
    if not legacy_index:
        raise SystemExit(f"No legacy snips found under {args.legacy_dir}")
    if not regenerated_index:
        raise SystemExit(f"No regenerated snips found under {args.snips_root}")

    pairs, legacy_only, regenerated_only, ambiguous = pair_by_key(legacy_index, regenerated_index)
    shared = pairs

    rows = []
    shape_mismatches = []
    for snip_id, regenerated_id in pairs:
        legacy = load_gray(legacy_index[snip_id])
        regenerated = load_gray(regenerated_index[regenerated_id])
        if legacy.shape != regenerated.shape:
            # Not diffable. Recorded, never silently skipped.
            shape_mismatches.append((snip_id, legacy.shape, regenerated.shape))
            rows.append({"snip_id": snip_id, "regenerated_snip_id": regenerated_id,
                         "comparable": False,
                         "legacy_shape": str(legacy.shape),
                         "regenerated_shape": str(regenerated.shape)})
            continue
        rows.append({"snip_id": snip_id, "regenerated_snip_id": regenerated_id,
                     "comparable": True,
                     "legacy_shape": str(legacy.shape),
                     "regenerated_shape": str(regenerated.shape),
                     **compare_pair(legacy, regenerated, args.noise_floor)})

    results = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)

    print(f"legacy snips        : {len(legacy_index)}")
    print(f"regenerated snips   : {len(regenerated_index)}")
    print(f"paired              : {len(shared)}")
    print(f"legacy only         : {len(legacy_only)}{' -> ' + ', '.join(legacy_only[:5]) if legacy_only else ''}")
    print(f"regenerated only    : {len(regenerated_only)}{' -> ' + ', '.join(regenerated_only[:5]) if regenerated_only else ''}")
    print(f"shape mismatches    : {len(shape_mismatches)}")
    print(f"ambiguous (n:m)     : {len(ambiguous)}"
          + ("".join(f"\n    {k} legacy={l} regenerated={r}" for k, l, r in ambiguous[:5])
             if ambiguous else ""))
    print(f"per-snip CSV        : {args.output_csv}")

    comparable = results[results["comparable"]] if "comparable" in results else results
    if comparable.empty:
        print("\nNo comparable pairs. Nothing was measured.")
        return 1

    flipped = comparable[comparable["orientation_flipped"]]
    aligned = comparable[~comparable["orientation_flipped"]]
    print(f"\norientation-flipped : {len(flipped)} / {len(comparable)}"
          f" ({100.0 * len(flipped) / len(comparable):.1f}%) -- excluded from the diff summary")

    if aligned.empty:
        print("Every comparable pair is flipped; no aligned pixel comparison to report.")
        return 1

    print(f"\nAligned pairs (n={len(aligned)}), per-snip mean|diff|:")
    print(f"  min {aligned['mean_abs_diff'].min():.3f}"
          f" | median {aligned['mean_abs_diff'].median():.3f}"
          f" | max {aligned['mean_abs_diff'].max():.3f}")
    identical = int((aligned["n_pixels_above_floor"] == 0).sum())
    print(f"  identical within noise floor ({args.noise_floor}): {identical} / {len(aligned)}")
    print("\nWorst 10 by mean|diff| (inspect these, not the median):")
    worst = aligned.nlargest(min(10, len(aligned)), "mean_abs_diff")
    for row in worst.itertuples():
        print(f"  {row.snip_id:<45} mean {row.mean_abs_diff:7.3f}"
              f"  max {row.max_abs_diff:3d}  frac>floor {row.frac_pixels_above_floor:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
