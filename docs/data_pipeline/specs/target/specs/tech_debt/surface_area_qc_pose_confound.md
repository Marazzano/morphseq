# Tech Debt: `surface_area_qc` Confounds Bad Masks With Real Dorsal-Pose Embryos

**Status:** known gap, mdcolon 2026-07-02. Recorded as an honest tech debt entry. `k_lower` was
raised 0.7 -> 0.9 as an interim mitigation (see below); the underlying confound is not fixed.

---

## The gap

`surface_area_qc` (`src/data_pipeline/quality_control/surface_area_qc/`) gates on `area_um2` vs a
stage-interpolated reference band (`[k_lower*p5, k_upper*p95]`). Area alone cannot distinguish two
very different failure/pose populations that land in the same low-area region of the band:

1. **Bad masks** — SAM2 mask captures only the yolk ball, missing the tail/trunk entirely (e.g.
   `20250912_H04_e01_BF_t0020`, `20250912_H05_e02_BF_t0016`). These are genuinely bad segmentation
   and should fail.
2. **Real, healthy embryos in a dorsal/thin pose** — later-stage larvae that have turned to a
   slender/elongated orientation have naturally low footprint area relative to body length (e.g.
   `20250912_H05_e02_BF_t0096`, aspect ratio ~8.2, area 0.82 mm² at 95.9 hpf — a real fish with
   head/eye/tail clearly visible, just captured edge-on).

Both populations can sit at the same `area_um2` for their stage, so a scalar area threshold cannot
cleanly separate them.

---

## Why `k_lower` alone can't fix it (empirical sweep, 20250912 experiment, alive embryos only)

Swept `k_lower` from 0.7 (current) upward and classified each newly-failing snip by `aspect =
length_um / width_um` as a rough proxy (aspect < 2 ~ round/compact, likely bad mask; aspect >= 2 ~
elongated, likely real embryo):

| k_lower | n flipped to fail | round/compact (aspect<2) | elongated (aspect>=2) |
|---|---|---|---|
| 0.75 | 113 | 49 | 64 |
| 0.85 | 451 | 160 | 291 |
| 0.95 | 774 | 196 | 578 |
| 1.05 | 1198 | 331 | 867 |
| 1.15 | 1720 | 548 | 1172 |

Even at the smallest bump needed to catch the two originally-flagged bad snips (`k_lower ~0.71`),
more real elongated embryos flip to fail than bad round-blob masks, and the imbalance gets worse
the higher `k_lower` goes. Thin/dorsal-pose larvae have naturally low area-per-length and are
caught *before* the round yolk-only blobs, whose aspect ratio (~1.2-1.5) is actually closer to the
"normal" compact-embryo aspect than the real slender larvae are.

---

## Decision (2026-07-02)

Raise `k_lower` anyway, as an interim, deliberate tradeoff: **prefer excluding thin/dorsal-pose
embryos (real animals, but lower morphological information content in that pose) over keeping
bad-mask yolk-only fragments** (garbage data with no usable body shape). This is an explicit
call — losing some real, low-information embryos is judged better than passing genuinely broken
segmentations through downstream analysis.

This is NOT a full fix. It does not resolve the underlying confound; it just accepts more
collateral loss on the "real but skinny" side to buy a partial reduction in bad-mask false passes.

Landed at **`k_lower=0.9`** (0.7 -> 0.75 -> 0.8 -> 0.85 -> 0.9), reached by iterating against
concrete flagged snips rather than the sweep table alone. At each step, specific yolk-only masks
that were still passing were pulled up as full-population galleries
(`results/mcolon/20260702_qc_reporting_v1/scratch_scripts/lower_border_gallery_k080.py`, reused/
edited in place across iterations — filename is stale, content is current) and visually confirmed:

- `0.7` (original default): `20250912_H04_e01_BF_t0020`, `20250912_H05_e02_BF_t0016` pass — both
  yolk-only.
- `0.75`: those two now fail, but `20250912_H04_e02_BF_t0015` still passes by a 2% margin — also
  yolk-only (this also caught a bug in the debug gallery script itself: bucketing quartiles from a
  pre-filtered 64-snip window mislabeled a 2%-margin pass as "clear_pass"; fixed to rank against
  the full population).
- `0.8`: `20250912_H04_e02_BF_t0015` now fails, but `20250912_E07_e01_BF_t0008` and
  `20250912_H04_e02_BF_t0014` still pass (~0.2% margin) — both yolk-only.
- `0.85`: those two now fail, but round-blob masks (`20250912_C11_e02_BF_t0048`,
  `20250912_C11_e02_BF_t0049`) still pass comfortably (not even a boundary case), while real
  elongated larvae started appearing in `borderline_fail` as collateral.
- `0.9`: all 7 flagged yolk-only snips fail; the round-blob population is visually absent from
  `borderline_pass`/`clear_pass` in the full-population gallery. One residual round blob
  (`20250912_H12_e01_BF_t0021`) appears in `borderline_fail`, correctly failing. Held here.

Collateral at 0.9 was not re-run through the aspect-ratio sweep table above (built for 0.7-1.15 in
0.05 steps, alive population) — extrapolate roughly from the 0.85 row (451 flipped, ~2:1 real
elongated vs round/compact) as a lower bound; the true 0.9 flip count is higher.

---

## The real fix (deferred)

Area-based gating cannot resolve this confound because it has no shape information. Candidate
shape-aware signals that could distinguish "yolk-only blob" from "real slender embryo" without
conflating them:

- **Solidity** (mask area / convex hull area) — a yolk ball is close to its own convex hull
  (solidity ~1); a real embryo with a curled or extended tail has a lower solidity.
- **Circularity** (`4*pi*area / perimeter^2`, cheap — `perimeter_um` already in `mask_geometry`) —
  yolk-only blobs should score high (near-circular); real embryos, even slender ones, should not.
- **Yolk-fraction ratio** (yolk UNet mask area vs whole-body mask area) — the most direct signal,
  but blocked today: the UNet `via` (viability) mask is NOT a body mask (it's dead-tissue
  detection) and cannot stand in for "whole body." The yolk UNet mask also lives in snip-crop
  coordinates while the SAM2 embryo mask (the one `mask_geometry`/`surface_area_qc` actually
  consumes) lives in full-frame coordinates — reprojecting one into the other's frame is unsolved
  and would need the snip crop/rotation transform to be recorded and read back.

None of these have been implemented. Circularity/solidity are the cheapest next step since they
need no cross-mask alignment — just the existing SAM2-derived `mask_geometry` fields (or a fresh
pass over the RLE mask for exact circularity/solidity, since `mask_geometry` does not currently
store convex hull area).

---

## Where to look when paying this debt

- Config / current `k_lower`, `k_upper`: `src/data_pipeline/quality_control/surface_area_qc/config.py`
- Band computation: `src/data_pipeline/quality_control/surface_area_qc/reference.py`
  (`interpolate_reference_band`)
- Area/length/width source: `src/data_pipeline/feature_extraction/mask_geometry/compute.py` (decodes
  the SAM2 `mask_rle` from `frame_masks`, full-frame coordinates)
- Yolk UNet mask (snip-crop coordinates, NOT currently aligned to the SAM2 mask):
  `src/data_pipeline/object_extraction/segmentation/backends/unet_snip/` +
  `snip_auxiliary_masks_contract.py` (`ALLOWED_AUXILIARY_MASK_TYPES` includes `yolk`)
- Debug galleries and the empirical `k_lower` sweep script (ad-hoc, not wired into the DAG):
  `results/mcolon/20260702_qc_reporting_v1/scratch_scripts/`
