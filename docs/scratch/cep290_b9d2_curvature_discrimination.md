# cep290 LH/HL vs b9d2 CE — localizing the phenotype (WHERE + WHAT) via OT

*Scratch note — reflecting back the meeting goal, then inventorying what we already have.*

## The biological question (restated)

The concrete, defensible use case for this meeting is **discrimination**, not description:

1. **Separate the two cep290 phenotypes from each other** — the LH and HL directions
   of the same allele. This is *probably a gross morphological difference in
   curvature* (not a fine/pixel-level one).
2. **Separate cep290 from b9d2** — the b9d2 CE phenotype could be *either* gross
   morphological *or* pixel-level. We don't know a priori, which is exactly why we
   want a method that can localize the signal wherever it lives.
3. **Say WHERE and WHAT the phenotype is** — not just "these are different," but
   which region along the body carries the signal and what is changing there.
   **Morphological embeddings can't tell you this directly** — they give you a
   distance, not a location or a mechanism. That's the whole motivation.

**The enabling requirement:** to compare embryos we need a **shared, stage-matched
reference frame** so that "curvature at position S" means the same thing across
embryos. That's the canonical grid + OT machinery below.

## Method chain (the actual plan)

The way we get from "different" to "where and what":

1. **OT on the binary mask** of each embryo → a **stage-matched WT reference**.
   (WT-referenced UOT on the canonical grid. Stage-matching the reference is the key
   move — compare like-to-like developmental stage, not raw hpf.)
2. **Look at the raw OT map first.** If the phenotype signal (cost / displacement /
   delta-mass field) is already visually obvious on the map, we're done — that *is*
   the where-and-what.
3. **If it's not clear from the raw map, apply the restricted / regularized linear
   classifier** (already implemented in `results/`) to localize *where* along the
   body axis the signal lives (per-S-bin AUROC, interval search, TV-regularized
   weight map). The classifier finds the discriminating region so we don't have to
   eyeball it.

This is how OT + a restricted classifier answers **WHERE** and **WHAT**, which a
morphological embedding fundamentally cannot.

## Meeting constraint noted

Attendance / grouping constraint the user flagged — need to find groupings that work
given who's in the meeting. *(Placeholder — clarify what grouping means: analysis
cohorts? presentation audience? Left as a TODO.)*

---

## What we already have wired (but haven't fully used for this)

We've put a lot of work into the canonical grid and optimal-transport utilities. The
pieces below already exist and map directly onto the question above.

### 1. Shared reference frame — canonical grid (Stage 1)
- `src/analyze/utils/coord/grids/canonical.py` — `CanonicalGridMapper`,
  `to_canonical_grid_{mask,image,frame}`. Puts every embryo into a common
  256×576 (≈10 µm/px) frame. Convention: yolk on LEFT, head on LEFT, all → flip=True.
- `back_direction.py` — back/dorsal direction estimation feeding the flip decision.
- `register.py` (Stage 2) — explicit rotate / pivot-translate registration on top of
  the canonical grid (never called implicitly by UOT).

**This is the "shared grid" the meeting needs.** It already gives a common coordinate
system to compare curvature across embryos.

### 2. What-is-changing engine — OT feature maps (Phase 0)
`results/mcolon/20260215_roi_discovery_via_ot_feature_maps/`
- `p0_ot_maps.py` — WT-referenced UOT → per-embryo maps of **cost density,
  displacement field, delta-mass** on the canonical grid. This is literally "how does
  this embryo's mass have to move to become the reference" = a shape-change field.
- `p0_s_coordinate.py` — builds the **S coordinate** (0=head → 1=tail) along the body
  axis via centerline extraction. Turns 2D maps into a **1D rostral-caudal axis**.
- `p0_sbin_features.py` — aggregates the OT maps into per-embryo, per-S-bin features
  (the small reusable `features_sbins.parquet`). Includes displacement decomposed into
  parallel/perpendicular (tangent/normal) — **perpendicular displacement ≈ curvature
  change**, which is exactly the cep290 signal.
- `p0_classification.py` / `p0_interval_search.py` — AUROC per S-bin: *where along the
  axis* does the phenotype separate. `p0_nulls.py` — permutation + bootstrap validity.

**This is the "what is changing, and where" engine.** It was built for WT-vs-cep290;
it has not yet been pointed at (cep290-LH vs cep290-HL) or (cep290 vs b9d2).

---

## The gap / next steps

We have the stage-matched grid, the OT maps, and the restricted classifier, but
haven't run the chain on the pairs that matter. Concretely:

1. **Generate stage-matched WT-referenced OT maps** for the cep290 and b9d2 embryos
   (this is the `p0_ot_maps.py` step — confirm the reference is stage-matched, not
   raw-hpf-matched).
2. **Eyeball the raw OT maps first** for each contrast. If the cep290 curvature
   signal (likely gross) is already obvious, that answers where/what directly.
3. **Re-target the restricted classifier** (per-S-bin AUROC / interval search /
   TV weight map, already in `results/`) to the pairs, for wherever the raw map is
   ambiguous — most likely needed for **b9d2**, since its signal may be pixel-level:
   - cep290-LH vs cep290-HL
   - cep290 vs b9d2 (esp. at 18 hpf)
4. **Curvature readout:** the tangent/normal (perpendicular) displacement component
   in the S-bin features is the curvature axis — foreground it for cep290.
5. Resolve the meeting **grouping constraint** placeholder above.

## Pointers
- Phase 0 plan / gates: `results/mcolon/20260215_roi_discovery_via_ot_feature_maps/PLAN.md`
- Canonical aligner debug: `results/mcolon/20260216_canonical_aligner_debug/`
- Key embryos: `20251113_A05_e01` (Not Penetrant), `20251113_E04_e01` (CEP290)
