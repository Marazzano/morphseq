# 20260729 — 20250612 (GENE7) QC failure diagnosis

Why the six 20250612 plates pass only 74–96% of snips when manual inspection says the imaging is
good. Asked as: is segmentation failing, or are QC flags firing on good data?

**Answer: QC flags, and it is one check.** `sa_outlier_flag` is implicated in 96 of 106 failures.
87 are "too small", and 75 of those sit within 20% of the cut — a tight-threshold signature, not a
broken-mask one. The montages confirm the masks track the embryo boundary.

## Files

- `qc_diagnostic_utils.py` — all the logic: loaders, band reconstruction, montage builder, plots.
  Importable outside Jupyter; the notebook is a thin narrative wrapper.
- `build_qc_diagnostic_nb.py` — regenerates `qc_flag_diagnosis.ipynb` (does not execute it).
- `qc_flag_diagnosis.ipynb` — the executed analysis.
- `figures/` — every figure as `.png` + `.pdf`, including one montage per plate.

## Running it

Needs pandas + pyarrow + matplotlib + skimage together; `points-ml` is the only env on the cluster
with all of them (`morphseq-env` has no pyarrow, `vae-env-cluster` has no nbclient).

```bash
cd results/nlammers/20260729_morphseq_integration
PYTHONPATH=../../../src:. \
  /net/trapnell/vol1/home/nlammers/micromamba/envs/points-ml/bin/python build_qc_diagnostic_nb.py
# then execute the notebook in that env (~1 min)
```

The surface-area band is re-derived from the packaged reference curve using the product's own
`k_upper`/`k_lower`, imported rather than copied. The notebook asserts `band_agrees` on all 586
snips — i.e. the reconstruction reproduces the flag the pipeline actually wrote — so nothing
downstream rests on a re-implementation that might have drifted.

## Findings

| | |
|---|---|
| Snips | 586 across 6 plates |
| Failures | 106 (81.9% pass) |
| `sa_outlier_flag` | 96 of 106 failures |
| Direction | 87 too small · 9 too large |
| Marginal (≥80% of cut) | 75 of 87 |
| Worst cell | 30 hpf @ 24 °C — 65% flagged |
| Best cell | 34 °C — ~0–2% flagged across all stages |

1. **Failures track rearing temperature, not stage.** Each plate is a 4-temperature block design.
   24 °C fails at 10 / 65 / 56% (24 / 30 / 36 hpf); 34 °C at ~0–2%.

2. **Reference-population mismatch, not a bug.** `k_lower = 0.9 × p5` is evaluated against a
   wildtype curve built at ~28.5 °C. Cold-reared embryos are genuinely smaller, so a healthy 24 °C
   cohort lands just under the 5th percentile of a reference it does not belong to. Confirmed
   independently: snip mask fill fraction rises monotonically with temperature in *passing* embryos
   too (0.073 → 0.089 → 0.103 → 0.098 at 24 / 28.5 / 34 / 35 °C).

3. **The obvious fix does not work.** Re-indexing the band on Arrhenius-corrected developmental
   stage recovers 55 cold-cohort snips but newly flags 45 hot-cohort ones — net 10. Temperature
   advances stage without proportionally increasing projected area, so no stage index rescues a
   tight two-sided area band on a temperature series.

4. **Relaxing `k_lower` does work.** 0.90 → 0.80 takes the pass rate 81.9% → 91.5%; → 0.75 gives
   93.9%, after which the curve plateaus with ~16–23 residual flags. `k_lower` was 0.7 before being
   raised to 0.9 on 2026-07-02.

5. **The 9 "too large" failures are real.** `length_um ≈ 4400`, `width_um ≈ 2000` — the full field
   of view, 5–7× the upper cut. Whole-frame masks where the embryo was not found and SAM2 segmented
   the background. All 9 are independently flagged out of focus. The check is right to catch these.

## Caveats

- The raised `k_lower` exists for a documented reason: separating yolk-only SAM2 masks from real
  small embryos, which area alone cannot do (`surface_area_qc_pose_confound.md`). Relaxing it
  globally would re-admit whatever population motivated the change, and that population is not
  visible in these six plates. Narrower options: make the reference temperature-aware (fit p5/p95
  per rearing temperature instead of pooling), or make the tolerance per-experiment-class. Both are
  product decisions.
- `predicted_stage_hpf` silently ignores temperature for any single-frame experiment: the rate
  multiplies an elapsed time of zero, so the value collapses to nominal clock time. Correct
  arithmetic, misleading column name — and it affects anything downstream that reads it as a
  developmental stage for a snapshot plate, not just this QC check.
- The executed notebook is ~15 MB because the six montages are embedded. Strip outputs before
  committing if that is unwelcome; `figures/` already holds them at higher resolution.
