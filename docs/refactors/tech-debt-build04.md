**Build04 Technical Debt (2025-09-14)**

**Context**
- The per‑experiment Build04 path (`build04_stage_per_experiment`) is the intended production workflow.
- Legacy combined workflow (`perform_embryo_qc`) remains in code and diverges from the production logic.
- Recent changes restored legacy‑consistent QC while making behavior explicit and debuggable.

**Current Behavior**
- Stage inference: honors `stage_ref_df.csv` path and fails loudly when missing.
- SA QC: internal controls first; stage_ref fallback when controls unavailable.
  - Internal controls: `(phenotype == 'wt' OR control_flag) AND use_embryo_flag`.
  - Time axis: `predicted_stage_hpf` (avoids SA→stage circularity; matches legacy parity).
  - Binning: 0–72 hpf, 0.5 hpf step; window ±0.75 hpf; Savitzky–Golay smoothing with safe odd/clamped window.
  - Fallback: `threshold(stage) = scale × margin_k × sa_ref(stage)` using robust per‑experiment scale; logs when fallback used.
- Death lead‑time QC: uses `predicted_stage_hpf` (legacy parity).
- Logging: prints concise summary including percent rows and percent snip_ids flagged by SA QC.

**Technical Debt Items**
- Single Source of Truth
  - Action: Move legacy workflow (`perform_embryo_qc`) to `src/build/build04_legacy.py` with a deprecation notice; keep `build04_stage_per_experiment(...)` as the only production path.

- Parameterize Magic Values / Remove Hardcoded Exceptions
  - Replace date/well special cases with configuration or documented, data‑driven fallbacks.
  - Expose only high‑value knobs via config (YAML/JSON):
    - `sa_qc.percentile`, `sa_qc.hpf_window`, `sa_qc.bin_step`, `sa_qc.sg_window`, `sa_qc.sg_poly`,
    - `sa_qc.fallback.margin_k`, `sa_qc.fallback.calibrate_scale`.
  - Keep domain‑stable defaults in code to avoid CLI sprawl.

- Stage Inference Variants
  - Remove or archive `infer_embryo_stage_orig` and `infer_embryo_stage_sigmoid` unless test‑referenced; maintain a single `infer_embryo_stage` that honors `stage_ref`.

- CLI/Docs Alignment
  - Align output handling: prefer `--out-dir` + fixed filename or re‑introduce `out_csv` in the library and honor it.
  - Ensure discovery paths are consistent everywhere: `metadata/build03_output/` and `metadata/build04_output/`.

- Tests and Logging
  - Add small synthetic tests for:
    - Internal‑control SA QC path
    - Stage_ref fallback path
    - Strict input validation
    - Summary logging (percent rows and snip_ids flagged)
  - Keep logs concise, deterministic, and helpful for post‑hoc analysis.

**Notes**
- For now, the legacy functions remain to avoid disruption; use the per‑experiment path for new work.
- When ready, migrate callers off the legacy workflow, then delete it after parity validation.

