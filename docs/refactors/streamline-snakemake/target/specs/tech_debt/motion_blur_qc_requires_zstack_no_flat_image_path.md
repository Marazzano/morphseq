# Tech Debt: motion_blur_qc (and therefore snip_qc) hard-requires z-stacks — no graceful path when a scope/dataset can't produce them

**Status:** resolved 2026-07-12. Keyence now materializes `BF__z_stack`. For datasets without
z-stacks, DAG planning fails with an intentional explanation when `motion_blur_flag` is requested.
Users may explicitly remove that flag from `snip_qc.exclusion_flags`; the pipeline never removes a
QC flag automatically.

---

## The dependency chain

```
motion_blur_qc  ──HARD input──►  BF__z_stack frame_inventory shard (+ .validated sentinel)
snip_qc         ──aggregates──►  {death_detection, surface_area, mask_quality, focus, motion_blur} QC flags
analysis_ready  ──►  snip_qc
```

- `build_motion_blur_qc_for_well` (`rules/motion_blur_qc.smk`) declares the per-well
  `BF__z_stack` frame_inventory product + its `.validated` sentinel as rule `input:`. Motion blur is
  a **between-Z-slice** metric — it fundamentally needs the z planes. No z_stack product → the rule
  cannot build → `MissingInputException` at DAG execution.
- `snip_qc` does **not** hard-require motion_blur_qc: `resolve_snip_qc_flag_sources`
  (`quality_control/snip_qc/flag_input_resolver.py`) pulls only the QC sources whose flag columns
  appear in the requested `exclusion_flags`. But the default `SNIP_QC_EXCLUSION_FLAGS`
  (`quality_control/snip_qc/contract.py`) **includes `motion_blur_flag`**, so by default snip_qc
  *does* request the motion_blur_qc shard — which transitively requires z_stack.

## Who hits this

- **Keyence native is resolved.** Keyence materializes and validates per-Z mosaics from acquisition
  inventory rows, so its configured `BF__z_stack` product satisfies motion blur QC.
- **Any flat-image drop-in dataset.** The drop-in handoff contract deliberately has no
  materialization/z-stack step ("the user already has images"), so a flat-image drop-in has the same
  problem. The DAG dry-run verification confirmed drop-in `through_line` fails with exactly this
  `MissingInputException` on `{well}_BF__z_stack_frame_inventory.csv`.

## Implemented policy

When a scope/dataset cannot produce z-stacks, `motion_blur_flag` remains requested by default and
planning fails before execution with two explicit remedies: configure `BF__z_stack`, or remove
`motion_blur_flag` from `snip_qc.exclusion_flags`. Removing it also removes `motion_blur_qc` from the
resolved snip-QC source inputs through the existing flag resolver.

The requirement is declared in `snip_qc.flag_input_resolver` and checked against the composed
materialization product keys at DAG parse time. The error names the requested flag, the required
product, the configured products, and both supported remedies. There is no availability-based
implicit mutation of the exclusion policy.
