# Tech Debt: motion_blur_qc (and therefore snip_qc) hard-requires z-stacks — no graceful path when a scope/dataset can't produce them

**Status:** open design decision, mdcolon 2026-07-10. Today it **fails loud**, which is arguably
correct — but the failure mode is a mid-DAG `MissingInputException`, not an intentional, explained
error, and there is no supported way to run the pipeline to `snip_qc` / `analysis_ready` on a
dataset that legitimately has no z-stacks. Decide the intended behavior.

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

- **Keyence native.** Keyence z_stack materialization is not wired
  (`materialize_well_keyence.py` raises `NotImplementedError`; see
  `../front_end/...` and the "wire Keyence z_stack" follow-up). Keyence configs therefore request
  projection only. A full Keyence run on the `all` target will fail at `motion_blur_qc` for every
  well (no `BF__z_stack` shard), and — because default `exclusion_flags` includes `motion_blur_flag`
  — snip_qc for those wells is also blocked. **This is expected today** and is why the Keyence full
  run (job 22211226, 2026-07-10) does not cleanly reach snip_qc.
- **Any flat-image drop-in dataset.** The drop-in handoff contract deliberately has no
  materialization/z-stack step ("the user already has images"), so a flat-image drop-in has the same
  problem. The DAG dry-run verification confirmed drop-in `through_line` fails with exactly this
  `MissingInputException` on `{well}_BF__z_stack_frame_inventory.csv`.

## The decision to make

When a scope/dataset **cannot** produce z-stacks, what should happen to motion_blur_qc and to
snip_qc?

Candidate answers (pick and specify):

1. **Fail loud, but intentionally.** Detect at DAG-planning time that `motion_blur_flag` is
   requested while `BF__z_stack` is **not** in the configured product set, and raise a clear message
   — e.g. *"snip_qc requested `motion_blur_flag`, but no `BF__z_stack` product is configured, so
   motion_blur_qc cannot be produced. Remove `motion_blur_flag` from `exclusion_flags`, or add the
   `BF__z_stack` product."* This replaces a confusing mid-run `MissingInputException` with an
   explained, early failure. (Least behavior change; keeps the honest failure mdcolon wants.)

2. **Config-gate motion_blur_qc on z_stack presence.** Make `motion_blur_qc` (and its
   `motion_blur_flag` contribution to snip_qc) **conditional** on `BF__z_stack` being in the product
   set. Absent z_stack → motion_blur_qc is not in the DAG and snip_qc aggregates the remaining four
   sources. snip_qc / analysis_ready then complete on flat-image / Keyence datasets. (The
   "graceful" path; more wiring — target chain + resolver + `exclusion_flags` derivation.)

3. **Implement Keyence z_stack** so the question is moot for Keyence (does not help flat-image
   drop-in).

## Recommendation (not yet implemented)

Option 2 is the structurally correct fix ("z-stack-dependent QC is conditional on the scope actually
producing z-stacks"), and it generalizes to every flat-image drop-in. Option 1 is a cheap
intermediate that at least turns the confusing `MissingInputException` into an explained error.
mdcolon (2026-07-10) chose to **defer** building either for now and let the run fail — recording the
decision here. Revisit before promising a Keyence or flat-image drop-in run to `snip_qc` /
`analysis_ready`.
