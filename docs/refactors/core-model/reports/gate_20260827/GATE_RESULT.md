# Image gate — legacy vs regenerated snips, `20250612_30hpf_ctrl_atf6`

**Run date:** 2026-08-27 · **Tree:** `core-model-refactor` post-`origin/main` merge (`19061cbf`)
**Harness:** `scripts/snip_legacy_diff.py` · **Per-snip data:** `legacy_vs_regenerated_30hpf.csv`

## Verdict

**Restoring 6.5 um/px and 75 um restores the geometry and the saturation. It does not reproduce
legacy pixels.** Zero of 90 comparable pairs fall within the JPEG noise floor.

This is the outcome `SNIP_IMAGE_REGRESSION_STATUS.md` anticipated: *"Treat exact legacy parity as an
acceptance question, not as an implication of restoring two defaults."* It is now measured rather
than assumed.

## What was run

No full pipeline rerun. The renderer was driven directly against existing upstream artifacts
(`frame_masks`, `frame_inventory`, `physical_embryo_registry`), writing to a fresh root
`pipeline/output_gate_20260827/`. Production output was read, never written.

- 96 wells, 96 succeeded, 0 failed.
- 97 snips, all `is_valid_snip = True`.
- Confirmed settings from the emitted inventory: `snip_micrometers_per_pixel = 6.5`,
  `target_um_per_px = 6.5`, `source_micrometers_per_pixel = 1.887`.

The merged renderer **requires** the geometry table; calling it without one fails with
`snip_transform_id ... not found in the snip transform table`. So the gate runs
`run_snip_geometry` per well first.

## Pairing

Legacy and regenerated snip ids differ: `..._A01_e00_t0000` vs `..._A01_e01_BF_t0000` — 0-based vs
1-based embryo index, plus a channel token. Pairing is therefore on `(experiment, well, timepoint)`,
never on embryo index, and only where exactly one embryo exists on each side.

| | count |
|---|---:|
| legacy snips | 93 |
| regenerated snips | 97 |
| **paired** | **91** |
| ambiguous (2 embryos regenerated, 1 legacy) | 2 — `E09`, `F12` |
| regenerated with no legacy counterpart | 6 |
| shape mismatches | 0 |
| orientation-flipped (excluded) | 1 / 91 (1.1%) |
| **compared** | **90** |

The 2 ambiguous wells are reported, not guessed at — the legacy side kept one embryo where the
pipeline now finds two, so any forced pairing would compare different animals.

## Result

Per-snip mean absolute difference, aligned pairs (n=90):

| | grey levels |
|---|---:|
| min | 4.62 |
| median | 8.19 |
| max | 25.49 |
| **within noise floor (<=3)** | **0 / 90** |

The JPEG confound is not the explanation. Legacy snips are JPEG, regenerated are PNG; the measured
q95 re-encode floor on this corpus is mean 0.123, p99 1, max 3 grey levels — two orders of magnitude
below the observed difference.

### What the difference is made of

| measure | result | reading |
|---|---|---|
| foreground pixel ratio (regen/legacy) | median **1.0026**, range 1.001–1.003 | **Geometry restored.** Embryos are the same size — the 6.5 um/px correction worked. |
| saturation fraction | legacy 0.0004 -> regen **0.0002** | **Blend radius restored.** The regression's 3.63x saturation excess is gone; regenerated is marginally *below* legacy. |
| mean intensity delta | median **+4.28**, range -0.00 to +11.41 | Systematic brightness offset, regenerated brighter. |
| mean abs diff after removing that offset | 9.62 -> **5.68** | ~40% of the difference is the global offset; ~5.7 grey levels is structural. |
| foreground-only mean abs diff | 9.59 vs 9.62 raw | The difference lives **inside the embryo**, not in the background halo. |

So the two headline regressions are genuinely fixed, and a third, previously undiagnosed difference
remains: a systematic brightness offset plus residual structure within the embryo.

## What the panels show

`panels/` holds four contact sheets: `panels_worst.png`, `panels_median.png`, `panels_best.png`,
`panels_flipped.png`. Each row is legacy | regenerated | |difference|, difference on its own colour
scale.

The visual signature is consistent across the whole range, including the **best** pairs:

- **Embryo position, pose, and size are visually identical.** No whole-object shift. This matches
  the 1.0026 foreground ratio — the geometry really is restored.
- **The background is clean.** Difference outside the embryo is essentially zero, so the halo/blend
  treatment is not the source.
- **The difference is inside the tissue and gradient-weighted** — diffuse across the body with
  sharp bright rims on internal boundaries (yolk edge, notochord, tail margin). Present even at
  mean 4.62, the closest pair in the set.

That is not the signature of a uniform brightness shift, and not of a misregistration. It is what
local contrast processing looks like when it redistributes intensity slightly differently.

**Hypothesis, not yet established:** CLAHE. It is adaptive, so any change in the crop window or the
mask feeding it redistributes intensities across the entire embryo while leaving position and
extent untouched — producing exactly this pattern. The background model and main's new
channel-intensity/exposure work are alternative candidates. **Confirming this requires an ablation
(render with `apply_clahe=False` on both sides), which has not been run.**

### One pair is a different problem

`F11` (mean 25.49, nearly double the next worst at 14.01) does not fit the pattern. Legacy and
regenerated show visibly different embryo content, not a reprocessed version of the same content.
That looks like a mask or embryo-selection difference rather than a rendering difference, and
should be triaged separately — averaging it in with the rendering question would mislead both.

## What this does not establish

- **Cause of the residual.** CLAHE, the background model, and main's new channel-intensity/exposure
  work are all candidates. Not investigated here.
- **Whether the residual matters to the model.** That is the model gate — encode both sets through
  `20241107_ds_sweep01_optimum` and compare embeddings. Not run.
- **Whether legacy is the right target.** For a model trained from scratch, exact legacy parity may
  not be required. That is a science decision, not a measurement.

## Reproduce

```bash
python scripts/snip_legacy_diff.py \
  --legacy-dir .../training_data/bf_embryo_snips/20250612_30hpf_ctrl_atf6 \
  --snips-root .../pipeline/output_gate_20260827/snips \
  --output-csv docs/refactors/core-model/reports/gate_20260827/legacy_vs_regenerated_30hpf.csv
```

The harness self-tests against known answers: identical inputs report 93/93 identical; 5 injected
180-degree flips and 3 injected +10 grey offsets are each recovered exactly.
