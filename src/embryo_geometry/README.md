# embryo_geometry

Biological policy for embryo geometry. Currently: the shared orientation engine.

## Why this is a root-level package

Orientation is decided in two divergent places, and a third would be invented for the next consumer:

- **Pipeline** — `data_pipeline/object_extraction/snip_processing/rotation.py`: a skimage
  `regionprops` axis, then a 180° flip decided by a single scalar latch (`vert_ratio >= 0.5`).
- **Analysis** — `analyze/utils/coord/grids/canonical.py` (`CanonicalAligner`): a PCA axis, then a
  four-candidate enumeration, yolk-farthest-left for AP, then back-above-yolk for DV.

Both answer the same biological question — *which way is the fish facing?* — and have drifted. The
answer belongs to neither client, so it lives at the root, exactly like `image_geometry`:

```text
analyze ─────┐
             ├── embryo_geometry ── image_geometry
data_pipeline┘
```

`embryo_geometry` must not import `data_pipeline` or `analyze`. `test_layering_contract.py` asserts
it, and also asserts that no anatomical identifier or selection verb leaked *down* into
`image_geometry`.

## The border

Three layers, and the seams are the point:

| layer | owns | lives in |
|---|---|---|
| candidate **mechanics** | enumerate rot×flip, place on a grid, PCA axis | `image_geometry.candidates` |
| embryo **policy** | yolk-left AP, back-up DV, the fallbacks, the evidence | `embryo_geometry.orientation` |
| **rendering** | snip frame vs 256×576 canvas, µm/px, anchor, clipping | each caller |

A PCA major axis fixes an elongated object's pose only up to a 180° rotation, and mirroring is a
second free choice — four candidates. Enumerating them is generic raster work. *Choosing* among them
means knowing which end is the head, which is domain knowledge.

## There are THREE production rules, not one

This is the correction that shapes the whole design. It is tempting to say "the anatomical rule, with
a fallback" — but the fallback is not one thing:

| # | rule | who | when |
|---|---|---|---|
| 1 | yolk-left AP, then back-above-yolk DV | `CanonicalAligner` | yolk present |
| 2 | `-(com_y + com_x)`, mass toward upper-left | `CanonicalAligner` | yolk missing |
| 3 | `vert_ratio >= 0.5` | snip pipeline | yolk missing |

Rules 2 and 3 are **both** "the no-yolk policy" — of different callers — and they do not agree.
(The snip pipeline also has a *fourth* rule for the yolk-present case: yolk above the embryo
centroid. That is a different anatomical claim from rule 1, comparing different landmarks.)

So `no_yolk_policy` is a required, explicitly-chosen parameter from a closed set:

```python
orientation_policy(mask, yolk, no_yolk_policy="legacy_upper_left_com")  # analysis today
orientation_policy(mask, yolk, no_yolk_policy="legacy_vert_ratio")     # pipeline today
```

Each caller keeps its current behavior exactly. **Converging on one shared fallback then becomes a
later, attributable behavior change** rather than something that happens by accident inside an
extraction. A test (`test_the_two_no_yolk_policies_actually_disagree`) fails if they ever silently
become equivalent, since that would remove the reason for the parameter.

Fallback is provenance, never hidden control flow:

```
orientation_source = "yolk_back"
fallback_policy    = "legacy_vert_ratio"
fallback_reason    = "missing_yolk"        # vs "yolk_disabled" / "yolk_warped_off_grid"
```

Note that on experiment `20250416` every yolk mask is empty, so a fallback — *which* one depending on
the caller — is the entire path there. The anatomical rule is not exercised as widely as its
prominence suggests.

## The decision is canvas-independent

`EmbryoOrientationDecision` describes **orientation, not rendering**. It carries no grid shape, no
µm/px, no anchor, no clipping policy, so the pipeline can apply the same decision to its own snip
canvas. It reports the axis and the 0/180 choice separately:

```python
final = d.final_rotation_deg(target_angle_deg=0.0)   # or compose it yourself
```

`target_angle = 0 if is_landscape else 90` in `canonical.py` is a property of the **canvas aspect
ratio**, not of the embryo — that logic stays with the renderer.

(`candidate_shape_yx` is an *evaluation* frame, not a rendering target. It is required for the
candidate-based policies because rule 2 literally scores positions on a canvas; `legacy_vert_ratio`
needs no frame at all, since its rotation expands its own bounds.)

## Evidence is raw, and no confidence number is emitted

Every decision point was an unreported threshold latch: AP is an `argmin` with no margin requirement,
DV a strict sign test with no deadband, `vert_ratio >= 0.5` a hard latch on a continuous statistic.
Distance from the cliff was computed and thrown away. It is now returned: `ap_margin_px`,
`dv_margin_px`, `vert_ratio`, `vert_ratio_margin`, `back_support_pixels`, `back_estimation_status`,
`dv_override_ap_choice`.

This matters because orientation instability is real and pre-existing: embryo `20250416_D09` flips
180° **three times within one timelapse**, with `vert_ratio` of 0.4967 / 0.5068 / 0.4843 — three coin
flips reported as three confident decisions.

**No single confidence score is returned, deliberately.** The margins are not commensurable:
`ap_margin_px` is pixels of yolk-x separation, `dv_margin_px` is `|back_y − yolk_y|` in pixels, and
`vert_ratio_margin` is dimensionless. A 10 px AP margin is not "the same amount of evidence" as a
0.03 vert-ratio margin, and combining them needs calibration against embryo size, yolk radius, and
empirical error rates that nobody has done. A test asserts no `confidence` field reappears casually.

## Ambiguities in the current rules

Found while trying to state them precisely. All are preserved as-is and reported, not fixed:

1. **Step 2 can undo Step 1.** The vertical-flip partner mirrors horizontally too, so enforcing
   dorsal-up can move the yolk to the *right* — violating the rule Step 1 just applied. The two rules
   are not jointly satisfiable in general and DV silently wins. **Yolk-left is therefore not an
   invariant of the output, despite being stated as the rule.** `dv_override_ap_choice` reports it,
   and it fires on real synthetic cases (the AP test skips itself there, which is the finding).
2. **`back_y == yolk_y` exactly** — the back level with the yolk, no dorsal information — keeps Step
   1's answer rather than flagging the axis undetermined, because the test is `> 0`.
3. **The two yolk-present rules disagree in kind.** `CanonicalAligner` asks where the *back* is
   relative to the *yolk*; the pipeline asks where the *yolk* is relative to the *embryo centroid*.
   Different landmarks, so they can select opposite poses on the same animal.
4. **"Yolk present" is decided after warping**, per candidate. A yolk nonempty in source coordinates
   but warped off the grid degrades to the fallback with no error (`yolk_warped_off_grid`).
5. **The yolk landmark silently becomes the embryo centroid** when the yolk is unusable, so a
   coordinate alone cannot distinguish "yolk here" from "no yolk". Read `yolk_used`.
6. **`vert_ratio` counts occupied ROWS, not pixels** — it measures vertical extent, not mass, so a
   long thin tail counts as much as a bulky head.
7. **The two axis measurements are different quantities.** skimage's `regionprops.orientation`
   measures from the vertical; the PCA helper measures from the horizontal via `arctan2`, giving
   `skimage ≈ −(pca + 90)` mod 180. `legacy_vert_ratio` therefore keeps skimage's own measurement
   rather than deriving it, or equivalence would break. `axis_convention` records which was used.
8. **Empty masks return the evaluation-grid center** as a centroid — a shape-dependent sentinel
   indistinguishable from a real measurement. The caller now passes it in explicitly.

## Status

An **extraction with evidence reporting, not a policy change**. The equivalence gate is **two-sided**
and both halves are load-bearing (`test_orientation_equivalence.py`):

- side A — matches `CanonicalAligner._coarse_candidate_select`, on rotation, flip, *and* landmarks;
- side B — matches `get_embryo_rotation_angle`, including a sweep of near-latch cases
  (`|vert_ratio − 0.5| ≤ 0.05`), the regime where the legacy code actually flips.

Neither caller is wired to this engine yet — that is a separate reviewed step, and
`test_callers_are_not_yet_rewired` will fail when it happens, deliberately.

`canonical.py` also carries two placement defects (naive `w_out/2` centering with no half-pixel
correction, and a large downscale fused into `warpAffine` where it cannot anti-alias). Those are
tracked separately and are **deliberately reproduced** in `centered_placement_affine` so the
equivalence proof holds. Do not "fix" them here without re-running it.

## Tests

`tests/embryo_geometry/` — which must **not** contain an `__init__.py`, or it would shadow the real
package and break every import in it. `tests/image_geometry/` follows the same rule; a test asserts
both.
