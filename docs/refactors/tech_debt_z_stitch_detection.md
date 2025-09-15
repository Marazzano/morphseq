# Tech Debt: Z-Stitch Detection and Surfacing

Context
- We show per‑experiment image generation status in CLI/UX. “FF” reflects stitched FF images under `built_image_data/stitched_FF_images/{exp}` and is used consistently.
- Keyence workflows can also produce a stitched Z stack (“Z stitch”), while YX1 does not follow the same convention.

Current MVP Behavior
- We expose a best‑effort indicator named `Z_STITCH` in status and planners.
  - It checks for the presence of `stitch_z_path` (Keyence). If present → `Z_STITCH✅`, otherwise `Z_STITCH❌`.
  - There is no attempt to infer complex microscope‑specific Z workflows for YX1.
- We intentionally avoid deeper detection logic that could be expensive or brittle across microscopes.

Tradeoffs
- Pros: Simple, fast, predictable; keeps UX consistent during refactor.
- Cons: Might show `Z_STITCH❌` for valid YX1 experiments that were never intended to produce Z stitch outputs.

Current Decision (2025‑09‑14)
- Z‑stitch is not required for SAM2. The orchestrated `pipeline e2e` no longer triggers Z‑stitch automatically and omits it from the per‑run step display to reduce confusion.
- The `status` command may still show `Z_STITCH` as a presence indicator, but it should be treated as informational only.

Future Work
- Define per‑microscope expectations for Z stitching and codify them in `paths.py` + `Experiment` accessors.
- Add explicit metadata to experiments indicating whether Z stitch is applicable.
- Consider a three‑state indicator: `N/A` (not applicable), `✅` (present), `❌` (missing when expected).

Scope & Rationale
- Out of MVP scope to implement microscope‑aware Z stitch semantics across legacy and new flows.
- The current indicator is sufficient for visibility without blocking orchestration.
