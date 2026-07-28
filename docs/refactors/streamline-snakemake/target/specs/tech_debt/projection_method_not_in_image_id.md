# Tech Debt: `projection_method` Not Encoded in `image_id`

**Status:** known gap, mdcolon 2026-06-26. Recorded as an honest tech debt entry — not a blocker
for current work, but named so the invariant is clear and the refactor path is understood before
it is needed.

---

## The gap

`image_id` is built from `(well_id, channel_id, time_index)`. It does NOT include
`projection_method`. That means two projection products — `focus_stack` and (future) `max` — that
share the same `(well, channel, time)` atoms produce the **same `image_id`**, even though they are
different files on disk.

On disk they are separated by the path (`projection/focus_stack/` vs `projection/max/`), and in
the `frame_inventory` they are separated by the `projection_method` column. But any downstream
consumer that identifies an image by `image_id` alone cannot tell which projection type it is
looking at.

---

## Why it was not fixed now

Adding `projection_method` to `image_id` would require:

- Changing `build_image_id` (the identity minting function)
- Updating every downstream consumer of `image_id` (segmentation, snip registry, snip_qc, feature
  extraction, tracking, analysis)
- Deciding how non-projection rows (`z_stack`) carry a null method in the id grammar

The blast radius is large. The MVP ships only `focus_stack` BF projection, so no collision is
possible yet. The debt was accepted to keep forward progress.

---

## The mitigation (enforced now)

Since the method is NOT in `image_id`, the `frame_inventory` contract must enforce that no two rows
share the same `image_id`. This acts as a "one product per projection type" gate:

> **If two projection methods ever produce the same `(well, channel, time)` tuple, the
> `frame_inventory` uniqueness check on `image_id` will fail loud at validate time.**

This is not silent. The collision surfaces immediately at the frame_inventory validation step. Do
NOT relax this uniqueness check to accommodate multiple projection methods — that is the moment to
pay down this debt instead.

---

## Future refactor paths (if the collision is ever real)

**Option A — Extended ID.** Add `projection_method` (or a short token like `fs`/`mx`) to
`build_image_id`. All downstream consumers that keyed on `image_id` gain the method in their
key automatically. Large but mechanical refactor.

**Option B — Enforce one product per projection type per `image_id`.** Keep `image_id` as-is but
add a contract rule: the pipeline may only ever land ONE projection method for a given
`(well, channel, time)` in any single experiment run. Multiple methods require separate runs
(or separate experiment namespaces). Simpler, but constrains operational flexibility.

The right option depends on whether multiple projection methods per acquisition are ever a real
need (scientific want) or a theoretical one. If they remain theoretical, Option B (the existing
uniqueness gate) is the debt payment deferred; if they become real, Option A is the honest fix.

---

## Where to look when paying this debt

- `image_id` grammar: `shared/identifiers/constructors.py` (`build_image_id`)
- Frame inventory contract: `image_materialization/stitched/contracts/frame_inventory_contract.py`
  (uniqueness check on `image_id`)
- Layout: `image_materialization/stitched/materialized_image_paths.py` (path constructor uses
  `projection_method` as a path level — already correct; the path is NOT ambiguous, only the ID is)
- All consumers of `image_id` as a key (segmentation, snip world, snip_qc, feature extraction)
