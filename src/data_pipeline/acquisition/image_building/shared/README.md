# Focus-stack image processing

This directory owns focus-projection **image math** shared by all microscope
adapters. Keyence and YX1 code should own raw-data access, channel/Z selection,
well/time grouping, and (for Keyence) tile composition. They should not implement
their own contrast normalization or LoG-to-uint8 conversion.

## Why this boundary exists

A 2026-07 audit found that Keyence focus projections had drifted from legacy
Build01 behavior:

- legacy Keyence used one 0.1--99.9% intensity range shared by every tile in a
  well/time point;
- the new Keyence materializer normalized each tile independently;
- two Keyence call sites normalized once before calling a YX1 projection helper,
  which normalized a second time;
- the LoG implementation itself remained byte-identical to the legacy stacker.

Independent normalization produced inconsistent tile contrast and changed the
selected Z plane at roughly 2--3% of pixels in the audited frame. The redundant
second normalization contributed very little, but made the ownership error easy
to miss. YX1 did not double-normalize, although it still coupled focus scoring to
the displayed normalized pixels.

The design below makes these decisions explicit and testable.

## Processing contract

For one logical output frame, callers provide a **stack group**:

- YX1: one raw `(Z, Y, X)` uint16 stack;
- Keyence: all raw tile stacks for one `(well, channel, time)` frame, each
  `(Z, Y, X)` uint16 and ordered consistently with its `z_indices`.

The shared implementation must perform these steps exactly once:

1. Compute deterministic intensity bounds across the complete stack group.
   Use an exact 65,536-bin uint16 histogram and the 0.1 and 99.9 percentiles.
   Do not use random or strided sampling.
2. Build a float32 LoG scoring tensor from the raw values and shared bounds.
   The conservative migration mode clips scores to `[0, 1]` because this best
   matches the visually validated legacy result. An unclipped affine scoring mode
   may remain available for evaluation, but must not silently become the default.
3. Run the shared LoG stacker once and derive an integer stack-axis focus-index
   map with `argmax` over Z.
4. Gather focused pixel values from the **raw uint16 stack**, not from the scoring
   tensor. This separates focus selection from rendering.
5. Apply one shared display transform to the focused raw pixels using the same
   bounds, clip to `[0, 1]`, and convert to uint8.
6. Return the uint8 focused tiles, focus-index maps, and intensity/algorithm
   provenance. Keyence may then stitch the returned tiles; YX1 writes its single
   returned image.

Invariant: changing display encoding must not change the focus-index map.

## Proposed shared API

Keep the low-level `LoG_focus_stacker` private to this layer. Expose a higher-level
group operation with explicit configuration and structured results, conceptually:

```python
@dataclass(frozen=True)
class FocusStackConfig:
    low_percentile: float = 0.1
    high_percentile: float = 99.9
    filter_size: int = 3
    scoring_mode: Literal["shared_clipped", "shared_unclipped"] = "shared_clipped"
    algorithm_version: str = "shared_raw_gather_v1"


@dataclass(frozen=True)
class FocusStackResult:
    projection_u8: np.ndarray
    focus_index_map: np.ndarray


@dataclass(frozen=True)
class FocusStackGroupResult:
    tiles: tuple[FocusStackResult, ...]
    intensity_lo: int
    intensity_hi: int
    config: FocusStackConfig


focus_stack_group(
    stacks_zyx: Sequence[np.ndarray],
    *,
    config: FocusStackConfig,
    device: str,
) -> FocusStackGroupResult
```

The real implementation may stream histogram construction and process tiles in
GPU-sized batches. Batching must not change bounds or output values.

## Provenance

Every emitted focus projection already carries a `focus_index_map` NPZ. Extend
that construction-provenance artifact (or a colocated versioned sidecar) with:

- `intensity_lo` and `intensity_hi`;
- percentile values and bound method (`exact_uint16_histogram`);
- scoring mode;
- filter size;
- algorithm version.

This is construction provenance, not a new primary image product. Existing
`focus_index_map` and `z_indices` meanings must remain unchanged.

## Migration plan

### Phase 1 — shared primitive and unit tests

1. Add exact uint16 histogram-bound calculation.
2. Add raw-pixel gathering from a focus-index map.
3. Add shared display mapping and the structured group API.
4. Test:
   - exact/deterministic percentile bounds;
   - one bound pair is used for every tile in a group;
   - input stacks are not mutated;
   - focus indices are stack-axis offsets with the expected shape/dtype;
   - projection pixels come from raw data before display mapping;
   - display-only changes leave focus indices unchanged;
   - batching produces identical bytes;
   - CPU results remain equivalent to the existing LoG kernel implementation.

### Phase 2 — Keyence candidate route

1. Replace per-tile `im_rescale` and the imported YX1 projection helper in
   `materialize_well_keyence.py` with one `focus_stack_group` call per frame.
2. Use the same shared route in `build_keyence_stitch_map.py`.
3. Keep tile ordering and stitch geometry in the Keyence adapter.
4. Initially write candidate outputs without overwriting production images.
5. Validate A02 from `20260702_hotchem_30hpf_plate01` against the existing
   legacy/shared-clean montage, then review a cohort covering 2-, 3-, and 6-tile
   frames, chamber edges, debris, and high-contrast embryos.

Acceptance gates:

- no tile-specific contrast normalization;
- no repeated normalization call;
- candidate output has the expected geometry and dtype;
- focus-index provenance validates;
- visual review shows no washed-out blocks or new stitch seams;
- metrics and image differences are recorded, but MAE alone is not a sharpness
  criterion.

### Phase 3 — YX1 candidate route

1. Replace `materialize_ff_projection` internals with the group API called on a
   one-stack group.
2. Compare representative YX1 frames across acquisition ages and intensity
   ranges against the current and legacy outputs.
3. Confirm that results do not depend on GPU batch membership (legacy Build01
   could derive bounds across multiple batched frames).
4. Preserve identity composition: YX1 must not acquire Keyence tiling concerns.

Acceptance gates:

- one deterministic bound pair per YX1 well/time stack;
- raw-pixel gathering and one display transform;
- focus-map Z offsets remain valid against inventory `z_indices`;
- candidate frames are at least as sharp under image review, without new clipping
  artifacts.

### Phase 4 — switch defaults and remove duplicate routes

1. Flip Keyence and YX1 materialization to `shared_raw_gather_v1` only after both
   candidate gates pass.
2. Regenerate affected frame inventories/provenance when images are regenerated;
   never mix a new image with an old focus-index sidecar.
3. Route or retire the compatibility modules under
   `metadata_ingest/stitched_index/` and `image_building/scope/*` so there is one
   production owner for focus projection math.
4. Remove adapter imports of `im_rescale`; keep any low-level compatibility
   helper private and clearly deprecated until legacy consumers are gone.
5. Add a repository search/test gate ensuring microscope adapters do not call
   `im_rescale` or `LoG_focus_stacker` directly.

## Non-goals

- Tile alignment, seam smoothing, orientation, cropping, and image-product write
  policy remain outside this shared focus-math layer.
- Fluorescence max projection is a separate projection method.
- Percentile choices must not be changed opportunistically during the migration;
  evaluate alternative tone curves as versioned algorithms with candidate images.
