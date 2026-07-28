# Materialized Image Write Policy (TARGET)

**Status:** target spec, mdcolon 2026-06-30. This locks the MVP write-policy shape for materialized
image products, especially `BF__z_stack`, where full-resolution PNG storage would explode the output
budget.

**Companion to:** `z_stack_materialization_wire_through.md`, `frame_inventory_handoff_contract.md`,
and `quality_control/z_stack_focus_motion_blur_qc_and_slice_selection.md`.

---

## Decision

Materialized image products use a **per-product write policy**. Product selection stays in
`image_materialization.products`; encoding/writing details live in the sibling map
`image_materialization.write_policies`, keyed by canonical product key.

Policy says format, scale, dtype, and optional final presentation orientation. The shared writer
handles the route. `frame_inventory` records the derived bytes that actually landed on disk, and
downstream consumers read the recorded `image_path`.

Migration priority: make native materialization correct before the full frame-inventory rename.
Keyence stitching must render from canonical tile coordinates first; the writer may then orient the
final array as `horizontal` or `vertical`. The writer must not force a stitched image into a legacy
canvas shape and must not require fake `source_` dimensions in the core frame inventory.
YX1 must use the same writer boundary: record ND2 raw provenance separately, apply write policy to
the constructed image array, write it, and populate core frame-inventory fields from the actual
written file.

---

## Config Shape

```yaml
image_materialization:
  products:
    - channel_id: BF
      image_product_type: projection
      projection_method: focus_stack
    - channel_id: BF
      image_product_type: z_stack

  write_policies:
    BF__z_stack:
      file_format: jpg
      jpeg_quality: 85
      downsample_factor: 4
      downsample_method: area_resize
      pixel_dtype: uint8
```

`products` answers **which products** are built. `write_policies` answers **how active products are
encoded/written**.

---

## Policy Shape

Live module:

```text
src/data_pipeline/image_materialization/materialized_image_write_policy.py
```

Six-field dataclass:

```python
@dataclass(frozen=True)
class ImageWritePolicy:
    file_format: Literal["png", "jpg", "tif"]
    downsample_factor: int
    downsample_method: Literal["none", "block_mean", "area_resize"]
    pixel_dtype: Literal["uint8", "uint16"]
    orientation: Literal["none", "horizontal", "vertical"] = "none"
    jpeg_quality: int | None = None
```

Defaults:

```yaml
default:
  file_format: png
  downsample_factor: 1
  downsample_method: none
  pixel_dtype: uint8
  orientation: none
  jpeg_quality: null

BF__z_stack:
  file_format: jpg
  jpeg_quality: 85
  downsample_factor: 4
  downsample_method: area_resize
  pixel_dtype: uint8
  orientation: none

BF__projection__focus_stack:
  file_format: png
  downsample_factor: 1
  downsample_method: none
  pixel_dtype: uint8
  orientation: none
  jpeg_quality: null
```

Rules:

- `jpg` requires `pixel_dtype: uint8` and non-null `jpeg_quality`.
- `png` and `tif` require `jpeg_quality: null`.
- `downsample_method: none` or factor `1` is identity.
- `downsample_method: block_mean` requires native width/height divisible by `downsample_factor`.
- `downsample_method: area_resize` uses the shared image-resize seam for non-divisible native
  dimensions; the target dimensions are `round(source / downsample_factor)`.
- Write order is downsample, then fixed dtype conversion, then route-specific write.
- Orientation is the only presentation transform. `none` writes the canonical image as produced;
  `horizontal` rotates only when needed so the written image is wider than tall; `vertical` rotates
  only when needed so the written image is taller than wide.
- Keyence stitch geometry is not writer policy. Stitching produces a canonical mosaic from tile
  coordinates; writer orientation acts only on that final array before disk write.
- `uint16 -> uint8` conversion is fixed full-range conversion, with no per-image or per-plane
  min/max normalization.
- `pixel_dtype` means encoder/read-back dtype, not raw acquisition dtype.

---

## Frame Inventory Contract

Core frame-inventory image columns record on-disk reality. Raw/source acquisition facts are optional
provenance columns outside the core frame contract.

Required core image/write columns:

```text
image_path
image_width_px
image_height_px
image_micrometers_per_pixel
image_file_format
pixel_dtype
downsample_factor
downsample_method
```

`jpeg_quality` is format-conditional: required only for JPEG rows, null or absent otherwise.

Optional raw provenance examples:

```text
Keyence: raw_tile_path/raw_tile_manifest_path, raw_tile_width_px, raw_tile_height_px,
         raw_tile_count, raw_micrometers_per_pixel
YX1:     raw_image_source_path, raw_image_width_px, raw_image_height_px,
         raw_micrometers_per_pixel
```

L4 validation:

- `image_path` suffix must match `image_file_format`.
- JPEG rows require non-null `jpeg_quality`.
- Non-JPEG rows require null `jpeg_quality`.
- `image_width_px` / `image_height_px` must match the file header.
- `image_width_px` / `image_height_px` are computed from the final written array after writer
  orientation/downsample/dtype conversion, then self-checked against the file header.
- `downsample_factor >= 1`.

---

## Storage Rationale

Real probes on the 2025 experiments showed:

```text
PNG z-stack planes:       ~619 GB across 20250305 + 20251125 + 20260206
JPEG q90 z-stack planes:  ~140 GB
JPEG q85 z-stack planes:   ~94 GB
JPEG q75 z-stack planes:   ~56 GB
```

The `BF__z_stack` default also downscales by factor 4, reducing pixel count about 16x before JPEG
encoding. YX1 smoke frames are `2189x2189`, so the default method is `area_resize`, not strict
`block_mean`; the written ds4 shape is `547x547`. That is the real storage lever for keeping z-stack
persistence practical while preserving the full field of view.

---

## Downstream Consumer Rule

`motion_blur_qc` and later consumers query `frame_inventory` for `BF__z_stack` rows and read the
recorded `image_path`. During migration, code may fall back to `source_image_path` for old shards,
but new shards should author `image_path`. Consumers must not reconstruct paths or infer encoding
from config.

At `downsample_factor != 1`, mask-to-z-plane alignment belongs in the consumer: use nearest-neighbor
mask resampling onto the validated loaded z-plane dimensions.
