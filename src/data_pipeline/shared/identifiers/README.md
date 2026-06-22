# Shared Identifiers

This package owns canonical string grammar for shared pipeline identifiers. Callers should treat
identifier strings as opaque and use the constructors/parsers exported from
`data_pipeline.shared.identifiers`.

## Mask IDs

- Actual masks: `<image_id>_m####`
- No-mask placeholders: `<image_id>_mask_none`

`parse_mask_id(mask_id)` returns `(image_id, local_mask_index, is_no_mask)`. Placeholder rows return
`local_mask_index is None`.

## Track IDs

- Tracks: `<well_id>_track####`

Track indices are zero-based. `build_track_id("WELL", 0)` returns `WELL_track0000`.
