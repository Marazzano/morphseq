# Distribution catalog tidy-up

## Prioritize updates

The catalog API is a sound starting point and most of the construction,
comparison, pooling, and plotting paths are implemented. Before treating
`DISTRIBUTION_CATALOG_API.md` as a frozen public contract, complete the following
cleanup in priority order.

1. **Preserve pooled-coordinate provenance per sample.** When
   `pool_by("experiment")` removes a coordinate, retain each sample's original
   experiment value as a sample-aligned label. At present this only survives if
   callers also include the split coordinate in `label_columns`, so the promised
   sample-grain lineage is not automatic.
2. **Implement `DistributionComparisons.to_peak_dataframe()`.** The API document
   presents comparisons as reusable analysis structures, but the documented peak
   table surface is not implemented. Emit explicit held-constant coordinates,
   `across`, `across_value`, `distribution_id`, and `local_peak_id`; do not imply
   cross-distribution peak correspondence.
3. **Enforce catalog invariants at construction boundaries.** Validate that each
   distribution has the catalog's coordinate keys, that coordinate combinations
   and distribution IDs are unique, and that IDs agree with coordinates. Apply
   the same validation after `map_distributions()` and `pool_by()`.
4. **Bring the decided document in sync with the live API.** Replace stale
   `discover_modes()` examples with `detect_peaks()` and remove the earlier text
   describing pooling as future work now that `pool_by()` is implemented.
5. **Add one end-to-end regression test.** Cover
   `from_dataframe -> pool_by -> detect_peaks -> compare -> to_peak_dataframe`,
   including recovery of each pooled sample's experiment lineage.
6. **Disambiguate the two `DistributionComparison` concepts.** The engine catalog
   and core records packages use the same name for different structures. Rename
   one or establish an explicit namespace before expanding the public API.

The coordinate-versus-label distinction, contextual reference/target roles, and
the separation between population matching and peak matching should remain the
core design constraints.
