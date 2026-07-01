"""Cross-scope vocabularies for the MorphSeq pipeline.

This is NOT the old shared-schema bucket. Under the co-location doctrine (feature_world.md), every
product owns its own ``contract.py`` next to where it mints its artifact — the former
``REQUIRED_COLUMNS_*`` holders here were retired into their product folders.

What legitimately remains is ``channel_normalization`` — a canonical channel *vocabulary* (a
cross-scope language, not a per-product artifact schema). Do not re-add per-product column contracts
to this package.
"""
