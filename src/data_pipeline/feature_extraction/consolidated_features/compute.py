"""consolidated_features compute — merge validated feature tables one-to-one on snip_id.

Pure merge logic: no path knowledge, no file IO. Consumes a list of already-loaded feature
DataFrames and joins them on snip_id, failing loud on duplicate keys or column collisions (the
spine columns are the shared, expected overlap and are de-duplicated, not treated as collisions).
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.shared.feature_table_utils import SNIP_FEATURE_TABLE_SPINE_COLUMNS

_SPINE = set(SNIP_FEATURE_TABLE_SPINE_COLUMNS)


def assert_feature_table_compatible(df: pd.DataFrame, *, key: str, feature_name: str) -> None:
    """Fail loud unless ``df`` has a unique, non-null ``key`` column."""
    if key not in df.columns:
        raise ValueError(f"consolidated_features: table {feature_name!r} is missing join key {key!r}.")
    if df[key].isna().any():
        raise ValueError(f"consolidated_features: table {feature_name!r} has null {key!r}.")
    if df[key].duplicated().any():
        dupes = df.loc[df[key].duplicated(keep=False), key].head(5).tolist()
        raise ValueError(
            f"consolidated_features: table {feature_name!r} has duplicate {key!r}: {dupes}."
        )


def consolidate_feature_tables(
    feature_tables: dict[str, pd.DataFrame],
    *,
    key: str = "snip_id",
) -> pd.DataFrame:
    """Merge named feature tables one-to-one on ``key``.

    ``feature_tables`` maps product name -> its DataFrame. The first table seeds the spine; each
    subsequent table contributes only its NON-spine feature columns (the spine is the expected
    shared overlap, de-duplicated). A non-spine column appearing in two tables is a collision and
    fails loud.
    """
    if not feature_tables:
        raise ValueError("consolidated_features: no feature tables provided to merge.")

    names = list(feature_tables)
    base_name = names[0]
    merged = feature_tables[base_name].copy()
    assert_feature_table_compatible(merged, key=key, feature_name=base_name)

    seen_feature_columns: set[str] = {c for c in merged.columns if c not in _SPINE and c != key}

    for name in names[1:]:
        table = feature_tables[name]
        assert_feature_table_compatible(table, key=key, feature_name=name)

        new_feature_cols = [c for c in table.columns if c not in _SPINE and c != key]
        collisions = [c for c in new_feature_cols if c in seen_feature_columns]
        if collisions:
            raise ValueError(
                f"consolidated_features: column collision merging {name!r}: {collisions}. "
                "Two products must not emit the same feature column."
            )
        seen_feature_columns.update(new_feature_cols)

        merged = merged.merge(table[[key, *new_feature_cols]], on=key, how="left", validate="1:1")

    return merged
