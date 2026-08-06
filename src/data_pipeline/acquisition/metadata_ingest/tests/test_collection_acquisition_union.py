"""Collection ACQ UNION — N one-read-per-source inventories → ONE unioned inventory.

Proves Merge-A (identity, not pixels): N sources of one plate become ONE experiment_id with a
shared well_id, distinct per-source time_index blocks, correct per-source start_age_hpf, and a
per-well ``n_sources`` merge count. Per-frame source LABELS (source_child / source_scope /
source_time_index) are deliberately NOT emitted — only the count survives the seam. Sources are read
via an INJECTED reader returning tiny synthetic frames — no real ND2/TIFF is opened, and each source
is asserted to be read exactly once.

Run with:
    PYTHONPATH=src pytest \
      src/data_pipeline/acquisition/metadata_ingest/tests/test_collection_acquisition_union.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.collection_acquisition_union import (
    PlateSource,
    union_collection_acquisition_inventories,
)
from data_pipeline.acquisition.metadata_ingest.scope.yx1.acquisition_inventory import (
    YX1_ACQUISITION_INVENTORY_COLUMNS,
    validate_yx1_acquisition_inventory,
)

_COLLECTION = "cilia_snapshots_coll"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Synthetic per-source inventory builders (mock the one-read-per-source scope read).
# ─────────────────────────────────────────────────────────────────────────────────────────────


def _yx1_like_source_df(*, experiment_id: str, source_tag: str, n_t: int = 1) -> pd.DataFrame:
    """A tiny YX1-shaped acquisition inventory: NO well_id at ingest (attached later), 2 positions.

    Carries the full YX1 core+Tier-2 schema so the UNIONED result can be re-validated against the
    real YX1 validator (proving the union preserves the acquisition contract).
    """
    rows = []
    for position_index in (0, 1):
        for time_index in range(n_t):
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "raw_position_label": str(position_index),
                    "position_index": position_index,
                    "z_index": 0,
                    "channel_index": 0,
                    "channel_id": "BF",
                    "raw_channel_name": "Brightfield",
                    "time_index": time_index,
                    "acquisition_time_s": 1000.0 + time_index * 1800.0,
                    "elapsed_time_s": float(time_index * 1800.0),
                    "x_um": 10.0,
                    "y_um": 20.0,
                    "micrometers_per_pixel": 3.25,
                    "image_width_px": 640,
                    "image_height_px": 480,
                    "objective_magnification": "4x",
                    "microscope_id": "YX1",
                    "n_z": 1,
                    "source_nd2_path": f"/raw/{source_tag}.nd2",
                }
            )
    df = pd.DataFrame(rows)
    return df.reindex(columns=list(YX1_ACQUISITION_INVENTORY_COLUMNS))


def _keyence_like_source_df(*, experiment_id: str, source_tag: str) -> pd.DataFrame:
    """A tiny Keyence-shaped inventory: well_id BAKED at ingest off the per-source experiment_id.

    Only the columns the union touches/needs (well_index, well_id, time_index, provenance) — enough
    to prove well_id rebuild + provenance preservation without the full Keyence schema.
    """
    from data_pipeline.shared.identifiers.constructors import build_well_id

    rows = []
    for well_index in ("A01", "B02"):
        rows.append(
            {
                "experiment_id": experiment_id,
                "position_index": 0,
                "well_index": well_index,
                # Baked off the PER-SOURCE experiment_id — the union must rebuild this.
                "well_id": build_well_id(experiment_id, well_index),
                "time_index": 0,
                "channel_id": "BF",
                "microscope_id": "Keyence",
                "source_tiff_path": f"/raw/{source_tag}/{well_index}.tif",
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────────────────────


def test_two_snapshot_sources_become_one_inventory_shared_well_distinct_time():
    """t45 + t72 snapshots of plate01 → ONE experiment_id, shared well_id, distinct time_index."""
    read_calls: list[str] = []

    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        read_calls.append(source.source_id)
        return _keyence_like_source_df(experiment_id=experiment_id, source_tag=source.source_id)

    sources = [
        PlateSource("20260608_plate01_t72hpf", scope="Keyence"),  # deliberately out of age order
        PlateSource("20260607_plate01_t45hpf", scope="Keyence"),
    ]
    out = union_collection_acquisition_inventories(
        collection_name=_COLLECTION, sources=sources, read_source=read_source
    )

    # ONE experiment_id = {coll}_{plate}, no date / no event_label in identity.
    assert set(out["experiment_id"].unique()) == {"cilia_snapshots_coll_plate01"}

    # Each source read EXACTLY once.
    assert sorted(read_calls) == ["20260607_plate01_t45hpf", "20260608_plate01_t72hpf"]

    # well_id shared across the two sources (A01@t45 and A01@t72 are the same well).
    a01 = out[out["well_index"] == "A01"]
    assert set(a01["well_id"].unique()) == {"cilia_snapshots_coll_plate01_A01"}
    assert len(a01) == 2  # one row per source

    # Distinct time_index per source: t45 (declared 45) is block 0, t72 (declared 72) is block 1.
    # No source LABEL survives — select the source's rows by its declared age / time block instead.
    t45 = out[out["start_age_hpf"] == 45]
    t72 = out[out["start_age_hpf"] == 72]
    assert set(t45["time_index"].unique()) == {0}
    assert set(t72["time_index"].unique()) == {1}
    assert set(t45["time_index"]).isdisjoint(set(t72["time_index"]))

    # The only merge fact that survives the seam: n_sources == 2 on EVERY row (per-well constant).
    assert set(out["n_sources"].unique()) == {2}

    # Per-frame source LABELS are GONE (they existed only to feed the removed physical_embryo bridge).
    for gone in ("source_child", "source_scope", "source_time_index"):
        assert gone not in out.columns


def test_time_index_claimed_atom_is_offset_in_lockstep_with_time_index():
    """The raw ``time_index_claimed`` atom rides the SAME block offset as ``time_index``.

    Regression (found on real Keyence data): both single-snapshot sources carry
    ``time_index_claimed == 0``; the acquisition-inventory cell-key uniqueness check includes that
    atom, so if the union offsets only ``time_index`` and not ``time_index_claimed`` the two
    timepoints of a well COLLIDE on the cell key. The atom must be re-namespaced per source.
    """
    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        df = _keyence_like_source_df(experiment_id=experiment_id, source_tag=source.source_id)
        df["time_index_claimed"] = 0  # every single-snapshot source claims 0 (the collision setup)
        return df

    sources = [
        PlateSource("20260607_plate01_t45hpf", scope="Keyence"),
        PlateSource("20260608_plate01_t72hpf", scope="Keyence"),
    ]
    out = union_collection_acquisition_inventories(
        collection_name=_COLLECTION, sources=sources, read_source=read_source
    )

    # The atom is offset to match its block: block 0 → 0, block 1 → 1 (not both 0).
    assert set(out["time_index_claimed"].unique()) == {0, 1}
    # And it tracks time_index exactly (same offset applied), so the cell key stays unique.
    assert (out["time_index_claimed"] == out["time_index"]).all()


def test_start_age_hpf_is_per_source_from_declared_hpf():
    """start_age_hpf = parse_declared_hpf(event_label) per source (the age escape hatch)."""

    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        return _keyence_like_source_df(experiment_id=experiment_id, source_tag=source.source_id)

    sources = [
        PlateSource("20260607_plate01_t45hpf", scope="Keyence"),
        PlateSource("20260608_plate01_t72hpf", scope="Keyence"),
    ]
    out = union_collection_acquisition_inventories(
        collection_name=_COLLECTION, sources=sources, read_source=read_source
    )

    # No source LABEL survives — the age escape hatch is keyed by the time block, so group by it.
    ages = out.groupby("time_index")["start_age_hpf"].unique()
    assert list(ages[0]) == [45]  # block 0 = t45 source
    assert list(ages[1]) == [72]  # block 1 = t72 source


def test_no_per_frame_source_labels_but_scope_audit_path_survives():
    """Per-frame source LABELS are dropped; each scope's own source_*_path audit column rides through."""

    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        return _keyence_like_source_df(experiment_id=experiment_id, source_tag=source.source_id)

    sources = [
        PlateSource("20260607_plate01_t45hpf", scope="Keyence"),
        PlateSource("20260608_plate01_t72hpf", scope="Keyence"),
    ]
    out = union_collection_acquisition_inventories(
        collection_name=_COLLECTION, sources=sources, read_source=read_source
    )

    # The removed per-frame source labels (fed the now-removed physical_embryo bridge).
    for gone in ("source_child", "source_scope", "source_time_index"):
        assert gone not in out.columns

    # The scope's OWN raw-path audit column is NOT a merge source label — it rides through untouched.
    assert "source_tiff_path" in out.columns
    t45_paths = out.loc[out["start_age_hpf"] == 45, "source_tiff_path"]
    assert all("20260607_plate01_t45hpf" in p for p in t45_paths)


def test_timelapse_source_keeps_contiguous_time_block():
    """A multi-timepoint source occupies a contiguous block; the next source starts past it."""

    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        n_t = 3 if source.source_id.endswith("t45hpf") else 1
        return _yx1_like_source_df(
            experiment_id=experiment_id, source_tag=source.source_id, n_t=n_t
        )

    sources = [
        PlateSource("20260607_plate01_t45hpf", scope="YX1"),  # 3 timepoints → block {0,1,2}
        PlateSource("20260608_plate01_t72hpf", scope="YX1"),  # 1 timepoint  → block {3}
    ]
    out = union_collection_acquisition_inventories(
        collection_name=_COLLECTION, sources=sources, read_source=read_source
    )

    # Select by declared age (block), not a source label — the label no longer exists.
    t45 = out[out["start_age_hpf"] == 45]  # 3-timepoint source → block {0,1,2}
    t72 = out[out["start_age_hpf"] == 72]  # 1-timepoint source → block {3}
    assert sorted(t45["time_index"].unique()) == [0, 1, 2]
    assert sorted(t72["time_index"].unique()) == [3]
    # The original per-source time_index is NOT preserved as a column (no source label survives).
    assert "source_time_index" not in out.columns
    # n_sources counts the merged acquisitions regardless of how many timepoints each carries.
    assert set(out["n_sources"].unique()) == {2}


def test_unioned_yx1_inventory_still_satisfies_the_real_validator():
    """The unioned table satisfies the SAME acquisition contract the pipeline already validates."""

    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        return _yx1_like_source_df(experiment_id=experiment_id, source_tag=source.source_id)

    sources = [
        PlateSource("20260607_plate01_t45hpf", scope="YX1"),
        PlateSource("20260608_plate01_t72hpf", scope="YX1"),
    ]
    out = union_collection_acquisition_inventories(
        collection_name=_COLLECTION, sources=sources, read_source=read_source
    )

    # Drop the union-added provenance/time cols and re-check the core YX1 acquisition contract:
    # schema-complete, calibrated, and a CLEAN tensor (cell key unique across the union — proving
    # the block-offset gives each source a disjoint time_index band).
    core = out.reindex(columns=list(YX1_ACQUISITION_INVENTORY_COLUMNS))
    validate_yx1_acquisition_inventory(core)  # raises if the union broke the contract


def test_single_source_gives_n_sources_one():
    """A plate with ONE acquisition → n_sources == 1 on every row (the legacy/timelapse case)."""

    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        return _keyence_like_source_df(experiment_id=experiment_id, source_tag=source.source_id)

    sources = [PlateSource("20260607_plate01_t45hpf", scope="Keyence")]
    out = union_collection_acquisition_inventories(
        collection_name=_COLLECTION, sources=sources, read_source=read_source
    )
    assert set(out["n_sources"].unique()) == {1}


def test_n_sources_is_per_well_constant_across_the_union():
    """n_sources is a per-well fact — the SAME value on every frame row of the merged well."""

    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        return _keyence_like_source_df(experiment_id=experiment_id, source_tag=source.source_id)

    sources = [
        PlateSource("20260607_plate01_t45hpf", scope="Keyence"),
        PlateSource("20260608_plate01_t72hpf", scope="Keyence"),
    ]
    out = union_collection_acquisition_inventories(
        collection_name=_COLLECTION, sources=sources, read_source=read_source
    )
    # Constant within every well_id (and here, across the whole union = one plate).
    assert (out.groupby("well_id")["n_sources"].nunique() == 1).all()
    assert set(out["n_sources"].unique()) == {2}


def test_mixed_plate_tokens_fail_loud():
    """Sources from different plates must not be unioned into one experiment_id."""

    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        return _keyence_like_source_df(experiment_id=experiment_id, source_tag=source.source_id)

    sources = [
        PlateSource("20260607_plate01_t45hpf", scope="Keyence"),
        PlateSource("20260607_plate02_t45hpf", scope="Keyence"),  # different plate!
    ]
    with pytest.raises(ValueError, match="multiple plate tokens"):
        union_collection_acquisition_inventories(
            collection_name=_COLLECTION, sources=sources, read_source=read_source
        )


def test_empty_sources_fail_loud():
    with pytest.raises(ValueError, match="no sources given"):
        union_collection_acquisition_inventories(
            collection_name=_COLLECTION, sources=[], read_source=lambda s, e: pd.DataFrame()
        )


def test_duplicate_source_read_fails_loud():
    """The same source child may not be handed in twice — one read per source.

    A repeated child necessarily repeats its declared age, so the ambiguity guard rejects it first
    (before any read). Either way the union refuses; this pins that it refuses.
    """

    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        return _keyence_like_source_df(experiment_id=experiment_id, source_tag=source.source_id)

    sources = [
        PlateSource("20260607_plate01_t45hpf", scope="Keyence"),
        PlateSource("20260607_plate01_t45hpf", scope="Keyence"),
    ]
    with pytest.raises(ValueError, match="AMBIGUOUS|exactly once"):
        union_collection_acquisition_inventories(
            collection_name=_COLLECTION, sources=sources, read_source=read_source
        )


def test_repeated_child_among_valid_sources_fails_loud():
    """A repeated child is rejected even when other sources are perfectly well-formed.

    Since a repeated source_id repeats its declared age, the ambiguity guard is what fires — the
    point here is that one bad pair poisons an otherwise valid source list rather than being
    silently deduped into a shorter union.
    """

    def read_source(source: PlateSource, experiment_id: str) -> pd.DataFrame:
        return _keyence_like_source_df(experiment_id=experiment_id, source_tag=source.source_id)

    sources = [
        PlateSource("20260607_plate01_t45hpf", scope="Keyence"),
        PlateSource("20260608_plate01_t72hpf", scope="Keyence"),  # fine on its own
        PlateSource("20260607_plate01_t45hpf", scope="Keyence"),  # repeat of the first
    ]
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        union_collection_acquisition_inventories(
            collection_name=_COLLECTION, sources=sources, read_source=read_source
        )
