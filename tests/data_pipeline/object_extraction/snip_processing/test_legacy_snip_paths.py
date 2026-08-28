"""The legacy flat-path compatibility alias.

Four properties matter, and they are the four ways this can go wrong on a shared cluster
filesystem: ownership (which path holds the bytes), collision with pre-existing real files,
relative-link correctness under a moved tree, and idempotent reruns.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from data_pipeline.object_extraction.snip_processing.legacy_snip_paths import (
    LegacySnipPathConflict,
    legacy_flat_snip_path,
    link_legacy_flat_path,
)

PRODUCT = "BF__projection__focus_stack__clahe_blend"
EMBRYO = "20250912_B01_e01"
SNIP = "20250912_B01_e01_BF_t0000.png"


def _canonical(snips_dir: Path) -> Path:
    path = snips_dir / EMBRYO / PRODUCT / SNIP
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"canonical-pixels")
    return path


def _legacy(snips_dir: Path) -> Path:
    return legacy_flat_snip_path(snips_dir=snips_dir, physical_embryo_id=EMBRYO, filename=SNIP)


class TestOwnership:
    def test_the_product_path_holds_the_bytes(self, tmp_path):
        # THE DIRECTION OF THE ALIAS. Writing real bytes flat and pointing the product directory
        # back at them would keep the pre-migration layout as the authority and make the product
        # hierarchy decorative.
        canonical, legacy = _canonical(tmp_path), _legacy(tmp_path)
        link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy)

        assert not canonical.is_symlink(), "the canonical path must be a real file"
        assert legacy.is_symlink(), "the legacy path must be an alias, not a copy"

    def test_an_ordinary_reader_gets_the_same_bytes(self, tmp_path):
        # What an EXTERNAL consumer actually relies on: plain open(), not Path.resolve(). A test
        # that only compared resolved paths would pass even if the link pointed at nothing.
        canonical, legacy = _canonical(tmp_path), _legacy(tmp_path)
        link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy)
        assert legacy.read_bytes() == canonical.read_bytes() == b"canonical-pixels"


class TestRelativeLink:
    def test_the_link_body_is_relative_and_local(self, tmp_path):
        # Absolute links do not survive a copied, moved, or differently-mounted experiment tree --
        # all routine on this cluster. And because the product dir is a CHILD of the embryo dir,
        # the body needs no ".." at all: a stray parent segment would mean the layout regressed to
        # product-first, where the alias has to climb out of one subtree and back into another.
        canonical, legacy = _canonical(tmp_path), _legacy(tmp_path)
        link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy)
        body = os.readlink(legacy)
        assert not os.path.isabs(body)
        assert ".." not in body
        assert body == f"{PRODUCT}/{SNIP}"

    def test_the_alias_survives_the_tree_moving(self, tmp_path):
        # The property the relative link buys: rename the whole tree and the alias still resolves.
        snips = tmp_path / "before"
        canonical, legacy = _canonical(snips), _legacy(snips)
        link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy)

        moved = tmp_path / "after"
        snips.rename(moved)
        relocated = legacy_flat_snip_path(
            snips_dir=moved, physical_embryo_id=EMBRYO, filename=SNIP
        )
        assert relocated.read_bytes() == b"canonical-pixels"


class TestNeverClobbers:
    def test_a_pre_existing_real_file_is_replaced_once_canonical_exists(self, tmp_path):
        # CONTRACT CHANGE 2026-08-28. This used to fail loud on any real file, reasoning that
        # replacing it would destroy the only copy. With the canonical render present that premise
        # is false -- and since every snip in the corpus had a pre-migration file here, the refusal
        # fired on all of them, was recorded as is_valid_snip=False by the caller's bare
        # `except Exception`, and took down whole experiments over an un-creatable SYMLINK.
        canonical, legacy = _canonical(tmp_path), _legacy(tmp_path)
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"pre-migration-pixels")

        link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy)

        assert legacy.is_symlink(), "the stale real file must give way to the alias"
        assert legacy.read_bytes() == b"canonical-pixels", "the alias must resolve to the canonical bytes"

    def test_a_pre_existing_real_file_fails_loud_with_no_canonical(self, tmp_path):
        # THE CASE THE GUARD WAS ACTUALLY WRITTEN FOR, and it still holds: with no canonical render
        # on disk those pre-migration pixels really may be the only copy, so refuse rather than
        # destroy them.
        legacy = _legacy(tmp_path)
        canonical = tmp_path / EMBRYO / PRODUCT / SNIP  # deliberately NOT written
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"pre-migration-pixels")

        with pytest.raises(LegacySnipPathConflict, match="no canonical render exists"):
            link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy)
        assert legacy.read_bytes() == b"pre-migration-pixels", "the original data must survive"

    def test_a_link_to_another_product_fails_loud(self, tmp_path):
        # Two products fighting over one alias is a configuration error, not something to resolve
        # silently by last-write-wins.
        canonical, legacy = _canonical(tmp_path), _legacy(tmp_path)
        other = tmp_path / EMBRYO / "RFP__projection__max__no_change" / SNIP
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_bytes(b"rfp-pixels")
        link_legacy_flat_path(canonical_path=other, legacy_path=legacy)

        with pytest.raises(LegacySnipPathConflict, match="claiming one compatibility alias"):
            link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy)


class TestIdempotence:
    def test_rerunning_does_not_churn_the_filesystem(self, tmp_path):
        # Snakemake reruns rules routinely; the second pass must recognize its own work rather than
        # recreating or failing on it.
        canonical, legacy = _canonical(tmp_path), _legacy(tmp_path)
        assert link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy) is True
        assert link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy) is False
        assert legacy.read_bytes() == b"canonical-pixels"


class TestManifest:
    def test_created_aliases_are_recorded(self, tmp_path):
        # Turns "an external reader broke" from archaeology into a lookup.
        canonical, legacy = _canonical(tmp_path), _legacy(tmp_path)
        manifest: list[dict[str, str]] = []
        link_legacy_flat_path(
            canonical_path=canonical,
            legacy_path=legacy,
            log_manifest=manifest,
            snip_product_key=PRODUCT,
        )
        assert len(manifest) == 1
        assert manifest[0]["snip_product_key"] == PRODUCT
        assert manifest[0]["legacy_path"] == str(legacy)

    def test_an_idempotent_rerun_adds_no_entry(self, tmp_path):
        canonical, legacy = _canonical(tmp_path), _legacy(tmp_path)
        manifest: list[dict[str, str]] = []
        link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy, log_manifest=manifest)
        link_legacy_flat_path(canonical_path=canonical, legacy_path=legacy, log_manifest=manifest)
        assert len(manifest) == 1
