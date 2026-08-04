"""The geometry gate, asserted structurally rather than by convention.

THE GATE: geometry is derived exactly once per embryo-time, by snip_geometry, and every render job
resolves that shared recipe onto its own product grid. A render job that derived its own would
reopen the failure the gate closes -- two products straddling a frame_masks regeneration produce
siblings that no longer register, with no error anywhere.

Runtime enforcement already exists: a render with no transform row fails loud. But that only holds
while nobody adds a fallback. These tests make the boundary structural, so a future edit that
reintroduces derivation fails the build rather than passing review because the diff looked
reasonable.

AST-based, not string-matching. `test_layering_contract.py` greps canonical.py for a literal, and a
mere COMMENT tripped it during the coordinate work -- a text match cannot tell an import from a
sentence about one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[4] / "src"
SNIP_PROCESSING = SRC / "data_pipeline" / "object_extraction" / "snip_processing"
ENTRYPOINTS = SNIP_PROCESSING / "entrypoints"

RENDER_ENTRYPOINT = ENTRYPOINTS / "run_snip_processing.py"
GEOMETRY_ENTRYPOINT = ENTRYPOINTS / "run_snip_geometry.py"

#: The one function that mints geometry from a mask. Only snip_geometry may call it.
DERIVATION_SYMBOL = "derive_snip_transform"


def _imported_names(path: Path) -> set[str]:
    """Every name bound by an import in this module, at any level."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _called_names(path: Path) -> set[str]:
    """Every simple or attribute call target, so `mod.derive_snip_transform(...)` is caught too."""
    called: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    return called


class TestOnlyGeometryDerives:
    def test_the_render_entrypoint_does_not_import_derivation(self):
        # Importing it is the step before calling it. Blocking the import is what makes the
        # boundary visible at review time rather than at runtime.
        assert DERIVATION_SYMBOL not in _imported_names(RENDER_ENTRYPOINT), (
            f"{RENDER_ENTRYPOINT.name} imports {DERIVATION_SYMBOL}. Geometry is derived ONCE by "
            "snip_geometry; a render job may only deserialize the transform and resolve it for its "
            "own product grid."
        )

    def test_the_render_entrypoint_does_not_call_derivation(self):
        # Belt and braces: an import could be added indirectly, or reached through a module object.
        assert DERIVATION_SYMBOL not in _called_names(RENDER_ENTRYPOINT), (
            f"{RENDER_ENTRYPOINT.name} calls {DERIVATION_SYMBOL}. If a render job can derive its "
            "own geometry, two product jobs can derive DIFFERENT geometry and produce silently "
            "unregisterable siblings -- exactly what the gate exists to prevent."
        )

    def test_the_geometry_entrypoint_does_derive(self):
        # The complement, and it is not redundant: if derivation moved or was renamed, the two
        # negative assertions above would both pass while nothing derived geometry at all.
        assert DERIVATION_SYMBOL in _called_names(GEOMETRY_ENTRYPOINT), (
            f"{GEOMETRY_ENTRYPOINT.name} no longer calls {DERIVATION_SYMBOL}. The gate has a "
            "verifier and no producer."
        )

    def test_the_render_entrypoint_reads_the_table(self):
        # The positive half of the contract: it must actually consume the persisted recipe, not
        # merely abstain from deriving one.
        imported = _imported_names(RENDER_ENTRYPOINT)
        assert "canonical_from_row" in imported, (
            f"{RENDER_ENTRYPOINT.name} does not import canonical_from_row, so it cannot be "
            "reconstructing the canonical transform from the table."
        )


class TestOneFingerprintDefinition:
    def test_both_sides_share_the_hash(self):
        # I WROTE THIS BUG. Writer and verifier once had separate copies of the hash arithmetic --
        # one over bare tobytes(), one over shape+dtype+bytes -- so every mask "mismatched" on a
        # completely unchanged well. Two copies of an invariant's arithmetic is how the invariant
        # becomes decorative; a behavioral test cannot stop someone adding a third.
        for path in (GEOMETRY_ENTRYPOINT, RENDER_ENTRYPOINT):
            assert "mask_content_fingerprint" in _imported_names(path), (
                f"{path.name} does not import the shared mask_content_fingerprint. If it computes "
                "its own hash, the freeze compares two different quantities and silently passes or "
                "silently fails everything."
            )

    def test_neither_entrypoint_hashes_directly(self):
        # The mechanism, not just the symptom: reaching for sha256 here is how a second definition
        # gets written.
        for path in (GEOMETRY_ENTRYPOINT, RENDER_ENTRYPOINT):
            assert "sha256" not in _imported_names(path), (
                f"{path.name} imports sha256 directly. Hash the mask through "
                "mask_content_fingerprint so both sides compute the same thing by construction."
            )


class TestTheGateIsReachableOnlyThroughTheTable:
    @pytest.mark.parametrize(
        "symbol", ["read_snip_transform_table", "canonical_from_row"], ids=lambda s: s
    )
    def test_render_consumes_persisted_geometry(self, symbol):
        assert symbol in _imported_names(RENDER_ENTRYPOINT)

    def test_geometry_writes_the_table(self):
        assert "write_snip_transform_table" in _imported_names(GEOMETRY_ENTRYPOINT)

    def test_geometry_reads_no_image_pixels(self):
        # snip_geometry needs masks and calibration only. If it started READING frames it would
        # acquire a dependency on materialized products, and the RFP render could no longer run in
        # parallel with the BF one -- both would queue behind the same image reads.
        #
        # THE BAN IS ON READING, NOT ON THE IMPORT. This test used to assert `"skio" not in
        # imports`, which was a proxy that stopped being accurate once the gate began WRITING the
        # embryo mask: writing a raster derived from the mask it already decoded adds no dependency
        # on any materialized image product, and the parallelism argument above is untouched. An
        # import check cannot tell a read from a write, so it is the call that is banned.
        source = GEOMETRY_ENTRYPOINT.read_text(encoding="utf-8")
        for reader in ("skio.imread", "cv2.imread", "imageio.imread"):
            assert reader not in source, (
                f"run_snip_geometry calls {reader}. Geometry comes from the MASK alone; reading a "
                "materialized image here would make sibling product jobs queue behind it."
            )
