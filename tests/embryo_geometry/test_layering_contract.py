"""The border, asserted rather than documented.

A package boundary that only exists in a README erodes the first time someone needs a
symbol. These tests fail the build instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
EMBRYO_GEOMETRY = SRC / "embryo_geometry"
IMAGE_GEOMETRY = SRC / "image_geometry"

FORBIDDEN_ROOTS = {"data_pipeline", "analyze"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                roots.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", sorted(EMBRYO_GEOMETRY.glob("*.py")), ids=lambda p: p.name)
def test_embryo_geometry_does_not_import_its_clients(path):
    """``analyze`` and ``data_pipeline`` are CLIENTS. Importing either inverts the
    dependency and reintroduces the cycle this package exists to break."""
    offenders = _imported_roots(path) & FORBIDDEN_ROOTS
    assert not offenders, f"{path.name} imports {sorted(offenders)}"


@pytest.mark.parametrize("path", sorted(IMAGE_GEOMETRY.glob("*.py")), ids=lambda p: p.name)
def test_image_geometry_does_not_import_biology(path):
    """Candidate mechanics must stay domain-free -- otherwise the split is cosmetic."""
    offenders = _imported_roots(path) & (FORBIDDEN_ROOTS | {"embryo_geometry"})
    assert not offenders, f"{path.name} imports {sorted(offenders)}"


ANATOMY_WORDS = ("yolk", "dorsal", "ventral", "anterior", "posterior")


def test_no_anatomical_identifiers_in_image_geometry():
    """Anatomy must not appear in image_geometry's API surface.

    Checked against IDENTIFIERS rather than raw text: the package docstring legitimately
    names the biological concepts it EXCLUDES ("the yolk-aware aligner stayed in
    analyze"), and a substring scan cannot tell a border statement from a leak. A
    parameter or function actually named after anatomy is unambiguous.
    """
    offenders = []
    for path in IMAGE_GEOMETRY.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            name = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.arg):
                name = node.arg
            elif isinstance(node, ast.Name):
                name = node.id
            if name and any(w in name.lower() for w in ANATOMY_WORDS):
                offenders.append(f"{path.name}:{name}")
    assert not offenders, f"anatomy leaked into generic mechanics: {offenders}"


def test_selection_policy_did_not_leak_into_image_geometry():
    """Enumeration belongs below the border; CHOOSING a winner does not.

    image_geometry may build the four candidates. The moment it also decides which one is
    right, the split is cosmetic and the next consumer inherits a hidden policy.
    """
    for path in IMAGE_GEOMETRY.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n = node.name.lower()
                assert not any(
                    tok in n for tok in ("select", "choose", "decide", "best_")
                ), f"{path.name}:{node.name} looks like selection, which is policy"


def test_tests_directory_has_no_init_file():
    """An ``__init__.py`` here would shadow the real ``embryo_geometry`` package and break
    every import in it. This exact mistake has already been made once."""
    here = Path(__file__).resolve().parent
    assert not (here / "__init__.py").exists()
    assert not (here.parent / "image_geometry" / "__init__.py").exists()


def test_callers_are_not_yet_rewired():
    """This commit lands the engine only; wiring is a separate reviewed step.

    If this starts failing, the wiring happened -- delete this test as part of that
    change, deliberately.
    """
    canonical = (SRC / "analyze/utils/coord/grids/canonical.py").read_text()
    rotation = (SRC / "data_pipeline/object_extraction/snip_processing/rotation.py").read_text()
    assert "embryo_geometry" not in canonical
    assert "embryo_geometry" not in rotation
