"""The OT backends must stay geometry-free: solver-only, no coord/working-grid dependency.

Checked by parsing each backend's AST and inspecting its real ``import`` statements, NOT by
scanning the raw source text. A substring scan cannot tell an import from prose: it fails on a
docstring that merely NAMES the caller (e.g. ott_backend's "bucketing already done by
solve_working_grid_batch() at the outer pipeline level"), which is documentation of the layering
this test exists to protect — exactly the thing it should not flag.
"""

import ast
from pathlib import Path


def _imported_modules(tree: ast.AST) -> set[str]:
    """Every module name this file actually imports (dotted, as written)."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            # `from x import y` — y may itself be the forbidden submodule.
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def test_backends_do_not_import_coord_or_working_grid():
    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "src" / "analyze" / "utils" / "optimal_transport" / "backends"

    forbidden = ("analyze.utils.coord", "working_grid")
    for p in sorted(backend_dir.glob("*.py")):
        imported = _imported_modules(ast.parse(p.read_text(encoding="utf-8"), filename=str(p)))
        for needle in forbidden:
            offenders = sorted(m for m in imported if needle in m)
            assert not offenders, (
                f"Backend module {p} imports forbidden geometry module(s) matching "
                f"{needle!r}: {offenders}"
            )
