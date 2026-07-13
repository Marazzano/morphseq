"""Structural regression guards for the one authoritative peak-analysis path."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2]
ENGINE = PACKAGE / "engine"


def _source(name: str) -> str:
    return (ENGINE / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.AST:
    return ast.parse(_source(name), filename=name)


def _called_names(name: str) -> set[str]:
    called = set()
    for node in ast.walk(_tree(name)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    return called


def test_retired_label_ontologies_do_not_exist_in_engine_sources():
    retired = (
        "LabelColumn",
        "LabelGroupArtifacts",
        "LabelProvenance",
        "DistributionLabelGroup",
    )
    violations = {}
    for path in ENGINE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        found = tuple(token for token in retired if token in source)
        if found:
            violations[path.name] = found
    assert violations == {}


def test_labeling_provenance_has_no_hidden_geometry_field():
    tree = _tree("objects.py")
    provenance = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LabelingProvenance"
    )
    annotated_fields = {
        node.target.id for node in provenance.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "geometry" not in annotated_fields


def test_catalog_and_plotting_never_import_or_call_kde_evaluation():
    forbidden_calls = {
        "evaluate_density", "gaussian_kde",
        "evaluate_isotropic_gaussian_kde_from_dist2",
        "_evaluate_density_for_spec",
    }
    for module in ("catalog.py", "plotting.py"):
        tree = _tree(module)
        imported = {
            alias.name.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert imported.isdisjoint(forbidden_calls), module
        assert _called_names(module).isdisjoint(forbidden_calls), module


def test_catalog_routes_peak_assignment_through_the_single_chain():
    catalog = _source("catalog.py")
    objects = _source("objects.py")
    labelers = _source("labelers.py")

    # Catalog delegates each member to the Distribution API.
    assert "distribution.detect_peaks(" in catalog
    # Distribution delegates to the sole engine labeler.
    assert "from .labelers import detect_peaks as _detect_peaks" in objects
    assert "group = _detect_peaks(" in objects
    # The labeler resolves through core then crosses exactly one adapter.
    assert "resolved_record = compute_resolved_peaks(" in labelers
    assert "return label_group_from_resolved_peaks(" in labelers
    assert "from .peak_adapter import label_group_from_resolved_peaks" in labelers


def test_catalog_does_not_reconstruct_basins_hdrs_or_recount_peaks():
    source = _source("catalog.py")
    tree = _tree("catalog.py")
    forbidden_names = {
        "assign_cells_to_peaks", "assign_points_to_peaks", "detect_peaks",
        "PeakGeometry", "HDR", "empirical_basin_labels",
    }
    # Method delegation as ``distribution.detect_peaks`` is allowed; a naked
    # detector/reconstruction call or import is not.
    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    }
    assert imported.isdisjoint(forbidden_names)
    naked_calls = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert naked_calls.isdisjoint(forbidden_names)
    assert "len(resolved" not in source
    assert "len(peaks" not in source
    assert "basin" not in source.lower()
    assert "hdr" not in source.lower()
