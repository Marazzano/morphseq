"""Structural regression guards for the one authoritative peak-analysis path."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2]
ENGINE = PACKAGE / "engine"
CORE = PACKAGE / "core"


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


def test_active_density_paths_share_one_kde_kernel_and_no_scipy_backend():
    """The supported engine/resolver surface has one evaluator implementation.

    ``support_geometry`` is intentionally excluded: it is a legacy research
    diagnostic module and labels its historical alternate estimators as such.
    """
    grid_source = (ENGINE / "grid.py").read_text(encoding="utf-8")
    resolver_source = (CORE / "resolved_peak_analysis.py").read_text(encoding="utf-8")
    for source in (grid_source, resolver_source):
        assert "evaluate_isotropic_gaussian_kde_from_dist2(" in source
        assert "scipy.stats" not in source
        assert "from scipy" not in source
        assert "scipy_default" not in source
    assert resolver_source.count("evaluate_isotropic_gaussian_kde_from_dist2(") == 1


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


def test_single_pass_resolvers_are_internal_and_not_publicly_exported():
    source = (CORE / "resolved_peak_analysis.py").read_text(encoding="utf-8")
    assert "def resolve_points_with_analysis_spec(" not in source
    assert "def resolve_density_grid_with_analysis_spec(" not in source
    assert "def _resolve_points_single_pass(" in source
    assert "def _resolve_density_grid_single_pass(" in source
    exported = source.partition("__all__ = [")[2]
    assert "_resolve_points_single_pass" not in exported
    assert "_resolve_density_grid_single_pass" not in exported


def test_engine_labeler_accepts_only_one_resolution_configuration_path():
    source = _source("labelers.py")
    signature = source.partition("def detect_peaks(")[2].partition(") -> LabelGroup:")[0]
    assert "resolution_config: PeakResolutionConfig" in signature
    assert "voting_spec" not in signature
    assert "robustness_policy" not in signature


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


def test_catalog_comparison_is_matching_only_and_delegates_numeric_work():
    tree = _tree("catalog.py")
    calls = _called_names("catalog.py")
    forbidden = {
        "build_shared_grid", "evaluate_density", "_bandwidth", "_overlap",
        "test_nulls", "gaussian_kde",
    }
    imported = {
        alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint(forbidden)
    assert calls.isdisjoint(forbidden)
    assert calls >= {"compare_distributions"}
