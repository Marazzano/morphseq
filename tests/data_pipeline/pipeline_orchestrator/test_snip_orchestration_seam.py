"""The orchestration seam: constructed CLI arguments -> parser -> entrypoint call.

WHY THIS EXISTS. The snip tests exercise the Python entrypoints directly, and the Snakemake rules
are only reachable on a cluster, so the layer BETWEEN them -- tasks.py building an argument list, the
parser accepting it, the verb calling the entrypoint -- had no coverage at all. Two stale-keyword
bugs reached committed code through that gap in a single session:

    snip_transform_table_path   survived after the render side stopped WRITING the table
    resolved=                   survived after the row builder started taking a canonical

Both would have failed on the first real invocation and neither was visible to any test, because
every test called run_snip_processing(...) directly with keyword arguments it chose itself.

These tests drive the REAL parser with a REAL argv list and assert on what the entrypoint receives.
No well, no rendering, no disk: the entrypoint is mocked, because the question is whether the
arguments survive the trip, not whether the pixels are right.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from data_pipeline.object_extraction.snip_processing.snip_product_keys import (
    DEFAULT_BF_SNIP_PRODUCT_KEY,
)
from data_pipeline.pipeline_orchestrator.tasks import build_parser

RFP_PRODUCT = "RFP__projection__max__no_change"


def _render_argv(**overrides) -> list[str]:
    """The argv a Snakemake shell: block assembles for one render job."""
    values = {
        "--frame-masks-csv": "/w/frame_masks.csv",
        "--frame-inventory-csv": "/w/frame_inventory.csv",
        "--physical-embryo-registry-csv": "/w/registry.csv",
        "--snip-transform-table-csv": "/w/snip_transforms.parquet",
        "--output-csv": "/w/BF__projection__focus_stack__clahe_blend/snip_inventory.csv",
        "--snips-dir": "/w/snips",
        "--output-root": "/root",
        "--snip-product-key": DEFAULT_BF_SNIP_PRODUCT_KEY,
    }
    values.update(overrides)
    argv = ["snip-processing"]
    for flag, value in values.items():
        argv += [flag, str(value)]
    return argv


def _geometry_argv(**overrides) -> list[str]:
    values = {
        "--frame-masks-csv": "/w/frame_masks.csv",
        "--frame-inventory-csv": "/w/frame_inventory.csv",
        "--physical-embryo-registry-csv": "/w/registry.csv",
        "--output-path": "/w/snip_geometry/snip_transforms.parquet",
    }
    values.update(overrides)
    argv = ["snip-geometry"]
    for flag, value in values.items():
        argv += [flag, str(value)]
    return argv


def _dispatch(argv: list[str], target: str):
    """Parse argv for real, dispatch the verb, and capture the entrypoint call."""
    args = build_parser().parse_args(argv)
    with patch(target) as entrypoint:
        args.func(args)
    assert entrypoint.call_count == 1, "the verb did not call its entrypoint exactly once"
    return entrypoint.call_args.kwargs


RENDER_TARGET = (
    "data_pipeline.object_extraction.snip_processing.entrypoints"
    ".run_snip_processing.run_snip_processing"
)
GEOMETRY_TARGET = (
    "data_pipeline.object_extraction.snip_processing.entrypoints"
    ".run_snip_geometry.run_snip_geometry"
)


class TestEveryArgumentSurvivesTheTrip:
    """The bug class that reached committed code twice: a name that stopped matching."""

    def test_the_render_verb_passes_what_the_entrypoint_expects(self):
        import inspect

        from data_pipeline.object_extraction.snip_processing.entrypoints.run_snip_processing import (
            run_snip_processing,
        )

        kwargs = _dispatch(_render_argv(), RENDER_TARGET)
        accepted = set(inspect.signature(run_snip_processing).parameters)
        stale = set(kwargs) - accepted
        assert not stale, (
            f"tasks.py passes {sorted(stale)}, which run_snip_processing does not accept. This is "
            "the exact failure that reached committed code twice -- a keyword survived a rename on "
            "the entrypoint side because no test drove the CLI path."
        )

    def test_the_geometry_verb_passes_what_the_entrypoint_expects(self):
        import inspect

        from data_pipeline.object_extraction.snip_processing.entrypoints.run_snip_geometry import (
            run_snip_geometry,
        )

        kwargs = _dispatch(_geometry_argv(), GEOMETRY_TARGET)
        stale = set(kwargs) - set(inspect.signature(run_snip_geometry).parameters)
        assert not stale, f"tasks.py passes {sorted(stale)} to run_snip_geometry"

    def test_a_removed_flag_is_rejected_by_the_parser(self):
        # The negative case, proving the parser is really in the loop: the OLD flag name from before
        # the render side became a table consumer must no longer parse. If this passed, the test
        # above would be exercising a parser that accepts anything.
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                _render_argv() + ["--snip-transform-table-path", "/w/t.parquet"]
            )


class TestTheProductKeyCrossesEveryLayer:
    def test_the_key_arrives_unchanged(self):
        kwargs = _dispatch(_render_argv(**{"--snip-product-key": RFP_PRODUCT}), RENDER_TARGET)
        assert kwargs["snip_product_key"] == RFP_PRODUCT

    def test_the_default_is_the_bf_product(self):
        # A render job with no explicit key must render what the pipeline rendered before product
        # keys existed -- otherwise the migration silently changes the default output.
        argv = [a for a in _render_argv() if a not in ("--snip-product-key", DEFAULT_BF_SNIP_PRODUCT_KEY)]
        kwargs = _dispatch(argv, RENDER_TARGET)
        assert kwargs["snip_product_key"] == DEFAULT_BF_SNIP_PRODUCT_KEY

    def test_the_output_path_and_the_key_agree(self):
        # The fanout writes one inventory per product; a key that disagreed with the path it was
        # given would put one product's rows in another product's shard.
        kwargs = _dispatch(_render_argv(**{
            "--snip-product-key": RFP_PRODUCT,
            "--output-csv": f"/w/{RFP_PRODUCT}/snip_inventory.csv",
        }), RENDER_TARGET)
        assert RFP_PRODUCT in str(kwargs["output_csv"])
        assert kwargs["snip_product_key"] == RFP_PRODUCT


class TestTheTransformTableIsAnInput:
    def test_the_render_verb_receives_it_as_a_table_to_read(self):
        kwargs = _dispatch(_render_argv(), RENDER_TARGET)
        assert kwargs["snip_transform_table_csv"] == Path("/w/snip_transforms.parquet")

    def test_the_render_verb_has_no_way_to_write_one(self):
        # THE GATE, at the orchestration layer. If the render verb could be handed an output path
        # for the transform table, a caller could make it write geometry -- reopening the drift the
        # gate closes. The parser must have no such flag at all.
        kwargs = _dispatch(_render_argv(), RENDER_TARGET)
        assert not any("table_path" in k for k in kwargs), sorted(kwargs)

    def test_the_geometry_verb_writes_it(self):
        kwargs = _dispatch(_geometry_argv(), GEOMETRY_TARGET)
        assert kwargs["output_path"] == Path("/w/snip_geometry/snip_transforms.parquet")

    def test_the_geometry_verb_takes_no_product_key(self):
        # Per-well only, deliberately: one embryo-time has one canonical transform, so a product
        # dimension here would derive it N times.
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                _geometry_argv() + ["--snip-product-key", RFP_PRODUCT]
            )


class TestMalformedKeysFailBeforeRendering:
    @pytest.mark.parametrize(
        "bad", ["BF", "BF__no_change", "not a key"], ids=["too-few-fields", "no-source", "garbage"]
    )
    def test_an_unparseable_key_is_rejected(self, bad):
        # Fail at the seam, where the message can name the key, rather than after a job has read
        # frames and written partial output.
        from data_pipeline.object_extraction.snip_processing.snip_product_keys import (
            parse_snip_product_key,
        )

        with pytest.raises(ValueError):
            parse_snip_product_key(bad)
