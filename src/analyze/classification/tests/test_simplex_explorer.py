"""
Tests for analyze.classification.viz.simplex_explorer + simplex_adapters

Grouped into:
- TestBuildSimplexPayload: payload shape, validation, group/penetrance handling
- TestNClassGeneralization: N=1, 2, 3 all render (nothing hardcodes three)
- TestRenderSimplexExplorerHtml: self-containment and output contract
- TestPenetranceRate: the predicted-penetrant fraction the page draws
- TestResolveProbCols: explicit per-convention column resolution
- TestAdapters: one adapter per producer, mismatches fail loudly

Synthetic fixture: 4 embryos x 3 timepoints, classes A / B / "Not Penetrant".
Note the class is "Not Penetrant" (a phenotype) while the ``genotype`` facet
carries "wildtype"/"homozygous" (genotypes) — that mismatch is intended.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from analyze.classification.viz.simplex_explorer import (
    build_simplex_payload,
    render_simplex_explorer_html,
    render_simplex_explorer_html_from_scores,
)
from analyze.classification.viz.simplex_adapters import (
    PROB_COL_CONVENTIONS,
    from_label_transfer,
    from_run_classification,
    resolve_prob_cols,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Rows are (embryo, group, P(A), P(B), P(WT)); argmax gives the predicted class.
_EMBRYOS = [
    ("e1", "homozygous", 0.80, 0.10, 0.10),   # pred A  -> penetrant
    ("e2", "homozygous", 0.10, 0.70, 0.20),   # pred B  -> penetrant
    ("e3", "wildtype",   0.05, 0.15, 0.80),   # pred Not Penetrant
    ("e4", "wildtype",   0.20, 0.20, 0.60),   # pred Not Penetrant
]

CLASS_PROB_COLS = {"A": "p_a", "B": "p_b", "Not Penetrant": "p_wt"}


def _frame(n_timepoints: int = 3) -> pd.DataFrame:
    rows = []
    for eid, group, pa, pb, pwt in _EMBRYOS:
        for i in range(n_timepoints):
            rows.append({
                "embryo_id": eid,
                "hpf": 24.0 + 6.0 * i,
                "curvature": 0.1 * i + (0.5 if eid == "e1" else 0.0),
                "length_um": 1000.0 + 10.0 * i,
                "genotype": group,
                "curated": "manual" if eid == "e1" else "",
                "in_train": eid in ("e1", "e3"),
                "p_a": pa, "p_b": pb, "p_wt": pwt,
            })
    return pd.DataFrame(rows)


def _payload(**kwargs) -> dict:
    defaults = dict(
        id_col="embryo_id",
        time_col="hpf",
        features=["curvature", "length_um"],
        class_prob_cols=CLASS_PROB_COLS,
        group_col="genotype",
        curated_col="curated",
        in_train_col="in_train",
        non_penetrant_class="Not Penetrant",
    )
    defaults.update(kwargs)
    return build_simplex_payload(_frame(), **defaults)


# ---------------------------------------------------------------------------


class TestBuildSimplexPayload:
    def test_top_level_shape(self):
        p = _payload()
        for field in ("embryos", "classes", "featureLabels", "zygOrder",
                      "zygColor", "nonPenetrantClass", "timeLabel", "defaultColorBy"):
            assert field in p
        assert len(p["embryos"]) == len(_EMBRYOS)
        assert len(p["classes"]) == 3
        assert p["nonPenetrantClass"] == "Not Penetrant"

    def test_embryo_record_shape(self):
        p = _payload()
        e = next(e for e in p["embryos"] if e["id"] == "e1")
        assert e["zyg"] == "homozygous"
        assert e["pred"] == "A"
        assert e["inTrain"] is True
        assert set(e["p"]) == {c["key"] for c in p["classes"]}
        # One y series per declared feature, each aligned to the time series.
        assert len(e["y"]) == len(p["featureLabels"]) == 2
        assert all(len(series) == len(e["t"]) for series in e["y"])

    def test_class_keys_are_unique_and_js_safe(self):
        p = build_simplex_payload(
            _frame().rename(columns={"p_a": "x1", "p_b": "x2", "p_wt": "x3"}),
            id_col="embryo_id", time_col="hpf", features=["curvature"],
            class_prob_cols={"homozygous NON-CE": "x1", "homozygous non ce": "x2", "Not Penetrant": "x3"},
        )
        keys = [c["key"] for c in p["classes"]]
        assert len(set(keys)) == 3
        assert all(re.fullmatch(r"[a-z0-9_]+", k) for k in keys)

    def test_predicted_class_is_argmax(self):
        p = _payload()
        pred = {e["id"]: e["pred"] for e in p["embryos"]}
        assert pred == {"e1": "A", "e2": "B", "e3": "Not Penetrant", "e4": "Not Penetrant"}

    def test_group_order_respected_and_filtered(self):
        p = _payload(group_order=["wildtype", "heterozygous", "homozygous"])
        # "heterozygous" is absent from the data and must be dropped.
        assert p["zygOrder"] == ["wildtype", "homozygous"]

    def test_group_order_none_uses_first_appearance(self):
        p = _payload()
        assert p["zygOrder"] == ["homozygous", "wildtype"]

    def test_downsampling_caps_points(self):
        df = _frame(n_timepoints=200)
        p = build_simplex_payload(
            df, id_col="embryo_id", time_col="hpf", features=["curvature"],
            class_prob_cols=CLASS_PROB_COLS, max_points=25,
        )
        assert all(len(e["t"]) == 25 for e in p["embryos"])

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            build_simplex_payload(
                _frame(), id_col="embryo_id", time_col="hpf",
                features=["nope"], class_prob_cols=CLASS_PROB_COLS,
            )

    def test_empty_class_map_raises(self):
        with pytest.raises(ValueError, match="at least one class"):
            build_simplex_payload(
                _frame(), id_col="embryo_id", time_col="hpf",
                features=["curvature"], class_prob_cols={},
            )

    def test_bad_non_penetrant_class_raises(self):
        with pytest.raises(ValueError, match="non_penetrant_class"):
            _payload(non_penetrant_class="not-a-class")

    def test_unscored_embryos_dropped(self):
        df = _frame()
        df.loc[df["embryo_id"] == "e2", "p_a"] = np.nan
        p = build_simplex_payload(
            df, id_col="embryo_id", time_col="hpf", features=["curvature"],
            class_prob_cols=CLASS_PROB_COLS,
        )
        assert {e["id"] for e in p["embryos"]} == {"e1", "e3", "e4"}


class TestNClassGeneralization:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_payload_and_html_for_n_classes(self, n):
        cols = dict(list(CLASS_PROB_COLS.items())[:n])
        p = build_simplex_payload(
            _frame(), id_col="embryo_id", time_col="hpf",
            features=["curvature"], class_prob_cols=cols, group_col="genotype",
        )
        assert len(p["classes"]) == n
        html = render_simplex_explorer_html(p)
        # The sliders are built client-side from DATA.classes, so the guarantee
        # we can assert statically is that all N classes reached the payload.
        for name in cols:
            assert f'"name":"{name}"' in html

    def test_single_class_defaults_non_penetrant_to_last(self):
        p = build_simplex_payload(
            _frame(), id_col="embryo_id", time_col="hpf",
            features=["curvature"], class_prob_cols={"A": "p_a"},
        )
        assert p["nonPenetrantClass"] == "A"

    def test_nothing_hardcodes_three(self):
        cols = dict(CLASS_PROB_COLS)
        cols["extra"] = "p_a"  # a 4th class reusing a column is still 4 sliders
        p = build_simplex_payload(
            _frame(), id_col="embryo_id", time_col="hpf",
            features=["curvature"], class_prob_cols=cols, non_penetrant_class="Not Penetrant",
        )
        assert len(p["classes"]) == 4


class TestRenderSimplexExplorerHtml:
    def test_no_external_references(self):
        html = render_simplex_explorer_html(_payload())
        assert not re.search(r'(src|href)\s*=\s*"https?://', html)
        # Also no runtime fetching of any kind.
        for token in ("cdn.", "fetch(", "XMLHttpRequest", "importScripts"):
            assert token not in html

    def test_is_a_full_document(self):
        html = render_simplex_explorer_html(_payload(), title="My Explorer")
        assert html.lstrip().startswith("<!doctype html>")
        assert "<title>My Explorer</title>" in html
        assert "__PAYLOAD__" not in html
        assert "__TITLE__" not in html
        assert "__SUBTITLE__" not in html

    def test_title_is_escaped(self):
        html = render_simplex_explorer_html(_payload(), title="a <script> & b")
        assert "<title>a &lt;script&gt; &amp; b</title>" in html

    def test_subtitle_html_passes_through(self):
        html = render_simplex_explorer_html(_payload(), subtitle="try <em>P(A)</em>")
        assert "try <em>P(A)</em>" in html

    def test_writes_output_path(self, tmp_path):
        out = tmp_path / "nested" / "explorer.html"
        html = render_simplex_explorer_html(_payload(), output_path=out)
        assert out.read_text(encoding="utf-8") == html

    def test_from_scores_convenience(self, tmp_path):
        out = tmp_path / "explorer.html"
        html = render_simplex_explorer_html_from_scores(
            _frame(), id_col="embryo_id", time_col="hpf",
            features=["curvature"], class_prob_cols=CLASS_PROB_COLS,
            group_col="genotype", non_penetrant_class="Not Penetrant", output_path=out,
        )
        assert out.exists()
        assert '"nonPenetrantClass":"Not Penetrant"' in html

    def test_mismatched_feature_series_raises(self):
        p = _payload()
        p["embryos"][0]["y"] = p["embryos"][0]["y"][:1]
        with pytest.raises(ValueError, match="feature series"):
            render_simplex_explorer_html(p)

    def test_duplicate_class_keys_raise(self):
        p = _payload()
        p["classes"][1]["key"] = p["classes"][0]["key"]
        with pytest.raises(ValueError, match="unique"):
            render_simplex_explorer_html(p)


class TestPenetranceRate:
    """The page draws a divider at (# predicted non-wildtype) / (# in panel)."""

    @staticmethod
    def _rate(embryos, non_penetrant: str) -> float:
        return sum(e["pred"] != non_penetrant for e in embryos) / len(embryos)

    def test_overall_rate(self):
        p = _payload()
        # e1 -> A, e2 -> B penetrant; e3, e4 -> WT not.
        assert self._rate(p["embryos"], p["nonPenetrantClass"]) == pytest.approx(0.5)

    def test_per_facet_rate(self):
        p = _payload()
        by_group: dict[str, list] = {}
        for e in p["embryos"]:
            by_group.setdefault(e["zyg"], []).append(e)
        assert self._rate(by_group["homozygous"], "Not Penetrant") == pytest.approx(1.0)
        assert self._rate(by_group["wildtype"], "Not Penetrant") == pytest.approx(0.0)

    def test_rate_follows_non_penetrant_class_choice(self):
        p = _payload(non_penetrant_class="A")
        # Only e1 is predicted A, so 3 of 4 count as "penetrant".
        assert self._rate(p["embryos"], "A") == pytest.approx(0.75)

    def test_label_wording_is_model_output_not_biology(self):
        html = render_simplex_explorer_html(_payload())
        assert "predicted penetrant:" in html
        # Guard against regressing to a bare "penetrance" claim on the divider.
        assert "penetrance: " not in html


class TestResolveProbCols:
    """Column resolution is explicit: the caller states which convention it reads."""

    @staticmethod
    def _renamed(prefix: str) -> pd.DataFrame:
        return _frame().rename(columns={
            "p_a": f"{prefix}CE", "p_b": f"{prefix}HTA",
            "p_wt": f"{prefix}Not Penetrant",
        })

    @pytest.mark.parametrize("producer,prefix", sorted(PROB_COL_CONVENTIONS.items()))
    def test_resolves_each_known_convention(self, producer, prefix):
        found = resolve_prob_cols(self._renamed(prefix), prefix)
        assert found == {
            "CE": f"{prefix}CE", "HTA": f"{prefix}HTA",
            "Not Penetrant": f"{prefix}Not Penetrant",
        }

    def test_class_order_controls_order_and_subset(self):
        found = resolve_prob_cols(
            self._renamed("prob_"), "prob_", class_order=["Not Penetrant", "CE"]
        )
        assert list(found) == ["Not Penetrant", "CE"]

    def test_unknown_class_in_order_raises(self):
        with pytest.raises(ValueError, match="no 'prob_' column"):
            resolve_prob_cols(self._renamed("prob_"), "prob_", class_order=["nope"])

    def test_conventions_do_not_bleed(self):
        # A table carrying both conventions resolves strictly to the one asked for.
        df = self._renamed("prob_")
        df["pred_proba_CE"] = df["prob_CE"]
        assert resolve_prob_cols(df, "pred_proba_") == {"CE": "pred_proba_CE"}
        assert set(resolve_prob_cols(df, "prob_")) == {"CE", "HTA", "Not Penetrant"}

    def test_non_numeric_columns_ignored(self):
        df = self._renamed("prob_")
        df["prob_note"] = "text"
        assert "prob_note" not in resolve_prob_cols(df, "prob_").values()

    def test_missing_convention_raises(self):
        with pytest.raises(ValueError, match="No numeric 'prob_' columns"):
            resolve_prob_cols(_frame()[["embryo_id", "hpf", "curvature"]], "prob_")


class TestAdapters:
    """One adapter per producer; each knows exactly one output convention."""

    @staticmethod
    def _renamed(prefix: str) -> pd.DataFrame:
        return _frame().rename(columns={
            "p_a": f"{prefix}CE", "p_b": f"{prefix}HTA",
            "p_wt": f"{prefix}Not Penetrant",
        })

    def test_from_label_transfer(self, tmp_path):
        out = tmp_path / "explorer.html"
        html = from_label_transfer(
            self._renamed("prob_").rename(columns={"embryo_id": "query_embryo_id"}),
            time_col="hpf", features=["curvature"],
            class_order=["CE", "HTA", "Not Penetrant"],
            non_penetrant_class="Not Penetrant",
            group_col="genotype", output_path=out,
        )
        assert out.exists()
        assert '"name":"CE"' in html and '"name":"HTA"' in html
        assert '"nonPenetrantClass":"Not Penetrant"' in html

    def test_from_run_classification(self, tmp_path):
        out = tmp_path / "explorer.html"
        html = from_run_classification(
            self._renamed("pred_proba_"),
            id_col="embryo_id", time_col="hpf", features=["curvature"],
            class_order=["CE", "HTA", "Not Penetrant"],
            non_penetrant_class="Not Penetrant", output_path=out,
        )
        assert out.exists()
        assert '"name":"CE"' in html

    def test_wrong_adapter_for_the_table_fails_loudly(self):
        # The whole point of naming the producer: a mismatch errors, not guesses.
        with pytest.raises(ValueError, match="No numeric 'pred_proba_' columns"):
            from_run_classification(
                self._renamed("prob_"), id_col="embryo_id", time_col="hpf",
                features=["curvature"],
            )

    def test_adapter_output_is_self_contained(self):
        html = from_label_transfer(
            self._renamed("prob_").rename(columns={"embryo_id": "query_embryo_id"}),
            time_col="hpf", features=["curvature"],
        )
        assert not re.search(r'(src|href)\s*=\s*"https?://', html)
