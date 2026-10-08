"""The evaluation suite itself: metric maths, task generation, gold-set integrity and scoring."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from matsci_agent.agents.nodes.extraction import ExtractionAgent
from matsci_agent.domain.units import normalise
from matsci_agent.evaluation import extraction_eval as ee
from matsci_agent.evaluation import retrieval_eval as re_

GOLD = Path(__file__).resolve().parents[2] / "evaluation" / "data" / "extraction_gold.json"


# ---- retrieval metrics ---------------------------------------------------------------------------


def test_ranking_metrics_on_a_known_ranking() -> None:
    ranked, relevant = ["a", "b", "c", "d"], {"b", "d"}
    assert re_.recall_at_k(ranked, relevant, 2) == 0.5 and re_.recall_at_k(ranked, relevant, 4) == 1.0
    assert re_.precision_at_k(ranked, relevant, 4) == 0.5
    assert re_.reciprocal_rank(ranked, relevant) == 0.5 and re_.reciprocal_rank(["x"], relevant) == 0.0
    assert re_.ndcg_at_k(["b", "d"], relevant, 2) == pytest.approx(1.0)  # ideal order
    assert 0 < re_.ndcg_at_k(ranked, relevant, 4) < 1.0


def test_ranked_papers_counts_each_paper_once_in_rank_order() -> None:
    hits = [{"id": "p1_abs", "metadata": {"arxiv_id": "p1"}}, {"id": "p2_3", "metadata": {}}, {"id": "p1_4", "metadata": {"arxiv_id": "p1"}}]
    assert re_.ranked_papers(hits) == ["p1", "p2"]


def test_sentence_tasks_skip_the_title_and_are_reproducible() -> None:
    docs = [{"id": f"p{i}_abs", "metadata": {"arxiv_id": f"p{i}"},
             "text": f"Title of paper {i}. " + "The first finding is about the material and its measured property here. " * 1
                     + "A second finding discusses the synthesis route used for the samples in detail. Final words."} for i in range(20)]
    a, b = re_.sentence_tasks(docs, 5, seed=3), re_.sentence_tasks(docs, 5, seed=3)
    assert a == b and len(a) == 5
    assert all("Title of paper" not in t["query"] for t in a)
    assert all(len(t["relevant"]) == 1 for t in a)


def test_topic_tasks_need_enough_relevant_papers() -> None:
    manifest = {"papers": [{"arxiv_id": f"p{i}", "topics": ["big"]} for i in range(10)] + [{"arxiv_id": "q", "topics": ["tiny"]}]}
    assert [t["query"] for t in re_.topic_tasks(manifest)] == ["big"]


def test_evaluate_scores_a_perfect_and_a_useless_retriever() -> None:
    tasks = [{"query": f"q{i}", "relevant": {f"p{i}"}} for i in range(4)]
    perfect = lambda q, n: [{"id": f"p{q[1:]}_abs", "metadata": {"arxiv_id": f"p{q[1:]}"}}]  # noqa: E731
    useless = lambda q, n: [{"id": "zzz_abs", "metadata": {"arxiv_id": "zzz"}}]  # noqa: E731
    assert re_.evaluate(perfect, tasks, topic=False)["mrr"] == 1.0
    assert re_.evaluate(useless, tasks, topic=False)["recall10"] == 0.0


# ---- extraction gold set -------------------------------------------------------------------------


def test_every_gold_value_agrees_with_the_unit_normaliser() -> None:
    """The gold labels were written by hand; this proves the normaliser reproduces them from the as-written value."""
    from matsci_agent.domain.units import parse_value

    gold = ee.load_gold(GOLD)
    checked = 0
    for item in gold:
        for rec in item["records"]:
            parsed = parse_value(rec["as_written"]["value"]).value
            _, canonical, _, _ = normalise(rec["property"], parsed, rec["as_written"]["unit"])
            assert canonical == pytest.approx(rec["value"], rel=1e-6), (item["id"], rec)
            checked += 1
    assert checked == 64


def test_the_gold_set_has_unique_ids_and_negative_controls() -> None:
    gold = ee.load_gold(GOLD)
    assert len({g["id"] for g in gold}) == len(gold)
    assert sum(not g["records"] for g in gold) >= 5  # passages that must yield nothing


def test_formula_canonicalisation_ignores_case_phase_prefixes_and_spacing() -> None:
    assert ee.canon_formula("β-Ga2O3") == ee.canon_formula("ga2o3") == "ga2o3"
    assert ee.canon_formula(" Li7La3Zr2O12 ") == "li7la3zr2o12" and ee.canon_formula(None) == ""


def test_scoring_counts_hits_wrong_values_misses_and_false_alarms() -> None:
    gold = [
        {"id": "a", "records": [{"formula": "GaN", "property": "bandgap", "value": 3.4}, {"formula": "AlN", "property": "bandgap", "value": 6.2}]},
        {"id": "b", "records": []},
    ]
    preds = {
        "a": [{"formula": "GaN", "property": "bandgap", "value": 3.41},      # within 2%: true positive
              {"formula": "AlN", "property": "bandgap", "value": 62.0}],     # right material, wrong value (unit slip)
        "b": [{"formula": "X", "property": "density", "value": 1.0}],         # false alarm on a negative passage
    }
    out = ee.score(gold, preds)
    assert out["overall"]["tp"] == 1 and out["overall"]["fp"] == 2 and out["overall"]["fn"] == 1
    assert out["value_accuracy"] == 0.5 and out["negative_items_with_false_alarms"] == 1


# ---- the extractors against the gold set ------------------------------------------------------------


class NoLLM:
    available = False


class NoDB:
    online = False

    def get_properties(self, formula):
        return []


def test_the_heuristic_extractor_clears_a_quality_floor_and_never_fires_on_negative_passages() -> None:
    """A regression guard: the rule-based fallback must stay at least this good (measured: P 0.90, R 0.58, F1 0.71)."""
    gold = ee.load_gold(GOLD)
    records = asyncio.run(ExtractionAgent(NoLLM(), NoDB()).extract_chunks(ee.as_chunks(gold), []))
    result = ee.score(gold, ee.records_by_item(records))

    assert result["overall"]["precision"] >= 0.85 and result["overall"]["recall"] >= 0.5
    assert result["negative_items_with_false_alarms"] == 0
    assert result["value_accuracy"] >= 0.9


def test_a_gold_record_accepts_its_listed_aliases() -> None:
    gold = [{"id": "a", "records": [{"formula": "LLZO", "property": "ionic_conductivity", "value": 9e-4, "also": ["Li7La3Zr2O12"]}]}]
    for name in ("LLZO", "Li7La3Zr2O12"):
        out = ee.score(gold, {"a": [{"formula": name, "property": "ionic_conductivity", "value": 9e-4}]})
        assert out["overall"]["tp"] == 1 and out["overall"]["fp"] == 0
    wrong = ee.score(gold, {"a": [{"formula": "GaN", "property": "ionic_conductivity", "value": 9e-4}]})
    assert wrong["overall"]["tp"] == 0
