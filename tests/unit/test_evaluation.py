"""Evaluation metrics, the comparison runner, and the shipped dataset (fake LLM, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from matsci_agent.evaluation import metrics
from matsci_agent.evaluation.runner import (
    METRICS, compare, evaluate_question, load_questions, markdown_table, summarise,
)

DATASET = Path(__file__).resolve().parents[2] / "evaluation" / "eval_dataset.json"


class JudgeLLM:
    """Returns canned judgements by schema type and records the prompts it was given."""

    model = "fake-judge"

    def __init__(self, **canned):
        self.canned, self.prompts = canned, []

    async def structured(self, prompt, schema, **kwargs):
        self.prompts.append((schema.__name__, prompt))
        return self.canned[schema.__name__]


def claims(*flags):
    return metrics.Claims(claims=[metrics.ClaimCheck(claim=f"c{i}", supported=f) for i, f in enumerate(flags)])


# ---- metrics --------------------------------------------------------------------------------


def test_average_precision_rewards_relevant_contexts_ranked_first() -> None:
    assert metrics.average_precision([True, False, True]) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert metrics.average_precision([False, False, True]) == pytest.approx(1 / 3)
    assert metrics.average_precision([True, True, False]) == pytest.approx(1.0)
    assert metrics.average_precision([False, False]) == 0.0
    assert metrics.average_precision([]) is None


async def test_faithfulness_is_the_supported_share_of_claims() -> None:
    llm = JudgeLLM(Claims=claims(True, True, False, True))
    assert await metrics.faithfulness(llm, "answer", ["ctx"]) == pytest.approx(0.75)
    assert await metrics.faithfulness(JudgeLLM(Claims=claims()), "answer", ["ctx"]) is None  # nothing to judge
    assert await metrics.faithfulness(llm, "answer", []) is None  # no context to be faithful to


async def test_context_recall_is_the_attributable_share_and_zero_without_context() -> None:
    stmts = metrics.Statements(statements=[
        metrics.StatementCheck(statement="a", attributable=True),
        metrics.StatementCheck(statement="b", attributable=False),
    ])
    assert await metrics.context_recall(JudgeLLM(Statements=stmts), "gt", ["ctx"]) == 0.5
    assert await metrics.context_recall(JudgeLLM(), "gt", []) == 0.0  # retrieving nothing recalls nothing


async def test_context_precision_tolerates_a_misaligned_judge_reply() -> None:
    llm = JudgeLLM(Relevance=metrics.Relevance(relevant=[True]))  # judge returned 1 flag for 3 contexts
    assert await metrics.context_precision(llm, "q", "gt", ["a", "b", "c"]) == pytest.approx(1.0)
    assert await metrics.context_precision(llm, "q", "gt", []) is None


async def test_answer_relevancy_is_clamped_to_the_unit_interval() -> None:
    assert await metrics.answer_relevancy(JudgeLLM(Score=metrics.Score(score=1.7)), "q", "a") == 1.0
    assert await metrics.answer_relevancy(JudgeLLM(Score=metrics.Score(score=-0.2)), "q", "a") == 0.0


async def test_no_context_means_the_answer_admits_it_cannot_answer_without_calling_the_llm() -> None:
    llm = JudgeLLM()
    assert "cannot determine" in await metrics.generate_answer(llm, "q", [])
    assert llm.prompts == []


async def test_answers_are_generated_from_the_contexts_only() -> None:
    llm = JudgeLLM(Answer=metrics.Answer(answer="3.4 eV"))
    assert await metrics.generate_answer(llm, "GaN gap?", ["GaN has a band gap of 3.4 eV"]) == "3.4 eV"
    assert "ONLY the context" in llm.prompts[0][1]


# ---- runner ---------------------------------------------------------------------------------------


def full_judge() -> JudgeLLM:
    return JudgeLLM(
        Answer=metrics.Answer(answer="an answer"), Claims=claims(True, False), Score=metrics.Score(score=0.8),
        Relevance=metrics.Relevance(relevant=[True, False]),
        Statements=metrics.Statements(statements=[metrics.StatementCheck(statement="s", attributable=True)]),
    )


QUESTION = {"id": "q1", "question": "What is the GaN gap?", "ground_truth": "3.4 eV"}


async def test_evaluate_question_returns_all_four_metrics() -> None:
    row = await evaluate_question(full_judge(), QUESTION, ["ctx a", "ctx b"])
    assert row["faithfulness"] == 0.5 and row["answer_relevancy"] == 0.8
    assert row["context_precision"] == 1.0 and row["context_recall"] == 1.0 and row["n_contexts"] == 2


def test_summarise_ignores_metrics_that_could_not_be_computed() -> None:
    rows = [
        {"faithfulness": 1.0, "answer_relevancy": None, "context_precision": 0.5, "context_recall": 0.0},
        {"faithfulness": 0.5, "answer_relevancy": None, "context_precision": None, "context_recall": 1.0},
    ]
    assert summarise(rows) == {"faithfulness": 0.75, "answer_relevancy": None, "context_precision": 0.5, "context_recall": 0.5}


async def test_compare_scores_each_system_and_reports_the_difference() -> None:
    async def strong(question):
        return ["a", "b"]

    async def empty(question):
        return []

    seen = []
    report = await compare(
        full_judge(), [QUESTION, {**QUESTION, "id": "q2"}], {"plain": empty, "agentic": strong},
        on_progress=lambda system, qid: seen.append((system, qid)),
    )
    assert report["n_questions"] == 2 and report["answer_and_judge_model"] == "fake-judge"
    assert report["systems"]["plain"]["summary"]["context_recall"] == 0.0  # nothing retrieved
    assert report["systems"]["agentic"]["summary"]["context_recall"] == 1.0
    assert report["delta_vs_first"]["context_recall"] == 1.0
    assert len(seen) == 4 and "not the ragas library" in report["note"]
    assert json.loads(json.dumps(report))["systems"]["agentic"]["questions"][0]["id"] == "q1"  # serialisable


def test_markdown_table_lists_every_metric_with_signed_changes() -> None:
    summary = {m: 0.5 for m in METRICS}
    report = {
        "systems": {"plain": {"summary": summary}, "agentic": {"summary": {**summary, "faithfulness": None}}},
        "delta_vs_first": {**{m: 0.1 for m in METRICS}, "faithfulness": None},
    }
    table = markdown_table(report)
    assert table.count("\n") == len(METRICS) + 1
    assert "| faithfulness | 0.500 | n/a | n/a |" in table and "| context_recall | 0.500 | 0.500 | +0.100 |" in table


# ---- dataset -------------------------------------------------------------------------------------


def test_shipped_dataset_has_20_unique_complete_questions() -> None:
    questions = load_questions(DATASET)
    assert len(questions) == 20
    assert len({q["id"] for q in questions}) == 20
    assert all(q["question"].endswith("?") and len(q["ground_truth"]) > 10 for q in questions)


def test_load_questions_rejects_duplicates_and_incomplete_entries(tmp_path) -> None:
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps({"questions": [QUESTION, QUESTION]}))
    with pytest.raises(ValueError, match="Duplicate"):
        load_questions(dup)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "x", "question": "q?"}]))
    with pytest.raises(ValueError, match="ground_truth"):
        load_questions(bad)
