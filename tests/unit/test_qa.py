"""Grounded follow-up Q&A over a finished report, and the in-memory report store."""

from __future__ import annotations

import asyncio

import pytest

from matsci_agent.agents.nodes.qa import QAAnswer, ReportQA, build_context, rank_context
from matsci_agent.infrastructure.llm.anthropic_client import LLMError
from matsci_agent.services.report_store import ReportStore

REPORT = {
    "query": "Find wide band gap materials above 5 eV",
    "search_results": [
        {"text": "AlN is an ultrawide band gap semiconductor with a gap of 6.2 eV.", "metadata": {"title": "AlN review", "year": 2023, "arxiv_id": "2301.00001"}},
        {"text": "Perovskite solar cells reach 25 percent efficiency.", "metadata": {"title": "PV paper", "year": 2022, "arxiv_id": "2201.00002"}},
    ],
    "hypotheses": [{"statement": "AlN qualifies because its gap exceeds 5 eV.", "confidence": "high", "evidence": [{"id": "E1"}], "suggested_experiment": "UV-Vis"}],
    "candidates": [{"formula": "AlN", "status": "qualified", "origin": "literature",
                    "criteria": [{"label": "Bandgap > 5 eV", "status": "met", "value": 6.2, "unit": "eV"}]}],
    "extracted_properties": [
        {"material_formula": "AlN", "material_name": "AlN", "property_name": "bandgap", "value_raw": "6.2 eV", "method": "literature", "source_title": "AlN review", "source_id": "2301.00001", "is_predicted": False},
    ],
    "knowledge_gaps": ["No measured stability data for AlN."],
    "criteria": [{"label": "Bandgap > 5 eV"}],
}


class ScriptedLLM:
    available = True

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.prompts = answer, error, []

    async def structured(self, prompt, schema, **kw):
        self.prompts.append((prompt, kw.get("system")))
        if self.error:
            raise self.error
        return self.answer


def ask(qa, question="What is the band gap of AlN?", history=None):
    return asyncio.run(qa.ask(REPORT, question, history))


def test_the_context_covers_papers_hypotheses_candidates_values_and_gaps() -> None:
    ctx = build_context(REPORT)
    assert {"P1", "P2", "H1", "C1", "R1", "G1", "K1", "Q1"} <= set(ctx)
    assert "6.2 eV" in ctx["P1"] and "qualified" in ctx["C1"] and "AlN: bandgap = 6.2 eV" in ctx["R1"]


def test_context_is_ranked_by_overlap_with_the_question() -> None:
    top = list(rank_context(build_context(REPORT), "band gap AlN", limit=3))
    assert "P1" in top and "P2" not in top


def test_an_answer_with_real_citations_is_grounded_and_citations_carry_their_text() -> None:
    llm = ScriptedLLM(QAAnswer(answer="AlN has a 6.2 eV gap [P1].", citations=["P1", "P1", "H1"], insufficient_evidence=False))
    out = ask(ReportQA(llm))

    assert out["grounded"] and out["mode"] == "llm" and [c["id"] for c in out["citations"]] == ["P1", "H1"]
    assert "6.2 eV" in out["citations"][0]["text"]
    prompt, system = llm.prompts[0]
    assert "<document>" in prompt and "untrusted data" in system  # evidence is fenced as data


def test_invented_citation_ids_are_dropped_and_an_uncited_answer_is_not_grounded() -> None:
    out = ask(ReportQA(ScriptedLLM(QAAnswer(answer="It is 9 eV [Z9].", citations=["Z9"], insufficient_evidence=False))))
    assert out["citations"] == [] and out["grounded"] is False and out["insufficient_evidence"] is True


def test_the_model_can_say_the_evidence_is_insufficient() -> None:
    out = ask(ReportQA(ScriptedLLM(QAAnswer(answer="The report does not say.", citations=["Q1"], insufficient_evidence=True))))
    assert out["grounded"] is False and out["insufficient_evidence"] is True


def test_recent_turns_are_passed_back_as_conversation_memory() -> None:
    llm = ScriptedLLM(QAAnswer(answer="ok [P1]", citations=["P1"], insufficient_evidence=False))
    history = [{"role": "user", "content": "Which material is best?"}, {"role": "assistant", "content": "AlN."}, {"role": "system", "content": "ignored"}]
    ask(ReportQA(llm), "And its stability?", history)
    prompt = llm.prompts[0][0]
    assert "User: Which material is best?" in prompt and "Assistant: AlN." in prompt and "ignored" not in prompt


def test_without_an_llm_the_best_matching_passages_are_returned_extractively() -> None:
    class NoLLM:
        available = False

    out = ask(ReportQA(NoLLM()))
    assert out["mode"] == "extractive" and out["grounded"] is False and len(out["citations"]) == 3


def test_an_llm_error_falls_back_to_extractive_mode() -> None:
    assert ask(ReportQA(ScriptedLLM(error=LLMError("down"))))["mode"] == "extractive"


def test_the_report_store_is_a_bounded_lru() -> None:
    store = ReportStore(capacity=2)
    store.put("a", {"n": 1})
    store.put("b", {"n": 2})
    assert store.get("a") == {"n": 1}  # touching a makes b the oldest
    store.put("c", {"n": 3})
    assert store.get("b") is None and store.get("a") and store.get("c")


@pytest.mark.parametrize("question", ["ab", "x" * 501])
def test_the_ask_endpoint_validates_the_question(question) -> None:
    from fastapi.testclient import TestClient

    from matsci_agent.api.server import app

    client = TestClient(app)
    assert client.post("/api/runs/whatever/ask", json={"question": question}).status_code == 422
    assert client.post("/api/runs/unknown-run/ask", json={"question": "valid question?"}).status_code in (404, 503)
