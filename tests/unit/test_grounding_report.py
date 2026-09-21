"""Grounding checker, citations, and report builder."""

from __future__ import annotations

import pytest

from matsci_agent.agents.nodes.report import ReportAgent, build_citations
from matsci_agent.domain.state import initial_state
from matsci_agent.infrastructure.llm.anthropic_client import LLMError
from matsci_agent.services.grounding import GroundingChecker, Verdict, Verdicts
from matsci_agent.services.report_builder import build_report


class FakeLLM:
    def __init__(self, available=True, verdicts=None, error=None):
        self.available, self.verdicts, self.error, self.prompt = available, verdicts or [], error, ""

    async def structured(self, prompt, schema, **kwargs):
        self.prompt = prompt
        if self.error:
            raise self.error
        return Verdicts(verdicts=self.verdicts)


def ev(snippet, source_id="p1", title="Paper", doi=""):
    return {"id": "E1", "kind": "chunk", "ref": "c", "title": title, "doi": doi, "source_id": source_id, "snippet": snippet}


def hyp(statement, evidence):
    return {"statement": statement, "evidence": evidence, "confidence": "medium", "suggested_experiment": "x", "origin": "llm"}


# ---- grounding -------------------------------------------------------------------------------


async def test_rules_flag_uncited_hypotheses_and_missing_figures() -> None:
    hyps = [
        hyp("Garnets conduct well.", []),
        hyp("LLZO reaches 1.2 mS/cm and a 6.0 V window.", [ev("LLZO: conductivity = 1.2 mS/cm")]),
        hyp("LLZO reaches 1.2 mS/cm.", [ev("LLZO: conductivity = 1.2 mS/cm")]),
    ]
    checked, summary, warnings = await GroundingChecker(FakeLLM(available=False)).check(hyps)

    verdicts = [c["verdict"] for c in summary["claims"]]
    assert verdicts == ["unsupported", "partial", "supported"]
    assert [h["grounded"] for h in checked] == [False, True, True]
    assert "6.0" in summary["claims"][1]["reason"]
    assert summary["is_grounded"] is False and summary["unsupported"] == 1 and summary["method"] == "rules"
    assert summary["score"] == pytest.approx(0.5)
    assert len(warnings) == 1 and "Hypothesis 1" in warnings[0]


async def test_counts_and_years_are_not_mistaken_for_claims() -> None:
    _, summary, _ = await GroundingChecker(FakeLLM(available=False)).check(
        [hyp("Three of the 12 candidates share oxygen, as reported in 2024.", [ev("anything")])]
    )
    assert summary["claims"][0]["verdict"] == "supported"


async def test_llm_can_only_make_the_verdict_stricter() -> None:
    hyps = [
        hyp("A claim.", [ev("evidence one")]),
        hyp("Another.", [ev("evidence two")]),
    ]
    llm = FakeLLM(verdicts=[
        Verdict(index=0, verdict="unsupported", reason="evidence says the opposite"),
        Verdict(index=1, verdict="supported", reason="fine"),
    ])
    _, summary, warnings = await GroundingChecker(llm).check(hyps)
    assert [c["verdict"] for c in summary["claims"]] == ["unsupported", "supported"]
    assert "the opposite" in summary["claims"][0]["reason"]
    assert summary["method"] == "llm+rules" and warnings
    assert "evidence one" in llm.prompt


async def test_llm_failure_falls_back_to_rules_with_a_warning() -> None:
    _, summary, warnings = await GroundingChecker(FakeLLM(error=LLMError("down"))).check(
        [hyp("Fine.", [ev("x")])]
    )
    assert summary["method"] == "rules" and summary["is_grounded"]
    assert any("rule checks only" in w for w in warnings)


async def test_no_hypotheses_is_vacuously_grounded() -> None:
    _, summary, warnings = await GroundingChecker(FakeLLM(available=False)).check([])
    assert summary["is_grounded"] and summary["checked"] == 0 and warnings == []


# ---- citations ---------------------------------------------------------------------------------


def state_for_citations():
    state = initial_state("goal")
    state["retrieved_chunks"] = [{
        "id": "2401.1_0", "text": "t",
        "metadata": {"arxiv_id": "2401.1", "title": "Garnets", "authors": "A, B, C", "year": 2024},
    }]
    state["retrieved_papers"] = state["retrieved_chunks"]
    state["hypotheses"] = [
        {"statement": "s", "evidence": [ev("x", "2401.1", "Garnets"), ev("y", "mp-1", "Materials Project", "10.1063/1.4812323")]},
        {"statement": "s2", "evidence": [ev("z", "2401.1", "Garnets")]},
    ]
    state["extracted_properties"] = [
        {"method": "offline reference", "source_id": "offline-reference", "source_title": "Offline", "source_doi": "", "material_formula": "GaN"},
        {"method": "Kernel Ridge Regression (RBF kernel)", "source_id": "", "source_title": "KRR prediction", "source_doi": "", "material_formula": "GaN"},
    ]
    return state


def test_citations_are_deduplicated_linked_and_track_who_cites_them() -> None:
    cites = {c["source_id"]: c for c in build_citations(state_for_citations())}
    assert set(cites) == {"2401.1", "mp-1"}  # offline table and KRR predictions are not citations
    garnets = cites["2401.1"]
    assert garnets["url"] == "https://arxiv.org/abs/2401.1" and garnets["authors"] == "A, B, C"
    assert garnets["year"] == 2024 and garnets["cited_by_hypotheses"] == [1, 2]
    assert cites["mp-1"]["url"] == "https://doi.org/10.1063/1.4812323"
    assert [c["id"] for c in build_citations(state_for_citations())] == ["C1", "C2"]


async def test_report_agent_emits_grounding_event_and_updates_state() -> None:
    seen = []

    async def listener(event):
        seen.append(event)

    out = await ReportAgent(GroundingChecker(FakeLLM(available=False))).run(
        state_for_citations(), {"configurable": {"emit": listener}}
    )
    assert seen == [{"event": "grounding_check", "is_grounded": True, "score": 1.0}]
    assert out["grounding"]["checked"] == 2 and len(out["citations"]) == 2
    assert all("grounded" in h for h in out["hypotheses"])


# ---- report builder ------------------------------------------------------------------------------


def rec(formula, prop, value, unit, method="literature", predicted=False, **extra):
    return {
        "material_name": formula, "material_formula": formula, "property_name": prop,
        "value_raw": f"{value} {unit}", "value_numeric": value, "unit": unit, "source_title": "Src",
        "source_doi": "", "source_id": "id", "confidence": "medium", "method": method,
        "is_predicted": predicted, **extra,
    }


def test_report_keeps_legacy_fields_and_adds_the_structured_report() -> None:
    state = initial_state("bandgap above 3 eV")
    state.update(
        retrieved_papers=[{
            "id": "p_0", "text": "text", "similarity": None, "relevance_score": 0.8,
            "metadata": {"arxiv_id": "p", "title": "T", "authors": "A", "year": 2024, "section": "Results"},
        }],
        screened_candidates=[{"formula": "GaN", "name": "GaN", "status": "qualified", "rank": 1, "criteria": [], "sources": []}],
        extracted_properties=[
            rec("GaN", "bandgap", 3.4, "eV"),
            rec("GaN", "bandgap", 1.9, "eV", "Materials Project (DFT)", is_direct_gap=True),
            rec("GaN", "bandgap", 3.3, "eV", "Kernel Ridge Regression (RBF kernel)", True, uncertainty=0.4),
            rec("GaN", "ionic_conductivity", 1e-3, "S/cm"),  # not a chart property
            rec("Unlisted", "bandgap", 9.9, "eV"),  # not a screened candidate
        ],
        warnings=["a", "b", "a"], agent_log=[{"step": "S", "details": "d"}], retry_count=2,
        hypotheses=[{"statement": "Top hypothesis", "evidence": []}],
    )
    report = build_report(state, session_id="sid", query="q", max_results=5, mode={"llm": None}, duration_ms=12)

    assert report["session_id"] == "sid" and report["status"] == "COMPLETED"
    assert report["candidate_formulas"] == ["GaN"]
    rows = report["property_records"]["GaN"]
    assert [r["method"] for r in rows] == ["Materials Project (DFT)", "literature"]  # database first
    assert rows[0]["is_direct_gap"] is True and all(r["property_name"] == "bandgap" for r in rows)
    pred = report["krr_predictions"]["GaN"]
    assert (pred["property"], pred["predicted_value"], pred["uncertainty"]) == ("band_gap", 3.3, 0.4)
    assert "±0.4" in pred["note"]
    assert report["search_results"][0]["similarity"] == 0.8  # falls back to relevance when no cosine score
    assert report["warnings"] == ["a", "b"] and report["retrieval_attempts"] == 3
    assert report["hypothesis"] == "Top hypothesis"
    assert report["agent_logs"] == [{"step": "S", "details": "d"}] and report["duration_ms"] == 12


def test_headline_summarises_screening_when_there_are_no_hypotheses() -> None:
    state = initial_state("g")
    state["screened_candidates"] = [{"formula": "A", "name": "A", "status": "partial", "criteria": [], "sources": []}]
    report = build_report(state, session_id="s", query="g", max_results=5, mode={}, duration_ms=0)
    assert "1 candidate(s) were screened" in report["hypothesis"] and "0 fully qualified" in report["hypothesis"]
