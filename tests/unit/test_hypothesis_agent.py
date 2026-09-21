"""Unit tests for HypothesisAgent (fake LLM)."""

from __future__ import annotations

from matsci_agent.agents.nodes.hypothesis import HypothesisAgent, HypothesisDraft, HypothesisOutput
from matsci_agent.domain.state import initial_state
from matsci_agent.infrastructure.llm.anthropic_client import LLMError


class FakeLLM:
    def __init__(self, available=True, output=None, error=None):
        self.available, self.output, self.error, self.prompt = available, output, error, ""

    async def structured(self, prompt, schema, **kwargs):
        self.prompt = prompt
        if self.error:
            raise self.error
        return self.output


def candidate(rank, formula, status="partial", criteria=None):
    return {
        "rank": rank, "formula": formula, "name": formula, "status": status, "score": 0.5,
        "origin": "literature", "flags": [], "sources": [], "record_count": 1,
        "criteria": criteria or [{
            "label": "Bandgap > 3 eV", "kind": "numeric", "status": "met", "value": 3.4,
            "unit": "eV", "is_predicted": False, "source": "Paper", "detail": "",
        }],
    }


def record(formula, value, method="literature", predicted=False):
    return {
        "material_name": formula, "material_formula": formula, "property_name": "bandgap",
        "value_raw": f"{value} eV", "value_numeric": value, "unit": "eV", "source_title": f"Paper on {formula}",
        "source_doi": "10.1/x", "source_id": "p1", "chunk_id": "p1_0", "method": method, "is_predicted": predicted,
    }


def chunk(cid="p1_0"):
    return {"id": cid, "text": "GaN has a wide band gap.", "metadata": {"title": "Wide gap", "doi": "10.1/x", "arxiv_id": "p1"}}


def state(candidates, records=None, criteria=None):
    s = initial_state("bandgap above 3 eV")
    s.update(
        screened_candidates=candidates, extracted_properties=records or [],
        retrieved_papers=[chunk()], criteria=criteria or [{"kind": "numeric", "label": "Bandgap > 3 eV", "property_name": "bandgap"}],
    )
    return s


async def test_llm_hypotheses_resolve_cited_evidence_and_flag_unknown_ids() -> None:
    output = HypothesisOutput(
        hypotheses=[
            HypothesisDraft(statement="Nitrides win", evidence_ids=["E1", "R1", "E99"], suggested_experiment="XRD", confidence="high"),
            HypothesisDraft(statement="Made up", evidence_ids=["E42"], suggested_experiment="?", confidence="high"),
        ],
        knowledge_gaps=["Air stability is unknown"],
    )
    llm = FakeLLM(output=output)
    out = await HypothesisAgent(llm).run(state([candidate(1, "GaN")], [record("GaN", 3.4)]))

    good, bad = out["hypotheses"]
    assert [e["id"] for e in good["evidence"]] == ["E1", "R1"]
    assert good["unsupported_ids"] == ["E99"] and good["confidence"] == "high" and not good["ungrounded"]
    assert bad["ungrounded"] and bad["confidence"] == "low"  # no valid citation: confidence capped
    assert "Air stability is unknown" in out["knowledge_gaps"]
    assert "E1:" in llm.prompt and "R1:" in llm.prompt  # evidence ids are offered to the model


async def test_scarcity_instruction_is_sent_when_fewer_than_three_candidates_qualify() -> None:
    llm = FakeLLM(output=HypothesisOutput(hypotheses=[], knowledge_gaps=[]))
    await HypothesisAgent(llm).run(state([candidate(1, "GaN", "qualified")]))
    assert "scarce" in llm.prompt


async def test_llm_error_falls_back_to_templates_with_a_warning() -> None:
    out = await HypothesisAgent(FakeLLM(error=LLMError("down"))).run(
        state([candidate(1, "GaN"), candidate(2, "AlN")], [record("GaN", 3.4), record("AlN", 6.2)])
    )
    assert any("fell back to templates" in w for w in out["warnings"])
    assert out["hypotheses"] and all(h["origin"] == "template" and h["confidence"] == "low" for h in out["hypotheses"])


async def test_template_finds_shared_chemistry_and_cites_records() -> None:
    out = await HypothesisAgent(FakeLLM(available=False)).run(
        state([candidate(1, "ZnO"), candidate(2, "TiO2")], [record("ZnO", 3.37), record("TiO2", 3.2)])
    )
    first = out["hypotheses"][0]
    assert "oxides" in first["statement"] and "ZnO" in first["statement"]
    assert first["evidence"] and not first["ungrounded"]
    assert "UV-Vis" in first["suggested_experiment"]
    assert any("template-based" in w for w in out["warnings"])


async def test_template_flags_predicted_or_missing_data_for_the_top_candidate() -> None:
    weak = candidate(1, "ZnO", criteria=[{
        "label": "Bandgap > 3 eV", "kind": "numeric", "status": "predicted_met", "value": 3.3,
        "unit": "eV", "is_predicted": True, "source": "KRR prediction", "detail": "",
    }])
    out = await HypothesisAgent(FakeLLM(available=False)).run(state([weak], [record("ZnO", 3.3, "Kernel Ridge Regression (RBF kernel)", True)]))
    assert any("only estimated by machine learning" in h["statement"] for h in out["hypotheses"])


async def test_knowledge_gaps_cover_missing_predicted_and_qualitative_items() -> None:
    cands = [candidate(1, "ZnO", criteria=[
        {"label": "Conductivity", "kind": "numeric", "status": "missing", "value": None, "unit": "", "is_predicted": False, "source": "", "detail": ""},
        {"label": "Bandgap", "kind": "numeric", "status": "predicted_met", "value": 3.3, "unit": "eV", "is_predicted": True, "source": "", "detail": ""},
    ])]
    crit = [{"kind": "qualitative", "label": "Air stable"}]
    out = await HypothesisAgent(FakeLLM(available=False)).run(state(cands, criteria=crit))
    gaps = " | ".join(out["knowledge_gaps"])
    assert "No measured value for 'Conductivity' was found for ZnO" in gaps
    assert "rests on a KRR estimate" in gaps
    assert "'Air stable' cannot be verified" in gaps


async def test_database_only_candidates_are_flagged_as_unconfirmed() -> None:
    cand = {**candidate(1, "AlN", "qualified"), "origin": "database"}
    out = await HypothesisAgent(FakeLLM(available=False)).run(state([cand], [record("AlN", 6.2, "offline reference")]))
    assert any("no retrieved paper reports it" in h["statement"] for h in out["hypotheses"])
    assert any("came from a database screen only" in g for g in out["knowledge_gaps"])
    assert all(h["evidence"] for h in out["hypotheses"])  # cites the database record


async def test_no_candidates_reports_a_gap_and_does_not_crash() -> None:
    out = await HypothesisAgent(FakeLLM(available=True, output=HypothesisOutput(hypotheses=[], knowledge_gaps=[]))).run(state([]))
    assert out["hypotheses"] == []
    assert any("No materials with reported values" in g for g in out["knowledge_gaps"])
