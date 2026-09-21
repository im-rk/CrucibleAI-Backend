"""End-to-end pipeline with the REAL agents, graph and service.

Only the outside world is faked: retrieval returns a small in-memory corpus, and no LLM or
Materials Project key is configured, so this exercises the documented degraded ("heuristic")
mode, including the re-retrieval loop, from goal to grounded report.
"""

from __future__ import annotations

import pytest

from matsci_agent.agents.nodes.extraction import ExtractionAgent
from matsci_agent.agents.nodes.hypothesis import HypothesisAgent
from matsci_agent.agents.nodes.report import ReportAgent
from matsci_agent.agents.nodes.screening import ScreeningAgent
from matsci_agent.agents.nodes.search import SearchAgent
from matsci_agent.agents.orchestrator import build_research_graph
from matsci_agent.domain.schemas import ResearchRequest
from matsci_agent.infrastructure.clients.mp_api import MaterialsProjectClient
from matsci_agent.infrastructure.llm.anthropic_client import LLMClient
from matsci_agent.infrastructure.ml.krr import KRRPropertyPredictor
from matsci_agent.services.grounding import GroundingChecker
from matsci_agent.services.research_srv import ResearchService

GOAL = (
    "Find solid-state electrolyte materials with ionic conductivity above 10⁻³ S/cm and "
    "electrochemical stability above 5V, with no rare earth elements"
)


def paper(arxiv_id: str, title: str, text: str, sim: float) -> dict:
    return {
        "id": f"{arxiv_id}_0", "text": text, "similarity": sim, "rrf_score": 0.03,
        "metadata": {"arxiv_id": arxiv_id, "title": title, "authors": "A. Author", "year": 2024, "section": "Results"},
    }


CORPUS = [
    paper("2401.00001", "Garnet electrolytes",
          "The garnet solid electrolyte Li7La3Zr2O12 reaches an ionic conductivity of 1.2 × 10⁻³ S/cm at room "
          "temperature. Its electrochemical stability window is 6.0 V.", 0.62),
    paper("2401.00002", "Superionic sulfides",
          "The solid electrolyte Li10GeP2S12 exhibits an ionic conductivity of 1.2 × 10⁻² S/cm. Its "
          "electrochemical stability window is 5.5 V.", 0.6),
    paper("2401.00003", "Thiophosphates",
          "The solid electrolyte Li3PS4 shows an ionic conductivity of 1 × 10⁻⁴ S/cm and an electrochemical "
          "stability window of 5.2 V.", 0.58),
    paper("2401.00004", "Baking", "Sourdough bread needs a long, cool fermentation to develop flavour.", 0.16),
]


class InMemorySearch:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, n_results):
        self.queries.append(query)
        return [dict(c) for c in CORPUS]


@pytest.fixture
def service(tmp_path):
    llm = LLMClient(api_key=None, model="unused")
    search = InMemorySearch()
    krr = KRRPropertyPredictor(model_dir=tmp_path)
    krr._featurizer_failed = True
    mp = MaterialsProjectClient(api_key=None)
    graph = build_research_graph(
        SearchAgent(llm, search, arxiv=None, live_arxiv=False),
        ExtractionAgent(llm, mp),
        ScreeningAgent(llm, krr, mp),
        HypothesisAgent(llm),
        ReportAgent(GroundingChecker(llm)),
    )
    svc = ResearchService(graph, lambda: {"llm": None, "materials_project": "offline"}, timeout_s=120)
    svc.search = search
    return svc


async def collect(service, goal=GOAL):
    return [e async for e in service.stream(ResearchRequest(query=goal, max_results=10))]


async def test_full_pipeline_produces_a_grounded_ranked_report(service) -> None:
    events = await collect(service)
    report = events[-1]["report"]

    # The stream tells the story in order.
    names = [e["event"] for e in events]
    assert names[0] == "session_started" and names[-1] == "complete"
    started = [e["agent"] for e in events if e["event"] == "agent_start"]
    assert started[0] == "SearchAgent" and started[-4:] == [
        "ExtractionAgent", "ScreeningAgent", "HypothesisAgent", "ReportBuilder",
    ]
    assert names.count("re_retrieving") == 3  # weak results (< 5 relevant) -> 3 retries, then move on
    assert "candidates_found" in names and "grounding_check" in names and "hypotheses_generated" in names

    # Screening did the real work on the extracted numbers.
    by_formula = {c["formula"]: c for c in report["candidates"]}
    assert by_formula["Li10GeP2S12"]["status"] == "qualified"
    assert by_formula["Li7La3Zr2O12"]["status"] == "rejected"  # numbers pass, but contains lanthanum
    assert by_formula["Li3PS4"]["status"] == "rejected"  # conductivity 1e-4 < 1e-3
    assert report["candidates"][0]["formula"] == "Li10GeP2S12" and report["candidates"][0]["rank"] == 1
    conductivity = next(r for r in by_formula["Li10GeP2S12"]["criteria"] if r["kind"] == "numeric")
    assert conductivity["status"] == "met" and conductivity["value"] == pytest.approx(1.2e-2)

    # The irrelevant passage was filtered out before extraction.
    relevant_ids = {c["id"] for c in report["search_results"]}
    assert "2401.00004_0" not in relevant_ids and len(relevant_ids) == 3
    assert report["retrieval_attempts"] == 4

    # Honest about running without an LLM / MP key.
    assert report["mode"] == {"llm": None, "materials_project": "offline"}
    assert any("LLM not configured" in w for w in report["warnings"])
    assert len(report["warnings"]) == len(set(report["warnings"]))  # deduplicated

    # Hypotheses exist, are cited, and passed the grounding check.
    assert report["hypotheses"] and all(h["evidence"] for h in report["hypotheses"])
    assert report["grounding"]["is_grounded"] and report["grounding"]["method"] == "rules"
    assert report["hypothesis"] == report["hypotheses"][0]["statement"]

    # Citations resolve to real links; legacy fields the first UI reads are still present.
    urls = {c["url"] for c in report["citations"]}
    assert "https://arxiv.org/abs/2401.00002" in urls
    assert set(report["candidate_formulas"]) == set(by_formula)
    assert set(report["property_records"]) == set(by_formula)
    assert report["agent_logs"] and all("agent" in log and "duration_ms" in log for log in report["agent_logs"])


async def test_run_returns_the_same_report_without_streaming(service) -> None:
    report = await service.run(ResearchRequest(query=GOAL))
    assert report["status"] == "COMPLETED" and report["candidates"]


async def test_pipeline_survives_an_agent_crash_and_says_so(service, monkeypatch, tmp_path) -> None:
    async def boom(state, config=None):
        raise RuntimeError("extractor exploded")

    # Rebuild the graph with a broken extraction agent.
    llm = LLMClient(api_key=None, model="unused")
    broken = ExtractionAgent(llm, MaterialsProjectClient(api_key=None))
    monkeypatch.setattr(broken, "run", boom)
    krr = KRRPropertyPredictor(model_dir=tmp_path)
    service.graph = build_research_graph(
        SearchAgent(llm, service.search, arxiv=None, live_arxiv=False),
        broken, ScreeningAgent(llm, krr, MaterialsProjectClient(api_key=None)),
        HypothesisAgent(llm), ReportAgent(GroundingChecker(llm)),
    )
    events = await collect(service)
    report = events[-1]["report"]

    assert events[-1]["event"] == "complete"  # the run still finishes
    assert any(e["event"] == "agent_error" and e["agent"] == "ExtractionAgent" for e in events)
    assert any("ExtractionAgent failed and was skipped" in w for w in report["warnings"])
    assert report["candidates"] == []
    assert any("No materials with reported values" in g for g in report["knowledge_gaps"])
