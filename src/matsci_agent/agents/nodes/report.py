"""Report node: grounding check plus a deduplicated citation list for everything used."""

from __future__ import annotations

from matsci_agent.agents.events import emit
from matsci_agent.domain.state import ResearchState
from matsci_agent.services.grounding import GroundingChecker


def _doi_url(doi: str) -> str:
    return doi if doi.startswith("http") else f"https://doi.org/{doi}"


def build_citations(state: ResearchState) -> list[dict]:
    """Every source that backs a hypothesis, candidate, or extracted value, cited-first."""
    meta_by_paper = {
        c["metadata"].get("arxiv_id"): c["metadata"] for c in state.get("retrieved_chunks", [])
    }
    citations: dict[str, dict] = {}

    def add(source_id: str, title: str, doi: str, *, hypothesis: int | None = None, formula: str | None = None) -> None:
        if not (source_id or title) or "offline" in source_id or title == "KRR prediction":
            return
        entry = citations.get(source_id or title)
        if entry is None:
            meta = meta_by_paper.get(source_id, {})
            is_arxiv = source_id in meta_by_paper or source_id[:2].isdigit()
            entry = {
                "id": f"C{len(citations) + 1}",
                "source_id": source_id,
                "title": meta.get("title") or title or source_id,
                "authors": meta.get("authors", ""),
                "year": meta.get("year"),
                "doi": doi,
                "url": _doi_url(doi) if doi else (f"https://arxiv.org/abs/{source_id}" if is_arxiv else ""),
                "cited_by_hypotheses": [],
                "supports_candidates": [],
            }
            citations[source_id or title] = entry
        if hypothesis is not None and hypothesis not in entry["cited_by_hypotheses"]:
            entry["cited_by_hypotheses"].append(hypothesis)
        if formula and formula not in entry["supports_candidates"]:
            entry["supports_candidates"].append(formula)

    for n, hyp in enumerate(state.get("hypotheses", []), 1):
        for ev in hyp.get("evidence", []):
            add(ev.get("source_id", ""), ev.get("title", ""), ev.get("doi", ""), hypothesis=n)
    for cand in state.get("screened_candidates", []):
        for src in cand.get("sources", []):
            add(src.get("source_id", ""), src.get("title", ""), src.get("doi", ""), formula=cand["formula"] or cand["name"])
    for rec in state.get("extracted_properties", []):
        if rec["method"] == "Materials Project (DFT)":
            add(rec["source_id"], "Materials Project", rec["source_doi"], formula=rec["material_formula"])
    for chunk in state.get("retrieved_papers", [])[:10]:
        meta = chunk["metadata"]
        add(meta.get("arxiv_id", ""), meta.get("title", ""), meta.get("doi", ""))
    return list(citations.values())


class ReportAgent:
    """Final graph node: verifies hypotheses against their evidence and assembles citations."""

    def __init__(self, checker: GroundingChecker) -> None:
        self.checker = checker

    async def run(self, state: ResearchState, config=None) -> dict:
        if state.get("grounding"):  # a verify node already checked (and possibly revised) the hypotheses
            hypotheses, grounding, warnings = state.get("hypotheses", []), state["grounding"], []
        else:
            hypotheses, grounding, warnings = await self.checker.check(state.get("hypotheses", []))
            await emit(config, "grounding_check", is_grounded=grounding["is_grounded"], score=grounding["score"])
        citations = build_citations({**state, "hypotheses": hypotheses})
        return {
            "hypotheses": hypotheses,
            "grounding": grounding,
            "citations": citations,
            "warnings": warnings,
            "agent_log": [{
                "step": "GROUNDING_CHECK",
                "details": (
                    f"Checked {grounding['checked']} hypotheses ({grounding['method']}): "
                    f"{grounding['unsupported']} unsupported; {len(citations)} sources cited."
                ),
            }],
        }
