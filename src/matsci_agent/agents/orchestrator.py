"""LangGraph orchestration for the end-to-end research workflow."""

from __future__ import annotations

from matsci_agent.domain.state import ResearchState


def build_research_graph() -> str:
    """Return a placeholder graph handle until the workflow is wired up."""

    return "research-graph"


def initialize_state(query: str) -> ResearchState:
    """Create the initial state for a research run."""

    return {"query": query, "search_results": [], "extracted_records": []}
