"""Extraction node used to normalize agent outputs into structured records."""

from __future__ import annotations

from matsci_agent.domain.state import ResearchState


async def run_extraction_node(state: ResearchState) -> ResearchState:
    """Populate the state with extracted records."""

    state["extracted_records"] = []
    return state
