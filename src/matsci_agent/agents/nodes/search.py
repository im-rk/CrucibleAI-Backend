"""Search node used to retrieve relevant papers and records."""

from __future__ import annotations

from matsci_agent.domain.state import ResearchState


async def run_search_node(state: ResearchState) -> ResearchState:
    """Populate the state with placeholder search results."""

    state["search_results"] = []
    return state
