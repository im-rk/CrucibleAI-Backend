"""Hypothesis node that turns screened evidence into a candidate pattern."""

from __future__ import annotations

from matsci_agent.domain.state import ResearchState


async def run_hypothesis_node(state: ResearchState) -> ResearchState:
    """Attach a placeholder hypothesis string."""

    state["hypothesis"] = "No hypothesis generated yet."
    return state
