"""Screening node for applying criteria filters to candidate records."""

from __future__ import annotations

from matsci_agent.domain.state import ResearchState


async def run_screening_node(state: ResearchState) -> ResearchState:
    """Return the state unchanged for the initial scaffold."""

    return state
