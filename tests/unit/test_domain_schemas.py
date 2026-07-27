"""Unit tests for the public domain models."""

from __future__ import annotations

from matsci_agent.domain.schemas import ResearchRequest


def test_research_request_defaults() -> None:
    request = ResearchRequest(query="battery")
    assert request.max_results == 10
