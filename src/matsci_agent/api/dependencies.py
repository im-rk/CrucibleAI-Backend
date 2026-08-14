"""FastAPI dependencies for sessions, services, and shared infrastructure."""

from __future__ import annotations

from matsci_agent.services.research_srv import ResearchService


def get_research_service() -> ResearchService:
    """Create the research service for a request."""

    return ResearchService()

