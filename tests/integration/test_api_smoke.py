"""Integration smoke tests for the FastAPI application."""

from __future__ import annotations

from fastapi.testclient import TestClient

from matsci_agent.api.dependencies import get_research_service
from matsci_agent.api.server import app


class StubResearchService:
    """Avoids loading the embedding model / real corpus for a routing smoke test."""

    async def run(self, request):
        return {"session_id": "test", "status": "COMPLETED", "query": request.query}


def test_research_route_returns_ok() -> None:
    app.dependency_overrides[get_research_service] = StubResearchService
    try:
        response = TestClient(app).post("/api/research", json={"query": "solar cell"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["query"] == "solar cell"
