"""Integration smoke tests for the FastAPI application."""

from __future__ import annotations

from fastapi.testclient import TestClient

from matsci_agent.api.server import app


def test_research_route_returns_ok() -> None:
    client = TestClient(app)
    response = client.post("/api/research", json={"query": "solar cell"})
    assert response.status_code == 200
