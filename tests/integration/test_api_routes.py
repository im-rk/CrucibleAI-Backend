"""HTTP-level tests: SSE framing, validation, CORS, papers, materials, health."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from matsci_agent.api import dependencies as deps
from matsci_agent.api.routers.research import format_sse, sse_frames
from matsci_agent.api.server import app
from matsci_agent.domain.state import initial_state
from matsci_agent.infrastructure.clients.mp_api import MaterialsProjectClient
from matsci_agent.services.papers_srv import PaperCatalog
from matsci_agent.services.persistence import PersistenceService
from matsci_agent.services.report_builder import build_report
from matsci_agent.services.research_srv import PipelineError


class StubService:
    def __init__(self, fail=False):
        self.fail = fail

    async def stream(self, request):
        yield {"event": "session_started", "session_id": "s1", "query": request.query}
        yield {"event": "papers_found", "count": 3}
        report = build_report(initial_state(request.query), session_id="s1", query=request.query,
                              max_results=request.max_results, mode={}, duration_ms=1)
        yield {"event": "complete", "report": report}

    async def run(self, request):
        if self.fail:
            raise PipelineError("pipeline broke")
        return {"session_id": "s1", "query": request.query}


class FakeStore:
    def __init__(self):
        self.docs = [
            {"id": "a_0", "text": "Solid electrolytes intro text", "metadata": {"arxiv_id": "a", "title": "Garnet Electrolytes", "authors": "Ann", "year": 2024, "section": "Introduction", "chunk_index": 0}},
            {"id": "a_1", "text": "Results text", "metadata": {"arxiv_id": "a", "title": "Garnet Electrolytes", "authors": "Ann", "year": 2024, "section": "Results", "chunk_index": 1}},
            {"id": "b_0", "text": "Perovskite text", "metadata": {"arxiv_id": "b", "title": "Perovskite Solar", "authors": "Bob", "year": 2025, "section": "Abstract", "chunk_index": 0}},
        ]

    def count(self):
        return len(self.docs)

    def iter_documents(self):
        yield from self.docs


@pytest.fixture
def client():
    app.dependency_overrides[deps.get_research_service] = StubService
    app.dependency_overrides[deps.get_paper_catalog] = lambda: PaperCatalog(FakeStore())
    app.dependency_overrides[deps.get_persistence] = lambda: PersistenceService(None)  # hermetic: no real database
    yield TestClient(app)
    app.dependency_overrides.clear()


def parse_sse(text: str) -> list[tuple[str, dict]]:
    frames = []
    for block in text.strip().split("\n\n"):
        lines = [line for line in block.splitlines() if not line.startswith(":")]
        if lines:
            frames.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return frames


def test_research_stream_emits_named_sse_events_ending_with_the_report(client) -> None:
    response = client.post("/api/research/stream", json={"query": "solid electrolytes"})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    frames = parse_sse(response.text)
    assert [name for name, _ in frames] == ["session_started", "papers_found", "complete"]
    assert frames[-1][1]["report"]["query"] == "solid electrolytes"


def test_research_json_endpoint_and_pipeline_failure(client) -> None:
    assert client.post("/api/research", json={"query": "solid electrolytes"}).json()["session_id"] == "s1"
    app.dependency_overrides[deps.get_research_service] = lambda: StubService(fail=True)
    failed = client.post("/api/research", json={"query": "solid electrolytes"})
    assert failed.status_code == 500 and failed.json()["detail"] == "pipeline broke"


@pytest.mark.parametrize("body", [{"query": "ab"}, {"query": "x" * 1001}, {"query": "fine query", "max_results": 0}, {"query": "fine query", "max_results": 31}, {}])
def test_invalid_research_requests_are_rejected(client, body) -> None:
    assert client.post("/api/research", json=body).status_code == 422
    assert client.post("/api/research/stream", json=body).status_code == 422


def test_cors_allows_the_dev_frontend_but_not_arbitrary_origins(client) -> None:
    ok = client.options("/api/research", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-credentials" not in ok.headers
    bad = client.options("/api/research", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_papers_listing_search_and_detail(client) -> None:
    listing = client.get("/api/papers").json()
    assert listing["total"] == 2 and [p["arxiv_id"] for p in listing["papers"]] == ["b", "a"]  # newest first
    garnet = next(p for p in listing["papers"] if p["arxiv_id"] == "a")
    assert garnet["chunk_count"] == 2 and garnet["sections"] == ["Introduction", "Results"]
    assert garnet["snippet"].startswith("Solid electrolytes")  # first chunk, not last
    assert client.get("/api/papers", params={"q": "perovskite"}).json()["total"] == 1
    assert client.get("/api/papers/a").json()["title"] == "Garnet Electrolytes"
    assert client.get("/api/papers/nope").status_code == 404


def test_corpus_and_health_report_real_state(client, monkeypatch) -> None:
    monkeypatch.setattr(deps, "get_mp_client", lambda: MaterialsProjectClient(api_key=None))
    corpus = client.get("/api/corpus").json()
    assert corpus["total_chunks"] == 3 and corpus["total_papers"] == 2
    assert corpus["sample_papers"][0]["arxiv_id"] == "b"
    health = client.get("/api/health").json()
    assert health["status"] == "online" and health["vector_store_chunks"] == 3
    assert "offline" in health["materials_project_api"]


def test_materials_lookup_screen_and_predict(client, monkeypatch) -> None:
    monkeypatch.setattr(deps, "get_mp_client", lambda: MaterialsProjectClient(api_key=None))
    import matsci_agent.api.routers.materials as materials

    monkeypatch.setattr(materials, "get_mp_client", lambda: MaterialsProjectClient(api_key=None))
    lookup = client.get("/api/materials/lookup", params={"formula": "GaN"}).json()
    assert lookup["source"] == "offline reference table" and lookup["properties"][0]["value"] == 3.4
    assert client.get("/api/materials/lookup", params={"formula": "not-a-formula"}).status_code == 422
    screened = client.get("/api/materials/screen", params={"bandgap_min": 6}).json()
    assert {r["formula"] for r in screened["results"]} == {"AlN", "Al2O3", "MgO"}


def test_predict_validates_the_formula_and_property(client, monkeypatch, tmp_path) -> None:
    import matsci_agent.api.routers.materials as materials
    from matsci_agent.infrastructure.ml.krr import KRRPropertyPredictor

    krr = KRRPropertyPredictor(model_dir=tmp_path)
    krr._featurizer_failed = True
    monkeypatch.setattr(materials, "get_krr_predictor", lambda: krr)
    ok = client.post("/api/predict", json={"formula": "GaN", "property_name": "band_gap"})
    assert ok.status_code == 200 and ok.json()["is_predicted"] and ok.json()["uncertainty"] > 0
    assert client.post("/api/predict", json={"formula": "Xx9"}).status_code == 422
    unsupported = client.post("/api/predict", json={"formula": "GaN", "property_name": "ionic_conductivity"})
    assert unsupported.status_code == 422 and "Unsupported property" in unsupported.json()["detail"]


# ---- SSE framing & heartbeat ------------------------------------------------------------------


def test_format_sse_puts_the_event_name_in_the_frame_and_the_rest_in_data() -> None:
    frame = format_sse({"event": "papers_found", "count": 3})
    assert frame == 'event: papers_found\ndata: {"count": 3}\n\n'


async def test_sse_frames_send_keepalives_while_quiet_and_never_drop_events() -> None:
    async def slow_events():
        await asyncio.sleep(0.25)
        yield {"event": "late", "n": 1}

    frames = [f async for f in sse_frames(slow_events(), heartbeat_s=0.05)]
    assert frames[-1] == 'event: late\ndata: {"n": 1}\n\n'
    assert frames.count(": keep-alive\n\n") >= 2


# ---- sessions & health with a database --------------------------------------------------------------


@pytest.fixture
def sqlite_store():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    from matsci_agent.infrastructure.db.models import Base
    from matsci_agent.services.persistence import PersistenceService

    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})

    async def create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(create())
    return PersistenceService(async_sessionmaker(engine, expire_on_commit=False))


def test_sessions_endpoints_return_stored_runs_404_and_503(client, sqlite_store) -> None:
    import uuid

    sid = str(uuid.uuid4())
    asyncio.run(sqlite_store.start(sid, "solid electrolytes"))
    asyncio.run(sqlite_store.finish(sid, {"agent_logs": [{"agent": "A", "step": "S", "details": "d", "duration_ms": 1}], "extracted_properties": []}))
    app.dependency_overrides[deps.get_persistence] = lambda: sqlite_store

    got = client.get(f"/api/sessions/{sid}").json()
    assert got["status"] == "COMPLETED" and got["agent_logs"][0]["agent"] == "A"
    assert client.get("/api/sessions").json()["sessions"][0]["session_id"] == sid
    assert client.get(f"/api/sessions/{uuid.uuid4()}").status_code == 404

    from matsci_agent.services.persistence import PersistenceService

    app.dependency_overrides[deps.get_persistence] = lambda: PersistenceService(None)
    assert client.get(f"/api/sessions/{sid}").status_code == 503


def test_health_reports_the_database_state(client, sqlite_store, monkeypatch) -> None:
    from matsci_agent.services.persistence import PersistenceService

    monkeypatch.setattr(deps, "get_mp_client", lambda: MaterialsProjectClient(api_key=None))
    app.dependency_overrides[deps.get_persistence] = lambda: sqlite_store
    assert client.get("/api/health").json()["database"] == "connected"
    app.dependency_overrides[deps.get_persistence] = lambda: PersistenceService(None)
    assert client.get("/api/health").json()["database"] == "disabled"
