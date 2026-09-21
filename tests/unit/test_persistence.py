"""Persistence: SQLite always; real PostgreSQL too when TEST_DATABASE_URL is set."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from matsci_agent.infrastructure.db.models import Base, Paper, PropertyRecord
from matsci_agent.services.persistence import PersistenceService, PersistenceUnavailable

POSTGRES_URL = os.getenv("TEST_DATABASE_URL")  # e.g. postgresql+asyncpg://postgres:postgres@localhost:55432/matsci_agent


def report(**over) -> dict:
    base = {
        "session_id": "x", "query": "solid electrolytes",
        "agent_logs": [
            {"agent": "SearchAgent", "step": "LITERATURE_SEARCH", "details": "found 5", "duration_ms": 12},
            {"agent": "ScreeningAgent", "step": "SCREENING", "details": "3 candidates", "duration_ms": 4},
        ],
        "extracted_properties": [
            {"material_name": "GaN", "material_formula": "GaN", "property_name": "bandgap", "value_raw": "3.4 eV",
             "value_numeric": 3.4, "unit": "eV", "conditions": "", "source_id": "2401.1", "source_doi": "10.1/x",
             "is_predicted": False, "uncertainty": None, "method": "literature", "confidence": "high"},
            {"material_name": "ZnO", "material_formula": "ZnO", "property_name": "bandgap", "value_raw": "3.3 eV",
             "value_numeric": 3.3, "unit": "eV", "source_id": "", "source_doi": "", "is_predicted": True,
             "uncertainty": 0.42, "method": "Kernel Ridge Regression (RBF kernel)", "confidence": "low"},
        ],
    }
    return {**base, **over}


@pytest.fixture(params=["sqlite"] + (["postgres"] if POSTGRES_URL else []))
async def factory(request):
    if request.param == "sqlite":
        engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    else:  # schema comes from `alembic upgrade head`
        engine = create_async_engine(POSTGRES_URL)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def test_a_completed_run_is_stored_with_its_audit_trail(factory) -> None:
    store, sid = PersistenceService(factory), str(uuid.uuid4())
    arxiv_id = f"t-{uuid.uuid4().hex[:10]}"  # unique: a real Postgres database persists between runs
    async with factory() as s:  # a known paper, so records can link to it
        s.add(Paper(arxiv_id=arxiv_id, title="Wide gap", authors="A", chunk_count=3))
        await s.commit()
    run_report = report()
    run_report["extracted_properties"][0]["source_id"] = arxiv_id

    await store.start(sid, "solid electrolytes")
    assert (await store.get(sid))["status"] == "RUNNING"
    await store.finish(sid, run_report)

    stored = await store.get(sid)
    assert stored["status"] == "COMPLETED" and stored["research_goal"] == "solid electrolytes"
    assert stored["completed_at"] is not None and stored["report"]["query"] == "solid electrolytes"
    assert [(l["agent"], l["duration_ms"]) for l in stored["agent_logs"]] == [("SearchAgent", 12), ("ScreeningAgent", 4)]

    async with factory() as s:
        rows = (await s.execute(select(PropertyRecord).where(PropertyRecord.session_id == uuid.UUID(sid)))).scalars().all()
        by_name = {r.material_name: r for r in rows}
        paper = (await s.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))).scalar_one()
        assert by_name["GaN"].source_paper_id == paper.id and not by_name["GaN"].is_predicted
        assert by_name["ZnO"].is_predicted and by_name["ZnO"].prediction_confidence == pytest.approx(0.42)
        assert by_name["ZnO"].source_paper_id is None


async def test_a_failed_run_records_the_error(factory) -> None:
    store, sid = PersistenceService(factory), str(uuid.uuid4())
    await store.start(sid, "goal")
    await store.fail(sid, "Research timed out after 300s.")
    stored = await store.get(sid)
    assert stored["status"] == "FAILED" and "timed out" in stored["error_message"]


async def test_lookup_of_unknown_or_malformed_ids_returns_none(factory) -> None:
    store = PersistenceService(factory)
    assert await store.get(str(uuid.uuid4())) is None
    assert await store.get("not-a-uuid") is None


async def test_recent_sessions_are_newest_first(factory) -> None:
    store = PersistenceService(factory)
    ids = [str(uuid.uuid4()) for _ in range(3)]
    for i, sid in enumerate(ids):
        await store.start(sid, f"goal {i}")
        await asyncio.sleep(0.02)
    recent = [r["session_id"] for r in await store.recent(limit=50)]
    assert [i for i in recent if i in ids] == ids[::-1]


async def test_ingested_papers_are_upserted_idempotently(factory) -> None:
    async with factory() as s:
        from matsci_agent.infrastructure.db.repository import ResearchRepository

        repo = ResearchRepository(s)
        meta = {"arxiv_id": f"upsert-{uuid.uuid4()}", "title": "T", "authors": ["A", "B"], "year": 2024}
        await repo.upsert_paper(meta, chunk_count=3)
        await s.commit()
        await repo.upsert_paper(meta, chunk_count=9)
        await s.commit()
        rows = (await s.execute(select(Paper).where(Paper.arxiv_id == meta["arxiv_id"]))).scalars().all()
        assert len(rows) == 1 and rows[0].chunk_count == 9 and rows[0].authors == "A, B"


# ---- outages must never break a research run ----------------------------------------------------


class ExplodingFactory:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise ConnectionRefusedError("postgres is down")


async def test_writes_swallow_an_outage_and_back_off_instead_of_retrying_every_call() -> None:
    factory = ExplodingFactory()
    store = PersistenceService(factory, retry_after_s=60)
    await store.start(str(uuid.uuid4()), "goal")  # must not raise
    assert factory.calls == 1 and not store.available
    await store.finish(str(uuid.uuid4()), report())
    await store.fail(str(uuid.uuid4()), "x")
    assert factory.calls == 1  # backing off: no repeated connection attempts


async def test_the_service_tries_again_after_the_backoff_expires() -> None:
    factory = ExplodingFactory()
    store = PersistenceService(factory, retry_after_s=0.05)
    await store.start(str(uuid.uuid4()), "goal")
    await asyncio.sleep(0.08)
    await store.start(str(uuid.uuid4()), "goal")
    assert factory.calls == 2


async def test_reads_raise_persistence_unavailable_during_an_outage() -> None:
    store = PersistenceService(ExplodingFactory())
    with pytest.raises(PersistenceUnavailable):
        await store.get(str(uuid.uuid4()))
    with pytest.raises(PersistenceUnavailable):
        await store.recent()


async def test_a_disabled_service_is_a_silent_noop_for_writes() -> None:
    store = PersistenceService(None)
    await store.start(str(uuid.uuid4()), "goal")
    assert await store.ping() == "disabled" and not store.enabled
    with pytest.raises(PersistenceUnavailable):
        await store.recent()


async def test_a_malformed_report_is_logged_but_neither_raises_nor_disables_persistence(factory, caplog) -> None:
    store, sid = PersistenceService(factory), str(uuid.uuid4())
    await store.start(sid, "goal")
    await store.finish(sid, {"agent_logs": [], "extracted_properties": [{"material_name": "only-this"}]})
    assert "Persisting session result failed" in caplog.text
    assert store.available  # a data bug is not a database outage


async def test_ping_reports_connected_and_unavailable(factory) -> None:
    assert await PersistenceService(factory).ping() == "connected"
    assert await PersistenceService(ExplodingFactory()).ping() == "unavailable"


class AuthFailingSession:
    """Mimics a database that is up but rejects the credentials (asyncpg InvalidPasswordError)."""

    class InvalidPasswordError(Exception):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def connection(self):
        raise self.InvalidPasswordError('password authentication failed for user "postgres"')

    async def execute(self, *args):
        await self.connection()


async def test_a_database_that_rejects_our_credentials_is_an_outage_not_a_crash() -> None:
    store = PersistenceService(AuthFailingSession)
    await store.start(str(uuid.uuid4()), "goal")  # must not raise
    assert not store.available
    with pytest.raises(PersistenceUnavailable):
        await store.get(str(uuid.uuid4()))
    assert await PersistenceService(AuthFailingSession).ping() == "unavailable"  # health never raises
