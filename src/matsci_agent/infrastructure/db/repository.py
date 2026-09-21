"""SQLAlchemy repository helpers for persistence operations."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from matsci_agent.infrastructure.db.models import AgentLog, Paper, PropertyRecord, ResearchSession


class ResearchRepository:
    """Data access wrapper for research sessions, their audit log, and extracted records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_session(self, session_id: uuid.UUID, goal: str) -> None:
        self._session.add(ResearchSession(
            id=session_id, research_goal=goal, status="RUNNING",
            started_at=datetime.now(timezone.utc),  # set here: microsecond precision on every database
        ))

    async def finish_session(self, session_id: uuid.UUID, report: dict) -> None:
        """Mark a session COMPLETED and store its agent log, property records and report."""
        row = await self._session.get(ResearchSession, session_id)
        if row is None:
            return
        row.status = "COMPLETED"
        row.report = json.loads(json.dumps(report, default=str))  # guarantee JSON-safe
        row.completed_at = datetime.now(timezone.utc)

        self._session.add_all(
            AgentLog(
                session_id=session_id, agent_name=str(entry.get("agent", "pipeline"))[:50],
                action=str(entry.get("step", ""))[:100], output_summary=entry.get("details"),
                duration_ms=entry.get("duration_ms"),
            )
            for entry in report.get("agent_logs", [])
        )
        arxiv_ids = {r["source_id"] for r in report.get("extracted_properties", []) if r.get("source_id")}
        papers = {}
        if arxiv_ids:
            found = await self._session.execute(select(Paper).where(Paper.arxiv_id.in_(arxiv_ids)))
            papers = {p.arxiv_id: p.id for p in found.scalars()}
        self._session.add_all(
            PropertyRecord(
                session_id=session_id,
                material_name=r["material_name"][:200],
                material_formula=(r.get("material_formula") or None) and r["material_formula"][:100],
                property_name=r["property_name"][:100],
                value_raw=r["value_raw"],
                value_numeric=r.get("value_numeric"),
                unit=(r.get("unit") or "")[:50],
                conditions=r.get("conditions"),
                source_paper_id=papers.get(r.get("source_id")),
                source_doi=(r.get("source_doi") or None) and r["source_doi"][:200],
                is_predicted=bool(r.get("is_predicted")),
                prediction_confidence=r.get("uncertainty"),
                method=(r.get("method") or "")[:100],
                confidence=(r.get("confidence") or "")[:10],
            )
            for r in report.get("extracted_properties", [])
        )

    async def fail_session(self, session_id: uuid.UUID, message: str) -> None:
        row = await self._session.get(ResearchSession, session_id)
        if row is not None:
            row.status, row.error_message = "FAILED", message[:2000]
            row.completed_at = datetime.now(timezone.utc)

    async def get_session(self, session_id: uuid.UUID) -> dict | None:
        row = await self._session.get(ResearchSession, session_id)
        if row is None:
            return None
        logs = await self._session.execute(
            select(AgentLog).where(AgentLog.session_id == session_id).order_by(AgentLog.logged_at)
        )
        return {
            "session_id": str(row.id), "research_goal": row.research_goal, "status": row.status,
            "started_at": row.started_at, "completed_at": row.completed_at,
            "error_message": row.error_message, "report": row.report,
            "agent_logs": [
                {"agent": log.agent_name, "step": log.action, "details": log.output_summary, "duration_ms": log.duration_ms}
                for log in logs.scalars()
            ],
        }

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        rows = await self._session.execute(
            select(ResearchSession).order_by(ResearchSession.started_at.desc()).limit(limit)
        )
        return [
            {"session_id": str(r.id), "research_goal": r.research_goal, "status": r.status,
             "started_at": r.started_at, "completed_at": r.completed_at}
            for r in rows.scalars()
        ]

    async def upsert_paper(self, meta: dict, chunk_count: int, pdf_path: str | None = None) -> None:
        """Record an ingested paper (idempotent on arxiv_id)."""
        existing = (await self._session.execute(select(Paper).where(Paper.arxiv_id == meta["arxiv_id"]))).scalar_one_or_none()
        authors = ", ".join(meta["authors"]) if isinstance(meta["authors"], list) else meta["authors"]
        if existing is None:
            self._session.add(Paper(
                arxiv_id=meta["arxiv_id"], title=meta["title"], authors=authors, abstract=meta.get("abstract"),
                year=meta.get("year"), doi=meta.get("doi"), pdf_path=pdf_path, chunk_count=chunk_count,
            ))
        else:
            existing.chunk_count, existing.pdf_path = chunk_count, pdf_path or existing.pdf_path
