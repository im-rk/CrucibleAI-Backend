"""Best-effort persistence of research sessions to PostgreSQL.

The database is an audit trail, not a dependency of the pipeline: if it is disabled or
unreachable, runs still complete and this service just logs and steps aside for a while.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import InterfaceError, OperationalError

from matsci_agent.infrastructure.db.repository import ResearchRepository

logger = logging.getLogger(__name__)

_OUTAGE_ERRORS = (OSError, asyncio.TimeoutError, OperationalError, InterfaceError)


class PersistenceUnavailable(RuntimeError):
    """The database is disabled or currently unreachable."""


class PersistenceService:
    def __init__(self, session_factory=None, *, retry_after_s: float = 60, op_timeout_s: float = 5) -> None:
        self._factory = session_factory
        self._retry_after_s = retry_after_s
        self._op_timeout_s = op_timeout_s
        self._down_until = 0.0

    @property
    def enabled(self) -> bool:
        return self._factory is not None

    @property
    def available(self) -> bool:
        return self.enabled and monotonic() >= self._down_until

    async def _run(self, name: str, op: Callable[[ResearchRepository], Awaitable[Any]]) -> Any:
        """Run `op` in a transaction. Raises PersistenceUnavailable on outage; other errors propagate."""
        if not self.available:
            raise PersistenceUnavailable("database disabled or recently unreachable")
        try:
            async with asyncio.timeout(self._op_timeout_s):
                async with self._factory() as session:
                    try:
                        await session.connection()  # any failure to connect (auth, no such db, DNS...) is an outage
                    except Exception as exc:  # noqa: BLE001
                        raise ConnectionError(f"{type(exc).__name__}: {exc}") from exc
                    result = await op(ResearchRepository(session))
                    await session.commit()
                    return result
        except _OUTAGE_ERRORS as exc:
            self._down_until = monotonic() + self._retry_after_s
            logger.warning(f"Database unreachable during {name} ({type(exc).__name__}); retrying in {self._retry_after_s:.0f}s")
            raise PersistenceUnavailable(str(exc)) from exc

    async def _best_effort(self, name: str, op: Callable[[ResearchRepository], Awaitable[Any]]) -> None:
        try:
            await self._run(name, op)
        except PersistenceUnavailable:
            pass
        except Exception:  # noqa: BLE001 - a bad record must never break a research run
            logger.exception(f"Persisting {name} failed")

    # ---- write path (called by the pipeline; never raises) -----------------------------

    async def start(self, session_id: str, goal: str) -> None:
        await self._best_effort("session start", lambda repo: repo.create_session(uuid.UUID(session_id), goal))

    async def finish(self, session_id: str, report: dict) -> None:
        await self._best_effort("session result", lambda repo: repo.finish_session(uuid.UUID(session_id), report))

    async def record_paper(self, meta: dict, chunk_count: int, pdf_path: str | None = None) -> None:
        await self._best_effort("paper", lambda repo: repo.upsert_paper(meta, chunk_count, pdf_path))

    async def fail(self, session_id: str, message: str) -> None:
        await self._best_effort("session failure", lambda repo: repo.fail_session(uuid.UUID(session_id), message))

    # ---- read path (called by the API; raises PersistenceUnavailable) ------------------

    async def get(self, session_id: str) -> dict | None:
        try:
            sid = uuid.UUID(session_id)
        except ValueError:
            return None
        return await self._run("session lookup", lambda repo: repo.get_session(sid))

    async def recent(self, limit: int = 20) -> list[dict]:
        return await self._run("session list", lambda repo: repo.list_sessions(limit))

    async def ping(self) -> str:
        """'connected' | 'unavailable' | 'disabled', from a real `SELECT 1` (ignores the back-off)."""
        if not self.enabled:
            return "disabled"
        try:
            async with asyncio.timeout(2):
                async with self._factory() as session:
                    await session.execute(text("SELECT 1"))
            self._down_until = 0.0
            return "connected"
        except Exception:  # noqa: BLE001 - a health probe must never raise, whatever the driver throws
            return "unavailable"
