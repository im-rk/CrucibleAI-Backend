"""SQLAlchemy repository helpers for persistence operations."""

# from __future__ import annotations

# from sqlalchemy.ext.asyncio import AsyncSession

# from matsci_agent.infrastructure.db.models import AgentLog, ResearchSession


# class ResearchRepository:
#     """Data access wrapper for research session records."""

#     def __init__(self, session: AsyncSession) -> None:
#         self._session = session

#     async def create_session(self, query: str) -> ResearchSession:
#         session = ResearchSession(query=query)
#         self._session.add(session)
#         await self._session.flush()
#         return session

#     async def add_log(self, session_id: int, event_type: str, payload: str) -> AgentLog:
#         log = AgentLog(session_id=session_id, event_type=event_type, payload=payload)
#         self._session.add(log)
#         await self._session.flush()
#         return log
