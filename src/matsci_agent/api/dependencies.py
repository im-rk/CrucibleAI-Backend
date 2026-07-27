"""FastAPI dependencies for sessions, services, and shared infrastructure."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from matsci_agent.infrastructure.db.session import get_session
from matsci_agent.services.research_srv import ResearchService


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Expose an async SQLAlchemy session to request handlers."""

    async for session in get_session():
        yield session


def get_research_service() -> ResearchService:
    """Create the research service for a request."""

    return ResearchService()
