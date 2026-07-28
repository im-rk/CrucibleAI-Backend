"""Async SQLAlchemy engine and session factory."""

# from __future__ import annotations

# from collections.abc import AsyncIterator

# from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# from matsci_agent.core.config import settings

# engine = create_async_engine(settings.database_url, pool_pre_ping=True)
# async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# async def get_session() -> AsyncIterator[AsyncSession]:
#     """Yield a database session for FastAPI dependencies."""

#     async with async_session_factory() as session:
#         yield session
