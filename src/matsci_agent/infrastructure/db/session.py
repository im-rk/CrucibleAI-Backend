"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def create_session_factory(url: str, *, connect_timeout_s: int = 3) -> async_sessionmaker[AsyncSession]:
    """Build a session factory. Connecting is lazy, so this never touches the network."""
    connect_args = {"timeout": connect_timeout_s} if url.startswith("postgresql") else {}
    engine = create_async_engine(url, pool_pre_ping=True, connect_args=connect_args)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
