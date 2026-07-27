"""SQLAlchemy ORM models for research sessions and agent logs."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class ResearchSession(Base):
    """Tracks a user research session."""

    __tablename__ = "research_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    logs: Mapped[list[AgentLog]] = relationship(back_populates="session", cascade="all, delete-orphan")


class AgentLog(Base):
    """Stores individual agent events for later auditing."""

    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("research_sessions.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    session: Mapped[ResearchSession] = relationship(back_populates="logs")
