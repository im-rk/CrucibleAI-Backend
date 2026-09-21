"""Read access to persisted research sessions (the audit trail)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from matsci_agent.api.dependencies import get_persistence
from matsci_agent.services.persistence import PersistenceService, PersistenceUnavailable

router = APIRouter(tags=["sessions"])


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(20, ge=1, le=100), store: PersistenceService = Depends(get_persistence)
) -> dict:
    try:
        return {"sessions": await store.recent(limit)}
    except PersistenceUnavailable as exc:
        raise HTTPException(status_code=503, detail="Session history is unavailable (database not reachable).") from exc


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, store: PersistenceService = Depends(get_persistence)) -> dict:
    try:
        session = await store.get(session_id)
    except PersistenceUnavailable as exc:
        raise HTTPException(status_code=503, detail="Session history is unavailable (database not reachable).") from exc
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")
    return session
