"""Follow-up questions about a finished research run."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from matsci_agent.agents.nodes.qa import ReportQA
from matsci_agent.api.dependencies import get_persistence, get_qa, get_report_store
from matsci_agent.services.persistence import PersistenceService, PersistenceUnavailable
from matsci_agent.services.report_store import ReportStore

router = APIRouter(tags=["qa"])


class Turn(BaseModel):
    role: str
    content: str = Field(max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=20)


@router.post("/runs/{session_id}/ask")
async def ask_about_run(
    session_id: str,
    body: AskRequest,
    qa: ReportQA = Depends(get_qa),
    reports: ReportStore = Depends(get_report_store),
    store: PersistenceService = Depends(get_persistence),
) -> dict:
    """Answer a question using only the evidence in that run's report, with citations."""
    report = reports.get(session_id)
    if report is None:  # evicted or from before a restart: fall back to the saved session
        try:
            saved = await store.get(session_id)
        except PersistenceUnavailable:
            saved = None
        report = (saved or {}).get("report")
    if not report:
        raise HTTPException(status_code=404, detail=f"No report found for run '{session_id}'.")
    return await qa.ask(report, body.question, [t.model_dump() for t in body.history])
