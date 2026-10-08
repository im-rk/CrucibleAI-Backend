"""Serve the latest evaluation results so the UI can show how well the system actually performs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from matsci_agent.evaluation import suite

router = APIRouter(tags=["evaluation"])


@router.get("/evaluation/latest")
def latest_evaluation() -> dict:
    """The most recent results of `scripts/run_evals.py` (retrieval, extraction, injection, end to end)."""
    results = suite.load()
    if results is None:
        raise HTTPException(status_code=404, detail="No evaluation results yet. Run: python scripts/run_evals.py")
    return results
