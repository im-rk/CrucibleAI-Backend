"""Browse the ingested paper corpus."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from matsci_agent.api.dependencies import get_paper_catalog
from matsci_agent.services.papers_srv import PaperCatalog

router = APIRouter(tags=["papers"])


@router.get("/papers")
def list_papers(
    q: str | None = Query(None, max_length=200, description="Filter by title, author or arXiv id"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    catalog: PaperCatalog = Depends(get_paper_catalog),
) -> dict:
    return catalog.list(q, limit, offset)


@router.get("/papers/{arxiv_id}")
def get_paper(arxiv_id: str, catalog: PaperCatalog = Depends(get_paper_catalog)) -> dict:
    paper = catalog.get(arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper '{arxiv_id}' is not in the corpus.")
    return paper
