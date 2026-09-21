"""Health and corpus status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from matsci_agent import __version__
from matsci_agent.api.dependencies import get_paper_catalog, runtime_mode
from matsci_agent.core.config import settings
from matsci_agent.services.papers_srv import PaperCatalog

router = APIRouter(tags=["system"])


@router.get("/health")
def health_check(catalog: PaperCatalog = Depends(get_paper_catalog)) -> dict:
    mode = runtime_mode()
    return {
        "status": "online",
        "version": __version__,
        "app_name": settings.app_name,
        "environment": settings.environment,
        "vector_store_chunks": catalog.stats()["total_chunks"],
        "krr_engine": "Kernel Ridge Regression (RBF kernel)",
        "materials_project_api": (
            "live API" if mode["materials_project"] == "live" else "offline reference table (no API key)"
        ),
        "llm": mode["llm"] or "not configured (heuristic mode)",
    }


@router.get("/corpus")
def corpus_stats(catalog: PaperCatalog = Depends(get_paper_catalog)) -> dict:
    stats = catalog.stats()
    return {
        **stats,
        "sample_papers": [
            {
                "id": p["arxiv_id"], "arxiv_id": p["arxiv_id"], "title": p["title"],
                "section": ", ".join(p["sections"][:2]), "year": p["year"], "snippet": p["snippet"],
            }
            for p in catalog.list(limit=5)["papers"]
        ],
    }
