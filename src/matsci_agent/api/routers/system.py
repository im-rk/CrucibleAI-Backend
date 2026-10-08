"""Health and corpus status endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from matsci_agent import __version__
from matsci_agent.api.dependencies import get_paper_catalog, get_persistence, runtime_mode
from matsci_agent.core.config import settings
from matsci_agent.observability.metrics import METRICS
from matsci_agent.services.papers_srv import ListingUnsupported, PaperCatalog
from matsci_agent.services.persistence import PersistenceService

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_check(
    catalog: PaperCatalog = Depends(get_paper_catalog),
    store: PersistenceService = Depends(get_persistence),
) -> dict:
    mode = runtime_mode()
    stats = await asyncio.to_thread(catalog.stats)
    return {
        "status": "online",
        "version": __version__,
        "app_name": settings.app_name,
        "environment": settings.environment,
        "vector_store_chunks": stats["total_chunks"],
        "krr_engine": "Kernel Ridge Regression (Laplacian kernel)",
        "materials_project_api": (
            "live API" if mode["materials_project"] == "live" else "offline reference table (no API key)"
        ),
        "llm": mode["llm"] or f"unavailable, heuristic mode ({mode['llm_unavailable_reason']})",
        "database": await store.ping(),
    }


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus text format: runs, LLM calls, tokens and latency since the process started."""
    return METRICS.render()


def _sample(catalog: PaperCatalog) -> list[dict]:
    try:
        return catalog.list(limit=5)["papers"]
    except ListingUnsupported:
        return []


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
            for p in _sample(catalog)
        ],
    }
