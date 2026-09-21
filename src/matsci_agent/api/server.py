"""FastAPI application factory and startup wiring."""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, Query

from matsci_agent.api.dependencies import get_krr_predictor, get_mp_client, get_vector_store
from matsci_agent.api.middleware import configure_middleware
from matsci_agent.api.routers.research import router as research_router
from matsci_agent.api.routers.stream import router as stream_router
from matsci_agent.core.config import settings

def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    app = FastAPI(title=settings.app_name)
    configure_middleware(app)

    # Core routers
    app.include_router(research_router, prefix=settings.api_prefix)
    app.include_router(stream_router, prefix=settings.api_prefix)

    # Health & System Status Endpoint
    @app.get("/api/health")
    async def health_check() -> dict[str, Any]:
        vector_store = get_vector_store()
        return {
            "status": "online",
            "app_name": settings.app_name,
            "environment": settings.environment,
            "vector_store_chunks": vector_store.count(),
            "krr_engine": "Kernel Ridge Regression (RBF kernel) Ready",
            "materials_project_api": "Active (Mock/Live Rester)",
        }

    # Direct KRR Predictor Endpoint
    @app.post("/api/predict")
    async def predict_property(
        formula: str = Body(..., embed=True),
        property_name: str = Body(default="band_gap", embed=True),
    ) -> dict[str, Any]:
        return get_krr_predictor().predict(formula, property_name=property_name)

    # Direct Materials Project Lookup & Screening
    @app.get("/api/materials/lookup")
    async def lookup_material(formula: str = Query(...)) -> dict[str, Any]:
        props = get_mp_client().get_properties(formula)
        return {
            "formula": formula,
            "properties": [
                {
                    "material_id": p.material_id,
                    "property_name": p.property_name,
                    "value": p.value,
                    "unit": p.unit,
                    "source": p.source,
                    "is_direct_gap": p.is_direct_gap,
                }
                for p in props
            ],
        }

    # Paper Corpus Statistics
    @app.get("/api/corpus")
    async def get_corpus_stats() -> dict[str, Any]:
        vector_store = get_vector_store()
        total_chunks = vector_store.count()
        sample_results = vector_store.search("solid state electrolyte", n_results=5)
        return {
            "total_chunks": total_chunks,
            "sample_papers": [
                {
                    "id": c["id"],
                    "arxiv_id": c["metadata"].get("arxiv_id", "N/A"),
                    "title": c["metadata"].get("title", "Untitled Paper"),
                    "section": c["metadata"].get("section", "General"),
                    "year": c["metadata"].get("year", 2026),
                    "snippet": c["text"][:150] + "...",
                }
                for c in sample_results
            ],
        }

    return app


app = create_app()

