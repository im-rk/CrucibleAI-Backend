"""API routes for submitting research requests."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from matsci_agent.api.dependencies import get_research_service
from matsci_agent.domain.schemas import ResearchRequest
from matsci_agent.services.research_srv import ResearchService

router = APIRouter(tags=["research"])


@router.post("/research")
async def research_endpoint(
    request: ResearchRequest,
    service: ResearchService = Depends(get_research_service),
) -> dict[str, object]:
    """Start a research workflow for the submitted query."""

    return await service.run(request)
