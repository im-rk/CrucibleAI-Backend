"""API routes for running research: a JSON endpoint and a Server-Sent Events stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from matsci_agent.agents.nodes.planner import PlannerAgent
from matsci_agent.api.dependencies import get_planner, get_research_service
from matsci_agent.domain.schemas import ResearchRequest
from matsci_agent.services.research_srv import PipelineError, ResearchService

router = APIRouter(tags=["research"])

HEARTBEAT_S = 15  # keeps idle proxies (nginx, ALB) from closing a long-running stream


def format_sse(event: dict) -> str:
    payload = {k: v for k, v in event.items() if k != "event"}
    return f"event: {event['event']}\ndata: {json.dumps(payload, default=str)}\n\n"


async def sse_frames(events: AsyncIterator[dict], heartbeat_s: float = HEARTBEAT_S) -> AsyncIterator[str]:
    """Encode events as SSE frames, emitting a comment heartbeat while the pipeline is quiet."""
    iterator = events.__aiter__()
    pending: asyncio.Future | None = None
    try:
        while True:
            pending = pending or asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=heartbeat_s)
            if not done:
                yield ": keep-alive\n\n"
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            pending = None
            yield format_sse(event)
    finally:
        if pending is not None:
            pending.cancel()
        await iterator.aclose()


@router.post("/research/plan")
async def preview_plan(request: ResearchRequest, planner: PlannerAgent = Depends(get_planner)) -> dict:
    """Plan only: the criteria, queries and material families a run would use, for review and editing.

    Send the (edited) result back as `plan` on `/research` or `/research/stream` to run it.
    """
    warnings: list[str] = []
    plan = await planner.plan(request.query, warnings, request.plan)
    return {"query": request.query, "plan": plan, "warnings": warnings}


@router.post("/research")
async def research_endpoint(
    request: ResearchRequest,
    service: ResearchService = Depends(get_research_service),
) -> dict:
    """Run the full agent pipeline and return the finished report."""
    try:
        return await service.run(request)
    except PipelineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/research/stream")
async def research_stream_endpoint(
    request: ResearchRequest,
    service: ResearchService = Depends(get_research_service),
) -> StreamingResponse:
    """Run the pipeline, streaming agent progress as SSE and the report in a final `complete` event."""
    return StreamingResponse(
        sse_frames(service.stream(request)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
