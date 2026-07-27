"""Server-Sent Events routes for streaming agent output."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["stream"])


@router.get("/stream")
async def stream_endpoint() -> StreamingResponse:
    """Stream a minimal SSE heartbeat for the initial scaffold."""

    async def event_source():
        yield "data: ready\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
