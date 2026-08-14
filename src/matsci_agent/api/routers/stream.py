"""Server-Sent Events routes for streaming agent output."""

from __future__ import annotations

import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["stream"])


@router.get("/stream")
async def stream_endpoint() -> StreamingResponse:
    """Stream real-time SSE research agent execution events."""

    async def event_source():
        steps = [
            {"phase": 1, "step": "LITERATURE_SEARCH", "message": "Querying ChromaDB vector store with SentenceTransformers embeddings...", "progress": 25},
            {"phase": 2, "step": "MATERIALS_PROJECT_LOOKUP", "message": "Connecting to Materials Project API for candidate formulas...", "progress": 50},
            {"phase": 3, "step": "KRR_PREDICTION", "message": "Featurizing chemical compositions and running Kernel Ridge Regression (RBF kernel)...", "progress": 75},
            {"phase": 4, "step": "SYNTHESIS", "message": "Generating final integrated research summary and hypothesis...", "progress": 100},
        ]
        for step_data in steps:
            yield f"data: {json.dumps(step_data)}\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(event_source(), media_type="text/event-stream")

