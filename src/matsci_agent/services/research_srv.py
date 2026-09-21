"""Service layer entry point for running the research workflow."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from time import perf_counter
from typing import Any

from matsci_agent.agents.orchestrator import initial_state, run_config
from matsci_agent.domain.schemas import ResearchRequest
from matsci_agent.services.report_builder import build_report

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """The research run failed as a whole (as opposed to one agent failing)."""


class ResearchService:
    """Runs the compiled agent graph and streams its progress as events."""

    def __init__(
        self,
        graph,
        mode: Callable[[], dict[str, Any]],
        *,
        timeout_s: float = 300,
        max_concurrent: int = 2,
    ) -> None:
        self.graph = graph
        self.mode = mode
        self.timeout_s = timeout_s
        self._slots = asyncio.Semaphore(max_concurrent)

    async def stream(self, request: ResearchRequest) -> AsyncIterator[dict]:
        """Yield progress events, ending with `complete` (the report) or `error`."""
        session_id = str(uuid.uuid4())
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        mode = self.mode()

        async def emit(event: dict) -> None:
            await queue.put(event)

        async def runner() -> None:
            try:
                async with self._slots:
                    started = perf_counter()
                    state = await asyncio.wait_for(
                        self.graph.ainvoke(
                            initial_state(request.query, request.max_results), run_config(emit)
                        ),
                        timeout=self.timeout_s,
                    )
                    report = build_report(
                        state, session_id=session_id, query=request.query,
                        max_results=request.max_results, mode=mode,
                        duration_ms=int((perf_counter() - started) * 1000),
                    )
                await queue.put({"event": "complete", "report": report})
            except asyncio.TimeoutError:
                await queue.put({"event": "error", "message": f"Research timed out after {self.timeout_s:.0f}s."})
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Research run {session_id} failed")
                await queue.put({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                await queue.put(None)

        logger.info(f"Starting research run {session_id}: {request.query!r}")
        yield {"event": "session_started", "session_id": session_id, "query": request.query, "mode": mode}
        if self._slots.locked():
            yield {"event": "queued", "message": "Waiting for a free research slot."}
        task = asyncio.create_task(runner())
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:  # client disconnected or finished: never leave the pipeline running unobserved
            if not task.done():
                task.cancel()

    async def run(self, request: ResearchRequest) -> dict[str, Any]:
        """Run to completion and return the report (non-streaming)."""
        async for event in self.stream(request):
            if event["event"] == "complete":
                return event["report"]
            if event["event"] == "error":
                raise PipelineError(event["message"])
        raise PipelineError("Research ended without a result.")
