"""Service layer entry point for running the research workflow."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from time import perf_counter
from typing import Any

from matsci_agent.agents.orchestrator import initial_state, run_config
from matsci_agent.core.config import settings
from matsci_agent.domain.schemas import ResearchRequest
from matsci_agent.observability.metrics import METRICS
from matsci_agent.observability.trace import RunTrace, current_trace
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
        persistence=None,
        timeout_s: float = 300,
        max_concurrent: int = 2,
        reports=None,
    ) -> None:
        self.graph = graph
        self.mode = mode
        self.persistence = persistence
        self.reports = reports
        self.timeout_s = timeout_s
        self._slots = asyncio.Semaphore(max_concurrent)

    async def stream(self, request: ResearchRequest) -> AsyncIterator[dict]:
        """Yield progress events, ending with `complete` (the report) or `error`."""
        session_id = str(uuid.uuid4())
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        mode = self.mode()

        async def emit(event: dict) -> None:
            await queue.put(event)

        trace = RunTrace(session_id, llm_call_budget=settings.llm_max_calls_per_run)

        async def runner() -> None:
            current_trace.set(trace)
            try:
                async with self._slots:
                    started = perf_counter()
                    if self.persistence:
                        await self.persistence.start(session_id, request.query)
                    state = await asyncio.wait_for(
                        self.graph.ainvoke(
                            initial_state(request.query, request.max_results, request.plan), run_config(emit)
                        ),
                        timeout=self.timeout_s,
                    )
                    report = build_report(
                        state, session_id=session_id, query=request.query,
                        # recomputed: the LLM may have been disabled mid-run (e.g. no credits)
                        max_results=request.max_results, mode=self.mode(),
                        duration_ms=int((perf_counter() - started) * 1000), trace=trace.to_dict(),
                    )
                    METRICS.inc("runs_total", outcome="completed")
                    METRICS.observe("run_seconds", report["duration_ms"] / 1000)
                if self.reports is not None:
                    self.reports.put(session_id, report)
                if self.persistence:
                    await self.persistence.finish(session_id, report)
                await queue.put({"event": "complete", "report": report})
            except asyncio.TimeoutError:
                message = f"Research timed out after {self.timeout_s:.0f}s."
                METRICS.inc("runs_total", outcome="timeout")
                await self._record_failure(session_id, message)
                await queue.put({"event": "error", "message": message})
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Research run {session_id} failed")
                METRICS.inc("runs_total", outcome="failed")
                message = f"{type(exc).__name__}: {exc}"
                await self._record_failure(session_id, message)
                await queue.put({"event": "error", "message": message})
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

    async def _record_failure(self, session_id: str, message: str) -> None:
        if self.persistence:
            await self.persistence.fail(session_id, message)

    async def run(self, request: ResearchRequest) -> dict[str, Any]:
        """Run to completion and return the report (non-streaming)."""
        async for event in self.stream(request):
            if event["event"] == "complete":
                return event["report"]
            if event["event"] == "error":
                raise PipelineError(event["message"])
        raise PipelineError("Research ended without a result.")
