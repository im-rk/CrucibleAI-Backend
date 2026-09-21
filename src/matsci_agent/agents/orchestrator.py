"""LangGraph orchestration for the end-to-end research workflow.

    search --(too few relevant chunks and retries left)--> search   (re-retrieval loop)
    search --> extract --> screen --> hypothesise --> report --> END

Every node is wrapped so it announces itself, is timed, and can fail without taking the run
down: a failure becomes a warning and the pipeline continues with whatever state it has.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

from langgraph.graph import END, START, StateGraph

from matsci_agent.agents.events import emit
from matsci_agent.domain.state import ResearchState, initial_state

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 40  # search retries (<=4 passes) + 5 other nodes, with wide headroom

Node = Callable[[ResearchState, Any], Awaitable[dict]]


def _instrument(
    label: str, step: str, fn: Node, on_error: Callable[[ResearchState], dict] | None = None
) -> Node:
    async def node(state: ResearchState, config) -> dict:
        await emit(config, "agent_start", agent=label)
        started = perf_counter()
        try:
            update = await fn(state, config)
        except Exception as exc:  # noqa: BLE001 - one agent failing must not kill the run
            logger.exception(f"{label} failed")
            update = {
                **(on_error(state) if on_error else {}),
                "warnings": [f"{label} failed and was skipped: {type(exc).__name__}: {exc}"],
                "agent_log": [{"step": step, "details": f"Failed: {type(exc).__name__}: {exc}"}],
            }
            await emit(config, "agent_error", agent=label, message=str(exc))
        elapsed = int((perf_counter() - started) * 1000)
        for entry in update.get("agent_log", []):
            entry.update(agent=label, duration_ms=elapsed)
        await emit(config, "agent_complete", agent=label, duration_ms=elapsed)
        return update

    return node


def build_research_graph(search_agent, extraction_agent, screening_agent, hypothesis_agent, report_agent):
    """Compile the workflow. Each argument is an object exposing `run(state, config)`."""
    graph = StateGraph(ResearchState)
    graph.add_node(
        "search",
        # A crashed search must still count as an attempt, or the retry edge would loop forever.
        _instrument("SearchAgent", "LITERATURE_SEARCH", search_agent.run,
                    on_error=lambda s: {"retry_count": s.get("retry_count", 0) + 1}),
    )
    graph.add_node("extract", _instrument("ExtractionAgent", "EXTRACTION", extraction_agent.run))
    graph.add_node("screen", _instrument("ScreeningAgent", "SCREENING", screening_agent.run))
    graph.add_node("hypothesise", _instrument("HypothesisAgent", "HYPOTHESIS", hypothesis_agent.run))
    graph.add_node("report", _instrument("ReportBuilder", "GROUNDING_CHECK", report_agent.run))

    graph.add_edge(START, "search")
    graph.add_conditional_edges(
        "search",
        lambda state: "search" if search_agent.should_retry(state) else "extract",
        {"search": "search", "extract": "extract"},
    )
    graph.add_edge("extract", "screen")
    graph.add_edge("screen", "hypothesise")
    graph.add_edge("hypothesise", "report")
    graph.add_edge("report", END)
    return graph.compile()


def run_config(emit_fn=None) -> dict:
    """Invocation config carrying the event listener and the recursion limit."""
    return {"configurable": {"emit": emit_fn}, "recursion_limit": RECURSION_LIMIT}


__all__ = ["build_research_graph", "initial_state", "run_config"]
