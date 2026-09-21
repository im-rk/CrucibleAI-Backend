"""Graph wiring (retry edge, failure isolation) and ResearchService behaviour with stub agents."""

from __future__ import annotations

import asyncio

import pytest

from matsci_agent.agents.orchestrator import build_research_graph, initial_state, run_config
from matsci_agent.domain.schemas import ResearchRequest
from matsci_agent.services.research_srv import PipelineError, ResearchService


class Recorder:
    """Stub agent that records that it ran and returns a canned update."""

    def __init__(self, name, update=None, fail=False, delay=0.0):
        self.name, self.update, self.fail, self.delay, self.calls = name, update or {}, fail, delay, 0

    async def run(self, state, config=None):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"{self.name} broke")
        return {"agent_log": [{"step": self.name, "details": "ran"}], **self.update}


class RetryingSearch(Recorder):
    """Finds nothing relevant for `weak_passes` passes, then enough."""

    def __init__(self, weak_passes, max_retries=3, fail=False):
        super().__init__("search", fail=fail)
        self.weak_passes, self.max_retries = weak_passes, max_retries

    async def run(self, state, config=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("search broke")
        strong = self.calls > self.weak_passes
        return {
            "search_queries": [*state.get("search_queries", []), f"q{self.calls}"],
            "retrieved_papers": [{"id": str(i)} for i in range(5 if strong else 1)],
            "retry_count": state.get("retry_count", 0) + (1 if state.get("search_queries") else 0),
            "agent_log": [{"step": "LITERATURE_SEARCH", "details": f"pass {self.calls}"}],
        }

    def should_retry(self, state):
        return len(state.get("retrieved_papers", [])) < 5 and state.get("retry_count", 0) < self.max_retries


def make_graph(search, **overrides):
    agents = {
        "extract": Recorder("extract"), "screen": Recorder("screen"),
        "hypothesis": Recorder("hypothesis"), "report": Recorder("report"), **overrides,
    }
    graph = build_research_graph(search, agents["extract"], agents["screen"], agents["hypothesis"], agents["report"])
    return graph, agents


async def invoke(graph, listener=None):
    return await graph.ainvoke(initial_state("goal"), run_config(listener))


async def test_search_loops_until_results_are_strong_then_runs_every_stage_once() -> None:
    search = RetryingSearch(weak_passes=2)
    graph, agents = make_graph(search)
    state = await invoke(graph)
    assert search.calls == 3  # two weak passes, then a strong one
    assert all(a.calls == 1 for a in agents.values())
    assert [e["step"] for e in state["agent_log"]].count("LITERATURE_SEARCH") == 3


async def test_retries_stop_at_the_limit_even_if_results_stay_weak() -> None:
    search = RetryingSearch(weak_passes=99, max_retries=3)
    graph, agents = make_graph(search)
    await invoke(graph)
    assert search.calls == 4 and agents["extract"].calls == 1  # 1 initial + 3 retries, then move on


async def test_a_crashing_search_cannot_loop_forever() -> None:
    search = RetryingSearch(weak_passes=0, fail=True)
    graph, agents = make_graph(search)
    state = await invoke(graph)
    assert 1 < search.calls <= 4  # each crash counts as an attempt, so the loop is bounded
    assert agents["report"].calls == 1
    assert any("SearchAgent failed" in w for w in state["warnings"])


async def test_a_crashing_agent_becomes_a_warning_and_the_rest_still_run() -> None:
    graph, agents = make_graph(RetryingSearch(0), screen=Recorder("screen", fail=True))
    events = []

    async def listener(event):
        events.append(event)

    state = await invoke(graph, listener)
    assert agents["hypothesis"].calls == 1 and agents["report"].calls == 1
    assert any("ScreeningAgent failed and was skipped" in w for w in state["warnings"])
    assert {"event": "agent_error", "agent": "ScreeningAgent", "message": "screen broke"} in events


async def test_log_entries_get_the_agent_name_and_duration() -> None:
    graph, _ = make_graph(RetryingSearch(0))
    state = await invoke(graph)
    assert all({"agent", "duration_ms"} <= set(entry) for entry in state["agent_log"])
    assert {e["agent"] for e in state["agent_log"]} >= {"SearchAgent", "ExtractionAgent", "ReportBuilder"}


# ---- ResearchService ---------------------------------------------------------------------------


class FakeGraph:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    async def ainvoke(self, state, config):
        return await self.behaviour(state, config)


def service_for(behaviour, **kwargs):
    return ResearchService(FakeGraph(behaviour), lambda: {"llm": None}, **kwargs)


REQUEST = ResearchRequest(query="solid electrolytes")


async def test_stream_forwards_node_events_then_completes_with_a_report() -> None:
    async def behaviour(state, config):
        await config["configurable"]["emit"]({"event": "papers_found", "count": 3})
        return initial_state("solid electrolytes")

    events = [e async for e in service_for(behaviour).stream(REQUEST)]
    assert [e["event"] for e in events] == ["session_started", "papers_found", "complete"]
    assert events[0]["session_id"] == events[-1]["report"]["session_id"]
    assert events[-1]["report"]["query"] == "solid electrolytes"


async def test_run_raises_pipeline_error_when_the_graph_fails() -> None:
    async def behaviour(state, config):
        raise ValueError("bad state")

    with pytest.raises(PipelineError, match="ValueError: bad state"):
        await service_for(behaviour).run(REQUEST)


async def test_timeout_is_reported_as_an_error_event() -> None:
    async def behaviour(state, config):
        await asyncio.sleep(5)

    events = [e async for e in service_for(behaviour, timeout_s=0.05).stream(REQUEST)]
    assert events[-1]["event"] == "error" and "timed out" in events[-1]["message"]


async def test_abandoning_the_stream_cancels_the_pipeline() -> None:
    cancelled = asyncio.Event()

    async def behaviour(state, config):
        await config["configurable"]["emit"]({"event": "papers_found", "count": 1})
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    stream = service_for(behaviour).stream(REQUEST)
    assert (await stream.__anext__())["event"] == "session_started"
    assert (await stream.__anext__())["event"] == "papers_found"  # the pipeline is now running
    await stream.aclose()  # client disconnects
    await asyncio.wait_for(cancelled.wait(), timeout=2)


async def test_concurrent_runs_are_limited_and_the_waiter_is_told_it_is_queued() -> None:
    gate = asyncio.Event()

    async def behaviour(state, config):
        await gate.wait()
        return initial_state("solid electrolytes")

    service = service_for(behaviour, max_concurrent=1)
    first = asyncio.create_task(service.run(REQUEST))
    await asyncio.sleep(0.05)
    events = []

    async def second():
        async for event in service.stream(REQUEST):
            events.append(event["event"])

    second_task = asyncio.create_task(second())
    await asyncio.sleep(0.05)
    assert "queued" in events and "complete" not in events
    gate.set()
    await asyncio.gather(first, second_task)
    assert events[-1] == "complete"
