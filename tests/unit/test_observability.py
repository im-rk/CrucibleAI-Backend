"""Run tracing, metrics and the instrumented LLM wrapper (cache, budget, token accounting)."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import BaseModel

from matsci_agent.infrastructure.llm.anthropic_client import LLMError
from matsci_agent.infrastructure.llm.instrumented import InstrumentedLLM, LLMCache, cache_key
from matsci_agent.observability import METRICS, Metrics, RunTrace, current_agent, current_trace, report_usage, trace_span


class Answer(BaseModel):
    words: list[str]


class FakeProvider:
    """A provider client that counts calls and can report exact token usage."""

    model = "fake-model"

    def __init__(self, *, report_tokens: bool = True, fail: Exception | None = None, available: bool = True) -> None:
        self.calls = 0
        self.report_tokens = report_tokens
        self.fail = fail
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    @property
    def unavailable_reason(self) -> str | None:
        return None if self._available else "no key"

    async def structured(self, prompt, schema, *, system=None, max_tokens=4096):
        self.calls += 1
        if self.fail:
            raise self.fail
        if self.report_tokens:
            report_usage(120, 30)
        return schema(words=[prompt[:3]])


@pytest.fixture(autouse=True)
def fresh_metrics():
    METRICS.reset()
    yield
    METRICS.reset()


def run_in_trace(coro_fn, trace: RunTrace, agent: str = "SearchAgent"):
    async def wrapper():
        current_trace.set(trace)
        current_agent.set(agent)
        return await coro_fn()

    return asyncio.run(wrapper())


# ---- trace ------------------------------------------------------------------------------------


def test_spans_are_timed_and_summarised_per_agent() -> None:
    trace = RunTrace("run-1", llm_call_budget=10)
    trace.record("agent", "SearchAgent", start_ms=0, duration_ms=500, agent="SearchAgent")
    trace.record("llm", "Answer", start_ms=10, duration_ms=200, agent="SearchAgent", model="m", prompt_tokens=100, completion_tokens=20)
    trace.record("llm", "Answer", start_ms=220, duration_ms=0, status="cache_hit", agent="SearchAgent", model="m")
    trace.record("retrieval", "hybrid_search", start_ms=5, duration_ms=40, agent="SearchAgent")
    summary = trace.summary()

    assert summary["llm"] == {
        "calls": 1, "errors": 0, "cache_hits": 1, "prompt_tokens": 100, "completion_tokens": 20,
        "total_tokens": 120, "latency_ms": 200, "rate_limit_wait_ms": 0, "estimated_tokens": False, "models": ["m"], "budget": 10,
    }
    assert summary["retrieval"] == {"calls": 1, "latency_ms": 40}
    assert summary["agents"]["SearchAgent"] == {"duration_ms": 500, "llm_calls": 1, "tokens": 120, "retrievals": 1}


def test_span_context_manager_records_status_and_extra_attributes() -> None:
    trace = RunTrace("r")
    with trace.span("tool", "lookup", formula="GaN") as extra:
        extra["rows"] = 3
    with pytest.raises(ValueError):
        with trace.span("tool", "boom"):
            raise ValueError("x")
    spans = trace.to_dict()["spans"]
    assert spans[0]["attrs"] == {"formula": "GaN", "rows": 3} and spans[0]["status"] == "ok"
    assert spans[1]["status"] == "error"


def test_trace_span_helper_is_a_no_op_outside_a_run() -> None:
    with trace_span("retrieval", "x") as extra:
        extra["a"] = 1  # must not raise


def test_the_span_list_is_capped() -> None:
    trace = RunTrace("r")
    for i in range(450):
        trace.record("tool", "t", start_ms=i, duration_ms=1)
    assert len(trace.to_dict()["spans"]) == 400 and trace.summary()["spans_dropped"] == 50


# ---- metrics ----------------------------------------------------------------------------------


def test_metrics_render_prometheus_text() -> None:
    m = Metrics()
    m.describe("x_total", "things")
    m.inc("x_total", 2, kind='a"b')
    m.inc("x_total", 1, kind='a"b')
    m.observe("lat_seconds", 0.5, model="m")
    m.observe("lat_seconds", 1.5, model="m")
    text = m.render()
    assert '# TYPE crucible_x_total counter' in text
    assert 'crucible_x_total{kind="a\\"b"} 3' in text
    assert 'crucible_lat_seconds_count{model="m"} 2' in text and 'crucible_lat_seconds_sum{model="m"} 2' in text


# ---- the instrumented client ---------------------------------------------------------------------


def test_a_call_is_traced_with_exact_token_counts_and_counted() -> None:
    llm = InstrumentedLLM(FakeProvider(), cache=None, cache_mode="off")
    trace = RunTrace("r")
    out = run_in_trace(lambda: llm.structured("hello", Answer), trace)

    assert out == Answer(words=["hel"])
    s = trace.summary()["llm"]
    assert (s["calls"], s["prompt_tokens"], s["completion_tokens"], s["estimated_tokens"]) == (1, 120, 30, False)
    assert 'crucible_llm_calls_total{agent="SearchAgent",model="fake-model",outcome="ok"} 1' in METRICS.render()
    assert 'crucible_llm_tokens_total{direction="prompt",model="fake-model"} 120' in METRICS.render()


def test_tokens_are_estimated_when_the_provider_reports_none() -> None:
    llm = InstrumentedLLM(FakeProvider(report_tokens=False), cache=None, cache_mode="off")
    trace = RunTrace("r")
    run_in_trace(lambda: llm.structured("x" * 400, Answer), trace)
    s = trace.summary()["llm"]
    assert s["estimated_tokens"] is True and s["prompt_tokens"] == 100


def test_identical_calls_are_answered_from_the_cache() -> None:
    provider = FakeProvider()
    llm = InstrumentedLLM(provider, cache=LLMCache(":memory:"), cache_mode="on")
    trace = RunTrace("r")

    async def twice():
        a = await llm.structured("same prompt", Answer, system="s")
        b = await llm.structured("same prompt", Answer, system="s")
        c = await llm.structured("different prompt", Answer, system="s")
        return a, b, c

    a, b, c = run_in_trace(twice, trace)
    assert a == b and c != a
    assert provider.calls == 2  # the second identical call never reached the provider
    s = trace.summary()["llm"]
    assert (s["calls"], s["cache_hits"]) == (2, 1)
    assert 'outcome="cache_hit"' in METRICS.render()


def test_refresh_mode_ignores_stored_answers_but_stores_new_ones() -> None:
    provider = FakeProvider()
    cache = LLMCache(":memory:")
    asyncio.run(InstrumentedLLM(provider, cache=cache, cache_mode="on").structured("p", Answer))
    asyncio.run(InstrumentedLLM(provider, cache=cache, cache_mode="refresh").structured("p", Answer))
    assert provider.calls == 2
    asyncio.run(InstrumentedLLM(provider, cache=cache, cache_mode="on").structured("p", Answer))
    assert provider.calls == 2 and cache.stats()["entries"] == 1


def test_failures_are_not_cached_and_are_recorded() -> None:
    provider = FakeProvider(fail=LLMError("quota"))
    cache = LLMCache(":memory:")
    llm = InstrumentedLLM(provider, cache=cache, cache_mode="on")
    trace = RunTrace("r")
    with pytest.raises(LLMError):
        run_in_trace(lambda: llm.structured("p", Answer), trace)
    assert cache.stats()["entries"] == 0
    assert trace.summary()["llm"]["errors"] == 1
    assert 'outcome="error"' in METRICS.render()


def test_cache_entries_expire() -> None:
    cache = LLMCache(":memory:", ttl_s=0.05)
    cache.put("k", "v")
    assert cache.get("k") == "v"
    time.sleep(0.08)
    assert cache.get("k") is None


def test_a_stale_cached_value_that_no_longer_fits_the_schema_is_a_miss() -> None:
    provider = FakeProvider()
    cache = LLMCache(":memory:")
    llm = InstrumentedLLM(provider, cache=cache, cache_mode="on")
    cache.put(cache_key("FakeProvider", "fake-model", None, "p", Answer), '{"words": "not a list"}')
    asyncio.run(llm.structured("p", Answer))
    assert provider.calls == 1


def test_the_per_run_call_budget_stops_further_real_calls_but_not_cache_hits() -> None:
    provider = FakeProvider()
    llm = InstrumentedLLM(provider, cache=LLMCache(":memory:"), cache_mode="on")
    trace = RunTrace("r", llm_call_budget=2)

    async def calls():
        await llm.structured("a", Answer)
        await llm.structured("b", Answer)
        await llm.structured("a", Answer)  # cached: free
        with pytest.raises(LLMError, match="budget"):
            await llm.structured("c", Answer)

    run_in_trace(calls, trace)
    assert provider.calls == 2


def test_an_unavailable_provider_is_reported_without_a_call() -> None:
    llm = InstrumentedLLM(FakeProvider(available=False), cache=None, cache_mode="off")
    assert not llm.available and llm.unavailable_reason == "no key"
    with pytest.raises(LLMError, match="no key"):
        asyncio.run(llm.structured("p", Answer))


def test_an_invalid_cache_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="cache_mode"):
        InstrumentedLLM(FakeProvider(), cache_mode="sometimes")
