"""Per-run tracing: every agent step, LLM call and retrieval becomes a span.

The model follows the OpenTelemetry GenAI idea (an operation span carrying model, token counts and
outcome) without needing a collector: spans are kept in memory, summarised into the report, and
shown in the UI. Everything reaches the current run through context variables, so code deep in the
call stack (an LLM client, a retriever) can report without being handed the trace.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Iterator

current_trace: ContextVar["RunTrace | None"] = ContextVar("current_trace", default=None)
current_agent: ContextVar[str | None] = ContextVar("current_agent", default=None)
# Set by the LLM wrapper around one call; provider clients write exact token usage into it.
usage_sink: ContextVar[dict | None] = ContextVar("usage_sink", default=None)

MAX_SPANS = 400


@dataclass
class Span:
    kind: str  # agent | llm | retrieval | tool
    name: str
    agent: str | None
    start_ms: int
    duration_ms: int
    status: str = "ok"  # ok | error | cache_hit
    attrs: dict[str, Any] = field(default_factory=dict)


class RunTrace:
    """Thread-safe span collector for one research run."""

    def __init__(self, run_id: str, *, llm_call_budget: int | None = None) -> None:
        self.run_id = run_id
        self.llm_call_budget = llm_call_budget
        self._t0 = perf_counter()
        self._lock = threading.Lock()
        self._spans: list[Span] = []
        self._dropped = 0
        self._llm_started = 0

    def now_ms(self) -> int:
        return int((perf_counter() - self._t0) * 1000)

    def reserve_llm_call(self) -> bool:
        """Count a real (non-cached) LLM call against the budget; False when the budget is used up."""
        with self._lock:
            if self.llm_call_budget is not None and self._llm_started >= self.llm_call_budget:
                return False
            self._llm_started += 1
            return True

    def record(self, kind: str, name: str, *, start_ms: int, duration_ms: int, status: str = "ok",
               agent: str | None = None, **attrs: Any) -> None:
        span = Span(kind, name, agent if agent is not None else current_agent.get(), start_ms, duration_ms, status, attrs)
        with self._lock:
            if len(self._spans) < MAX_SPANS:
                self._spans.append(span)
            else:
                self._dropped += 1

    @contextmanager
    def span(self, kind: str, name: str, **attrs: Any) -> Iterator[dict]:
        """Time a block; the yielded dict can be filled in with more attributes before it ends."""
        start = self.now_ms()
        extra: dict[str, Any] = dict(attrs)
        status = "ok"
        try:
            yield extra
        except Exception:
            status = "error"
            raise
        finally:
            self.record(kind, name, start_ms=start, duration_ms=self.now_ms() - start, status=status, **extra)

    # ---- reporting ------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        with self._lock:
            spans = list(self._spans)
        llm = [s for s in spans if s.kind == "llm"]
        real = [s for s in llm if s.status != "cache_hit"]
        prompt = sum(int(s.attrs.get("prompt_tokens") or 0) for s in real)
        completion = sum(int(s.attrs.get("completion_tokens") or 0) for s in real)
        per_agent: dict[str, dict[str, Any]] = defaultdict(lambda: {"duration_ms": 0, "llm_calls": 0, "tokens": 0, "retrievals": 0})
        for s in spans:
            if s.kind == "agent":
                per_agent[s.name]["duration_ms"] += s.duration_ms
            elif s.agent:
                bucket = per_agent[s.agent]
                if s.kind == "llm" and s.status != "cache_hit":
                    bucket["llm_calls"] += 1
                    bucket["tokens"] += int(s.attrs.get("prompt_tokens") or 0) + int(s.attrs.get("completion_tokens") or 0)
                elif s.kind == "retrieval":
                    bucket["retrievals"] += 1
        return {
            "run_id": self.run_id,
            "total_ms": self.now_ms(),
            "llm": {
                "calls": len(real),
                "errors": sum(s.status == "error" for s in real),
                "cache_hits": sum(s.status == "cache_hit" for s in llm),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "latency_ms": sum(s.duration_ms for s in real),
                "rate_limit_wait_ms": sum(int(s.attrs.get("rate_limit_wait_ms") or 0) for s in real),
                "estimated_tokens": any(s.attrs.get("tokens_estimated") for s in real),
                "models": sorted({str(s.attrs.get("model")) for s in llm if s.attrs.get("model")}),
                "budget": self.llm_call_budget,
            },
            "retrieval": {
                "calls": sum(s.kind == "retrieval" for s in spans),
                "latency_ms": sum(s.duration_ms for s in spans if s.kind == "retrieval"),
            },
            "tools": {"calls": sum(s.kind == "tool" for s in spans)},
            "agents": dict(per_agent),
            "spans_dropped": self._dropped,
        }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            spans = [asdict(s) for s in self._spans]
        return {"summary": self.summary(), "spans": spans}


# ---- helpers used by code that should not need to know about the trace ---------------------


@contextmanager
def trace_span(kind: str, name: str, **attrs: Any) -> Iterator[dict]:
    """A span on the current run's trace; does nothing outside a run."""
    trace = current_trace.get()
    if trace is None:
        yield dict(attrs)
        return
    with trace.span(kind, name, **attrs) as extra:
        yield extra


def report_usage(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    """Called by LLM clients with the provider's exact token counts for the call in progress."""
    sink = usage_sink.get()
    if sink is not None:
        sink["prompt_tokens"] = int(prompt_tokens or 0)
        sink["completion_tokens"] = int(completion_tokens or 0)
