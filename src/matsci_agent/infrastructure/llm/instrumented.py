"""Production wrapper around any provider client: response cache, token accounting, budget, tracing.

Agents keep calling `structured(...)`; this layer is where cross-cutting concerns live so no
provider client or agent has to know about them:

* **cache**: identical (provider, model, system, prompt, schema) calls are answered from a local
  SQLite file. It saves rate-limited free-tier quota, makes evaluations repeatable and turns a
  rehearsed demo query into an instant replay. Failures are never cached.
* **accounting**: exact token counts from the provider when it reports them, an estimate otherwise.
* **budget**: a per-run cap on real calls, so one runaway loop cannot burn a quota.
* **tracing and metrics**: every call is a span on the run's trace and a Prometheus counter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from matsci_agent.infrastructure.llm.anthropic_client import LLMError
from matsci_agent.infrastructure.llm.ratelimit import TokenRateLimiter, estimate_tokens
from matsci_agent.observability.metrics import METRICS
from matsci_agent.observability.trace import current_agent, current_trace, usage_sink

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMCache:
    """A tiny persistent key-value store for validated model outputs."""

    def __init__(self, path: str | Path, ttl_s: float = 7 * 24 * 3600) -> None:
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "created REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0)"
        )
        self._db.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value, created FROM llm_cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            if time.time() - row[1] > self.ttl_s:
                self._db.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                self._db.commit()
                return None
            self._db.execute("UPDATE llm_cache SET hits = hits + 1 WHERE key = ?", (key,))
            self._db.commit()
            return row[0]

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO llm_cache (key, value, created, hits) VALUES (?, ?, ?, 0)",
                (key, value, time.time()),
            )
            self._db.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            entries, hits = self._db.execute("SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM llm_cache").fetchone()
        return {"entries": int(entries), "hits": int(hits)}

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM llm_cache")
            self._db.commit()


def cache_key(provider: str, model: str, system: str | None, prompt: str, schema: type[BaseModel]) -> str:
    payload = json.dumps([provider, model, system or "", prompt, schema.model_json_schema()], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class InstrumentedLLM:
    """Same interface as the provider clients (`available`, `unavailable_reason`, `model`, `structured`)."""

    def __init__(
        self, inner: Any, *, cache: LLMCache | None = None, cache_mode: str = "on",
        limiter: TokenRateLimiter | None = None,
    ) -> None:
        if cache_mode not in {"on", "off", "refresh"}:
            raise ValueError(f"cache_mode must be 'on', 'off' or 'refresh', not '{cache_mode}'")
        self.inner = inner
        self.cache = cache if cache_mode != "off" else None
        self.cache_mode = cache_mode
        self.limiter = limiter
        self.provider = type(inner).__name__

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def available(self) -> bool:
        return self.inner.available

    @property
    def unavailable_reason(self) -> str | None:
        return self.inner.unavailable_reason

    async def structured(
        self, prompt: str, schema: type[T], *, system: str | None = None, max_tokens: int = 4096
    ) -> T:
        trace = current_trace.get()
        agent = current_agent.get() or "-"
        key = cache_key(self.provider, self.model, system, prompt, schema) if self.cache else ""

        if self.cache and self.cache_mode == "on":
            cached = self.cache.get(key)
            if cached is not None:
                try:
                    result = schema.model_validate_json(cached)
                except ValidationError:
                    cached = None  # schema changed since it was stored: treat as a miss
                else:
                    now = trace.now_ms() if trace else 0
                    if trace:
                        trace.record("llm", schema.__name__, start_ms=now, duration_ms=0, status="cache_hit", model=self.model)
                    METRICS.inc("llm_calls_total", model=self.model, agent=agent, outcome="cache_hit")
                    return result

        if not self.inner.available:
            raise LLMError(self.inner.unavailable_reason or "LLM unavailable")
        if trace is not None and not trace.reserve_llm_call():
            raise LLMError(f"the per-run LLM call budget ({trace.llm_call_budget}) is used up")

        reservation, waited = None, 0.0
        if self.limiter is not None:
            estimate = estimate_tokens(prompt, system, json.dumps(schema.model_json_schema()), max_tokens)
            reservation, waited = await self.limiter.acquire(estimate)
            if waited >= 0.5:
                logger.info(f"Waited {waited:.1f}s for the provider token budget before calling {self.model}")
        sink: dict = {}
        token = usage_sink.set(sink)
        start_ms = trace.now_ms() if trace else 0
        started = perf_counter()
        try:
            result = await self.inner.structured(prompt, schema, system=system, max_tokens=max_tokens)
        except LLMError as exc:
            elapsed = perf_counter() - started
            if reservation is not None:
                self.limiter.settle(reservation, 1)  # the failed call used none of the budget
            if trace:
                trace.record("llm", schema.__name__, start_ms=start_ms, duration_ms=int(elapsed * 1000),
                             status="error", model=self.model, error=str(exc)[:160])
            METRICS.inc("llm_calls_total", model=self.model, agent=agent, outcome="error")
            raise
        finally:
            usage_sink.reset(token)

        elapsed = perf_counter() - started
        estimated = "prompt_tokens" not in sink
        prompt_tokens = sink.get("prompt_tokens") or (len(prompt) + len(system or "")) // 4
        completion_tokens = sink.get("completion_tokens") or len(result.model_dump_json()) // 4
        if reservation is not None:
            self.limiter.settle(reservation, prompt_tokens + completion_tokens)
        if trace:
            trace.record("llm", schema.__name__, start_ms=start_ms, duration_ms=int(elapsed * 1000), model=self.model,
                         prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, tokens_estimated=estimated,
                         rate_limit_wait_ms=int(waited * 1000))
        METRICS.inc("llm_calls_total", model=self.model, agent=agent, outcome="ok")
        METRICS.inc("llm_tokens_total", prompt_tokens, model=self.model, direction="prompt")
        METRICS.inc("llm_tokens_total", completion_tokens, model=self.model, direction="completion")
        METRICS.observe("llm_call_seconds", elapsed, model=self.model)
        if self.cache:
            self.cache.put(key, result.model_dump_json())
        return result
