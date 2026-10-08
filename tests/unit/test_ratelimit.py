"""Client-side token-per-minute scheduling."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import BaseModel

from matsci_agent.infrastructure.llm.instrumented import InstrumentedLLM
from matsci_agent.infrastructure.llm.ratelimit import TokenRateLimiter, estimate_tokens
from matsci_agent.observability import RunTrace, current_trace, report_usage


def test_calls_that_fit_the_window_are_not_delayed() -> None:
    async def go():
        limiter = TokenRateLimiter(1000, window_s=5)
        waits = [(await limiter.acquire(300))[1] for _ in range(3)]
        return waits, limiter.used()

    waits, used = asyncio.run(go())
    assert max(waits) < 0.05 and used == 900


def test_a_call_that_does_not_fit_waits_for_the_window_to_free_up() -> None:
    async def go():
        limiter = TokenRateLimiter(100, window_s=0.3)
        await limiter.acquire(70)
        t = time.monotonic()
        _, waited = await limiter.acquire(70)
        return waited, time.monotonic() - t

    waited, elapsed = asyncio.run(go())
    assert 0.2 <= waited <= 0.6 and elapsed >= 0.2


def test_settling_with_the_exact_count_frees_over_reserved_budget() -> None:
    async def go():
        limiter = TokenRateLimiter(100, window_s=5)
        reservation, _ = await limiter.acquire(90)
        limiter.settle(reservation, 10)  # the call really used 10
        _, waited = await limiter.acquire(80)
        return waited

    assert asyncio.run(go()) < 0.05


def test_an_estimate_larger_than_the_whole_limit_is_clamped_instead_of_waiting_forever() -> None:
    async def go():
        limiter = TokenRateLimiter(100, window_s=5)
        reservation, waited = await limiter.acquire(10_000)
        return reservation.tokens, waited

    tokens, waited = asyncio.run(go())
    assert tokens == 100 and waited < 0.05


def test_reservations_are_served_in_order() -> None:
    async def go():
        limiter = TokenRateLimiter(100, window_s=0.2)
        await limiter.acquire(100)
        order: list[str] = []

        async def call(name: str, tokens: int):
            await limiter.acquire(tokens)
            order.append(name)

        first = asyncio.create_task(call("big", 90))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(call("small", 5))
        await asyncio.gather(first, second)
        return order

    assert asyncio.run(go()) == ["big", "small"]


def test_the_limit_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TokenRateLimiter(0)


def test_the_estimate_grows_with_the_prompt_and_is_capped_for_the_answer() -> None:
    small = estimate_tokens("x" * 320, None, "{}", 4000)
    big = estimate_tokens("x" * 3200, None, "{}", 4000)
    assert (small, big) == (1001, 1901)  # ~3.2 chars per token in, plus the answer allowance capped at 900


class Answer(BaseModel):
    words: list[str]


class Provider:
    model = "m"
    available = True
    unavailable_reason = None

    async def structured(self, prompt, schema, *, system=None, max_tokens=4096):
        report_usage(40, 10)
        return schema(words=["a"])


def test_the_wrapper_waits_for_the_budget_and_reports_the_wait_in_the_trace() -> None:
    limiter = TokenRateLimiter(120, window_s=0.3)
    llm = InstrumentedLLM(Provider(), cache=None, cache_mode="off", limiter=limiter)
    trace = RunTrace("r")

    async def go():
        current_trace.set(trace)
        await llm.structured("p" * 200, Answer, max_tokens=50)  # estimate ~ 63 + schema + 50 -> settles to 50
        await llm.structured("p" * 200, Answer, max_tokens=50)
        await llm.structured("p" * 200, Answer, max_tokens=50)

    asyncio.run(go())
    summary = trace.summary()["llm"]
    assert summary["calls"] == 3 and summary["total_tokens"] == 150
    assert summary["rate_limit_wait_ms"] > 0  # the third call had to wait for the window
