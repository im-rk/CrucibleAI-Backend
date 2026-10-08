"""Client-side rate limiting for providers that cap tokens per minute (e.g. Groq's free tier: 8,000).

Hitting the provider's limit costs a failed request and a retry wait; scheduling around it costs
only a short, deliberate delay. The limiter keeps a sliding window of the tokens each call used (the
estimate at reservation time, replaced by the provider's exact count afterwards) and makes a call wait
until it fits. Reservations are served in order, so one large call cannot be starved by small ones.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from time import monotonic


class Reservation:
    __slots__ = ("at", "tokens")

    def __init__(self, at: float, tokens: int) -> None:
        self.at = at
        self.tokens = tokens


class TokenRateLimiter:
    def __init__(self, tokens_per_minute: int, *, window_s: float = 60.0) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
        self.limit = tokens_per_minute
        self.window_s = window_s
        self._events: deque[Reservation] = deque()
        self._lock = asyncio.Lock()

    def _expire(self, now: float) -> None:
        while self._events and now - self._events[0].at >= self.window_s:
            self._events.popleft()

    def used(self) -> int:
        self._expire(monotonic())
        return sum(e.tokens for e in self._events)

    async def acquire(self, estimate: int) -> tuple[Reservation, float]:
        """Wait until `estimate` tokens fit in the window; return (reservation, seconds waited)."""
        estimate = min(max(int(estimate), 1), self.limit)  # a call larger than the whole limit can never fit
        started = monotonic()
        async with self._lock:
            while True:
                now = monotonic()
                self._expire(now)
                used = sum(e.tokens for e in self._events)
                if used + estimate <= self.limit or not self._events:
                    reservation = Reservation(now, estimate)
                    self._events.append(reservation)
                    return reservation, now - started
                # wait for enough of the oldest reservations to leave the window
                freed, needed = 0, used + estimate - self.limit
                wake = now + self.window_s
                for e in self._events:
                    freed += e.tokens
                    if freed >= needed:
                        wake = e.at + self.window_s
                        break
                await asyncio.sleep(max(0.01, min(wake - now + 0.02, 5.0)))

    @staticmethod
    def settle(reservation: Reservation, actual_tokens: int) -> None:
        """Replace the estimate with the provider's exact count once the call is done."""
        reservation.tokens = max(int(actual_tokens), 1)


def estimate_tokens(prompt: str, system: str | None, schema_json: str, max_tokens: int) -> int:
    """Rough upper-ish estimate: ~3.2 characters per token in, plus a capped allowance for the answer."""
    return math.ceil((len(prompt) + len(system or "") + len(schema_json)) / 3.2) + min(max_tokens, 900)
