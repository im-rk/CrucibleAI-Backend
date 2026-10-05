"""Groq client with the same interface as `LLMClient` (structured output via JSON Schema).

Selected with LLM_PROVIDER=groq. Groq serves an OpenAI-compatible chat-completions API, so this
calls it directly with httpx (already a dependency) rather than adding an SDK.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from matsci_agent.infrastructure.llm.anthropic_client import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
RETRYABLE = {429, 500, 502, 503}
MAX_RETRY_WAIT_S = 15.0  # longest single wait between attempts
MAX_TOTAL_WAIT_S = 20.0  # most one call may spend waiting before it gives up
# After a call gives up on a temporary failure (overload, per-minute token limit) the client pauses
# so the remaining steps of a run use rules at once instead of each repeating the same waits.
COOLDOWN_S = 180.0


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """Groq's Retry-After header (seconds) if present, else exponential backoff."""
    if response is not None:
        try:
            return min(float(response.headers["retry-after"]), MAX_RETRY_WAIT_S)
        except (KeyError, ValueError):
            pass
    return min(2.0 * 2**attempt, MAX_RETRY_WAIT_S)


def _is_daily_limit(message: str) -> bool:
    """Per-day limits (requests or tokens) reset only once a day, so retrying cannot help."""
    return bool(re.search(r"per day|\((?:RPD|TPD)\)", message, re.IGNORECASE))


class GroqClient:
    """`available` is False without a key, after Groq rejects the key, and while paused."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        max_retries: int = 3,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._timeout = timeout_s
        self._disabled_reason: str | None = None
        self._paused_until = 0.0
        self._paused_reason: str | None = None

    @property
    def available(self) -> bool:
        return bool(self._api_key) and self._disabled_reason is None and time.monotonic() >= self._paused_until

    @property
    def unavailable_reason(self) -> str | None:
        if not self._api_key:
            return "GROQ_API_KEY is not configured"
        if self._disabled_reason:
            return self._disabled_reason
        if time.monotonic() < self._paused_until:
            return self._paused_reason
        return None

    def _disable(self, reason: str) -> LLMError:
        if self._disabled_reason is None:
            logger.error(f"LLM disabled for this process: {reason}")
        self._disabled_reason = reason
        return LLMError(reason)

    def _pause(self, detail: str) -> LLMError:
        """Give up for a while after a temporary failure; unlike `_disable`, this heals itself."""
        self._paused_until = time.monotonic() + COOLDOWN_S
        self._paused_reason = (
            f"Groq is overloaded or rate limited ({detail}); paused for {COOLDOWN_S:.0f}s, using rules meanwhile"
        )
        logger.warning(self._paused_reason)
        return LLMError(f"Groq API error: {detail}")

    async def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        max_tokens: int = 4096,  # noqa: ARG002 - see below
    ) -> T:
        """Run one prompt and return the response parsed into `schema`."""
        if not self.available:
            raise LLMError(self.unavailable_reason or "LLM unavailable")

        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            # max_tokens is deliberately not forwarded: reasoning models count their hidden reasoning
            # against the completion cap, so the small caps tuned for Claude would truncate the JSON.
            "response_format": {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
        }
        if "gpt-oss" in self.model:
            body["reasoning_effort"] = "low"  # extraction and scoring do not need long reasoning; saves tokens

        data = (await self._post(body)).json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("Groq returned no answer")
        choice = choices[0]
        text = (choice.get("message") or {}).get("content") or ""
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(
                f"Model output did not match {schema.__name__} (finish_reason={choice.get('finish_reason')}): {exc}"
            ) from exc

    async def _post(self, body: dict) -> httpx.Response:
        waited = 0.0
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            for attempt in range(self._max_retries + 1):
                try:
                    response = await http.post(self._url, json=body, headers={"Authorization": f"Bearer {self._api_key}"})
                except httpx.HTTPError as exc:
                    delay = _retry_delay(None, attempt)
                    if attempt == self._max_retries or waited + delay > MAX_TOTAL_WAIT_S:
                        raise self._pause(f"request failed: {exc}") from exc
                    waited += delay
                    await asyncio.sleep(delay)
                    continue

                if response.status_code == 200:
                    return response
                message = self._error_message(response)
                if response.status_code in (401, 403):
                    raise self._disable(f"Groq rejected the API key or its permissions ({message})")
                if response.status_code == 404 or "model_not_found" in response.text:
                    raise self._disable(f"Groq model '{self.model}' is not available to this key ({message})")
                if response.status_code == 429 and _is_daily_limit(message):
                    raise self._disable(
                        f"the Groq daily limit for {self.model} is used up; it resets daily "
                        f"({message[:120]})"
                    )
                if response.status_code in RETRYABLE:
                    delay = _retry_delay(response, attempt)
                    if attempt < self._max_retries and waited + delay <= MAX_TOTAL_WAIT_S:
                        logger.warning(f"Groq {response.status_code}; retrying in {delay:.0f}s ({message[:80]})")
                        waited += delay
                        await asyncio.sleep(delay)
                        continue
                    raise self._pause(f"{response.status_code}: {message[:100]}")
                raise LLMError(f"Groq API error {response.status_code}: {message}")
        raise LLMError("Groq request failed")  # unreachable: the loop always returns or raises

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            return str(response.json().get("error", {}).get("message", response.text))[:300]
        except ValueError:
            return response.text[:300]
