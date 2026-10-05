"""Google Gemini client with the same interface as `LLMClient` (structured output via JSON Schema).

Selected with LLM_PROVIDER=gemini. Calls the Gemini REST API directly with httpx, which is already
a dependency, rather than adding an SDK.
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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE = {429, 500, 503}
MAX_RETRY_WAIT_S = 15.0  # longest single wait between attempts
MAX_TOTAL_WAIT_S = 20.0  # most one call may spend waiting before it gives up
# After a call gives up on a temporary failure (overload, per-minute rate limit) the client pauses
# so the remaining steps of the run use rules at once instead of each repeating the same waits,
# which used to push a whole run past the pipeline timeout.
COOLDOWN_S = 180.0


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """Gemini's suggested wait (RetryInfo.retryDelay, e.g. "23s"), else exponential backoff."""
    try:
        details = response.json().get("error", {}).get("details", []) if response is not None else []
        for detail in details:
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", "")))
            if match:
                return min(float(match.group(1)), MAX_RETRY_WAIT_S)
    except ValueError:
        pass
    return min(2.0 * 2**attempt, MAX_RETRY_WAIT_S)


def _daily_quota(response: httpx.Response) -> str | None:
    """The exhausted quota if it is a per-day one (retrying cannot help until it resets)."""
    try:
        details = response.json().get("error", {}).get("details", [])
    except ValueError:
        return None
    for detail in details:
        for violation in detail.get("violations", []) or []:
            quota_id = str(violation.get("quotaId", ""))
            if "PerDay" in quota_id:
                limit = violation.get("quotaValue")
                tier = "free-tier " if "FreeTier" in quota_id else ""
                return f"{tier}daily quota{f' of {limit} requests' if limit else ''}"
    return None


class GeminiClient:
    """`available` is False without a key, and after Gemini rejects the key or its permissions."""

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
        self._url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
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
            return "GEMINI_API_KEY is not configured"
        if self._disabled_reason:
            return self._disabled_reason
        if time.monotonic() < self._paused_until:
            return self._paused_reason
        return None

    def _pause(self, detail: str) -> LLMError:
        """Give up for a while after a temporary failure; unlike `_disable`, this heals itself."""
        self._paused_until = time.monotonic() + COOLDOWN_S
        self._paused_reason = (
            f"Gemini is overloaded or rate limited ({detail}); paused for {COOLDOWN_S:.0f}s, using rules meanwhile"
        )
        logger.warning(self._paused_reason)
        return LLMError(f"Gemini API error: {detail}")

    def _disable(self, reason: str) -> LLMError:
        if self._disabled_reason is None:
            logger.error(f"LLM disabled for this process: {reason}")
        self._disabled_reason = reason
        return LLMError(reason)

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

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            # max_tokens is deliberately not forwarded: Gemini counts its internal thinking against
            # maxOutputTokens, so the small caps tuned for Claude would truncate the JSON. The
            # schema bounds the output instead.
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": schema.model_json_schema()},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        response = await self._post(body)
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMError(f"Gemini returned no answer ({reason})")
        candidate = candidates[0]
        text = "".join(p.get("text", "") for p in (candidate.get("content") or {}).get("parts", []) if not p.get("thought"))
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(
                f"Model output did not match {schema.__name__} (finishReason={candidate.get('finishReason')}): {exc}"
            ) from exc

    async def _post(self, body: dict) -> httpx.Response:
        waited = 0.0
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            for attempt in range(self._max_retries + 1):
                try:
                    response = await http.post(self._url, json=body, headers={"x-goog-api-key": self._api_key})
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
                if response.status_code in (401, 403) or "API_KEY_INVALID" in response.text:
                    raise self._disable(f"Gemini rejected the API key or its permissions ({message})")
                if response.status_code == 404:
                    raise self._disable(f"Gemini model '{self.model}' is not available to this key ({message})")
                daily = _daily_quota(response) if response.status_code == 429 else None
                if daily:
                    raise self._disable(
                        f"the Gemini {daily} for {self.model} is used up; it resets daily "
                        "(or enable billing in Google AI Studio for higher limits)"
                    )
                if response.status_code in RETRYABLE:
                    delay = _retry_delay(response, attempt)
                    if attempt < self._max_retries and waited + delay <= MAX_TOTAL_WAIT_S:
                        logger.warning(f"Gemini {response.status_code}; retrying in {delay:.0f}s ({message[:80]})")
                        waited += delay
                        await asyncio.sleep(delay)
                        continue
                    raise self._pause(f"{response.status_code}: {message[:100]}")
                raise LLMError(f"Gemini API error {response.status_code}: {message}")
        raise LLMError("Gemini request failed")  # unreachable: the loop always returns or raises

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            return str(response.json().get("error", {}).get("message", response.text))[:300]
        except ValueError:
            return response.text[:300]
