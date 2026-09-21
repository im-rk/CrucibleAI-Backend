"""Async Anthropic client that returns schema-validated structured output."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM is unconfigured or a call fails; agents catch it and degrade."""


class LLMClient:
    """Thin wrapper over the Anthropic SDK. `available` is False when no API key is set."""

    def __init__(self, api_key: str | None, model: str, max_retries: int = 2) -> None:
        self.model = model
        self._client = None
        if api_key:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)

    @property
    def available(self) -> bool:
        return self._client is not None

    async def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> T:
        """Run one prompt and return the response parsed into `schema`."""
        if self._client is None:
            raise LLMError("ANTHROPIC_API_KEY is not configured")

        import anthropic

        kwargs: dict[str, Any] = {}
        if system:
            kwargs["system"] = system
        try:
            response = await self._client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                **kwargs,
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc
        except ValidationError as exc:  # e.g. output truncated at max_tokens
            raise LLMError(f"Model output did not match {schema.__name__}: {exc}") from exc

        if response.parsed_output is None:
            raise LLMError(f"No parsable output (stop_reason={response.stop_reason})")
        return response.parsed_output
