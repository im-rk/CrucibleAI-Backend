"""Async Anthropic client that returns schema-validated structured output."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from matsci_agent.observability.trace import report_usage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM is unconfigured or a call fails; agents catch it and degrade."""


class LLMClient:
    """Thin wrapper over the Anthropic SDK.

    `available` is False when no API key is set, and also once Anthropic has rejected the account
    itself (invalid key, no permission, no credits): those failures repeat on every call, so the
    client stops calling and `unavailable_reason` says why, instead of every agent re-trying.
    """

    def __init__(self, api_key: str | None, model: str, max_retries: int = 2) -> None:
        self.model = model
        self._client = None
        self._disabled_reason: str | None = None
        if api_key:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)

    @property
    def available(self) -> bool:
        return self._client is not None and self._disabled_reason is None

    @property
    def unavailable_reason(self) -> str | None:
        if self._client is None:
            return "ANTHROPIC_API_KEY is not configured"
        return self._disabled_reason

    def _disable(self, reason: str) -> None:
        if self._disabled_reason is None:
            logger.error(f"LLM disabled for this process: {reason}")
        self._disabled_reason = reason

    async def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> T:
        """Run one prompt and return the response parsed into `schema`."""
        if not self.available:
            raise LLMError(self.unavailable_reason or "LLM unavailable")

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
        except anthropic.AuthenticationError as exc:
            self._disable("Anthropic rejected the API key (check ANTHROPIC_API_KEY)")
            raise LLMError(self._disabled_reason) from exc
        except anthropic.PermissionDeniedError as exc:
            self._disable(f"the Anthropic API key lacks permission for {self.model}")
            raise LLMError(self._disabled_reason) from exc
        except anthropic.BadRequestError as exc:
            if "credit balance" in str(exc).lower():
                self._disable(
                    "the Anthropic account has no credits (add them at console.anthropic.com, Plans & Billing)"
                )
                raise LLMError(self._disabled_reason) from exc
            raise LLMError(f"Anthropic API error: {exc}") from exc
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc
        except ValidationError as exc:  # e.g. output truncated at max_tokens
            raise LLMError(f"Model output did not match {schema.__name__}: {exc}") from exc

        usage = getattr(response, "usage", None)
        report_usage(getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))
        if response.parsed_output is None:
            raise LLMError(f"No parsable output (stop_reason={response.stop_reason})")
        return response.parsed_output
