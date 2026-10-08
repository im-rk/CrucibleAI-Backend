"""Tools an agent may call: a named, described callable with timing, tracing and safe failure.

A tool never raises into the agent loop: a failure or a timeout becomes an observation the agent can
read and react to, which is how a tool-using agent recovers instead of crashing.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from matsci_agent.observability.trace import trace_span

logger = logging.getLogger(__name__)

MAX_OBSERVATION_CHARS = 900


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # parameter name -> what it means
    fn: Callable[..., Any | Awaitable[Any]]
    timeout_s: float = 30.0


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return list(self.tools)

    def describe(self) -> str:
        lines = []
        for t in self.tools.values():
            params = ", ".join(f"{k} ({v})" for k, v in t.params.items()) or "no arguments"
            lines.append(f"- {t.name}: {t.description} Arguments: {params}.")
        return "\n".join(lines)

    async def call(self, name: str, arguments: dict[str, str]) -> tuple[str, bool]:
        """Run a tool; return (observation text, ok)."""
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: there is no tool named '{name}'. Available: {', '.join(self.names)}.", False
        missing = [p for p in tool.params if not str(arguments.get(p, "")).strip()]
        if missing:
            return f"Error: tool '{name}' needs the argument(s): {', '.join(missing)}.", False
        kwargs = {p: str(arguments[p]).strip() for p in tool.params}
        with trace_span("tool", name, **{k: v[:80] for k, v in kwargs.items()}) as span:
            try:
                result = tool.fn(**kwargs)
                if inspect.isawaitable(result):
                    result = await asyncio.wait_for(result, timeout=tool.timeout_s)
                text, ok = str(result), True
            except asyncio.TimeoutError:
                text, ok = f"Error: tool '{name}' timed out after {tool.timeout_s:.0f}s.", False
            except Exception as exc:  # noqa: BLE001 - becomes an observation, never a crash
                logger.warning(f"Tool {name} failed: {exc}")
                text, ok = f"Error: tool '{name}' failed: {type(exc).__name__}: {exc}", False
            span["ok"] = ok
        return (text[:MAX_OBSERVATION_CHARS] + (" ..." if len(text) > MAX_OBSERVATION_CHARS else "")), ok
