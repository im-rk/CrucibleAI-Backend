"""A bounded ReAct-style loop: the model decides which tool to call next, reads the result, repeats.

Provider-agnostic: each step is one structured-output call (thought, action, arguments), so it works
with every LLM client here without relying on native function calling. The loop is deliberately
bounded (step cap, repeat detection, LLM budget) and returns its whole trajectory, which the report
shows and the evaluation suite scores.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, field_validator

from matsci_agent.agents.tools import ToolRegistry
from matsci_agent.infrastructure.llm.anthropic_client import LLMError
from matsci_agent.security.guardrails import UNTRUSTED_NOTICE, wrap_untrusted

logger = logging.getLogger(__name__)

FINISH = "finish"

LOOP_SYSTEM = (
    "You are a careful materials-science research agent. You work in short steps: think, call one "
    "tool, read the result, and continue until you have what you need. Use only what the tools return; "
    "never invent values. Stop as soon as the task is done or no tool can help." + UNTRUSTED_NOTICE
)

LOOP_PROMPT = """Task:
{task}

Tools you can call:
{tools}
- finish: stop and report. Arguments: none (put your summary in `answer`).

Steps so far:
{scratchpad}

Decide the next step. Return:
- thought: one sentence on what you need next and why
- action: the name of ONE tool from the list above, or "finish"
- arguments: the tool's arguments as name/value pairs (empty for finish)
- answer: only when action is "finish": what you found, in one or two sentences

You have {remaining} step(s) left."""


class Argument(BaseModel):
    name: str
    value: str


class AgentStep(BaseModel):
    thought: str
    action: str
    arguments: list[Argument]
    answer: str | None

    @field_validator("arguments", mode="before")
    @classmethod
    def _accept_an_object(cls, value):
        """Models often send {"query": "..."} instead of [{"name": "query", "value": "..."}]."""
        if isinstance(value, dict):
            return [{"name": str(k), "value": str(v)} for k, v in value.items()]
        return value


@dataclass
class LoopResult:
    steps: list[dict] = field(default_factory=list)
    answer: str = ""
    stopped: str = "max_steps"  # finish | max_steps | llm_error | repeat


async def run_tool_loop(
    llm,
    registry: ToolRegistry,
    task: str,
    *,
    max_steps: int = 5,
    on_step: Callable[[dict], Awaitable[None]] | None = None,
) -> LoopResult:
    result = LoopResult()
    seen: set[tuple] = set()
    for step_no in range(1, max_steps + 1):
        scratch = "\n".join(
            f"{s['n']}. thought: {s['thought']}\n   action: {s['action']}({', '.join(f'{k}={v!r}' for k, v in s['arguments'].items())})\n"
            f"   observation: {wrap_untrusted(s['observation'])}"
            for s in result.steps
        ) or "(none yet)"
        try:
            step = await llm.structured(
                LOOP_PROMPT.format(task=task, tools=registry.describe(), scratchpad=scratch, remaining=max_steps - step_no + 1),
                AgentStep, system=LOOP_SYSTEM, max_tokens=600,
            )
        except LLMError as exc:
            logger.warning(f"Tool loop stopped: {exc}")
            result.stopped = "llm_error"
            return result

        action = step.action.strip()
        args = {a.name.strip(): a.value for a in step.arguments if a.name.strip()}
        if action == FINISH:
            result.steps.append({"n": step_no, "thought": step.thought, "action": FINISH, "arguments": {}, "observation": step.answer or ""})
            result.answer, result.stopped = (step.answer or "").strip(), "finish"
            if on_step:
                await on_step(result.steps[-1])
            return result

        signature = (action, tuple(sorted(args.items())))
        if signature in seen:
            observation, ok = "Error: you already made this exact call; try something different or finish.", False
        else:
            seen.add(signature)
            observation, ok = await registry.call(action, args)
        result.steps.append({"n": step_no, "thought": step.thought, "action": action, "arguments": args, "observation": observation, "ok": ok})
        if on_step:
            await on_step(result.steps[-1])
        if signature in seen and not ok and "already made" in observation and sum("already made" in s["observation"] for s in result.steps) >= 2:
            result.stopped = "repeat"
            return result
    return result
