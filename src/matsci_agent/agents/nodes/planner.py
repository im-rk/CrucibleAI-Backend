"""PlannerAgent: turns the research goal into an explicit, inspectable plan before any work starts.

Plan-and-execute: one structured LLM call produces the criteria to screen against, the literature
queries to run, and the material families worth looking at. The plan is shown to the user and can be
edited and sent back (`ResearchRequest.plan`), which is the human-in-the-loop checkpoint. Without an
LLM the same plan is built from the regex criteria parser and keyword queries.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, ValidationError

from matsci_agent.agents import prompts
from matsci_agent.agents.events import emit
from matsci_agent.agents.nodes.screening import CriterionDraft, criterion_from_draft, parse_criteria_heuristic
from matsci_agent.agents.nodes.search import keywords
from matsci_agent.domain.schemas import Criterion
from matsci_agent.domain.state import ResearchState
from matsci_agent.infrastructure.llm.anthropic_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

MAX_QUERIES = 5


class PlanOutput(BaseModel):
    summary: str
    criteria: list[CriterionDraft]
    queries: list[str]
    material_families: list[str]


_ASKS_FOR_EVIDENCE = re.compile(
    r"experimental|demonstrated|measured|validated|verified|not\s+(?:just|only)\s+(?:predicted|computed|dft)|real[- ]world",
    re.IGNORECASE,
)


def _drop_unrequested(goal: str, criteria: list[Criterion], warnings: list[str]) -> list[Criterion]:
    """A planner model sometimes adds a requirement that sounds reasonable but was never asked for."""
    kept = [c for c in criteria if c.kind != "measured_evidence" or _ASKS_FOR_EVIDENCE.search(goal)]
    if len(kept) < len(criteria):
        warnings.append("Dropped a planned criterion about experimental evidence: the goal does not ask for it.")
    return kept


def _heuristic_queries(goal: str) -> list[str]:
    terms = " ".join(keywords(goal))
    return [goal] if terms == goal.lower() or not terms else [goal, terms]


class PlannerAgent:
    """Builds the research plan; also used on its own to preview a plan before a run."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def plan(self, goal: str, warnings: list[str], override: dict | None = None) -> dict:
        """Return {summary, criteria, queries, material_families, source}; `warnings` collects fallbacks."""
        if override:
            user = self._from_override(goal, override, warnings)
            if user:
                return user

        if self.llm.available:
            try:
                out = await self.llm.structured(
                    prompts.PLAN_PROMPT.format(goal=goal), PlanOutput, system=prompts.PLAN_SYSTEM, max_tokens=2000
                )
                criteria = _drop_unrequested(goal, [c for c in (criterion_from_draft(d, warnings) for d in out.criteria) if c], warnings)
                queries = list(dict.fromkeys(q.strip() for q in out.queries if q.strip()))[:MAX_QUERIES]
                if criteria or queries:
                    return {
                        "summary": out.summary.strip() or goal,
                        "criteria": [c.model_dump() for c in criteria] or [
                            c.model_dump() for c in parse_criteria_heuristic(goal)
                        ],
                        "queries": queries or _heuristic_queries(goal),
                        "material_families": [f.strip() for f in out.material_families if f.strip()][:6],
                        "source": "llm",
                    }
            except LLMError as exc:
                warnings.append(f"Planning fell back to heuristics: {exc}")
        else:
            warnings.append(
                "LLM unavailable: the plan was built with regex criteria parsing and keyword queries, "
                "and may be incomplete."
            )
        criteria = parse_criteria_heuristic(goal)
        return {
            "summary": goal,
            "criteria": [c.model_dump() for c in criteria],
            "queries": _heuristic_queries(goal),
            "material_families": [],
            "source": "heuristic",
        }

    @staticmethod
    def _from_override(goal: str, override: dict, warnings: list[str]) -> dict | None:
        """A plan the user reviewed and edited: trusted over the model, but validated."""
        try:
            criteria = [Criterion(**c) for c in override.get("criteria", [])]
        except (ValidationError, TypeError) as exc:
            warnings.append(f"The supplied plan had invalid criteria and was ignored: {exc}")
            return None
        queries = [q.strip() for q in override.get("queries", []) if isinstance(q, str) and q.strip()][:MAX_QUERIES]
        return {
            "summary": str(override.get("summary") or goal),
            "criteria": [c.model_dump() for c in criteria],
            "queries": queries or _heuristic_queries(goal),
            "material_families": [str(f) for f in override.get("material_families", [])][:6],
            "source": "user",
        }

    async def run(self, state: ResearchState, config=None) -> dict:
        warnings: list[str] = []
        plan = await self.plan(state["research_goal"], warnings, state.get("plan_override"))
        await emit(
            config, "plan_ready", source=plan["source"], summary=plan["summary"],
            criteria=[c["label"] for c in plan["criteria"]], queries=plan["queries"],
            families=plan["material_families"],
        )
        return {
            "plan": plan,
            "criteria": plan["criteria"],
            "planned_queries": plan["queries"],
            "warnings": warnings,
            "agent_log": [{
                "step": "PLANNING",
                "details": (
                    f"Plan from {plan['source']}: {len(plan['criteria'])} criteria, {len(plan['queries'])} queries"
                    + (f", families: {', '.join(plan['material_families'])}" if plan["material_families"] else "")
                ),
            }],
        }
