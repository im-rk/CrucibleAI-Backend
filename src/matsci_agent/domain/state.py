"""LangGraph state shared across the research workflow."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict, total=False):
    """State passed between graph nodes. Nodes return partial updates, never the whole state.

    `agent_log` and `warnings` use an additive reducer, so a node returns only its new entries.
    """

    research_goal: str
    max_results: int
    plan_override: dict | None  # a plan the user reviewed and edited

    # PlannerAgent
    plan: dict
    planned_queries: list[str]

    # SearchAgent
    search_queries: list[str]  # every query tried, in order
    retrieved_chunks: list[dict]  # every unique chunk seen, each with `relevance_score`
    retrieved_papers: list[dict]  # chunks judged relevant, best first
    retry_count: int  # re-retrieval attempts made so far

    # Downstream agents
    extracted_properties: list[dict]
    criteria: list[dict]
    screened_candidates: list[dict]
    investigation: dict  # InvestigatorAgent trajectory and outcome
    evidence_index: dict  # id -> evidence item the hypotheses may cite (E* chunks, R* records)
    revision_count: int
    grounding_history: list[dict]
    rescreen_count: int
    hypotheses: list[dict]
    knowledge_gaps: list[str]
    citations: list[dict]
    grounding: dict

    agent_log: Annotated[list[dict], operator.add]
    warnings: Annotated[list[str], operator.add]


def initial_state(goal: str, max_results: int = 10, plan: dict | None = None) -> ResearchState:
    return {
        "research_goal": goal,
        "max_results": max_results,
        "plan_override": plan,
        "plan": {},
        "planned_queries": [],
        "search_queries": [],
        "retrieved_chunks": [],
        "retrieved_papers": [],
        "retry_count": 0,
        "extracted_properties": [],
        "criteria": [],
        "screened_candidates": [],
        "investigation": {},
        "evidence_index": {},
        "revision_count": 0,
        "grounding_history": [],
        "rescreen_count": 0,
        "hypotheses": [],
        "knowledge_gaps": [],
        "citations": [],
        "grounding": {},
        "agent_log": [],
        "warnings": [],
    }
