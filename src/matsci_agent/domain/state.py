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

    # SearchAgent
    search_queries: list[str]  # every query tried, in order
    retrieved_chunks: list[dict]  # every unique chunk seen, each with `relevance_score`
    retrieved_papers: list[dict]  # chunks judged relevant, best first
    retry_count: int  # re-retrieval attempts made so far

    # Downstream agents
    extracted_properties: list[dict]
    criteria: list[dict]
    screened_candidates: list[dict]
    hypotheses: list[dict]
    knowledge_gaps: list[str]
    citations: list[dict]
    grounding: dict

    agent_log: Annotated[list[dict], operator.add]
    warnings: Annotated[list[str], operator.add]


def initial_state(goal: str, max_results: int = 10) -> ResearchState:
    return {
        "research_goal": goal,
        "max_results": max_results,
        "search_queries": [],
        "retrieved_chunks": [],
        "retrieved_papers": [],
        "retry_count": 0,
        "extracted_properties": [],
        "criteria": [],
        "screened_candidates": [],
        "hypotheses": [],
        "knowledge_gaps": [],
        "citations": [],
        "grounding": {},
        "agent_log": [],
        "warnings": [],
    }
