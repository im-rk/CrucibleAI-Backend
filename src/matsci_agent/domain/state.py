"""LangGraph state shared across the research workflow."""

from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict, total=False):
    """Mutable state passed between graph nodes."""

    query: str
    search_results: list[dict[str, object]]
    extracted_records: list[dict[str, object]]
    hypothesis: str
    error: str
