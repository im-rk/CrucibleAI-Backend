"""Pydantic models that define the API and internal data contracts."""

# from __future__ import annotations

# from datetime import datetime

# from pydantic import BaseModel, Field


# class ResearchRequest(BaseModel):
#     """Request payload for a research query."""

#     query: str = Field(min_length=1)
#     max_results: int = Field(default=10, ge=1, le=100)


# class PropertyRecord(BaseModel):
#     """Structured material property record returned by the agent."""

#     material_id: str
#     property_name: str
#     value: float | str
#     units: str | None = None
#     source: str | None = None
#     created_at: datetime = Field(default_factory=datetime.utcnow)
