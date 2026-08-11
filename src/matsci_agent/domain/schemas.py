"""Pydantic models that define the API and internal data contracts."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel,Field

class PaperMetadata(BaseModel):
    """Metadata contract for an ingested paper"""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    doi : Optional[str] = None
    pdf_path: Optional[str] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

class PaperChunk(BaseModel):
    """Contract for a searchable chunk stored in ChromaDb"""

    chunk_id: str
    arxiv_id: str
    text: str
    section_name: str
    chunk_index: int
    char_count: int
    metadata: PaperMetadata