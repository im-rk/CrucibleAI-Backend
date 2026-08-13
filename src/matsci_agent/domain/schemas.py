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


class ResearchRequest(BaseModel):
    """User input contract for starting a research run."""
    query: str
    max_results: int = 10


class MaterialProperty(BaseModel):
    """Material property data record contract."""
    material_id: str
    formula: str
    property_name: str
    value: float | str
    unit: str
    source: str = "Materials Project"
    doi: str = "https://doi.org/10.1038/s41524-020-00448-7"
    is_direct_gap: Optional[bool] = None
    is_predicted: bool = False
    confidence: str = "high"
