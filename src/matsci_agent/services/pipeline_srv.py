"""Pipeline service for PDF parsing, chunking, and embedding."""

from __future__ import annotations


class PipelineService:
    """Orchestrates ingestion-time document processing."""

    async def process_paper(self, paper_id: str) -> None:
        _ = paper_id
