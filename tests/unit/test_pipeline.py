"""Unit tests for the data pipeline PDF parser and section-aware chunker."""

from __future__ import annotations

from matsci_agent.domain.schemas import PaperMetadata
from matsci_agent.services.pipeline_srv import PDFParser, SectionAwareChunker


def test_pdf_parser_section_splitting() -> None:
    parser = PDFParser()
    sample_text = """Abstract
This is a sample abstract about lithium ion batteries with high conductivity.

1. Introduction
Lithium-ion batteries are widely used in energy storage applications.

References
1. Smith et al., 2023.
"""
    sections = parser._split_by_sections(sample_text)
    assert "abstract" in [s.lower() for s in sections]
    assert any("introduction" in s.lower() for s in sections)


def test_section_aware_chunker_basic() -> None:
    chunker = SectionAwareChunker(max_chars=200, overlap_chars=20)
    meta = PaperMetadata(
        arxiv_id="2401.00001",
        title="Test Battery Paper",
        authors=["Alice", "Bob"],
        abstract="Sample abstract",
        year=2024,
    )
    sections = {
        "1. Introduction": (
            "Solid state electrolytes offer high thermal stability. "
            "They replace liquid organic electrolytes to prevent thermal runaway. "
            "Ionic conductivity is a crucial parameter for performance."
        ),
        "References": "Smith et al., 2023. Journal of Energy.",
    }

    chunks = chunker.chunk_paper(sections, meta)

    # Should skip References section
    assert all("reference" not in c.section_name.lower() for c in chunks)
    # Should create valid chunks with metadata
    assert len(chunks) > 0
    assert chunks[0].arxiv_id == "2401.00001"
    assert chunks[0].metadata.title == "Test Battery Paper"
