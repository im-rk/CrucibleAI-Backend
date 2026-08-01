"""Pipeline service for PDF parsing, chunking, and embedding."""

import logging
import re
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
from matsci_agent.domain.schemas import PaperChunk, PaperMetadata

logger = logging.getLogger(__name__)

class PDFParser:
    SECTION_PATTERNS = [
        r"^\s*(abstract)\s*$",
        r"^\s*(1\.?\s*introduction)\s*",
        r"^\s*(2\.?\s*experimental|2\.?\s*methods|2\.?\s*computational)\s*",
        r"^\s*(3\.?\s*results|3\.?\s*results and discussion)\s*",
        r"^\s*(4\.?\s*discussion)\s*",
        r"^\s*(5\.?\s*conclusion|5\.?\s*conclusions)\s*",
    ]

    def parse(self, pdf_path: Path) -> Optional[dict[str, str]]:
        if not pdf_path.exists():
            return None
        try:
            doc = fitz.open(pdf_path)
            full_text = [page.get_text("text") for page in doc]
            doc.close()
            return self._split_by_sections("\n".join(full_text))
        except Exception as exc:
            logger.error(f"PyMuPDF failed to parse {pdf_path}: {exc}")
            return None

    def _split_by_sections(self, text: str) -> dict[str, str]:
        lines = text.split("\n")
        sections: dict[str, list[str]] = {"preamble": []}
        current_section = "preamble"

        for line in lines:
            clean_line = line.strip().lower()
            is_header = False
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, clean_line):
                    current_section = line.strip()
                    if current_section not in sections:
                        sections[current_section] = []
                    is_header = True
                    break
            if not is_header:
                sections[current_section].append(line)

        return {
            sec: "\n".join(content).strip()
            for sec, content in sections.items()
            if len("\n".join(content).strip()) > 50
        }

class SectionAwareChunker:
    def __init__(self, max_chars: int = 1500, overlap_chars: int = 200):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_paper(self, sections: dict[str, str], metadata: PaperMetadata) -> list[PaperChunk]:
        chunks: list[PaperChunk] = []
        global_index = 0

        for section_name, text in sections.items():
            if "reference" in section_name.lower():
                continue

            pos = 0
            while pos < len(text):
                end = min(pos + self.max_chars, len(text))
                if end < len(text):
                    last_period = text.rfind(".", pos, end)
                    if last_period > pos + (self.max_chars // 2):
                        end = last_period + 1

                chunk_text = text[pos:end].strip()
                if len(chunk_text) >= 100:
                    chunks.append(
                        PaperChunk(
                            chunk_id=f"{metadata.arxiv_id}_{global_index}",
                            arxiv_id=metadata.arxiv_id,
                            text=chunk_text,
                            section_name=section_name,
                            chunk_index=global_index,
                            char_count=len(chunk_text),
                            metadata=metadata,
                        )
                    )
                    global_index += 1
                pos = end - self.overlap_chars if end < len(text) else len(text)
        return chunks

class PaperPipelineService:
    def __init__(self):
        self.parser = PDFParser()
        self.chunker = SectionAwareChunker()

    def process_pdf(self, pdf_path: Path, metadata: PaperMetadata) -> list[PaperChunk]:
        sections = self.parser.parse(pdf_path)
        if not sections:
            return []
        return self.chunker.chunk_paper(sections, metadata)