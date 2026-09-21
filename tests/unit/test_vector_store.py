"""Unit tests for MaterialsEmbedder and ChromaVectorStore."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch
from matsci_agent.domain.schemas import PaperChunk, PaperMetadata
from matsci_agent.infrastructure.vector_store.chromadb import ChromaVectorStore


def test_chroma_vector_store_add_and_search() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:

        # Patch MaterialsEmbedder so unit tests run fast without network downloads
        with patch("matsci_agent.infrastructure.vector_store.chromadb.MaterialsEmbedder") as MockEmbedder:
            mock_instance = MagicMock()
            mock_instance.embed_texts.return_value = [[0.1] * 384]
            mock_instance.embed_single.return_value = [0.1] * 384
            MockEmbedder.return_value = mock_instance

            store = ChromaVectorStore(persist_dir=tmp_dir)

            meta = PaperMetadata(
                arxiv_id="2401.99999",
                title="Solid State Battery Electrolytes",
                authors=["Dr. Science"],
                abstract="Lithium ionic conductivity study.",
                year=2024,
            )

            chunk = PaperChunk(
                chunk_id="2401.99999_0",
                arxiv_id="2401.99999",
                text="High ionic conductivity in LLZO ceramic solid electrolytes was measured at room temperature.",
                section_name="Results",
                chunk_index=0,
                char_count=98,
                metadata=meta,
            )

            store.add_chunks([chunk])
            assert store.count() == 1

            results = store.search("ionic conductivity LLZO", n_results=1)
            assert len(results) == 1
            assert results[0]["id"] == "2401.99999_0"
            assert "LLZO" in results[0]["text"]
            assert "similarity" in results[0]


def test_reingesting_the_same_chunks_is_idempotent_and_keeps_the_doi() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        with patch("matsci_agent.infrastructure.vector_store.chromadb.MaterialsEmbedder") as MockEmbedder:
            mock_instance = MagicMock()
            mock_instance.embed_texts.return_value = [[0.1] * 384]
            mock_instance.embed_single.return_value = [0.1] * 384
            MockEmbedder.return_value = mock_instance
            store = ChromaVectorStore(persist_dir=tmp_dir)

            meta = PaperMetadata(arxiv_id="2401.5", title="T", authors=["A"], abstract="a", year=2024, doi="10.1/abc")
            chunk = PaperChunk(chunk_id="2401.5_0", arxiv_id="2401.5", text="LLZO conductivity text " * 6,
                               section_name="Results", chunk_index=0, char_count=138, metadata=meta)
            store.add_chunks([chunk])
            store.add_chunks([chunk])  # second ingestion run: must neither fail nor duplicate

            assert store.count() == 1
            assert store.search("LLZO", n_results=1)[0]["metadata"]["doi"] == "10.1/abc"
