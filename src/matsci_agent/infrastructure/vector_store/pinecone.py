"""Pinecone vector store for production (selected with VECTOR_STORE=pinecone).

Same interface as the Chroma store for what the pipeline needs: add_chunks, search, count and
similarities. Pinecone has no cheap "list every document", so `iter_documents` is unsupported:
with Pinecone the pipeline uses semantic search only (no BM25) and the paper catalogue is off.

Verified against the live service (pinecone client 10): search results and similarity scores match
the Chroma store on the same corpus. Unit tests use a fake client.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from matsci_agent.domain.schemas import PaperChunk
from matsci_agent.infrastructure.ml.embedder import MaterialsEmbedder

logger = logging.getLogger(__name__)

DIMENSION = 384  # all-MiniLM-L6-v2
UPSERT_BATCH = 100


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a client response object or a plain dict."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


class PineconeVectorStore:
    supports_listing = False

    def __init__(
        self,
        api_key: str | None,
        index_name: str,
        *,
        cloud: str = "aws",
        region: str = "us-east-1",
        embedder: MaterialsEmbedder | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("PINECONE_API_KEY is required when VECTOR_STORE=pinecone.")
            from pinecone import Pinecone

            client = Pinecone(api_key=api_key)
        self._client = client
        self._cloud, self._region = cloud, region
        self.index = client.Index(self._ensure_index(index_name))
        self.embedder = embedder or MaterialsEmbedder()

    def _ensure_index(self, name: str) -> str:
        """Create the index (cosine, 384-d, serverless) if it does not exist yet."""
        existing = {_get(i, "name") for i in self._client.list_indexes()}
        if name not in existing:
            from pinecone import ServerlessSpec

            logger.info(f"Creating Pinecone index '{name}' ({DIMENSION}-d, cosine, {self._cloud}/{self._region})")
            self._client.create_index(
                name=name, dimension=DIMENSION, metric="cosine",
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
        return name

    def add_chunks(self, chunks: list[PaperChunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed_texts([c.text for c in chunks])
        records = [
            {
                "id": c.chunk_id,
                "values": vec,
                # Pinecone has no document store, so the text travels in the metadata.
                "metadata": {
                    "text": c.text, "arxiv_id": c.arxiv_id, "section": c.section_name,
                    "chunk_index": c.chunk_index, "title": c.metadata.title,
                    "authors": ", ".join(c.metadata.authors[:3]), "year": c.metadata.year,
                    "doi": c.metadata.doi or "",
                },
            }
            for c, vec in zip(chunks, vectors)
        ]
        self.upsert_records(records)

    def upsert_records(self, records: list[dict]) -> None:
        """Upsert prepared {id, values, metadata} records in batches (also used by the migration script)."""
        for i in range(0, len(records), UPSERT_BATCH):
            self.index.upsert(vectors=records[i : i + UPSERT_BATCH])

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
        deduplicate_papers: bool = True,
        max_chunks_per_paper: int = 1,
    ) -> list[dict]:
        vector = self.embedder.embed_single(query)
        top_k = n_results * 5 if deduplicate_papers else n_results
        response = self.index.query(vector=vector, top_k=top_k, filter=where, include_metadata=True)

        chunks: list[dict] = []
        per_paper: dict[str, int] = {}
        for match in _get(response, "matches", []):
            metadata = dict(_get(match, "metadata", {}) or {})
            text = metadata.pop("text", "")
            if deduplicate_papers:
                paper = metadata.get("arxiv_id") or metadata.get("title") or _get(match, "id")
                if per_paper.get(paper, 0) >= max_chunks_per_paper:
                    continue
                per_paper[paper] = per_paper.get(paper, 0) + 1
            score = float(_get(match, "score", 0.0))  # cosine similarity for a cosine index
            chunks.append({
                "id": _get(match, "id"), "text": text, "metadata": metadata,
                "distance": 1.0 - score, "similarity": score,
            })
            if len(chunks) >= n_results:
                break
        return chunks

    def similarities(self, query: str, ids: list[str]) -> dict[str, float]:
        if not ids:
            return {}
        query_vec = np.asarray(self.embedder.embed_single(query))
        fetched = _get(self.index.fetch(ids=ids), "vectors", {})
        return {
            chunk_id: float(np.dot(query_vec, np.asarray(_get(vec, "values"))))
            for chunk_id, vec in fetched.items()
        }

    def count(self) -> int:
        return int(_get(self.index.describe_index_stats(), "total_vector_count", 0))

    def iter_documents(self) -> Iterator[dict]:
        raise NotImplementedError("Listing every document is not supported for the Pinecone store.")
