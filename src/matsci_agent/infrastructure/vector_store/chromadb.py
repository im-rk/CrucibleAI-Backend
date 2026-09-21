"""Local ChromaDB implementation used during development."""

import logging
from collections.abc import Iterator
from pathlib import Path

import chromadb
import numpy as np
from matsci_agent.domain.schemas import PaperChunk
from matsci_agent.infrastructure.ml.embedder import MaterialsEmbedder

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parents[4] / "chroma_db")

class ChromaVectorStore:
    COLLECTION_NAME = "materials_papers"

    def __init__(self, persist_dir: str | None = None):
        if persist_dir is None or persist_dir == "./chroma_db":
            persist_dir = DEFAULT_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = MaterialsEmbedder()

    def add_chunks(self, chunks: list[PaperChunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "arxiv_id": c.arxiv_id,
                "section": c.section_name,
                "chunk_index": c.chunk_index,
                "title": c.metadata.title,
                "authors": ", ".join(c.metadata.authors[:3]),
                "year": c.metadata.year,
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )
        logger.info(f"Indexed {len(chunks)} chunks into ChromaDB.")

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
        deduplicate_papers: bool = True,
        max_chunks_per_paper: int = 1,
    ) -> list[dict]:
        """Search for relevant paper chunks using vector similarity with paper deduplication."""
        query_embedding = self.embedder.embed_single(query)

        total_in_db = self.count() or n_results
        fetch_limit = min(n_results * 5 if deduplicate_papers else n_results, max(total_in_db, n_results))

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_limit,
            where=where,
        )

        chunks = []
        seen_paper_counts: dict[str, int] = {}

        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i]

                # Identify paper by unique arxiv_id or title
                paper_key = metadata.get("arxiv_id") or metadata.get("title") or f"paper_{i}"

                if deduplicate_papers:
                    count_so_far = seen_paper_counts.get(paper_key, 0)
                    if count_so_far >= max_chunks_per_paper:
                        continue
                    seen_paper_counts[paper_key] = count_so_far + 1

                chunks.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": metadata,
                    "distance": distance,
                    "similarity": 1.0 - distance,
                })

                if len(chunks) >= n_results:
                    break

        return chunks

    def count(self) -> int:
        return self.collection.count()

    def iter_documents(self, batch_size: int = 500) -> Iterator[dict]:
        """Yield every stored chunk as {id, text, metadata}; feeds the BM25 index."""
        offset = 0
        while True:
            batch = self.collection.get(
                limit=batch_size, offset=offset, include=["documents", "metadatas"]
            )
            ids = batch["ids"]
            if not ids:
                return
            for i, chunk_id in enumerate(ids):
                yield {
                    "id": chunk_id,
                    "text": batch["documents"][i],
                    "metadata": batch["metadatas"][i],
                }
            offset += len(ids)

    def similarities(self, query: str, ids: list[str]) -> dict[str, float]:
        """Cosine similarity of the query to specific stored chunks (embeddings are normalised)."""
        if not ids:
            return {}
        query_vec = np.asarray(self.embedder.embed_single(query))
        got = self.collection.get(ids=ids, include=["embeddings"])
        return {
            chunk_id: float(np.dot(query_vec, np.asarray(vec)))
            for chunk_id, vec in zip(got["ids"], got["embeddings"])
        }