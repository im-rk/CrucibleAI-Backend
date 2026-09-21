"""Hybrid retrieval: semantic + BM25 fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Protocol

from matsci_agent.infrastructure.retrieval.bm25_index import BM25Index
from matsci_agent.infrastructure.retrieval.reranker import CrossEncoderReranker


class SemanticStore(Protocol):
    def search(self, query: str, n_results: int, deduplicate_papers: bool) -> list[dict]: ...

    def similarities(self, query: str, ids: list[str]) -> dict[str, float]: ...


def rrf_merge(*ranked_lists: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over any number of ranked chunk lists (best first).

    Each chunk scores sum(1 / (k + rank)) across the lists it appears in; fields from every
    list a chunk appeared in are kept so callers see both `similarity` and `bm25_score`.
    """
    scores: dict[str, float] = {}
    merged: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            chunk_id = item["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            merged.setdefault(chunk_id, {}).update(item)
    order = sorted(scores, key=scores.__getitem__, reverse=True)
    return [{**merged[i], "rrf_score": round(scores[i], 6)} for i in order]


class HybridSearch:
    """Semantic + keyword retrieval, fused with RRF and optionally cross-encoder reranked."""

    def __init__(
        self,
        vector_store: SemanticStore,
        bm25: BM25Index | None = None,
        reranker: CrossEncoderReranker | None = None,
        pool_size: int = 15,
    ) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.pool_size = pool_size

    def search(self, query: str, n_results: int = 12) -> list[dict]:
        # Multiple chunks per paper are wanted here: extraction reads the Results sections.
        semantic = self.vector_store.search(
            query, n_results=self.pool_size, deduplicate_papers=False
        )
        keyword = self.bm25.search(query, n_results=self.pool_size) if self.bm25 else []
        merged = rrf_merge(semantic, keyword)

        pool = merged[: max(self.pool_size, n_results)]
        if self.reranker is not None:
            pool = self.reranker.rerank(query, pool, top_k=len(pool))
        top = pool[:n_results]

        # Keyword-only hits have no cosine score yet; fill it so every result is comparable.
        missing = [c["id"] for c in top if "similarity" not in c]
        if missing:
            sims = self.vector_store.similarities(query, missing)
            for chunk in top:
                chunk.setdefault("similarity", sims.get(chunk["id"], 0.0))
        return top
