"""Unit tests for BM25, RRF fusion, hybrid search, and the reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from matsci_agent.infrastructure.retrieval.bm25_index import BM25Index, tokenize
from matsci_agent.infrastructure.retrieval.hybrid_search import HybridSearch, rrf_merge
from matsci_agent.infrastructure.retrieval.reranker import CrossEncoderReranker

DOCS = [
    {"id": "a_0", "text": "LLZO garnet electrolyte shows high ionic conductivity", "metadata": {"arxiv_id": "a"}},
    {"id": "b_0", "text": "GaN wide bandgap semiconductor for power electronics", "metadata": {"arxiv_id": "b"}},
    {"id": "c_0", "text": "Perovskite solar cell efficiency and stability", "metadata": {"arxiv_id": "c"}},
]


class FakeStore:
    """Minimal vector-store double: fixed semantic ranking + document iteration."""

    def __init__(self, semantic_ids: list[str]) -> None:
        self.semantic_ids = semantic_ids
        self.iter_calls = 0

    def count(self) -> int:
        return len(DOCS)

    def iter_documents(self):
        self.iter_calls += 1
        yield from DOCS

    def search(self, query, n_results, deduplicate_papers):
        by_id = {d["id"]: d for d in DOCS}
        return [
            {**by_id[i], "similarity": 0.9 - 0.1 * rank}
            for rank, i in enumerate(self.semantic_ids[:n_results])
        ]

    def similarities(self, query, ids):
        return {i: 0.25 for i in ids}


def test_tokenize_keeps_formulas_whole() -> None:
    assert tokenize("Li7La3Zr2O12, 10-3 S/cm") == ["li7la3zr2o12", "10", "3", "s", "cm"]


def test_bm25_ranks_keyword_match_first_and_drops_non_matches() -> None:
    index = BM25Index(FakeStore([]))
    results = index.search("GaN bandgap", n_results=5)
    assert [r["id"] for r in results] == ["b_0"]
    assert results[0]["bm25_score"] > 0


def test_bm25_empty_query_returns_nothing() -> None:
    assert BM25Index(FakeStore([])).search("!!!") == []


def test_bm25_only_rebuilds_when_corpus_size_changes() -> None:
    store = FakeStore([])
    index = BM25Index(store)
    index.search("garnet")
    index.search("solar")
    assert store.iter_calls == 1


def test_rrf_rewards_agreement_between_lists() -> None:
    semantic = [{"id": "x", "similarity": 0.9}, {"id": "y", "similarity": 0.8}]
    keyword = [{"id": "y", "bm25_score": 3.0}, {"id": "z", "bm25_score": 1.0}]
    merged = rrf_merge(semantic, keyword)
    assert [m["id"] for m in merged] == ["y", "x", "z"]
    # fields from both lists survive the merge
    assert merged[0]["similarity"] == 0.8 and merged[0]["bm25_score"] == 3.0


def test_hybrid_search_fuses_and_fills_missing_similarity() -> None:
    store = FakeStore(["c_0", "a_0"])
    hybrid = HybridSearch(store, bm25=BM25Index(store))
    results = hybrid.search("GaN semiconductor", n_results=3)
    ids = [r["id"] for r in results]
    assert "b_0" in ids and "c_0" in ids  # keyword-only and semantic-only hits both present
    b_hit = next(r for r in results if r["id"] == "b_0")
    assert b_hit["similarity"] == 0.25  # backfilled for the keyword-only chunk
    assert all("rrf_score" in r for r in results)


def test_hybrid_search_respects_n_results() -> None:
    store = FakeStore(["a_0", "b_0", "c_0"])
    assert len(HybridSearch(store).search("anything", n_results=2)) == 2


def test_reranker_reorders_by_cross_encoder_score() -> None:
    chunks = [{"id": "1", "text": "weak"}, {"id": "2", "text": "strong"}]
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1, 0.9]
    reranker = CrossEncoderReranker("fake-model")
    reranker._model = fake_model
    out = reranker.rerank("q", chunks, top_k=2)
    assert [c["id"] for c in out] == ["2", "1"]
    assert out[0]["rerank_score"] == 0.9


def test_reranker_falls_back_when_model_cannot_load() -> None:
    chunks = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
    with patch("sentence_transformers.CrossEncoder", side_effect=OSError("offline")):
        reranker = CrossEncoderReranker("missing-model")
        assert reranker.rerank("q", chunks, top_k=1) == chunks[:1]
        assert reranker.rerank("q", chunks, top_k=2) == chunks  # no retry storm, still a no-op
