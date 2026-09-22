"""PineconeVectorStore against a fake client, plus graceful degradation for stores that cannot list."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from matsci_agent.api import dependencies as deps
from matsci_agent.api.server import app
from matsci_agent.domain.schemas import PaperChunk, PaperMetadata
from matsci_agent.infrastructure.vector_store.pinecone import PineconeVectorStore
from matsci_agent.services.papers_srv import ListingUnsupported, PaperCatalog
from matsci_agent.services.persistence import PersistenceService


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_single(self, text):
        return [1.0, 0.0]


class FakeIndex:
    def __init__(self, matches=None, vectors=None, total=0):
        self.upserts, self.queries = [], []
        self.matches, self.vectors, self.total = matches or [], vectors or {}, total

    def upsert(self, vectors):
        self.upserts.append(vectors)

    def query(self, vector, top_k, filter, include_metadata):
        self.queries.append({"top_k": top_k, "filter": filter})
        return types.SimpleNamespace(matches=self.matches)

    def fetch(self, ids):
        return types.SimpleNamespace(vectors={i: types.SimpleNamespace(values=self.vectors[i]) for i in ids})

    def describe_index_stats(self):
        return types.SimpleNamespace(total_vector_count=self.total)


class FakeClient:
    def __init__(self, existing=(), index=None):
        self.existing, self.created, self.index = list(existing), [], index or FakeIndex()

    def list_indexes(self):
        return [types.SimpleNamespace(name=n) for n in self.existing]

    def create_index(self, **kwargs):
        self.created.append(kwargs)

    def Index(self, name):  # noqa: N802 - mirrors the real client's method name
        return self.index


def match(cid, score, arxiv, text="t"):
    return {"id": cid, "score": score, "metadata": {"text": text, "arxiv_id": arxiv, "title": f"Paper {arxiv}"}}


def store(client=None, **kwargs):
    return PineconeVectorStore("key", "idx", embedder=FakeEmbedder(), client=client or FakeClient(existing=["idx"]), **kwargs)


def chunk(i):
    meta = PaperMetadata(arxiv_id="p1", title="T", authors=["A", "B", "C", "D"], abstract="a", year=2024, doi="10.1/x")
    return PaperChunk(chunk_id=f"p1_{i}", arxiv_id="p1", text=f"text {i}", section_name="Results",
                      chunk_index=i, char_count=6, metadata=meta)


def test_an_api_key_is_required_when_no_client_is_injected() -> None:
    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        PineconeVectorStore(None, "idx")


def test_a_missing_index_is_created_as_cosine_384d_serverless(monkeypatch) -> None:
    class ServerlessSpec:
        def __init__(self, cloud, region):
            self.cloud, self.region = cloud, region

    monkeypatch.setitem(sys.modules, "pinecone", types.SimpleNamespace(ServerlessSpec=ServerlessSpec))
    client = FakeClient(existing=["other"])
    store(client, cloud="aws", region="eu-west-1")
    created = client.created[0]
    assert (created["name"], created["dimension"], created["metric"]) == ("idx", 384, "cosine")
    assert (created["spec"].cloud, created["spec"].region) == ("aws", "eu-west-1")


def test_an_existing_index_is_reused() -> None:
    client = FakeClient(existing=["idx"])
    store(client)
    assert client.created == []


def test_chunks_are_upserted_in_batches_with_text_and_doi_in_the_metadata() -> None:
    index = FakeIndex()
    store(FakeClient(existing=["idx"], index=index)).add_chunks([chunk(i) for i in range(250)])
    assert [len(batch) for batch in index.upserts] == [100, 100, 50]
    first = index.upserts[0][0]
    assert first["id"] == "p1_0" and first["values"] == [1.0, 0.0]
    assert first["metadata"]["text"] == "text 0" and first["metadata"]["doi"] == "10.1/x"
    assert first["metadata"]["authors"] == "A, B, C"  # first three, as the Chroma store does


def test_search_dedupes_per_paper_by_default_and_reports_cosine_similarity() -> None:
    index = FakeIndex(matches=[match("a_0", 0.9, "a"), match("a_1", 0.8, "a"), match("b_0", 0.7, "b")])
    results = store(FakeClient(existing=["idx"], index=index)).search("q", n_results=5, where={"year": 2024})
    assert [r["id"] for r in results] == ["a_0", "b_0"]
    assert results[0]["similarity"] == 0.9 and results[0]["text"] == "t" and "text" not in results[0]["metadata"]
    assert index.queries[0] == {"top_k": 25, "filter": {"year": 2024}}  # over-fetch so dedup can still fill n


def test_search_without_dedup_keeps_every_chunk_and_stops_at_n() -> None:
    index = FakeIndex(matches=[match("a_0", 0.9, "a"), match("a_1", 0.8, "a"), match("a_2", 0.7, "a")])
    results = store(FakeClient(existing=["idx"], index=index)).search("q", n_results=2, deduplicate_papers=False)
    assert [r["id"] for r in results] == ["a_0", "a_1"] and index.queries[0]["top_k"] == 2


def test_similarities_are_dot_products_and_count_reads_index_stats() -> None:
    index = FakeIndex(vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]}, total=42)
    s = store(FakeClient(existing=["idx"], index=index))
    assert s.similarities("q", ["a", "b"]) == {"a": pytest.approx(1.0), "b": pytest.approx(0.0)}
    assert s.similarities("q", []) == {} and s.count() == 42


# ---- stores that cannot list documents degrade instead of failing ---------------------------------


def test_the_catalogue_reports_chunks_but_refuses_to_list_when_the_store_cannot_enumerate() -> None:
    catalog = PaperCatalog(store(FakeClient(existing=["idx"], index=FakeIndex(total=7))))
    assert catalog.stats() == {"total_papers": None, "total_chunks": 7}
    with pytest.raises(ListingUnsupported):
        catalog.list()


def test_hybrid_search_is_semantic_only_when_the_store_cannot_list(monkeypatch) -> None:
    deps.get_hybrid_search.cache_clear()
    monkeypatch.setattr(deps, "get_vector_store", lambda: store())
    try:
        assert deps.get_hybrid_search().bm25 is None
    finally:
        deps.get_hybrid_search.cache_clear()


def test_the_store_factory_rejects_an_unknown_backend(monkeypatch) -> None:
    deps.get_vector_store.cache_clear()
    monkeypatch.setattr(deps.settings, "vector_store", "faiss")
    try:
        with pytest.raises(ValueError, match="VECTOR_STORE"):
            deps.get_vector_store()
    finally:
        deps.get_vector_store.cache_clear()


def test_papers_api_answers_501_and_corpus_still_works_without_listing() -> None:
    catalog = PaperCatalog(store(FakeClient(existing=["idx"], index=FakeIndex(total=7))))
    app.dependency_overrides[deps.get_paper_catalog] = lambda: catalog
    app.dependency_overrides[deps.get_persistence] = lambda: PersistenceService(None)
    try:
        client = TestClient(app)
        assert client.get("/api/papers").status_code == 501
        assert client.get("/api/papers/x").status_code == 501
        corpus = client.get("/api/corpus").json()
        assert corpus["total_chunks"] == 7 and corpus["total_papers"] is None and corpus["sample_papers"] == []
    finally:
        app.dependency_overrides.clear()


def test_numpy_scalars_from_the_client_do_not_break_similarity() -> None:
    index = FakeIndex(vectors={"a": np.array([0.6, 0.8])})
    assert store(FakeClient(existing=["idx"], index=index)).similarities("q", ["a"])["a"] == pytest.approx(0.6)
