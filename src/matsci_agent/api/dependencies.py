"""FastAPI dependencies for sessions, services, and shared infrastructure.

Heavy resources (embedding model, Chroma client, KRR models) are built once per process;
building them per request re-loads the sentence-transformer every time.
"""

from __future__ import annotations

from functools import lru_cache

from matsci_agent.core.config import settings
from matsci_agent.infrastructure.clients.mp_api import MaterialsProjectClient
from matsci_agent.infrastructure.ml.krr import KRRPropertyPredictor
from matsci_agent.infrastructure.retrieval.bm25_index import BM25Index
from matsci_agent.infrastructure.retrieval.hybrid_search import HybridSearch
from matsci_agent.infrastructure.retrieval.reranker import CrossEncoderReranker
from matsci_agent.infrastructure.vector_store.chromadb import ChromaVectorStore
from matsci_agent.services.research_srv import ResearchService


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()


@lru_cache(maxsize=1)
def get_hybrid_search() -> HybridSearch:
    store = get_vector_store()
    reranker = (
        CrossEncoderReranker(settings.reranker_model) if settings.reranker_enabled else None
    )
    return HybridSearch(store, bm25=BM25Index(store), reranker=reranker)


@lru_cache(maxsize=1)
def get_mp_client() -> MaterialsProjectClient:
    return MaterialsProjectClient()


@lru_cache(maxsize=1)
def get_krr_predictor() -> KRRPropertyPredictor:
    return KRRPropertyPredictor()


@lru_cache(maxsize=1)
def get_research_service() -> ResearchService:
    return ResearchService(
        search=get_hybrid_search(),
        mp_client=get_mp_client(),
        krr_predictor=get_krr_predictor(),
    )
