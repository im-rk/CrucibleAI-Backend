"""FastAPI dependencies for sessions, services, and shared infrastructure.

Heavy resources (embedding model, Chroma client, KRR models, the compiled agent graph) are
built once per process; building them per request re-loads the sentence-transformer every time.
"""

from __future__ import annotations

from functools import lru_cache

from matsci_agent.agents.nodes.extraction import ExtractionAgent
from matsci_agent.agents.nodes.hypothesis import HypothesisAgent
from matsci_agent.agents.nodes.report import ReportAgent
from matsci_agent.agents.nodes.screening import ScreeningAgent
from matsci_agent.agents.nodes.search import SearchAgent
from matsci_agent.agents.orchestrator import build_research_graph
from matsci_agent.core.config import settings
from matsci_agent.infrastructure.clients.arxiv import ArxivClient
from matsci_agent.infrastructure.clients.mp_api import MaterialsProjectClient
from matsci_agent.infrastructure.db.session import create_session_factory
from matsci_agent.infrastructure.llm.anthropic_client import LLMClient
from matsci_agent.infrastructure.llm.gemini_client import GeminiClient
from matsci_agent.infrastructure.llm.groq_client import GroqClient
from matsci_agent.infrastructure.ml.krr import KRRPropertyPredictor
from matsci_agent.infrastructure.retrieval.bm25_index import BM25Index
from matsci_agent.infrastructure.retrieval.hybrid_search import HybridSearch
from matsci_agent.infrastructure.retrieval.reranker import CrossEncoderReranker
from matsci_agent.infrastructure.vector_store.chromadb import ChromaVectorStore
from matsci_agent.infrastructure.vector_store.pinecone import PineconeVectorStore
from matsci_agent.services.grounding import GroundingChecker
from matsci_agent.services.papers_srv import PaperCatalog
from matsci_agent.services.persistence import PersistenceService
from matsci_agent.services.research_srv import ResearchService


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore | PineconeVectorStore:
    if settings.vector_store == "pinecone":
        return PineconeVectorStore(
            settings.pinecone_api_key, settings.pinecone_index,
            cloud=settings.pinecone_cloud, region=settings.pinecone_region,
        )
    if settings.vector_store != "chroma":
        raise ValueError(f"VECTOR_STORE must be 'chroma' or 'pinecone', not '{settings.vector_store}'")
    return ChromaVectorStore()


@lru_cache(maxsize=1)
def get_hybrid_search() -> HybridSearch:
    store = get_vector_store()
    reranker = (
        CrossEncoderReranker(settings.reranker_model) if settings.reranker_enabled else None
    )
    # BM25 needs every document; stores that cannot list them (Pinecone) get semantic-only search.
    bm25 = BM25Index(store) if getattr(store, "supports_listing", True) else None
    return HybridSearch(store, bm25=bm25, reranker=reranker)


@lru_cache(maxsize=1)
def get_paper_catalog() -> PaperCatalog:
    return PaperCatalog(get_vector_store())


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient | GeminiClient | GroqClient:
    provider = settings.llm_provider.lower()
    if provider == "auto":
        provider = (
            "anthropic" if settings.anthropic_api_key
            else "gemini" if settings.gemini_api_key
            else "groq" if settings.groq_api_key
            else "anthropic"
        )
    if provider == "gemini":
        return GeminiClient(settings.gemini_api_key, settings.gemini_model)
    if provider == "groq":
        return GroqClient(settings.groq_api_key, settings.groq_model)
    if provider != "anthropic":
        raise ValueError(f"LLM_PROVIDER must be 'anthropic', 'gemini', 'groq' or 'auto', not '{settings.llm_provider}'")
    return LLMClient(settings.anthropic_api_key, settings.anthropic_model)


@lru_cache(maxsize=1)
def get_arxiv_client() -> ArxivClient:
    return ArxivClient()


@lru_cache(maxsize=1)
def get_mp_client() -> MaterialsProjectClient:
    return MaterialsProjectClient()


@lru_cache(maxsize=1)
def get_krr_predictor() -> KRRPropertyPredictor:
    return KRRPropertyPredictor()


@lru_cache(maxsize=1)
def get_persistence() -> PersistenceService:
    factory = create_session_factory(settings.database_url) if settings.persistence_enabled else None
    return PersistenceService(factory)


def runtime_mode() -> dict:
    """What the pipeline is actually running on; reported by /health and in every report."""
    llm = get_llm_client()
    return {
        "llm": llm.model if llm.available else None,
        "llm_unavailable_reason": llm.unavailable_reason,
        "materials_project": "live" if get_mp_client().online else "offline",
    }


@lru_cache(maxsize=1)
def get_research_service() -> ResearchService:
    llm, mp, krr = get_llm_client(), get_mp_client(), get_krr_predictor()
    graph = build_research_graph(
        SearchAgent(
            llm, get_hybrid_search(), get_arxiv_client(),
            relevance_threshold=settings.relevance_threshold,
            min_relevant=settings.min_relevant_chunks,
            max_retries=settings.max_retries,
            live_arxiv=settings.live_arxiv_fallback,
            concurrency=settings.llm_concurrency,
        ),
        ExtractionAgent(llm, mp, concurrency=settings.llm_concurrency),
        ScreeningAgent(llm, krr, mp),
        HypothesisAgent(llm),
        ReportAgent(GroundingChecker(llm)),
    )
    return ResearchService(
        graph, runtime_mode, persistence=get_persistence(),
        timeout_s=settings.pipeline_timeout_s, max_concurrent=settings.max_concurrent_runs,
    )
