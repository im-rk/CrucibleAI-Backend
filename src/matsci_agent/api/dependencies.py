"""FastAPI dependencies for sessions, services, and shared infrastructure.

Heavy resources (embedding model, Chroma client, KRR models, the compiled agent graph) are
built once per process; building them per request re-loads the sentence-transformer every time.
"""

from __future__ import annotations

from functools import lru_cache

from matsci_agent.agents.nodes.extraction import ExtractionAgent
from matsci_agent.agents.nodes.critic import ReviseAgent, VerifyAgent
from matsci_agent.agents.nodes.hypothesis import HypothesisAgent
from matsci_agent.agents.nodes.investigator import InvestigatorAgent
from matsci_agent.agents.nodes.planner import PlannerAgent
from matsci_agent.agents.nodes.qa import ReportQA
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
from matsci_agent.infrastructure.llm.instrumented import InstrumentedLLM, LLMCache
from matsci_agent.infrastructure.llm.ratelimit import TokenRateLimiter
from matsci_agent.infrastructure.ml.krr import KRRPropertyPredictor
from matsci_agent.infrastructure.retrieval.bm25_index import BM25Index
from matsci_agent.infrastructure.retrieval.hybrid_search import HybridSearch
from matsci_agent.infrastructure.retrieval.reranker import CrossEncoderReranker
from matsci_agent.infrastructure.vector_store.chromadb import ChromaVectorStore
from matsci_agent.infrastructure.vector_store.pinecone import PineconeVectorStore
from matsci_agent.services.grounding import GroundingChecker
from matsci_agent.services.papers_srv import PaperCatalog
from matsci_agent.services.persistence import PersistenceService
from matsci_agent.services.report_store import ReportStore
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
    return HybridSearch(store, bm25=bm25, reranker=reranker, bm25_weight=settings.bm25_weight)


@lru_cache(maxsize=1)
def get_paper_catalog() -> PaperCatalog:
    return PaperCatalog(get_vector_store())


def build_provider_client() -> LLMClient | GeminiClient | GroqClient:
    """The raw provider client selected by LLM_PROVIDER (no cache, tracing or budget)."""
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
def get_llm_client() -> InstrumentedLLM:
    """The provider client wrapped with the response cache, token accounting and run budget."""
    cache = LLMCache(settings.llm_cache_path, settings.llm_cache_ttl_s) if settings.llm_cache != "off" else None
    limiter = TokenRateLimiter(settings.llm_tokens_per_minute) if settings.llm_tokens_per_minute > 0 else None
    return InstrumentedLLM(build_provider_client(), cache=cache, cache_mode=settings.llm_cache, limiter=limiter)


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
def get_report_store() -> ReportStore:
    return ReportStore()


@lru_cache(maxsize=1)
def get_qa() -> ReportQA:
    return ReportQA(get_llm_client())


@lru_cache(maxsize=1)
def get_planner() -> PlannerAgent:
    return PlannerAgent(get_llm_client())


def build_research_service(llm, *, persistence=None, reports=None, mode=None, mp=None) -> ResearchService:
    """Assemble the whole agent graph around `llm` (any object with the LLM client interface).

    The evaluation suite uses this to run the identical pipeline with the LLM switched on or off.
    """
    mp, krr = mp or get_mp_client(), get_krr_predictor()
    search = SearchAgent(
        llm, get_hybrid_search(), get_arxiv_client(),
        relevance_threshold=settings.relevance_threshold,
        min_relevant=settings.min_relevant_chunks,
        max_retries=settings.max_retries,
        live_arxiv=settings.live_arxiv_fallback,
        concurrency=settings.llm_concurrency,
    )
    extraction = ExtractionAgent(llm, mp, concurrency=settings.llm_concurrency)
    checker = GroundingChecker(llm)
    graph = build_research_graph(
        search,
        extraction,
        ScreeningAgent(llm, krr, mp),
        HypothesisAgent(llm),
        ReportAgent(checker),
        planner=PlannerAgent(llm),
        investigator=InvestigatorAgent(llm, search, extraction, mp, krr),
        verifier=VerifyAgent(checker, lambda: llm.available),
        reviser=ReviseAgent(llm),
    )
    return ResearchService(
        graph, mode or runtime_mode, persistence=persistence,
        timeout_s=settings.pipeline_timeout_s, max_concurrent=settings.max_concurrent_runs,
        reports=reports,
    )


@lru_cache(maxsize=1)
def get_research_service() -> ResearchService:
    return build_research_service(
        get_llm_client(), persistence=get_persistence(), reports=get_report_store(),
    )
