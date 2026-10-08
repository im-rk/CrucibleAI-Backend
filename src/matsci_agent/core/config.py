"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings for the backend runtime."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "CrucibleAI Agent Backend"
    environment: str = "development"
    api_prefix: str = "/api"
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="matsci_agent", alias="POSTGRES_DB")
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    materials_project_api_key: str | None = Field(default=None, alias="MATERIALS_PROJECT_API_KEY")

    # LLM + agent tuning. Model defaults to the one named in the project spec; override freely.
    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL")
    # anthropic | gemini | groq | auto (auto: the first of Anthropic, Gemini, Groq whose key is set)
    llm_provider: str = Field(default="auto", alias="LLM_PROVIDER")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.8-flash", alias="GEMINI_MODEL")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    relevance_threshold: float = Field(default=0.65, alias="RELEVANCE_THRESHOLD")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    min_relevant_chunks: int = Field(default=5, alias="MIN_RELEVANT_CHUNKS")
    live_arxiv_fallback: bool = Field(default=True, alias="LIVE_ARXIV_FALLBACK")
    llm_concurrency: int = Field(default=4, alias="LLM_CONCURRENCY")
    # Response cache for LLM calls: on | off | refresh (ignore stored answers, store fresh ones)
    llm_cache: str = Field(default="on", alias="LLM_CACHE")
    llm_cache_path: str = Field(default=".cache/llm_cache.sqlite", alias="LLM_CACHE_PATH")
    llm_cache_ttl_s: float = Field(default=7 * 24 * 3600, alias="LLM_CACHE_TTL_S")
    # Provider token-per-minute cap to schedule around (0 = no limiter); e.g. 8000 for Groq's free tier
    llm_tokens_per_minute: int = Field(default=0, alias="LLM_TOKENS_PER_MINUTE")
    # Most real (non-cached) LLM calls one research run may make
    llm_max_calls_per_run: int = Field(default=60, alias="LLM_MAX_CALLS_PER_RUN")

    # API runtime
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:5173", alias="CORS_ORIGINS")
    pipeline_timeout_s: int = Field(default=300, alias="PIPELINE_TIMEOUT_S")
    max_concurrent_runs: int = Field(default=2, alias="MAX_CONCURRENT_RUNS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    persistence_enabled: bool = Field(default=True, alias="PERSISTENCE_ENABLED")

    # Vector store: embedded ChromaDB for development, Pinecone for production.
    vector_store: str = Field(default="chroma", alias="VECTOR_STORE")
    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index: str = Field(default="crucible-papers", alias="PINECONE_INDEX")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")

    # Retrieval tuning (hybrid BM25 + semantic search, optional cross-encoder rerank).
    # Weight of the keyword (BM25) list in hybrid fusion. 2.0 beat equal weights on every retrieval benchmark
    # task (docs/EVALUATION.md); raise it for keyword-style queries, lower it for conversational ones.
    bm25_weight: float = Field(default=2.0, alias="BM25_WEIGHT")
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    arxiv_verify_ssl: bool = Field(default=True, alias="ARXIV_VERIFY_SSL")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()


settings = get_settings()

