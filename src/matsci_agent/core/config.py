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
    relevance_threshold: float = Field(default=0.65, alias="RELEVANCE_THRESHOLD")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    min_relevant_chunks: int = Field(default=5, alias="MIN_RELEVANT_CHUNKS")
    live_arxiv_fallback: bool = Field(default=True, alias="LIVE_ARXIV_FALLBACK")
    llm_concurrency: int = Field(default=4, alias="LLM_CONCURRENCY")

    # API runtime
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:5173", alias="CORS_ORIGINS")
    pipeline_timeout_s: int = Field(default=300, alias="PIPELINE_TIMEOUT_S")
    max_concurrent_runs: int = Field(default=2, alias="MAX_CONCURRENT_RUNS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Retrieval tuning (hybrid BM25 + semantic search, optional cross-encoder rerank).
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

