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

    # Retrieval tuning (hybrid BM25 + semantic search, optional cross-encoder rerank).
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    arxiv_verify_ssl: bool = Field(default=True, alias="ARXIV_VERIFY_SSL")

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

