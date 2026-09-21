"""Logging configuration for the backend."""

from __future__ import annotations

import logging

from matsci_agent.core.config import settings


def configure_logging() -> None:
    """Plain stdlib logging at LOG_LEVEL; uvicorn keeps its own access log."""
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Chatty third-party loggers drown the pipeline's own progress lines.
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
