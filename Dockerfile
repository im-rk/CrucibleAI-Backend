# Backend image: FastAPI + LangGraph agents. Data (Chroma corpus, KRR models, downloaded
# papers, the embedding-model cache) lives in volumes, not in the image.
# Note: the locked dependency set includes PyTorch, so the image is large (several GB).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

FROM base AS builder

RUN pip install --no-cache-dir uv

# Dependencies first (cached unless the lockfile changes), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM base AS runtime

RUN useradd --create-home --uid 1000 app
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app pyproject.toml alembic.ini ./
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app src ./src
COPY --chown=app:app evaluation ./evaluation
RUN mkdir -p chroma_db papers models .cache && chown -R app:app chroma_db papers models .cache

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# Migrate first; a database that is down must not stop the API (persistence is best-effort).
CMD ["sh", "-c", "alembic upgrade head || echo 'alembic upgrade failed; continuing without persistence'; exec uvicorn matsci_agent.api.server:app --host 0.0.0.0 --port 8000"]
