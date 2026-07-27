# Multi-stage image for running the FastAPI backend in production.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

FROM base AS builder

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY uv.lock ./uv.lock
COPY src ./src
COPY scripts ./scripts
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini

RUN uv sync --frozen --no-dev

FROM base AS runtime

COPY --from=builder /app/.venv /app/.venv
COPY . .

EXPOSE 8000
CMD ["uvicorn", "matsci_agent.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
