# MatSci Agent Backend

This repository contains the backend scaffold for the MatSci research agent platform.

The layout is organized around:

- `src/matsci_agent/api` for FastAPI request handling and middleware.
- `src/matsci_agent/domain` for pure business models and interfaces.
- `src/matsci_agent/infrastructure` for databases, vector stores, and external APIs.
- `src/matsci_agent/agents` for LangGraph orchestration and agent nodes.
- `alembic` for database migrations.
- `scripts` for operational CLI helpers.

Run `uv sync --extra dev` after generating a lockfile to install the project dependencies.
