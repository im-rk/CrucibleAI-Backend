"""FastAPI application factory and startup wiring."""

from __future__ import annotations

from fastapi import FastAPI

from matsci_agent import __version__
from matsci_agent.api.middleware import configure_middleware
from matsci_agent.api.routers import materials, papers, research, system
from matsci_agent.core.config import settings
from matsci_agent.core.logging import configure_logging


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    configure_logging()
    app = FastAPI(title=settings.app_name, version=__version__)
    configure_middleware(app)
    for module in (research, materials, papers, system):
        app.include_router(module.router, prefix=settings.api_prefix)
    return app


app = create_app()
