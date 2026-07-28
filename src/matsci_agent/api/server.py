"""FastAPI application factory and startup wiring."""

# from __future__ import annotations

# from fastapi import FastAPI

# from matsci_agent.api.middleware import configure_middleware
# from matsci_agent.api.routers.research import router as research_router
# from matsci_agent.api.routers.stream import router as stream_router
# from matsci_agent.core.config import settings
# from matsci_agent.core.logging import configure_logging


# def create_app() -> FastAPI:
#     """Build and configure the FastAPI application."""

#     configure_logging()
#     app = FastAPI(title=settings.app_name)
#     configure_middleware(app)
#     app.include_router(research_router, prefix=settings.api_prefix)
#     app.include_router(stream_router, prefix=settings.api_prefix)
#     return app


# app = create_app()
