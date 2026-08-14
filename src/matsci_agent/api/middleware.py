"""Middleware for request logging, CORS, and error handling."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def configure_middleware(app: FastAPI) -> None:
    """Attach the baseline middleware stack to the app."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

