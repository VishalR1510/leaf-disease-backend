"""
Security-related utilities and CORS helpers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings


def configure_cors(app: FastAPI) -> None:
    """Add CORS middleware with origins from settings."""
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
