"""
FastAPI application entry-point.

Creates the application, configures middleware, and mounts routers.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.v1.endpoints.leaf_analysis import router as leaf_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.security import configure_cors


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    # Logging
    setup_logging(log_level="DEBUG" if settings.DEBUG else "INFO")
    logger.info("Starting {} v{}", settings.APP_NAME, settings.APP_VERSION)

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-powered leaf disease detection API",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ─────────────────────────────────────────
    configure_cors(app)

    @app.middleware("http")
    async def global_exception_handler(request: Request, call_next):
        """Catch any unhandled exception and return a standard error JSON."""
        try:
            return await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled error on {} {}", request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected error occurred. Please try again later.",
                    },
                },
            )

    # ── Routers ───────────────────────────────────────────
    app.include_router(leaf_router, prefix=settings.API_V1_PREFIX, tags=["Leaf Analysis"])

    # ── Health check ──────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}

    return app


app = create_app()
