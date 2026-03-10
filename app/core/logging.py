"""
Structured logging configuration using Loguru.

Intercepts stdlib logging (used by uvicorn) and routes everything
through loguru with a unified format.
"""

import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """Redirect standard-library log records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Map stdlib level to loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(*, log_level: str = "INFO", json_logs: bool = False) -> None:
    """Configure loguru as the sole logging backend."""
    # Remove default loguru handler
    logger.remove()

    # Console handler
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=fmt,
        level=log_level,
        colorize=True,
        serialize=json_logs,
    )

    # Intercept stdlib loggers (uvicorn, etc.)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [InterceptHandler()]
