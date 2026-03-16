"""
Application settings loaded from environment variables.

Uses pydantic-settings for type-safe configuration with .env support.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Leaf Disease Detection backend."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────
    APP_NAME: str = "LeafGuard AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── API ───────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"

    # ── CORS ──────────────────────────────────────────────
    CORS_ORIGINS: list[str] = [
        "https://leaf-disease-frontend.vercel.app",
    ]

    # ── Groq AI ───────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    GROQ_REASONING_MODEL: str = "openai/gpt-oss-120b"

    # ── LLM Inference Parameters ──────────────────────────
    LLM_TEMPERATURE: float = 0.3
    LLM_TOP_P: float = 0.8
    LLM_MAX_TOKENS: int = 2048

    # ── TFLite Model ──────────────────────────────────────
    TFLITE_MODEL_PATH: str = str(
        Path(__file__).resolve().parent.parent / "models" / "plant_disease_model.tflite"
    )

    # ── Image Validation ──────────────────────────────────
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: set[str] = {"jpg", "jpeg", "png", "webp"}
    IMAGE_MAX_DIMENSION: int = 1024

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
