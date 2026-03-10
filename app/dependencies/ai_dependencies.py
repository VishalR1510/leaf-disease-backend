"""
FastAPI dependency-injection factories.

Each function returns a fully-configured service instance
that can be injected into route handlers via `Depends(...)`.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.services.groq_service import GroqService
from app.services.tflite_model_service import TFLiteModelService
from app.services.image_validator import ImageValidator


@lru_cache()
def get_image_validator() -> ImageValidator:
    """Singleton image validator."""
    return ImageValidator(settings=get_settings())


@lru_cache()
def get_groq_service() -> GroqService:
    """Singleton Groq AI service."""
    return GroqService(settings=get_settings())


@lru_cache()
def get_keras_service() -> TFLiteModelService:
    """Singleton TFLite model service."""
    return TFLiteModelService(settings=get_settings())
