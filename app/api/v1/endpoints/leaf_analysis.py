"""
Leaf Analysis API endpoint.

POST /api/v1/analyze-leaf

Pipeline:
  1. Validate image format/size
  2. Groq llama-4-scout confirms it's a leaf
  3. TFLite CNN predicts disease name
  4. Groq gpt-oss-120b provides disease details
  5. Return merged result
"""

from fastapi import APIRouter, Depends, UploadFile, File
from loguru import logger

from app.dependencies.ai_dependencies import (
    get_groq_service,
    get_image_validator,
    get_keras_service,
)
from app.models.response_models import APIResponse, ErrorDetail, LeafAnalysisData
from app.services.groq_service import GroqService, GroqServiceError
from app.services.tflite_model_service import TFLiteModelService
from app.services.image_validator import ImageValidator, ImageValidationError
from app.utils.image_processing import encode_base64, normalize_format, resize_image

router = APIRouter()


@router.post(
    "/analyze-leaf",
    response_model=APIResponse,
    summary="Analyse a leaf image for diseases",
    description="Upload a leaf image and receive an AI-powered disease analysis.",
)
async def analyze_leaf(
    file: UploadFile = File(..., description="Leaf image (jpg, jpeg, png, webp)"),
    validator: ImageValidator = Depends(get_image_validator),
    groq: GroqService = Depends(get_groq_service),
    keras: TFLiteModelService = Depends(get_keras_service),
) -> APIResponse:
    """Full analysis pipeline: validate → leaf check → classify → details."""

    # 1. Read file bytes
    file_bytes = await file.read()
    filename = file.filename or "unknown"
    logger.info("Received file '{}' ({} bytes)", filename, len(file_bytes))

    # 2. Validate the image (format, size, decodable)
    try:
        validator.validate(filename=filename, file_bytes=file_bytes)
    except ImageValidationError as exc:
        logger.warning("Image validation failed: {}", exc.message)
        return APIResponse(
            success=False,
            error=ErrorDetail(code=exc.code, message=exc.message),
        )

    # 3. Groq vision model — is this a leaf?
    try:
        processed = normalize_format(file_bytes)
        processed = resize_image(processed, max_dim=512)
        b64 = encode_base64(processed)
        is_leaf = await groq.confirm_leaf(b64)
    except GroqServiceError as exc:
        logger.error("Leaf verification failed: {}", exc.message)
        return APIResponse(
            success=False,
            error=ErrorDetail(code=exc.code, message=exc.message),
        )

    if not is_leaf:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="NOT_A_LEAF",
                message="The uploaded image does not appear to contain a plant leaf. Please upload a plant image.",
            ),
        )

    # 4. TFLite model — predict disease name
    try:
        prediction = keras.predict(file_bytes)
    except Exception as exc:
        logger.error("TFLite prediction failed: {}", exc)
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_ERROR",
                message="Disease classification failed. Please try again.",
            ),
        )

    plant_name = prediction["plant_name"]
    disease_name = prediction["disease_name"]
    is_diseased = prediction["is_diseased"]

    logger.info("TFLite → plant='{}', disease='{}', diseased={}", plant_name, disease_name, is_diseased)

    # 5. Groq reasoning model — get disease details
    try:
        details = await groq.get_disease_details(plant_name, disease_name)
    except GroqServiceError as exc:
        logger.error("Disease detail generation failed: {}", exc.message)
        return APIResponse(
            success=False,
            error=ErrorDetail(code=exc.code, message=exc.message),
        )

    # 6. Merge and return
    analysis = LeafAnalysisData(
        plant_name=plant_name,
        is_leaf=True,
        is_diseased=is_diseased,
        disease_name=disease_name,
        symptoms=details.get("symptoms", ""),
        disease_description=details.get("disease_description", ""),
        possible_causes=details.get("possible_causes", ""),
        preventive_measures=details.get("preventive_measures", ""),
        treatment_suggestions=details.get("treatment_suggestions", ""),
    )

    logger.info("Analysis complete — plant={}, disease={}", plant_name, disease_name)
    return APIResponse(success=True, data=analysis)
