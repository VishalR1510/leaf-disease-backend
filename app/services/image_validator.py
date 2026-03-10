"""
Image validation service.

Validates uploaded files for:
  - Allowed file extensions
  - Maximum file size
  - Actual image decodability (OpenCV)
"""

import cv2
import numpy as np
from loguru import logger

from app.core.config import Settings


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class ImageValidator:
    """Validates uploaded image files."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate(self, *, filename: str, file_bytes: bytes) -> np.ndarray:
        """
        Run all validation checks and return the decoded image (BGR numpy array).

        Raises:
            ImageValidationError: if any check fails.
        """
        self._check_extension(filename)
        self._check_file_size(file_bytes)
        return self._decode_image(file_bytes)

    # ── Private helpers ───────────────────────────────────

    def _check_extension(self, filename: str) -> None:
        ext = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
        if ext not in self._settings.ALLOWED_EXTENSIONS:
            logger.warning("Rejected file extension: .{}", ext)
            raise ImageValidationError(
                code="INVALID_IMAGE",
                message=(
                    f"Unsupported file format '.{ext}'. "
                    f"Accepted formats: {', '.join(sorted(self._settings.ALLOWED_EXTENSIONS))}."
                ),
            )

    def _check_file_size(self, file_bytes: bytes) -> None:
        size = len(file_bytes)
        if size > self._settings.max_file_size_bytes:
            size_mb = round(size / (1024 * 1024), 2)
            logger.warning("Rejected file size: {} MB", size_mb)
            raise ImageValidationError(
                code="INVALID_IMAGE",
                message=(
                    f"File size ({size_mb} MB) exceeds the "
                    f"{self._settings.MAX_FILE_SIZE_MB} MB limit."
                ),
            )

    @staticmethod
    def _decode_image(file_bytes: bytes) -> np.ndarray:
        np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("File could not be decoded as a valid image")
            raise ImageValidationError(
                code="INVALID_IMAGE",
                message="Invalid image file. Please upload a valid leaf image.",
            )
        return img
