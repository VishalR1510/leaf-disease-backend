"""
Image processing utilities.

Provides helpers for resizing, base64-encoding, and format normalisation
using OpenCV.
"""

import base64

import cv2
import numpy as np
from loguru import logger


def resize_image(image_bytes: bytes, max_dim: int = 1024) -> bytes:
    """
    Resize the image so that neither width nor height exceeds *max_dim*.

    If the image is already within bounds it is returned unchanged.
    """
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        logger.warning("resize_image received invalid image bytes")
        return image_bytes

    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return image_bytes

    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.info("Resized image from {}x{} to {}x{}", w, h, new_w, new_h)

    success, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes() if success else image_bytes


def normalize_format(image_bytes: bytes, target_ext: str = ".jpg") -> bytes:
    """Re-encode image to the target format for consistency."""
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return image_bytes

    encode_params: list[int] = []
    if target_ext in (".jpg", ".jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    elif target_ext == ".png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    success, encoded = cv2.imencode(target_ext, img, encode_params)
    return encoded.tobytes() if success else image_bytes


def encode_base64(image_bytes: bytes) -> str:
    """Convert raw image bytes to a base64-encoded string."""
    return base64.b64encode(image_bytes).decode("utf-8")
