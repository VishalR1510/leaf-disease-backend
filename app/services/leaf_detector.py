"""
Leaf detection service.

Uses OpenCV colour-space analysis (HSV green-channel dominance)
to heuristically determine whether an image contains a plant leaf.
"""

import cv2
import numpy as np
from loguru import logger


class LeafDetectionError(Exception):
    """Raised when the image does not appear to contain a leaf."""

    def __init__(self, code: str = "NOT_A_LEAF", message: str = "") -> None:
        self.code = code
        self.message = message or "The uploaded image does not appear to contain a plant leaf."
        super().__init__(self.message)


class LeafDetector:
    """Heuristic plant-leaf detector using HSV colour analysis."""

    # HSV ranges that capture typical green foliage
    GREEN_LOWER = np.array([25, 30, 30])
    GREEN_UPPER = np.array([95, 255, 255])

    # Minimum fraction of green pixels to be considered a leaf image
    GREEN_THRESHOLD: float = 0.05  # 5 %

    def detect(self, img: np.ndarray) -> bool:
        """
        Return *True* if the image likely contains a plant leaf.

        Args:
            img: BGR image decoded by OpenCV.

        Returns:
            bool indicating leaf presence.
        """
        green_ratio = self._green_pixel_ratio(img)
        contour_found = self._has_leaf_contour(img)

        logger.info(
            "Leaf detection — green_ratio={:.3f}, contour_found={}",
            green_ratio,
            contour_found,
        )

        # Accept if either heuristic is positive
        return green_ratio >= self.GREEN_THRESHOLD or contour_found

    # ── Private helpers ───────────────────────────────────

    def _green_pixel_ratio(self, img: np.ndarray) -> float:
        """Fraction of pixels falling within the green HSV range."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
        return float(np.count_nonzero(mask)) / mask.size

    @staticmethod
    def _has_leaf_contour(img: np.ndarray) -> bool:
        """Check for a large, smooth contour typical of a leaf shape."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        img_area = img.shape[0] * img.shape[1]
        min_area = img_area * 0.02  # at least 2 % of the image

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    # Leaves typically have circularity between 0.1 and 0.9
                    if 0.05 <= circularity <= 0.95:
                        return True
        return False
