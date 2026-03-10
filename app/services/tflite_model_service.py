"""
TFLite model service for plant disease classification.

Loads the quantized TFLite model and predicts the disease class
from the 38-class PlantVillage dataset. TFLite provides faster inference
and lower memory usage compared to full Keras models.
"""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from app.core.config import Settings

# ── 38 PlantVillage class labels (alphabetical, standard order) ───────
CLASS_LABELS: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


class TFLiteModelService:
    """Loads and runs the TFLite plant disease classification model."""

    def __init__(self, settings: Settings) -> None:
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._model_path = settings.TFLITE_MODEL_PATH
        self._load_model()

    def _load_model(self) -> None:
        """Load the TFLite model at startup."""
        import tensorflow as tf

        model_path = Path(self._model_path)

        if not model_path.exists():
            logger.error("TFLite model not found at {}", model_path)
            raise FileNotFoundError(f"TFLite model not found at {model_path}")

        logger.info("Loading TFLite model from {} ...", model_path)

        # Load the TFLite model
        self._interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        # Get input and output details
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        logger.info(
            "TFLite model loaded — input_shape={}, output_shape={}, classes={}",
            self._input_details[0]["shape"],
            self._output_details[0]["shape"],
            len(CLASS_LABELS),
        )

    def predict(self, image_bytes: bytes) -> dict[str, str]:
        """
        Run inference on the image using TFLite interpreter.

        Returns:
            dict with 'plant_name', 'disease_name', 'is_diseased'
        """
        # Decode and preprocess
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image for TFLite prediction")

        # Convert BGR → RGB, resize to 224×224, normalize to [0, 1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)  # (1, 224, 224, 3)

        # Prepare input for TFLite
        self._interpreter.set_tensor(self._input_details[0]["index"], img_batch)
        self._interpreter.invoke()

        # Get predictions
        output_data = self._interpreter.get_tensor(self._output_details[0]["index"])
        predicted_index = int(np.argmax(output_data[0]))
        raw_label = CLASS_LABELS[predicted_index]

        # Parse label: "Plant___Disease" → plant_name, disease_name
        plant_name, disease_part = self._parse_label(raw_label)
        is_diseased = disease_part.lower() != "healthy"

        logger.info(
            "TFLite prediction — label='{}', plant='{}', disease='{}', is_diseased={}",
            raw_label, plant_name, disease_part, is_diseased,
        )

        return {
            "plant_name": plant_name,
            "disease_name": disease_part if is_diseased else "Healthy",
            "is_diseased": is_diseased,
        }

    @staticmethod
    def _parse_label(raw_label: str) -> tuple[str, str]:
        """
        Parse a PlantVillage label like 'Tomato___Early_blight'
        into ('Tomato', 'Early Blight').
        """
        parts = raw_label.split("___")
        plant = parts[0].replace("_", " ").replace(",", ",").strip()
        disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"

        # Title-case for nice display
        plant = plant.title() if plant.islower() else plant
        disease = disease.title() if disease.islower() else disease

        return plant, disease
