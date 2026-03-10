"""
Keras model service for plant disease classification.

Loads the trained CNN model and predicts the disease class
from the 38-class PlantVillage dataset.
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


class KerasModelService:
    """Loads and runs the Keras plant disease classification model."""

    def __init__(self, settings: Settings) -> None:
        self._model = None
        self._model_path = settings.KERAS_MODEL_PATH
        self._load_model()

    def _load_model(self) -> None:
        """Load the Keras model at startup from the expanded .keras directory."""
        import json as _json
        import tensorflow as tf

        model_dir = Path(self._model_path)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.weights.h5"

        if not model_dir.exists():
            logger.error("Keras model directory not found at {}", model_dir)
            raise FileNotFoundError(f"Keras model not found at {model_dir}")

        if not config_path.exists() or not weights_path.exists():
            logger.error("Missing config.json or model.weights.h5 in {}", model_dir)
            raise FileNotFoundError(f"Incomplete model at {model_dir}")

        logger.info("Loading Keras model from {} ...", model_dir)

        # Load architecture from config.json
        with open(config_path, "r") as f:
            config = _json.load(f)

        self._model = tf.keras.models.model_from_json(_json.dumps(config))

        # Load weights
        self._model.load_weights(str(weights_path))

        logger.info("Keras model loaded — input_shape={}, classes={}",
                     self._model.input_shape, len(CLASS_LABELS))

    def predict(self, image_bytes: bytes) -> dict[str, str]:
        """
        Run inference on the image.

        Returns:
            dict with 'plant_name', 'disease_name', 'is_diseased'
        """
        import tensorflow as tf

        # Decode and preprocess
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image for Keras prediction")

        # Convert BGR → RGB, resize to 224×224, normalize to [0, 1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)  # (1, 224, 224, 3)

        # Predict
        predictions = self._model.predict(img_batch, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        raw_label = CLASS_LABELS[predicted_index]

        # Parse label: "Plant___Disease" → plant_name, disease_name
        plant_name, disease_part = self._parse_label(raw_label)
        is_diseased = disease_part.lower() != "healthy"

        logger.info(
            "Keras prediction — label='{}', plant='{}', disease='{}', is_diseased={}",
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
