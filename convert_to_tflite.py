"""
Convert the expanded Keras model (.keras directory) to a quantized TFLite file.

Usage:
    python convert_to_tflite.py

Output:
    app/models/plant_disease_model.tflite
"""

import json
from pathlib import Path

import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "app" / "models" / "Plant_disease_prediction_model.keras"
OUTPUT_PATH = Path(__file__).parent / "app" / "models" / "plant_disease_model.tflite"

CONFIG_PATH = MODEL_DIR / "config.json"
WEIGHTS_PATH = MODEL_DIR / "model.weights.h5"


def main() -> None:
    # 1. Validate source files exist
    for p in (CONFIG_PATH, WEIGHTS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    print(f"[1/4] Loading model architecture from {CONFIG_PATH} ...")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model = tf.keras.models.model_from_json(json.dumps(config))

    print(f"[2/4] Loading weights from {WEIGHTS_PATH} ...")
    model.load_weights(str(WEIGHTS_PATH))
    print(f"       Model input shape : {model.input_shape}")
    print(f"       Model output shape: {model.output_shape}")

    # 2. Convert to TFLite with dynamic-range quantization
    print("[3/4] Converting to TFLite with dynamic-range quantization ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantization
    tflite_model = converter.convert()

    # 3. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(tflite_model)

    # 4. Report sizes
    weights_size_mb = WEIGHTS_PATH.stat().st_size / (1024 * 1024)
    tflite_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)

    print(f"[4/4] Done!")
    print(f"       Original weights : {weights_size_mb:>8.2f} MB")
    print(f"       TFLite output    : {tflite_size_mb:>8.2f} MB")
    print(f"       Compression ratio: {weights_size_mb / tflite_size_mb:.1f}x")
    print(f"       Saved to: {OUTPUT_PATH}")

    if tflite_size_mb > 100:
        print("\n⚠️  WARNING: TFLite file is still over 100 MB.")
        print("   Consider using float16 or full int8 quantization.")
    else:
        print(f"\n✅  File is under 100 MB — ready for GitHub!")


if __name__ == "__main__":
    main()
