"""
GeoCropAI -- Soil Classifier (Inference Module)
soil_model/soil_classifier.py
==============================================

Loads the trained MobileNetV2 model and provides
predict_soil() for use by the Flask application and
the image validation pipeline.

CRITICAL FIXES in v2.0:
  1. CLASS_NAMES now exactly match the training dataset folder
     names (Alluvial_Soil, Arid_Soil, Black_Soil, ...) in
     alphabetical order -- the same order used by
     image_dataset_from_directory during training.

  2. Preprocessing now uses
     tf.keras.applications.mobilenet_v2.preprocess_input()
     which scales pixels to [-1, 1] -- exactly what the
     MobileNetV2 backbone expects.
     The old code passed raw [0-255] values which caused the
     model to predict almost every image as Alluvial_Soil.

  3. SOIL_MAPPING is updated to cover all 7 classes with
     meaningful crop-recommendation system mappings.

  4. Top-3 predictions are returned for the UI display.

  5. Confidence threshold: 75% (aligned with validator).

Public API (unchanged for Flask compatibility):
    predict_soil(image_path)
        -> (cnn_soil: str, mapped_soil: str, confidence: float)
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image


# ==============================================================
# CONFIGURATION
# ==============================================================

MODEL_PATH          = "soil_model.keras"
CONFIDENCE_THRESHOLD = 75.0    # percent; enforced in image_validator.py


# ==============================================================
# CLASS NAMES
#
# MUST be in the same alphabetical order that Keras uses when
# loading the dataset with image_dataset_from_directory().
# Keras sorts folder names alphabetically, so the index mapping
# is: 0=Alluvial_Soil, 1=Arid_Soil, 2=Black_Soil, ...
#
# To verify: run train_soil_model.py and check "Class names
# (training order)" printed to the terminal, or read
# results/class_names.txt.
# ==============================================================

CLASS_NAMES = [
    "Alluvial_Soil",    # index 0
    "Arid_Soil",        # index 1
    "Black_Soil",       # index 2
    "Laterite_Soil",    # index 3
    "Mountain_Soil",    # index 4
    "Red_Soil",         # index 5
    "Yellow_Soil"       # index 6
]

# Display-friendly names for the result page
CLASS_DISPLAY_NAMES = {
    "Alluvial_Soil":  "Alluvial Soil",
    "Arid_Soil":      "Arid Soil",
    "Black_Soil":     "Black Soil",
    "Laterite_Soil":  "Laterite Soil",
    "Mountain_Soil":  "Mountain Soil",
    "Red_Soil":       "Red Soil",
    "Yellow_Soil":    "Yellow Soil"
}


# ==============================================================
# SOIL MAPPING
#
# Maps each predicted soil class to the soil type key used by
# the crop recommendation system (RecommendationAgent).
#
# These strings must match the "soil" field in CROP_DATABASE
# inside data/crop_database.py.
# ==============================================================

SOIL_MAPPING = {
    "Alluvial_Soil":  "loamy",
    "Arid_Soil":      "sandy",
    "Black_Soil":     "black",
    "Laterite_Soil":  "laterite",
    "Mountain_Soil":  "loamy",
    "Red_Soil":       "loamy",
    "Yellow_Soil":    "sandy"
}


# ==============================================================
# LOAD MODEL (once at module import -- not on every request)
# ==============================================================

print("[SoilClassifier] Loading model from: " + MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "[SoilClassifier] Model file not found: " + MODEL_PATH +
        "\nRun train_soil_model.py first to generate the model."
    )

_model = tf.keras.models.load_model(MODEL_PATH)

print("[SoilClassifier] Model loaded successfully.")
print("[SoilClassifier] Classes: " + str(CLASS_NAMES))


# ==============================================================
# PREPROCESSING FUNCTION
# ==============================================================

def _preprocess_image(image_path):
    """
    Loads an image, resizes to 224x224, converts to RGB, and
    prepares it for MobileNetV2.

    If the loaded model has an internal normalization layer
    (Rescaling/Preprocess), we keep pixels in [0, 255].
    Otherwise, we scale them to [-1, 1] using preprocess_input()
    for backward compatibility with older models.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)           # [0, 255]

    # Check if loaded model handles normalization internally
    has_internal_preprocess = any(
        "rescaling" in l.name or "preprocess" in l.name 
        for l in _model.layers
    )

    if not has_internal_preprocess:
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)  # [-1, 1]

    arr = np.expand_dims(arr, axis=0)                # add batch dim
    return arr


# ==============================================================
# PUBLIC API -- predict_soil()
# ==============================================================

def predict_soil(image_path):
    """
    Predicts the soil type from a single image.

    This function is called by:
      - soil_model/image_validator.py  (confidence gate)
      - app.py                         (via validation result)

    Parameters
    ----------
    image_path : str
        Path to the uploaded soil image.

    Returns
    -------
    cnn_soil   : str
        Display-friendly predicted soil name (e.g. "Red Soil").
    mapped_soil : str
        Crop-recommendation system key (e.g. "loamy").
    confidence  : float
        Top-1 prediction confidence as a percentage (0-100).

    Raises
    ------
    Exception
        If the image cannot be loaded or the model fails.
    """
    arr = _preprocess_image(image_path)

    predictions = _model.predict(arr, verbose=0)   # shape (1, 7)
    probs       = predictions[0]                   # shape (7,)

    predicted_index = int(np.argmax(probs))
    confidence      = float(probs[predicted_index] * 100)
    predicted_key   = CLASS_NAMES[predicted_index]

    cnn_soil    = CLASS_DISPLAY_NAMES[predicted_key]
    mapped_soil = SOIL_MAPPING[predicted_key]

    # Top-3 predictions for display
    top3_indices = np.argsort(probs)[::-1][:3]
    top3 = []
    for rank, idx in enumerate(top3_indices, start=1):
        top3.append({
            "rank":       rank,
            "name":       CLASS_DISPLAY_NAMES[CLASS_NAMES[idx]],
            "confidence": round(float(probs[idx] * 100), 2)
        })

    # Terminal output (ASCII-safe for Windows terminals)
    print("\n[SoilClassifier] Prediction Results:")
    print("  Top-1: " + cnn_soil + " (" + str(round(confidence, 2)) + "%)")
    for t in top3:
        print("  " + str(t["rank"]) + ". " + t["name"] +
              " : " + str(t["confidence"]) + "%")

    return cnn_soil, mapped_soil, confidence, top3