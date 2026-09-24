"""
GeoCropAI -- Image Validation Module v5.0
soil_model/image_validator.py
==========================================

Two-Stage Validation Pipeline:

STAGE 1 -- Non-Soil Guard (Hard Rules, Cannot be overridden):
  Checks that CANNOT be bypassed by CNN confidence.
  If ANY of these fire, the image is REJECTED immediately.

  G1. Grout/Tile Line Detection:
        Detects structural straight lines using Hough Transform.
        Floors, tiles, and roads contain repeating parallel lines.
        Threshold: > 6 structural lines detected.

  G2. ImageNet Hard Rejection:
        If the ImageNet model identifies a non-soil class with
        probability > 40%, reject immediately.
        This catches concrete, wood, metal, walls, flooring.

  G3. Gray/Neutral Non-Soil Color Guard:
        Concrete, cement, marble, tiles are gray/neutral (low saturation).
        Real soil always has some earthy hue (brownish, reddish, yellowish).
        If mean saturation is very low (< 25) AND mean brightness
        is mid-range (not very dark = not black soil), reject.

STAGE 2 -- Weighted Scoring (Only for images that pass Stage 1):
  Checks that SUPPORT the final decision.

  S1. Soil Color Analysis  (Weight 40%)
  S2. Feature Similarity   (Weight 35%)
  S3. CNN Confidence       (Weight 25%)

  Final acceptance: Score >= 60%.

Log file: logs/validation_log.csv

Author  : GeoCropAI Research Project
Version : 5.0 (Two-Stage Guard + Weighted Scoring)
"""

import os
import cv2
import csv
import numpy as np
from PIL import Image
from datetime import datetime
from dataclasses import dataclass, field

import tensorflow as tf


# ==============================================================
# CONFIGURATION
# ==============================================================

CONFIDENCE_THRESHOLD    = 75.0   # CNN top-1 confidence to show as PASS
GROUT_LINE_THRESHOLD    = 6      # Hough lines above this = structural floor
IMAGENET_REJECT_PROB    = 0.40   # ImageNet non-soil probability = hard reject
SATURATION_GRAY_THRESH  = 25     # HSV saturation below this = gray/concrete
WEIGHTED_PASS_SCORE     = 60     # Minimum weighted score to pass Stage 2

LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "validation_log.csv")

LOG_HEADERS = [
    "timestamp", "filename", "validation_status", "overall_score_pct",
    "blur_score", "earth_ratio_pct", "hough_lines", "mean_saturation",
    "imagenet_prob", "similarity_score", "cnn_confidence_pct",
    "predicted_soil", "failure_reason"
]

# ==============================================================
# IMAGENET NON-SOIL KEYWORDS
# ==============================================================

NON_SOIL_KEYWORDS = [
    # Floors / tiles / indoor surfaces
    "tile", "slate", "paving", "pavement", "sidewalk", "cobblestone",
    "floor", "carpet", "rug", "doormat", "grout", "linoleum",
    # Walls / structures
    "wall", "brick", "mason", "shingle", "roof", "fence", "facade", "concrete",
    # Buildings / indoor
    "building", "house", "barn", "monastery", "castle", "church", "mosque",
    "room", "studio", "restaurant", "office", "window", "door",
    # Materials / furniture
    "wood", "table", "chair", "bench", "lumber", "metal", "steel", "iron",
    "plastic", "glass", "paper", "book",
    # People / clothing
    "person", "man", "woman", "boy", "girl", "hand", "face", "skin",
    "jersey", "suit", "coat", "shirt", "pants", "dress", "shoe", "boot",
    # Animals
    "dog", "cat", "bird", "fish", "insect", "snake", "lizard", "spider",
    "frog", "turtle", "crocodile",
    # Plants (not soil)
    "flower", "tree", "leaf", "leaves", "grass", "shrub", "bush",
    "daisy", "rose", "tulip", "orchid", "sunflower",
    # Sky / water
    "water", "sea", "lake", "river", "sky", "cloud", "ocean", "ice", "snow"
]


# ==============================================================
# LOAD IMAGENET MODEL (once at startup)
# ==============================================================

print("[ImageValidator] Loading ImageNet MobileNetV2...")
_imagenet_model = tf.keras.applications.MobileNetV2(
    weights="imagenet", include_top=True, input_shape=(224, 224, 3)
)
_imagenet_model.trainable = False
print("[ImageValidator] ImageNet model loaded.")

from soil_model.soil_classifier import predict_soil


# ==============================================================
# RESULT DATACLASS
# ==============================================================

@dataclass
class ValidationResult:
    is_valid:      bool  = False
    confidence:    float = 0.0
    error_message: str   = ""
    report:        dict  = field(default_factory=dict)
    cnn_soil:      str   = ""
    mapped_soil:   str   = ""
    stage:         str   = "unknown"


# ==============================================================
# INTERNAL DETECTION FUNCTIONS
# ==============================================================

def _check_file_format(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png"}

def _check_image_quality(image_path):
    try:
        with Image.open(image_path) as img:
            w, h = img.size
        return (w >= 200) and (h >= 200)
    except Exception:
        return False

def _detect_blur(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _detect_hough_lines(image_path):
    """
    Detects structural straight lines (grout joints, tile edges, road markings).
    Uses a LARGE blur kernel to suppress fine soil cracks/pebbles, keeping
    only strong straight-edge structures typical of floors and walls.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0
    img_r = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    # Large kernel (21x21) smooths rocky soil texture; floor edges survive
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=120, minLineLength=180, maxLineGap=6
    )
    return len(lines) if lines is not None else 0

def _detect_color_stats(image_path):
    """
    Returns:
      earth_ratio   -- fraction of pixels in earthy HSV ranges
      mean_sat      -- mean saturation of the whole image (0-255)
      mean_val      -- mean brightness (value channel) of whole image
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, 0.0, 0.0
    img_r = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)

    total_px = 224 * 224
    mean_sat = float(np.mean(hsv[:, :, 1]))
    mean_val = float(np.mean(hsv[:, :, 2]))

    # Earthy soil color ranges:
    # Brown / Alluvial
    m_brown  = cv2.inRange(hsv, np.array([6,  50, 20]),  np.array([22, 255, 160]))
    # Red soil
    m_red    = cv2.inRange(hsv, np.array([0,  70, 25]),  np.array([10, 255, 200]))
    # Sandy / Yellow-orange
    m_sandy  = cv2.inRange(hsv, np.array([16, 30, 120]), np.array([36, 200, 240]))
    # Loamy mid-tones
    m_loamy  = cv2.inRange(hsv, np.array([8,  25, 60]),  np.array([30, 160, 195]))
    # Dark clay / Black soil
    m_clay   = cv2.inRange(hsv, np.array([5,  35, 8]),   np.array([25, 180, 80]))
    # Laterite (reddish-orange)
    m_lat    = cv2.inRange(hsv, np.array([8,  80, 40]),  np.array([20, 255, 210]))

    combined = m_brown | m_red | m_sandy | m_loamy | m_clay | m_lat
    earth_count = int(cv2.countNonZero(combined))
    earth_ratio = earth_count / total_px

    return earth_ratio, mean_sat, mean_val

def _check_imagenet(image_path):
    """Returns (max_non_soil_prob, matched_keyword)."""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    preds = _imagenet_model.predict(arr, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

    max_prob = 0.0
    matched_kw = ""
    for _, label, prob in decoded:
        label_lower = label.lower().replace("_", " ")
        for kw in NON_SOIL_KEYWORDS:
            if kw in label_lower and prob > max_prob:
                max_prob = prob
                matched_kw = kw
    return float(max_prob), matched_kw

def _check_feature_similarity(image_path):
    """Cosine similarity of image features vs. soil class centroids."""
    centroids_path = "results/soil_centroids.npy"
    if not os.path.exists(centroids_path):
        return 0.75  # neutral if no centroids yet

    centroids = np.load(centroids_path)
    D = centroids.shape[1]

    from soil_model.soil_classifier import _model, _preprocess_image
    layer_name = "fc_256" if D == 256 else "dense"
    try:
        feat_model = tf.keras.Model(
            inputs=_model.inputs,
            outputs=_model.get_layer(layer_name).output
        )
        arr = _preprocess_image(image_path)
        feat = feat_model.predict(arr, verbose=0)[0]
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return float(np.max(np.dot(centroids, feat)))
    except Exception:
        return 0.75


# ==============================================================
# LOG WRITER
# ==============================================================

def _write_log(filename, status, score, blur, earth, lines, sat,
               imgnet_prob, sim, cnn_conf, soil, reason):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_HEADERS)
            if not exists:
                writer.writeheader()
            writer.writerow({
                "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename":          filename,
                "validation_status": status,
                "overall_score_pct": round(score, 1),
                "blur_score":        round(blur, 2),
                "earth_ratio_pct":   round(earth * 100, 1),
                "hough_lines":       lines,
                "mean_saturation":   round(sat, 1),
                "imagenet_prob":     round(imgnet_prob, 4),
                "similarity_score":  round(sim, 4),
                "cnn_confidence_pct": round(cnn_conf, 2),
                "predicted_soil":    soil,
                "failure_reason":    reason
            })
    except Exception as e:
        print("[ImageValidator] Log error: " + str(e))


# ==============================================================
# PUBLIC INTERFACE
# ==============================================================

def validate_image(image_path):
    filename = os.path.basename(image_path)

    # --- Build report skeleton ---
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            file_size_kb = round(os.path.getsize(image_path) / 1024, 1)
            resolution_str = str(w) + " x " + str(h) + " px"
    except Exception:
        w = h = 0
        file_size_kb = 0.0
        resolution_str = "Unknown"

    report = {
        "filename":          filename,
        "file_size_kb":      file_size_kb,
        "resolution":        resolution_str,
        "blur_score":        0.0,
        "blur_status":       "",
        "earth_ratio":       0.0,
        "earth_status":      "",
        "hough_lines":       0,
        "hough_status":      "",
        "mean_sat":          0.0,
        "similarity_score":  0.0,
        "similarity_status": "",
        "confidence":        0.0,
        "soil_type":         "",
        "top3":              [],
        # Weighted contributions
        "color_contrib":      0,
        "similarity_contrib": 0,
        "confidence_contrib": 0,
        "overall_score":      0,
        # Row statuses
        "status_format":     "PASS",
        "status_quality":    "PASS",
        "status_blur":       "PENDING",
        "status_color":      "PENDING",
        "status_lines":      "PENDING",
        "status_imagenet":   "PENDING",
        "status_similarity": "PENDING",
        "status_confidence": "PENDING",
        "status_soil":       "PENDING",
    }

    # =========================================================
    # PRE-FLIGHT: FORMAT + QUALITY
    # =========================================================
    if not _check_file_format(image_path):
        return ValidationResult(
            is_valid=False, stage="format",
            error_message="Unsupported format. Please upload a JPG or PNG soil image.",
            report=report
        )

    if not _check_image_quality(image_path):
        return ValidationResult(
            is_valid=False, stage="quality",
            error_message="Image resolution too low. Please upload a clearer, higher-resolution image.",
            report=report
        )

    # =========================================================
    # BLUR CHECK (gate only — not in weighted score)
    # =========================================================
    blur_score = _detect_blur(image_path)
    report["blur_score"] = round(blur_score, 2)
    if blur_score >= 80.0:
        report["status_blur"] = "PASS"
        report["blur_status"] = "Clear (score: " + str(round(blur_score, 1)) + ")"
    elif blur_score >= 40.0:
        report["status_blur"] = "WARNING"
        report["blur_status"] = "Slightly blurry (score: " + str(round(blur_score, 1)) + ")"
    else:
        report["status_blur"] = "FAIL"
        _write_log(filename, "REJECTED", 0, blur_score, 0, 0, 0, 0, 0, 0, "", "Too blurry")
        return ValidationResult(
            is_valid=False, stage="blur",
            error_message="Image is too blurry. Please upload a clear, focused soil photo.",
            report=report
        )

    # =========================================================
    # STAGE 1 -- NON-SOIL HARD GUARDS
    # These checks CANNOT be overridden by CNN confidence.
    # Any single guard firing = REJECTED.
    # =========================================================

    # G1: Structural Line Detection
    lines_count = _detect_hough_lines(image_path)
    report["hough_lines"] = lines_count
    if lines_count > GROUT_LINE_THRESHOLD:
        report["status_lines"] = "FAIL"
        report["hough_status"] = (
            "Structural floor/tile lines detected (" + str(lines_count) + " lines)"
        )
    elif lines_count > 3:
        report["status_lines"] = "WARNING"
        report["hough_status"] = "Some straight lines (" + str(lines_count) + " lines)"
    else:
        report["status_lines"] = "PASS"
        report["hough_status"] = "Organic texture — no structural lines"

    # G2: ImageNet Non-Soil Detection
    non_soil_prob, matched_kw = _check_imagenet(image_path)
    if non_soil_prob > 0:
        print("[ImageValidator] ImageNet: '" + matched_kw + "' matched (" +
              str(round(non_soil_prob * 100, 1)) + "%)")
    if non_soil_prob >= IMAGENET_REJECT_PROB:
        report["status_imagenet"] = "FAIL"
    elif non_soil_prob >= 0.15:
        report["status_imagenet"] = "WARNING"
    else:
        report["status_imagenet"] = "PASS"

    # G3: Gray/Neutral Color Guard (concrete, cement, marble)
    earth_ratio, mean_sat, mean_val = _detect_color_stats(image_path)
    report["earth_ratio"] = round(earth_ratio * 100, 1)
    report["mean_sat"] = round(mean_sat, 1)

    # Concrete/tile/marble = low saturation + mid-range brightness
    # Black soil = very dark (mean_val < 60) — we must NOT reject it here
    is_gray_non_soil = (mean_sat < SATURATION_GRAY_THRESH) and (mean_val > 60)
    if is_gray_non_soil:
        report["status_color"] = "FAIL"
        report["earth_status"] = (
            "Gray/neutral surface detected (sat=" + str(round(mean_sat, 1)) + 
            ") — not soil"
        )
    elif earth_ratio >= 0.18:
        report["status_color"] = "PASS"
        report["earth_status"] = "Earth tones detected (" + str(round(earth_ratio * 100, 1)) + "%)"
    elif earth_ratio >= 0.08:
        report["status_color"] = "WARNING"
        report["earth_status"] = "Low earth tones (" + str(round(earth_ratio * 100, 1)) + "%)"
    else:
        report["status_color"] = "FAIL"
        report["earth_status"] = "No significant earth tones (" + str(round(earth_ratio * 100, 1)) + "%)"

    # --- Evaluate Stage 1 ---
    stage1_fails = []
    if lines_count > GROUT_LINE_THRESHOLD:
        stage1_fails.append("grout lines=" + str(lines_count))
    if non_soil_prob >= IMAGENET_REJECT_PROB:
        stage1_fails.append("ImageNet non-soil=" + matched_kw + "(" + str(round(non_soil_prob*100,1)) + "%)")
    if is_gray_non_soil:
        stage1_fails.append("gray surface (sat=" + str(round(mean_sat,1)) + ")")

    # Count how many Stage 1 signals are firing
    stage1_warnings = 0
    if lines_count > 3:
        stage1_warnings += 1
    if non_soil_prob >= 0.15:
        stage1_warnings += 1
    if report["status_color"] in ("FAIL", "WARNING"):
        stage1_warnings += 1

    if stage1_fails:
        # Hard reject — at least one definitive guard fired
        reason = "Stage 1 guard: " + "; ".join(stage1_fails)
        print("[ImageValidator] REJECTED by Stage 1: " + reason)
        _write_log(filename, "REJECTED", 0, blur_score, earth_ratio, lines_count,
                   mean_sat, non_soil_prob, 0.0, 0.0, "", reason)
        return ValidationResult(
            is_valid=False, stage="guard",
            error_message="Uploaded image is not recognized as a soil image. Please upload a clear soil image.",
            report=report
        )

    # If 2+ warnings fire in Stage 1, also reject (soft multi-signal rejection)
    if stage1_warnings >= 2:
        reason = "Stage 1 multi-warning: color=" + report["status_color"] + \
                 " lines=" + str(lines_count) + \
                 " imagenet=" + str(round(non_soil_prob*100,1)) + "%"
        print("[ImageValidator] REJECTED by Stage 1 multi-warning: " + reason)
        _write_log(filename, "REJECTED", 0, blur_score, earth_ratio, lines_count,
                   mean_sat, non_soil_prob, 0.0, 0.0, "", reason)
        return ValidationResult(
            is_valid=False, stage="guard",
            error_message="Uploaded image is not recognized as a soil image. Please upload a clear soil image.",
            report=report
        )

    # =========================================================
    # STAGE 2 -- WEIGHTED SCORING
    # Only reaches here if Stage 1 passes.
    # =========================================================

    # S1: Soil Color  (Weight 40%)
    if report["status_color"] == "PASS":
        color_val = 100
    elif report["status_color"] == "WARNING":
        color_val = int((earth_ratio / 0.18) * 100)
    else:
        color_val = 0
    report["color_contrib"] = int(color_val * 0.40)

    # S2: Feature Similarity  (Weight 35%)
    similarity = _check_feature_similarity(image_path)
    report["similarity_score"] = round(similarity, 4)
    if similarity >= 0.80:
        sim_val = 100
        report["status_similarity"] = "PASS"
        report["similarity_status"] = "Soil features matched (" + str(round(similarity, 3)) + ")"
    elif similarity >= 0.65:
        sim_val = int((similarity / 0.80) * 100)
        report["status_similarity"] = "WARNING"
        report["similarity_status"] = "Partial match (" + str(round(similarity, 3)) + ")"
    else:
        sim_val = int((similarity / 0.80) * 100)
        report["status_similarity"] = "FAIL"
        report["similarity_status"] = "Low similarity (" + str(round(similarity, 3)) + ")"
    report["similarity_contrib"] = int(sim_val * 0.35)

    # S3: CNN Soil Classification  (Weight 25%)
    try:
        cnn_soil, mapped_soil, confidence, top3 = predict_soil(image_path)
    except Exception as e:
        _write_log(filename, "ERROR", 0, blur_score, earth_ratio, lines_count,
                   mean_sat, non_soil_prob, similarity, 0.0, "", "Inference error: " + str(e))
        return ValidationResult(
            is_valid=False, stage="inference",
            error_message="Classification model error. Please try again.",
            report=report
        )

    report["confidence"] = round(confidence, 2)
    report["soil_type"]  = cnn_soil
    report["top3"]       = top3

    if confidence >= CONFIDENCE_THRESHOLD:
        conf_val = 100
        report["status_confidence"] = "PASS"
        report["status_soil"]       = "PASS"
    elif confidence >= 55.0:
        conf_val = int((confidence / CONFIDENCE_THRESHOLD) * 100)
        report["status_confidence"] = "WARNING"
        report["status_soil"]       = "WARNING"
    else:
        conf_val = int((confidence / CONFIDENCE_THRESHOLD) * 100)
        report["status_confidence"] = "FAIL"
        report["status_soil"]       = "FAIL"
    report["confidence_contrib"] = int(conf_val * 0.25)

    # Final weighted score
    overall_score = (
        report["color_contrib"] +
        report["similarity_contrib"] +
        report["confidence_contrib"]
    )
    report["overall_score"] = overall_score

    print("[ImageValidator] Stage 2 Score: " + str(overall_score) +
          "% (threshold=" + str(WEIGHTED_PASS_SCORE) + "%)")

    is_valid = (overall_score >= WEIGHTED_PASS_SCORE)

    if not is_valid:
        reason = "Stage 2 score " + str(overall_score) + "% < " + str(WEIGHTED_PASS_SCORE) + "%"
        _write_log(filename, "REJECTED", overall_score, blur_score, earth_ratio,
                   lines_count, mean_sat, non_soil_prob, similarity, confidence, cnn_soil, reason)
        return ValidationResult(
            is_valid=False, stage="weighted",
            error_message="Uploaded image is not recognized as a soil image. Please upload a clear soil image.",
            report=report
        )

    _write_log(filename, "VALID_SOIL", overall_score, blur_score, earth_ratio,
               lines_count, mean_sat, non_soil_prob, similarity, confidence, cnn_soil, "")
    return ValidationResult(
        is_valid=True,
        confidence=confidence,
        report=report,
        cnn_soil=cnn_soil,
        mapped_soil=mapped_soil,
        stage="complete"
    )
