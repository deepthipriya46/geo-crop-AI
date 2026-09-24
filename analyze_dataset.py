"""
GeoCropAI -- Complete Dataset Quality Analysis
analyze_dataset.py
==================================================

Performs a full quality audit of the soil image dataset
before training begins. Generates results/dataset_report.txt
and prints a summary to the terminal.

Checks performed:
  - Total images and images per class
  - Corrupted / unreadable images
  - Class imbalance ratio
  - Image size distribution
  - Duplicate file detection (MD5 hash)
  - Suggested improvements

Run standalone:
    python analyze_dataset.py
"""

import os
import hashlib
import numpy as np
from datetime import datetime
from collections import defaultdict

# PIL is used to attempt opening each image for corruption detection
from PIL import Image, UnidentifiedImageError


# ==============================================================
# CONFIGURATION
# ==============================================================

DATASET_PATH = "soil_dataset"
RESULTS_DIR  = "results"
REPORT_FILE  = os.path.join(RESULTS_DIR, "dataset_report.txt")


# ==============================================================
# HELPER -- MD5 hash for duplicate detection
# ==============================================================

def _file_md5(filepath):
    """Computes the MD5 hash of a file's raw bytes."""
    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# ==============================================================
# HELPER -- Try to open and get image size
# ==============================================================

def _read_image_info(filepath):
    """
    Attempts to open the image and return (width, height).
    Returns None if the image is corrupted.
    """
    try:
        with Image.open(filepath) as img:
            img.verify()        # detects truncated / corrupted files
        # Re-open after verify (verify closes the file)
        with Image.open(filepath) as img:
            return img.size     # (width, height)
    except (UnidentifiedImageError, Exception):
        return None


# ==============================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================

def analyze_dataset(dataset_path=DATASET_PATH):
    """
    Scans the dataset directory and returns a full quality report.

    Returns
    -------
    dict with keys:
        class_names, class_counts, total_images,
        corrupted, duplicates, size_stats, suggestions
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Discover class folders (sorted = reproducible order)
    class_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    class_counts    = {}
    corrupted       = {}
    all_widths      = []
    all_heights     = []
    hash_map        = defaultdict(list)   # md5 -> list of paths

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    print("\n" + "="*60)
    print("  GeoCropAI -- Dataset Quality Analysis")
    print("="*60)
    print("  Dataset  : " + dataset_path)
    print("  Started  : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    for cls in class_dirs:
        cls_path      = os.path.join(dataset_path, cls)
        all_files     = os.listdir(cls_path)
        img_files     = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        ]

        valid_count   = 0
        corrupt_list  = []

        for fname in img_files:
            fpath  = os.path.join(cls_path, fname)
            info   = _read_image_info(fpath)

            if info is None:
                corrupt_list.append(fname)
            else:
                valid_count += 1
                all_widths.append(info[0])
                all_heights.append(info[1])

                # Duplicate detection
                md5 = _file_md5(fpath)
                if md5:
                    hash_map[md5].append(fpath)

        class_counts[cls] = valid_count
        corrupted[cls]    = corrupt_list

        status = " [" + str(valid_count) + " valid"
        if corrupt_list:
            status += ", " + str(len(corrupt_list)) + " CORRUPTED"
        status += "]"
        print("  " + cls.ljust(20) + status)

    # Duplicate groups
    duplicates = {
        md5: paths for md5, paths in hash_map.items()
        if len(paths) > 1
    }
    dup_count = sum(len(v) - 1 for v in duplicates.values())

    total_valid    = sum(class_counts.values())
    total_corrupt  = sum(len(v) for v in corrupted.values())
    max_count      = max(class_counts.values()) if class_counts else 1
    min_count      = min(class_counts.values()) if class_counts else 0
    imbalance_ratio = round(max_count / max(min_count, 1), 2)

    # Image size statistics
    if all_widths:
        size_stats = {
            "min_w": int(min(all_widths)),  "max_w": int(max(all_widths)),
            "min_h": int(min(all_heights)), "max_h": int(max(all_heights)),
            "mean_w": int(np.mean(all_widths)),
            "mean_h": int(np.mean(all_heights)),
        }
    else:
        size_stats = {}

    # Suggestions
    suggestions = []
    if imbalance_ratio > 2.0:
        suggestions.append(
            "Class imbalance detected (ratio=" + str(imbalance_ratio) +
            "). Apply class_weight during training."
        )
    if min_count < 100:
        minority = [k for k, v in class_counts.items() if v < 100]
        suggestions.append(
            "Classes with < 100 images: " + ", ".join(minority) +
            ". Consider collecting more data or augmenting heavily."
        )
    if total_corrupt > 0:
        suggestions.append(
            str(total_corrupt) + " corrupted image(s) detected. "
            "Remove them before training."
        )
    if dup_count > 0:
        suggestions.append(
            str(dup_count) + " duplicate image(s) detected. "
            "Remove duplicates to avoid data leakage."
        )

    return {
        "class_names":     class_dirs,
        "class_counts":    class_counts,
        "total_images":    total_valid,
        "total_corrupted": total_corrupt,
        "corrupted":       corrupted,
        "duplicates":      dup_count,
        "size_stats":      size_stats,
        "imbalance_ratio": imbalance_ratio,
        "suggestions":     suggestions
    }


# ==============================================================
# REPORT WRITER
# ==============================================================

def write_report(report):
    """
    Saves the full dataset quality report to
    results/dataset_report.txt.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("  GeoCropAI -- Dataset Quality Report")
    lines.append("  Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("=" * 60)
    lines.append("")
    lines.append("DATASET OVERVIEW")
    lines.append("-" * 40)
    lines.append("Total Valid Images   : " + str(report["total_images"]))
    lines.append("Total Corrupted      : " + str(report["total_corrupted"]))
    lines.append("Duplicate Images     : " + str(report["duplicates"]))
    lines.append("Number of Classes    : " + str(len(report["class_names"])))
    lines.append("Imbalance Ratio      : " + str(report["imbalance_ratio"]) +
                 "  (max_class / min_class)")
    lines.append("")
    lines.append("IMAGES PER CLASS")
    lines.append("-" * 40)
    total = report["total_images"]
    for cls, count in report["class_counts"].items():
        pct  = round(count / max(total, 1) * 100, 1)
        bar  = "#" * int(pct / 2)
        lines.append(
            cls.ljust(20) + str(count).rjust(5) +
            "  (" + str(pct) + "%)  " + bar
        )
    lines.append("")
    lines.append("CORRUPTED IMAGES")
    lines.append("-" * 40)
    any_corrupt = False
    for cls, bad in report["corrupted"].items():
        if bad:
            any_corrupt = True
            for f in bad:
                lines.append("  " + cls + "/" + f)
    if not any_corrupt:
        lines.append("  None detected.")
    lines.append("")
    lines.append("IMAGE SIZE DISTRIBUTION")
    lines.append("-" * 40)
    ss = report.get("size_stats", {})
    if ss:
        lines.append("  Width  : min=" + str(ss["min_w"]) +
                     "  max=" + str(ss["max_w"]) +
                     "  mean=" + str(ss["mean_w"]))
        lines.append("  Height : min=" + str(ss["min_h"]) +
                     "  max=" + str(ss["max_h"]) +
                     "  mean=" + str(ss["mean_h"]))
    else:
        lines.append("  No valid images to analyze.")
    lines.append("")
    lines.append("SUGGESTED IMPROVEMENTS")
    lines.append("-" * 40)
    if report["suggestions"]:
        for s in report["suggestions"]:
            lines.append("  * " + s)
    else:
        lines.append("  Dataset looks healthy.")
    lines.append("")
    lines.append("CLASS NAMES (training order -- used in soil_classifier.py)")
    lines.append("-" * 40)
    for i, name in enumerate(report["class_names"]):
        lines.append("  [" + str(i) + "] " + name)
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    print("\n" + report_text)
    print("Report saved to: " + REPORT_FILE)

    return report_text


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    report = analyze_dataset(DATASET_PATH)
    write_report(report)
