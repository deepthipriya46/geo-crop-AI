"""
GeoCropAI -- Soil Classification Training Script v3.0
train_soil_model.py
=======================================================

Key design decisions for fixing Alluvial-only bias:
  1. Oversampling of minority classes (Yellow, Red, Alluvial) so every
     class is seen equally per epoch -- better than amplified class weights.
  2. Standard balanced class_weight as a secondary guard.
  3. Moderate augmentation including RandomHue and RandomSaturation
     to improve color robustness across soil photography conditions.
  4. SparseCategoricalCrossentropy (no label smoothing) -- small datasets
     suffer from underfitting with label smoothing.
  5. Dropout 0.35/0.20 (moderate, not excessive).
  6. Two-phase MobileNetV2 fine-tuning.

Usage:
    python train_soil_model.py

Output:
    soil_model.keras
    results/class_names.txt
    results/cnn_accuracy.png
    results/cnn_loss.png
    results/confusion_matrix_cnn.png
    results/evaluation_report.txt
    results/soil_centroids.npy

Author  : GeoCropAI Research Project
Version : 3.0 (Oversampling + Balanced Weights)
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from analyze_dataset import analyze_dataset, write_report


# ==============================================================
# REPRODUCIBILITY
# ==============================================================

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==============================================================
# CONFIGURATION
# ==============================================================

DATASET_PATH  = "soil_dataset"
MODEL_SAVE    = "soil_model.keras"
RESULTS_DIR   = "results"
BEST_CKPT     = os.path.join(RESULTS_DIR, "best_model.keras")

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
VAL_SPLIT     = 0.2

# Phase 1: frozen backbone, train head
PHASE1_EPOCHS = 12
PHASE1_LR     = 1e-3

# Phase 2: fine-tune last N backbone layers
PHASE2_EPOCHS = 35
PHASE2_LR     = 1e-5
UNFREEZE_LAST = 40

CONFIDENCE_THRESHOLD = 75.0
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================
# STEP 0 -- DATASET QUALITY ANALYSIS
# ==============================================================

print("\n" + "="*60)
print("  STEP 0 -- Dataset Quality Analysis")
print("="*60)

report = analyze_dataset(DATASET_PATH)
write_report(report)

CLASS_NAMES = report["class_names"]   # sorted alphabetically
NUM_CLASSES = len(CLASS_NAMES)

print("\nClass names (training order):")
for i, name in enumerate(CLASS_NAMES):
    print("  [" + str(i) + "] " + name)

# Save class names to file for inference alignment
class_names_file = os.path.join(RESULTS_DIR, "class_names.txt")
with open(class_names_file, "w", encoding="utf-8") as fh:
    for name in CLASS_NAMES:
        fh.write(name + "\n")
print("\nClass names saved to: " + class_names_file)


# ==============================================================
# STEP 1 -- LOAD DATASETS
# ==============================================================

print("\n" + "="*60)
print("  STEP 1 -- Loading Dataset")
print("="*60)

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

detected_classes = train_ds_raw.class_names
print("\nDataset class_names: " + str(detected_classes))

# CRITICAL CHECK: training class order must match CLASS_NAMES
assert detected_classes == CLASS_NAMES, (
    "MISMATCH: detected_classes != CLASS_NAMES\n"
    "Expected: " + str(CLASS_NAMES) + "\n"
    "Got     : " + str(detected_classes)
)
print("Class name order verified OK.")


# ==============================================================
# STEP 2 -- CLASS WEIGHTS + OVERSAMPLING
# ==============================================================

print("\n" + "="*60)
print("  STEP 2 -- Computing Class Weights & Oversampling")
print("="*60)

# Collect all integer labels from the raw training set
all_labels = np.concatenate([
    labels.numpy() for _, labels in train_ds_raw
])

# Standard balanced class weights (no amplification)
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=all_labels
)
class_weight_dict = {
    i: float(w) for i, w in enumerate(class_weights_array)
}

print("\nClass weights (balanced):")
for i, name in enumerate(CLASS_NAMES):
    count = int(np.sum(all_labels == i))
    print("  [" + str(i) + "] " + name.ljust(20) +
          " count=" + str(count).rjust(4) +
          "  weight=" + str(round(class_weight_dict[i], 3)))

# ---- OVERSAMPLING ----
# Load each sample individually and oversample minority classes
# so every class has max_class_count samples per epoch.
print("\nBuilding oversampled training dataset...")
per_imgs = {i: [] for i in range(NUM_CLASSES)}
per_lbls = {i: [] for i in range(NUM_CLASSES)}
for img, lbl in train_ds_raw.unbatch():
    i = int(lbl.numpy())
    per_imgs[i].append(img.numpy())
    per_lbls[i].append(i)

max_cnt = max(len(per_imgs[i]) for i in range(NUM_CLASSES))
print("Oversampling target per class: " + str(max_cnt))
bal_imgs, bal_lbls = [], []
for i in range(NUM_CLASSES):
    imgs = per_imgs[i]
    cnt  = len(imgs)
    reps = (max_cnt // cnt) + 1
    bal_imgs.extend((imgs * reps)[:max_cnt])
    bal_lbls.extend((per_lbls[i] * reps)[:max_cnt])

combined = list(zip(bal_imgs, bal_lbls))
random.shuffle(combined)
bal_imgs, bal_lbls = zip(*combined)
bal_imgs = np.array(bal_imgs, dtype=np.float32)
bal_lbls = np.array(bal_lbls, dtype=np.int32)
print("Total oversampled training samples: " + str(len(bal_imgs)))


# ==============================================================
# STEP 3 -- PREPROCESSING PIPELINE
# ==============================================================
#
# MobileNetV2 was trained with preprocess_input() which scales
# pixels to [-1, 1]. Using Rescaling(1/255) gives a [0,1] range
# that does NOT match MobileNetV2's expected distribution.
# This mismatch was the primary cause of poor accuracy.
#
# The correct pipeline:
#   raw pixels [0, 255]
#   -> preprocess_input()  -> [-1, 1]
#   -> MobileNetV2 backbone
# ==============================================================

def preprocess(image, label):
    """
    Casts raw uint8 pixels to float32, keeping them in the [0, 255] range
    so data augmentation layers (like RandomContrast) operate correctly.
    Converts integer label to one-hot so label smoothing works correctly
    with CategoricalCrossentropy.
    """
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((bal_imgs, bal_lbls))
train_ds = train_ds.shuffle(len(bal_imgs), seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)


# ==============================================================
# STEP 4 -- DATA AUGMENTATION LAYER
# ==============================================================

augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.12),
    tf.keras.layers.RandomContrast(0.15),
    tf.keras.layers.RandomBrightness(0.12),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
    tf.keras.layers.RandomSaturation(factor=(0.7, 1.3)),
    tf.keras.layers.RandomHue(factor=0.04),
], name="data_augmentation")


# ==============================================================
# STEP 5 -- BUILD MODEL
# ==============================================================

print("\n" + "="*60)
print("  STEP 3 -- Building Model")
print("="*60)

# Load MobileNetV2 backbone WITHOUT top classification head.
# Input is already preprocessed (float32, [-1,1]).
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False   # frozen during Phase 1

# Build the full model using Functional API for clarity
inputs  = tf.keras.Input(shape=IMG_SIZE + (3,), name="input_image")
x       = augmentation_layer(inputs)
x       = tf.keras.layers.Rescaling(scale=1.0/127.5, offset=-1.0, name="mobilenet_preprocess")(x)
x       = base_model(x, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
x       = tf.keras.layers.BatchNormalization(name="bn")(x)
x       = tf.keras.layers.Dropout(0.35, name="dropout_1")(x)
x       = tf.keras.layers.Dense(256, activation="relu", name="fc_256")(x)
x       = tf.keras.layers.Dropout(0.20, name="dropout_2")(x)
# Standard softmax output
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax",
                                name="predictions")(x)

model = tf.keras.Model(inputs, outputs, name="GeoCropAI_SoilCNN")

model.summary()


# ==============================================================
# STEP 6 -- CALLBACKS
# ==============================================================

callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_CKPT,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.TerminateOnNaN()
]

callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=4,
        min_lr=1e-9,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_CKPT,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.TerminateOnNaN()
]


# ==============================================================
# STEP 7 -- PHASE 1 TRAINING (frozen backbone)
# ==============================================================

print("\n" + "="*60)
print("  STEP 4 -- Phase 1: Training Head (backbone frozen)")
print("="*60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase1,
    verbose=1
)

print("\nPhase 1 complete.")


# ==============================================================
# STEP 8 -- PHASE 2 FINE-TUNING (unfreeze last N layers)
# ==============================================================

print("\n" + "="*60)
print("  STEP 5 -- Phase 2: Fine-Tuning Last " +
      str(UNFREEZE_LAST) + " Layers")
print("="*60)

# Unfreeze the last UNFREEZE_LAST layers of the backbone
base_model.trainable = True
total_layers         = len(base_model.layers)
freeze_until         = total_layers - UNFREEZE_LAST

for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= freeze_until)

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print("Backbone total layers     : " + str(total_layers))
print("Layers unfrozen           : " + str(trainable_count))
print("Fine-tuning learning rate : " + str(PHASE2_LR))

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase2,
    verbose=1
)

print("\nPhase 2 complete.")


# ==============================================================
# STEP 9 -- SAVE BEST MODEL
# ==============================================================

print("\n" + "="*60)
print("  STEP 6 -- Saving Model")
print("="*60)

# Load the best checkpoint and overwrite soil_model.keras
best_model = tf.keras.models.load_model(BEST_CKPT)
best_model.save(MODEL_SAVE)
print("Best model saved to: " + MODEL_SAVE)


# ==============================================================
# STEP 9.5 -- GENERATE & SAVE DEEP FEATURE CENTROIDS
# ==============================================================

print("\n" + "="*60)
print("  STEP 6.5 -- Extracting & Saving Feature Centroids")
print("="*60)

# Create a sub-model that outputs the feature representations from fc_256
feature_model = tf.keras.Model(
    inputs=best_model.inputs,
    outputs=best_model.get_layer("fc_256").output
)

print("Extracting training features...")
# Load without shuffle, labels as int for centroid grouping
train_ds_for_centroids = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

all_features = []
all_lbl_int  = []
for images, labels in train_ds_for_centroids:
    imgs_f = tf.cast(images, tf.float32)
    feats  = feature_model.predict(imgs_f, verbose=0)
    all_features.append(feats)
    all_lbl_int.extend(labels.numpy())

all_features = np.concatenate(all_features, axis=0)
all_lbl_int  = np.array(all_lbl_int)

centroids = []
for i in range(NUM_CLASSES):
    class_feats = all_features[all_lbl_int == i]
    if len(class_feats) > 0:
        centroid = np.mean(class_feats, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
    else:
        centroids.append(np.zeros(256))

centroids = np.array(centroids)
centroids_path = os.path.join(RESULTS_DIR, "soil_centroids.npy")
np.save(centroids_path, centroids)
print("Saved 256-D feature centroids to: " + centroids_path)




# ==============================================================
# STEP 10 -- COMBINE HISTORY AND PLOT GRAPHS
# ==============================================================

def _combine_histories(h1, h2, key):
    """Merges Phase 1 + Phase 2 history lists for a metric."""
    v1 = h1.history.get(key, [])
    v2 = h2.history.get(key, [])
    return v1 + v2


acc       = _combine_histories(history_phase1, history_phase2, "accuracy")
val_acc   = _combine_histories(history_phase1, history_phase2, "val_accuracy")
loss      = _combine_histories(history_phase1, history_phase2, "loss")
val_loss  = _combine_histories(history_phase1, history_phase2, "val_loss")
epochs_x  = list(range(1, len(acc) + 1))

# Mark the boundary between Phase 1 and Phase 2
phase1_end = len(history_phase1.history.get("accuracy", []))

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "train":  "#2E7D32",
    "val":    "#F57F17",
    "grid":   "#E0E0E0",
    "phase":  "#B0BEC5"
}

# -- Accuracy Graph --
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_x, acc,     color=COLORS["train"], linewidth=2.5,
        label="Training Accuracy",   marker="o", markersize=4)
ax.plot(epochs_x, val_acc, color=COLORS["val"],   linewidth=2.5,
        label="Validation Accuracy", marker="s", markersize=4,
        linestyle="--")
if phase1_end > 0:
    ax.axvline(x=phase1_end + 0.5, color=COLORS["phase"],
               linestyle=":", linewidth=2,
               label="Fine-tuning starts")
ax.set_title("GeoCropAI -- CNN Training & Validation Accuracy",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
plt.tight_layout()
acc_path = os.path.join(RESULTS_DIR, "cnn_accuracy.png")
plt.savefig(acc_path, dpi=150)
plt.close()
print("Saved: " + acc_path)

# -- Loss Graph --
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_x, loss,     color=COLORS["train"], linewidth=2.5,
        label="Training Loss",   marker="o", markersize=4)
ax.plot(epochs_x, val_loss, color=COLORS["val"],   linewidth=2.5,
        label="Validation Loss", marker="s", markersize=4,
        linestyle="--")
if phase1_end > 0:
    ax.axvline(x=phase1_end + 0.5, color=COLORS["phase"],
               linestyle=":", linewidth=2,
               label="Fine-tuning starts")
ax.set_title("GeoCropAI -- CNN Training & Validation Loss",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss",  fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
loss_path = os.path.join(RESULTS_DIR, "cnn_loss.png")
plt.savefig(loss_path, dpi=150)
plt.close()
print("Saved: " + loss_path)


# ==============================================================
# STEP 11 -- EVALUATION
# ==============================================================

print("\n" + "="*60)
print("  STEP 7 -- Model Evaluation")
print("="*60)

y_true, y_pred_probs = [], []

for images, labels in val_ds:
    preds = best_model.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_pred_probs.extend(preds.tolist())

y_pred = [np.argmax(p) for p in y_pred_probs]

# -- Overall metrics --
overall_acc   = accuracy_score(y_true, y_pred)
macro_f1      = f1_score(y_true, y_pred, average="macro")
weighted_f1   = f1_score(y_true, y_pred, average="weighted")

print("\nOverall Accuracy : " + str(round(overall_acc * 100, 2)) + "%")
print("Macro F1 Score   : " + str(round(macro_f1, 4)))
print("Weighted F1 Score: " + str(round(weighted_f1, 4)))

# -- Per-class classification report --
report_str = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, digits=4
)
print("\nClassification Report:\n")
print(report_str)

# Save report to file
eval_report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
with open(eval_report_path, "w", encoding="utf-8") as fh:
    fh.write("GeoCropAI -- Model Evaluation Report\n")
    fh.write("="*60 + "\n\n")
    fh.write("Overall Accuracy  : " + str(round(overall_acc * 100, 2)) + "%\n")
    fh.write("Macro F1 Score    : " + str(round(macro_f1, 4)) + "\n")
    fh.write("Weighted F1 Score : " + str(round(weighted_f1, 4)) + "\n\n")
    fh.write("Classification Report:\n")
    fh.write(report_str)
print("Evaluation report saved to: " + eval_report_path)


# -- Publication-quality Confusion Matrix --
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalized

# Short class labels for readability
short_names = [n.replace("_Soil", "").replace("_", " ") for n in CLASS_NAMES]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Raw counts
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=short_names, yticklabels=short_names,
    ax=axes[0], linewidths=0.5, linecolor="white",
    annot_kws={"size": 11, "weight": "bold"}
)
axes[0].set_title("Confusion Matrix (counts)",
                  fontsize=13, fontweight="bold", pad=10)
axes[0].set_xlabel("Predicted Class", fontsize=11)
axes[0].set_ylabel("True Class", fontsize=11)
axes[0].tick_params(axis="x", rotation=45)

# Row-normalized (shows class-level recall)
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Greens",
    xticklabels=short_names, yticklabels=short_names,
    ax=axes[1], linewidths=0.5, linecolor="white",
    vmin=0, vmax=1,
    annot_kws={"size": 11}
)
axes[1].set_title("Confusion Matrix (row-normalized recall)",
                  fontsize=13, fontweight="bold", pad=10)
axes[1].set_xlabel("Predicted Class", fontsize=11)
axes[1].set_ylabel("True Class", fontsize=11)
axes[1].tick_params(axis="x", rotation=45)

plt.suptitle("GeoCropAI -- MobileNetV2 Soil Classifier",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_cnn.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print("Saved: " + cm_path)


# ==============================================================
# STEP 12 -- FINAL SUMMARY
# ==============================================================

print("\n" + "="*60)
print("  TRAINING COMPLETE")
print("="*60)
print("  Model             : " + MODEL_SAVE)
print("  Overall Accuracy  : " + str(round(overall_acc * 100, 2)) + "%")
print("  Macro F1          : " + str(round(macro_f1, 4)))
print("  Weighted F1       : " + str(round(weighted_f1, 4)))
print("  Accuracy Graph    : " + acc_path)
print("  Loss Graph        : " + loss_path)
print("  Confusion Matrix  : " + cm_path)
print("  Eval Report       : " + eval_report_path)
print("  Dataset Report    : " + REPORT_FILE)
print("  Class Names File  : " + class_names_file)
print("="*60)
print("\nAll output saved to: results/")