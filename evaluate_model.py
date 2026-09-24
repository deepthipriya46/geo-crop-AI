import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load trained model
model = tf.keras.models.load_model("soil_model.keras")

# Load test dataset
test_ds = image_dataset_from_directory(
    "soil_dataset",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

class_names = test_ds.class_names

y_true = np.concatenate([y for x, y in test_ds], axis=0)

predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))