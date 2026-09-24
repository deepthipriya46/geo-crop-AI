import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

data = pd.read_csv("data/crop_weather_dataset.csv")
print("Columns in Dataset:")
print(data.columns)

print("\nFirst 5 Rows:")
print(data.head())

le = LabelEncoder()
data["crop"] = le.fit_transform(data["crop"])

X = data[["temperature", "humidity", "rainfall", "crop"]]
y = data["suitable"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)   

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("=" * 40)
print("XGBoost Model Performance")
print("=" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump((model, le), f)

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png")
plt.show()

importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(6,5))
plt.bar(features, importance)
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.savefig("results/feature_importance.png")
plt.show()

sns.countplot(x=pred)
plt.title("Prediction Distribution")
plt.savefig("results/prediction_distribution.png")
plt.show()