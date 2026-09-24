import pickle
import os
import pandas as pd


class RiskAgent:

    def __init__(self):

        model_path = os.path.join("models", "xgboost_model.pkl")

        with open(model_path, "rb") as f:
            self.model, self.encoder = pickle.load(f)

        print("\nSupported Crops:")
        print(self.encoder.classes_)

    def evaluate_risk(self, weather, crop_name):

        if crop_name not in self.encoder.classes_:
            print(f"Crop '{crop_name}' not present in ML model.")
            return 0

        crop_encoded = self.encoder.transform([crop_name])[0]

        features = pd.DataFrame([{
            "temperature": weather["temperature"],
            "humidity": weather["humidity"],
            "rainfall": weather["rainfall"],
            "crop": crop_encoded
        }])

        prediction = self.model.predict(features)[0]

        return prediction