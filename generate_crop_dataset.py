import pandas as pd
import random

random.seed(42)

# Crop specifications
crop_data = {
    "rice": (20, 35, 75, 200),
    "wheat": (15, 25, 50, 60),
    "maize": (18, 30, 60, 120),
    "millet": (25, 35, 40, 50),
    "sorghum": (26, 34, 50, 70),
    "cotton": (21, 32, 50, 80),
    "soybean": (20, 30, 65, 90),
    "groundnut": (22, 30, 55, 80),
    "sunflower": (20, 30, 50, 70),
    "mustard": (10, 25, 45, 40),
    "potato": (15, 20, 70, 80),
    "tomato": (18, 27, 70, 60),
    "onion": (13, 24, 60, 50),
    "banana": (26, 35, 80, 200),
    "mango": (24, 30, 60, 100),
    "sugarcane": (20, 38, 80, 200)
}

rows = []

for crop, values in crop_data.items():

    temp_min, temp_max, ideal_humidity, ideal_rainfall = values

    # 250 Suitable samples
    for _ in range(250):

        rows.append({
            "temperature": round(random.uniform(temp_min, temp_max), 2),
            "humidity": round(random.uniform(ideal_humidity - 5, ideal_humidity + 5), 2),
            "rainfall": round(random.uniform(ideal_rainfall - 20, ideal_rainfall + 20), 2),
            "crop": crop,
            "suitable": 1
        })

    # 250 Unsuitable samples
    for _ in range(250):

        condition = random.randint(1, 3)

        if condition == 1:
            temp = random.uniform(temp_max + 5, temp_max + 15)
            humidity = random.uniform(ideal_humidity - 30, ideal_humidity - 10)
            rainfall = random.uniform(0, ideal_rainfall - 40)

        elif condition == 2:
            temp = random.uniform(temp_min - 15, temp_min - 5)
            humidity = random.uniform(ideal_humidity + 10, ideal_humidity + 30)
            rainfall = random.uniform(ideal_rainfall + 40, ideal_rainfall + 120)

        else:
            temp = random.uniform(5, 45)
            humidity = random.uniform(20, 100)
            rainfall = random.uniform(0, 350)

        rows.append({
            "temperature": round(temp, 2),
            "humidity": round(humidity, 2),
            "rainfall": round(rainfall, 2),
            "crop": crop,
            "suitable": 0
        })

df = pd.DataFrame(rows)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/crop_weather_dataset.csv", index=False)

print("=" * 50)
print("Dataset Generated Successfully")
print("=" * 50)
print("Rows :", len(df))
print("Crops:", df["crop"].nunique())
print(df["crop"].value_counts())