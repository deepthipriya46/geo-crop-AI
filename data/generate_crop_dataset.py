import pandas as pd
import random

crops = {
    "rice": {"temp": (20,35), "humidity": (70,90), "rainfall": (150,300)},
    "wheat": {"temp": (15,25), "humidity": (40,60), "rainfall": (40,100)},
    "maize": {"temp": (18,30), "humidity": (50,70), "rainfall": (80,200)},
    "cotton": {"temp": (21,32), "humidity": (40,60), "rainfall": (50,120)},
    "soybean": {"temp": (20,30), "humidity": (60,80), "rainfall": (60,150)},
    "millet": {"temp": (25,35), "humidity": (30,50), "rainfall": (20,60)},
    "sorghum": {"temp": (26,34), "humidity": (40,60), "rainfall": (30,100)},
    "sunflower": {"temp": (20,30), "humidity": (40,60), "rainfall": (50,100)},
    "tomato": {"temp": (18,27), "humidity": (60,80), "rainfall": (40,100)},
    "potato": {"temp": (15,20), "humidity": (60,80), "rainfall": (50,120)}
}

data = []

for crop, values in crops.items():

    for i in range(300):

        temp = random.uniform(values["temp"][0]-10, values["temp"][1]+10)
        humidity = random.uniform(values["humidity"][0]-20, values["humidity"][1]+20)
        rainfall = random.uniform(values["rainfall"][0]-50, values["rainfall"][1]+50)

        suitable = (
            values["temp"][0] <= temp <= values["temp"][1]
            and values["humidity"][0] <= humidity <= values["humidity"][1]
            and values["rainfall"][0] <= rainfall <= values["rainfall"][1]
        )

        data.append([
            temp,
            humidity,
            rainfall,
            crop,
            int(suitable)
        ])

df = pd.DataFrame(
    data,
    columns=["temperature","humidity","rainfall","crop","suitable"]
)

df.to_csv("data/crop_weather_dataset.csv", index=False)

print("Dataset generated successfully with", len(df), "samples")