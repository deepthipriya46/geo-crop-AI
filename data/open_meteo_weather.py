import requests
from data.geocoding import get_coordinates


def get_weather(location):

    latitude, longitude = get_coordinates(location)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&current=temperature_2m,relative_humidity_2m,rain"
    )

    response = requests.get(url, timeout=10)

    data = response.json()

    current = data["current"]

    return {
        "location": location.title(),
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "rainfall": current["rain"]
    }