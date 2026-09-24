from data.open_meteo_weather import get_weather


class WeatherAgent:

    def get_weather(self, location):

        print("Fetching weather data...")

        weather = get_weather(location)

        return weather