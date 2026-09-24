from agents.weather_agent import WeatherAgent
from agents.crop_agent import CropAgent
from agents.risk_agent import RiskAgent
from agents.recommendation_agent import RecommendationAgent


class AgriOrchestrator:

    def __init__(self):

        self.weather_agent = WeatherAgent()
        self.crop_agent = CropAgent()
        self.risk_agent = RiskAgent()
        self.recommendation_agent = RecommendationAgent()

    def run(self, location, crop, soil, water):

        # Fetch weather
        weather = self.weather_agent.get_weather(location)

        # Get crop details
        crop_data = self.crop_agent.get_crop_data(crop)

        # Predict suitability
        prediction = self.risk_agent.evaluate_risk(
            weather,
            crop
        )

        # Generate recommendation
        result = self.recommendation_agent.generate(
            crop,
            prediction,
            weather,
            soil,
            water
        )

        return result