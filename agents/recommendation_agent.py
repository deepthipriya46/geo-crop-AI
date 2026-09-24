from data.crop_database import CROP_DATABASE


class RecommendationAgent:

    def generate(self, crop, prediction, weather, soil, water):

        crop = crop.lower()
        crop_info = CROP_DATABASE[crop]

        temp = weather["temperature"]
        humidity = weather["humidity"]
        rainfall = weather["rainfall"]

        reasons = []

        total_checks = 5
        passed_checks = 0

        # Temperature
        if crop_info["ideal_temp_min"] <= temp <= crop_info["ideal_temp_max"]:
            passed_checks += 1
        else:
            reasons.append("• Temperature is outside the optimal range")

        # Humidity
        if humidity >= crop_info["ideal_humidity"]:
            passed_checks += 1
        else:
            reasons.append("• Humidity is lower than required")

        # Rainfall
        if rainfall >= crop_info["ideal_rainfall"]:
            passed_checks += 1
        else:
            if water == "yes":
                passed_checks += 1
                reasons.append("• Rainfall is low, irrigation support required")
            else:
                reasons.append("• Insufficient rainfall and irrigation not available")

        # Soil
        if soil in crop_info["soil"]:
            passed_checks += 1
        else:
            reasons.append("• Soil type is not suitable for this crop")

        # Season
        month = weather.get("month", 6)

        if month in [6, 7, 8, 9]:
            season = "kharif"
        elif month in [10, 11, 12, 1]:
            season = "rabi"
        else:
            season = "zaid"

        if season == crop_info["season"]:
            passed_checks += 1
        else:
            reasons.append("• Current season is not ideal for this crop")

        suitability = round((passed_checks / total_checks) * 100)

        if suitability >= 80:
            status = "SUITABLE"
        elif suitability >= 60:
            status = "MODERATELY SUITABLE"
        else:
            status = "NOT SUITABLE"

        suggestions = []

        for name, info in CROP_DATABASE.items():

            if (
                info["ideal_temp_min"] - 3
                <= temp
                <= info["ideal_temp_max"] + 5
                and humidity >= info["ideal_humidity"] - 15
            ):
                suggestions.append(name)

        suggestions = list(set(suggestions))

        output = f"""
Current Weather Conditions

Temperature: {temp}°C
Humidity: {humidity}%
Rainfall: {rainfall} mm
Soil Type: {soil}
Water Availability: {water}
Detected Season: {season}

==================================
SUITABILITY SCORE : {suitability}%
==================================

Temperature : {"✔" if crop_info["ideal_temp_min"] <= temp <= crop_info["ideal_temp_max"] else "✘"}
Humidity    : {"✔" if humidity >= crop_info["ideal_humidity"] else "✘"}
Rainfall    : {"✔" if rainfall >= crop_info["ideal_rainfall"] or water=="yes" else "✘"}
Soil Type   : {"✔" if soil in crop_info["soil"] else "✘"}
Season      : {"✔" if season == crop_info["season"] else "✘"}

Result: {status}

Reasons:
"""

        if reasons:
            for reason in reasons:
                output += reason + "\n"
        else:
            output += "• Excellent conditions for cultivation.\n"

        output += f"""

Ideal Conditions for {crop}

Temperature : {crop_info['ideal_temp_min']}°C - {crop_info['ideal_temp_max']}°C
Humidity    : {crop_info['ideal_humidity']}%
Rainfall    : {crop_info['ideal_rainfall']} mm
Soil Type   : {", ".join(crop_info["soil"])}
Best Season : {crop_info["season"].upper()}
"""

        if suggestions:
            output += "\n\nSuggested Crops:\n"

            for s in suggestions[:5]:
                output += f"• {s}\n"

        if rainfall < 40:
            output += "\nNote: Rainfall is low. Irrigation may be required.\n"

        return output