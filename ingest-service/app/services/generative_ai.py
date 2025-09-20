from google.genai import Client
import re
import json

client = Client(api_key="AIzaSyBt_UZ7K3RMoJ-MSQestGuHaJ6agOkUG-E")


def generate_strategy(sensor, prediction):
    """
    Generate operational recommendations using Gemini + rule-based logic.
    """
    recommendations = []
    print("Sensor in Gen AI:", sensor)

    sensor_dict = sensor.dict()

    # ----------------------------
    # Rule-based recommendations
    # ----------------------------
    if sensor_dict.get("emissions", 0) > 400:
        recommendations.append({"recommendation": "Reduce emissions", "priority": "critical"})

    forecast_temp = prediction
    if forecast_temp > 1160:
        recommendations.append({"recommendation": "Adjust feed rate", "priority": "critical"})

    # ----------------------------
    # Gemini-based recommendations
    # ----------------------------
    try:
        prompt = f"""
        Given these kiln sensor readings: {sensor_dict}
        and a forecast temperature of {forecast_temp},
        suggest 2–3 operational recommendations with a priority level.
        Return ONLY JSON in format:
        [
          {{"recommendation": "...", "priority": "..."}},
          ...
        ]
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        raw_text = response.text.strip()

        # Remove ```json ... ``` wrappers if present
        clean_text = re.sub(r"^```json|```$", "", raw_text, flags=re.MULTILINE).strip()

        try:
            gemini_recs = json.loads(clean_text)
            if isinstance(gemini_recs, list):
                recommendations.extend(gemini_recs)
            else:
                recommendations.append({
                    "recommendation": clean_text,
                    "priority": "optional"
                })
        except json.JSONDecodeError:
            recommendations.append({
                "recommendation": raw_text,
                "priority": "optional"
            })

    except Exception as e:
        print("Gemini call failed:", e)

    return recommendations
