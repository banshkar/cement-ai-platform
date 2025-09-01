def generate_strategy(sensor, predictions):
    """
    Use Generative AI (Gemini) to provide quality / operational recommendations.
    This is a stub function; integrate with Gemini API.
    """
    recommendations = []
    if sensor.emissions > 400:
        recommendations.append("Consider alternative fuel blend to reduce emissions")
    if predictions.get("forecast_temp", 0) > 1160:
        recommendations.append("Pre-adjust feed rate to stabilize kiln temperature")
    return recommendations
