def holistic_optimization(sensor):
    """
    Combine raw mill, clinker, and utilities for overall optimization
    """
    result = {}
    # Simplified example: calculate energy efficiency score
    score = 100 - (sensor.kiln_temp - 1100)/5 - (sensor.motor_load - 80)/2
    result["energy_efficiency_score"] = max(min(score, 100), 0)
    # Recommend adjustments
    result["recommendation"] = "Reduce kiln temp 2% and optimize feed rate"
    return result
