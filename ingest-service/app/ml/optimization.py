def raw_mill(sensor):
    """
    Optimize raw material grinding efficiency.
    """
    result = {}
    # Predict adjustments based on sensor readings
    if sensor.kiln_temp > 1150:
        result["grinding_speed"] = "reduce 5%"
    else:
        result["grinding_speed"] = "increase 3%"
    result["feed_rate_adjustment"] = round(sensor.feeder_rate * 0.98, 1)
    return result


def fuel_mix(sensor):
    """
    Suggest optimal alternative fuel mix.
    """
    result = {}
    if sensor.emissions > 400:
        result["alternative_fuel_increase"] = "5%"
    else:
        result["alternative_fuel_increase"] = "0%"
    result["thermal_substitution_rate"] = round(min(80, 50 + sensor.emissions/10), 1)
    return result






def co2_optimization(sensor):
    """
    Suggest strategies to minimize CO2 emissions.
    """
    result = {}

    # Optimize alternative fuel usage
    if sensor.emissions > 400:
        result["increase_alternative_fuel"] = "5-10%"
    else:
        result["increase_alternative_fuel"] = "0-5%"

    # Kiln adjustment
    if sensor.kiln_temp > 1150:
        result["reduce_kiln_temp"] = "2-3%"
    else:
        result["reduce_kiln_temp"] = "0%"

    # Grinding energy efficiency
    if sensor.motor_load > 85:
        result["optimize_grinding_speed"] = "reduce 2-3%"
    else:
        result["optimize_grinding_speed"] = "maintain"

    # Expected CO2 reduction (simplified)
    base_co2 = sensor.emissions
    predicted_reduction = min(20, base_co2 * 0.05)
    result["predicted_co2_reduction_ppm"] = round(predicted_reduction, 1)

    return result
