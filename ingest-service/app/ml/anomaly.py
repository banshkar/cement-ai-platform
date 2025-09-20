def detect_anomaly(sensor):
    alerts = []
    if sensor.kiln_temp > 1200:
        alerts.append(" High kiln temperature")
    if sensor.motor_load > 90:
        alerts.append(" Motor overload")
    if sensor.emissions > 400:
        alerts.append("High emissions")
    return alerts
