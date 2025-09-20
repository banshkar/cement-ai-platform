# payload_generator.py
import random
from datetime import datetime

def generate_payload():
    """
    Generate random but realistic plant data payload.
    """
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "kiln_temp": round(random.uniform(1100, 1200), 3),
        "motor_load": round(random.uniform(10, 20), 3),
        "feeder_rate": round(random.uniform(5, 11), 3),
        "emissions": round(random.uniform(0.02, 0.08), 6),
        "vibration": round(random.uniform(0.1, 0.5), 3),
        "pressure": round(random.uniform(90, 110), 3),
        "fuel_rate": round(random.uniform(300, 360), 3),
        "raw_feed": round(random.uniform(410, 470), 3),
        "grinding_power": round(random.uniform(295, 335), 3),
    }
    return payload
