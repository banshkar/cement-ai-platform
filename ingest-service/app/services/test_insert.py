from app.services import bigquery_writer
from datetime import datetime

data = {
    "sensor_id": "S123",
    "kiln_temp": 1450.5,
    "motor_load": 78.2,
    "feeder_rate": 50.3,
    "emissions": 12.7,
    "vibration": 0.02,
    "pressure": 2.5,
    "timestamp": datetime.utcnow(),
    "created_at": datetime.utcnow()
}

bigquery_writer.save_sensor_data(data)
