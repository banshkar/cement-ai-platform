from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SensorData(BaseModel):
    sensor_id: str
    kiln_temp: float
    motor_load: float
    feeder_rate: float
    emissions: float
    vibration: Optional[float]
    pressure: Optional[float]
    timestamp: datetime
    created_at: datetime = datetime.utcnow()