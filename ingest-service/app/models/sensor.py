from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class SensorData(BaseModel):
    sensor_id: str
    kiln_temp: float
    motor_load: float
    feeder_rate: float
    emissions: float
    vibration: Optional[float] = None
    pressure: Optional[float] = None
    fuel_rate: Optional[float] = None
    raw_feed: Optional[float] = None
    grinding_power: Optional[float] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    hour_sin: Optional[float] = None
    hour_cos: Optional[float] = None
    minute_sin: Optional[float] = None
    minute_cos: Optional[float] = None
    prev_temp_1: Optional[float] = None
    prev_temp_2: Optional[float] = None
    prev_temp_3: Optional[float] = None
    prev_temp_4: Optional[float] = None
    prev_temp_5: Optional[float] = None
    prev_temp_6: Optional[float] = None
    rolling_3: Optional[float] = None
    rolling_5: Optional[float] = None
    rolling_7: Optional[float] = None
    rolling_10: Optional[float] = None
    motor_feeder: Optional[float] = None
    motor_fuel: Optional[float] = None
    motor_feeder_fuel: Optional[float] = None
    grind_raw_ratio: Optional[float] = None
    motor_emission: Optional[float] = None
    fuel_pressure: Optional[float] = None
    vibration_motor: Optional[float] = None
    timestamp: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
