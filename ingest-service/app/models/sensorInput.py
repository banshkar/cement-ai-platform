# app/models/sensorInput.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SensorInput(BaseModel):
    timestamp: datetime
    kiln_temp: float
    motor_load: float
    feeder_rate: float
    emissions: float
    vibration: Optional[float] = None
    pressure: Optional[float] = None
    fuel_rate: Optional[float] = None
    raw_feed: Optional[float] = None
    grinding_power: Optional[float] = None

    # Fuel properties
    af_share_pct: Optional[float] = 0.0   # % alternate fuel
    af_calorific: Optional[float] = 4000.0
    coal_calorific: Optional[float] = 6000.0
