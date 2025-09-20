import pandas as pd
import numpy as np
from datetime import datetime

def process_sensor_row(row: dict, prev_rows: pd.DataFrame = None, sensor_id="sensor-001") -> dict:
    """
    Convert raw sensor input into full feature set with lags, rolling, and interaction features.
    
    Args:
        row: dict with keys timestamp, kiln_temp, motor_load, feeder_rate, emissions, vibration, pressure, fuel_rate, raw_feed, grinding_power
        prev_rows: optional pd.DataFrame of previous sensor data to compute lag and rolling features
        sensor_id: sensor identifier

    Returns:
        dict: full sensor dictionary ready for prediction or storage
    """
    data = row.copy()
    
    ts = pd.to_datetime(data['timestamp'])
    data['hour'] = ts.hour
    data['minute'] = ts.minute
    data['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
    data['minute_sin'] = np.sin(2 * np.pi * ts.minute / 60)
    data['minute_cos'] = np.cos(2 * np.pi * ts.minute / 60)
    
    # Previous temperatures
    prev_temps = [0]*6
    if prev_rows is not None and len(prev_rows) >= 6:
        prev_temps = prev_rows['kiln_temp'].iloc[-6:].tolist()
    prev_temps = [0]*(6 - len(prev_temps)) + prev_temps
    for i, val in enumerate(prev_temps, 1):
        data[f'prev_temp_{i}'] = val

    # Rolling averages
    for w in [3,5,7,10]:
        if prev_rows is not None and len(prev_rows) >= w:
            roll = prev_rows['kiln_temp'].iloc[-w:].mean()
        else:
            roll = data['kiln_temp']
        data[f'rolling_{w}'] = roll

    # Interaction features
    data['motor_feeder'] = data['motor_load'] * data['feeder_rate']
    data['motor_fuel'] = data['motor_load'] / (data['fuel_rate'] + 1e-5)
    data['motor_feeder_fuel'] = data['motor_load'] * data['feeder_rate'] / (data['fuel_rate'] + 1e-5)
    data['grind_raw_ratio'] = data['grinding_power'] / (data['raw_feed'] + 1e-5)
    data['motor_emission'] = data['motor_load'] * data['emissions']
    data['fuel_pressure'] = data['fuel_rate'] * data['pressure']
    data['vibration_motor'] = data['vibration'] * data['motor_load']

    sensor_dict = {
        "sensor_id": sensor_id,
        "kiln_temp": float(data['kiln_temp']),
        "motor_load": float(data['motor_load']),
        "feeder_rate": float(data['feeder_rate']),
        "emissions": float(data['emissions']),
        "vibration": float(data['vibration']),
        "pressure": float(data['pressure']),
        "fuel_rate": float(data['fuel_rate']),
        "raw_feed": float(data['raw_feed']),
        "grinding_power": float(data['grinding_power']),
        "hour": int(data['hour']),
        "minute": int(data['minute']),
        "hour_sin": float(data['hour_sin']),
        "hour_cos": float(data['hour_cos']),
        "minute_sin": float(data['minute_sin']),
        "minute_cos": float(data['minute_cos']),
        "prev_temp_1": float(data['prev_temp_1']),
        "prev_temp_2": float(data['prev_temp_2']),
        "prev_temp_3": float(data['prev_temp_3']),
        "prev_temp_4": float(data['prev_temp_4']),
        "prev_temp_5": float(data['prev_temp_5']),
        "prev_temp_6": float(data['prev_temp_6']),
        "rolling_3": float(data['rolling_3']),
        "rolling_5": float(data['rolling_5']),
        "rolling_7": float(data['rolling_7']),
        "rolling_10": float(data['rolling_10']),
        "motor_feeder": float(data['motor_feeder']),
        "motor_fuel": float(data['motor_fuel']),
        "motor_feeder_fuel": float(data['motor_feeder_fuel']),
        "grind_raw_ratio": float(data['grind_raw_ratio']),
        "motor_emission": float(data['motor_emission']),
        "fuel_pressure": float(data['fuel_pressure']),
        "vibration_motor": float(data['vibration_motor']),
        "timestamp": ts.isoformat(),
        "created_at": datetime.utcnow().isoformat()
    }

    return sensor_dict



def prepare_vertex_payload(sensor):
    sequence = [
        "motor_load", "feeder_rate", "emissions", "vibration", "pressure", "fuel_rate",
        "raw_feed", "grinding_power", "hour", "minute", "hour_sin", "hour_cos",
        "minute_sin", "minute_cos", "prev_temp_1", "prev_temp_2", "prev_temp_3",
        "prev_temp_4", "prev_temp_5", "prev_temp_6", "rolling_3", "rolling_5",
        "rolling_7", "rolling_10", "motor_feeder", "motor_fuel", "motor_feeder_fuel",
        "grind_raw_ratio", "motor_emission", "fuel_pressure", "vibration_motor"
    ]

    # Convert Pydantic model to dict if needed
    if hasattr(sensor, "dict"):
        sensor = sensor.dict()

    #  Make sure this is a list, not tuple
    row = [float(sensor.get(col, 0)) for col in sequence]
    payload = {"instances": [row]}  # outer list ensures 2D array

    return payload



def sensor_to_list(sensor):
    """
    Convert a sensor dict or Pydantic model to a list of floats
    in the desired sequence.
    """
    sequence = [
        "motor_load", "feeder_rate", "emissions", "vibration", "pressure", "fuel_rate",
        "raw_feed", "grinding_power", "hour", "minute", "hour_sin", "hour_cos",
        "minute_sin", "minute_cos", "prev_temp_1", "prev_temp_2", "prev_temp_3",
        "prev_temp_4", "prev_temp_5", "prev_temp_6", "rolling_3", "rolling_5",
        "rolling_7", "rolling_10", "motor_feeder", "motor_fuel", "motor_feeder_fuel",
        "grind_raw_ratio", "motor_emission", "fuel_pressure", "vibration_motor"
    ]

    # If it's a Pydantic model, convert to dict
    if hasattr(sensor, "dict"):
        sensor = sensor.dict()

    # Build the list of values in order
    row = [float(sensor.get(col, 0)) for col in sequence]
    return row
