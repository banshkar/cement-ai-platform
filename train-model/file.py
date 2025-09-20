import pandas as pd
import numpy as np

# Parameters
n_rows = 1440  # e.g., 10-min intervals for 10 days
time_index = pd.date_range(start='2025-09-15', periods=n_rows, freq='10T')

# Base kiln temperature with smooth sinusoidal daily pattern
base_temp = 1150 + 50 * np.sin(2 * np.pi * time_index.hour / 24)

# Add random fluctuations
np.random.seed(42)
kiln_temp = base_temp + np.random.normal(0, 10, size=n_rows)  # smaller noise

# Other features correlated with temperature
motor_load = 80 + (kiln_temp - 1150) * 0.1 + np.random.normal(0, 2, n_rows)
feeder_rate = 10 + (kiln_temp - 1150) * 0.05 + np.random.normal(0, 1, n_rows)
emissions = 0.05 + (kiln_temp - 1150) * 0.0005 + np.random.normal(0, 0.005, n_rows)
vibration = 0.2 + (kiln_temp - 1150) * 0.001 + np.random.normal(0, 0.01, n_rows)
pressure = 1.2 + (kiln_temp - 1150) * 0.002 + np.random.normal(0, 0.05, n_rows)
fuel_rate = 30 + (kiln_temp - 1150) * 0.2 + np.random.normal(0, 3, n_rows)
raw_feed = 200 + (kiln_temp - 1150) * 0.3 + np.random.normal(0, 5, n_rows)
grinding_power = 300 + (kiln_temp - 1150) * 0.4 + np.random.normal(0, 10, n_rows)

# Create DataFrame
df = pd.DataFrame({
    'timestamp': time_index,
    'kiln_temp': kiln_temp,
    'motor_load': motor_load,
    'feeder_rate': feeder_rate,
    'emissions': emissions,
    'vibration': vibration,
    'pressure': pressure,
    'fuel_rate': fuel_rate,
    'raw_feed': raw_feed,
    'grinding_power': grinding_power
})

# Save CSV
df.to_csv('cement_synthetic_good_quality.csv', index=False)
print("CSV file 'cement_synthetic_good_quality.csv' generated successfully!")
