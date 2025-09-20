from google.cloud import bigquery
import os
import json
from datetime import datetime

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "app/keys/gen-ai-demo-468815-f202457614ef.json"

# BigQuery client
client = bigquery.Client()
dataset_id = "cement_plant"
table_id = "sensor_data"

# Final schema columns (must match BigQuery table schema!)
COLUMNS = [
    "sensor_id", "kiln_temp", "motor_load", "feeder_rate", "emissions", "vibration",
    "pressure", "fuel_rate", "raw_feed", "grinding_power",
    "hour", "minute", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "prev_temp_1", "prev_temp_2", "prev_temp_3", "prev_temp_4", "prev_temp_5", "prev_temp_6",
    "rolling_3", "rolling_5", "rolling_7", "rolling_10",
    "motor_feeder", "motor_fuel", "motor_feeder_fuel",
    "grind_raw_ratio", "motor_emission", "fuel_pressure", "vibration_motor",
    "timestamp", "created_at"
]

def save_sensor_data(data: dict):
    """
    Save sensor data (JSON object) to BigQuery via batch load.
    Automatically converts datetime to ISO format.
    """
    # Keep only valid schema columns
    row = {
        k: (v.isoformat() if isinstance(v, datetime) else v)
        for k, v in data.items() if k in COLUMNS
    }

    # Save as newline-delimited JSON
    tmp_file = "temp_row.json"
    with open(tmp_file, "w") as f:
        f.write(json.dumps(row) + "\n")  # single row

    # Load into BigQuery
    table_ref = client.dataset(dataset_id).table(table_id)
    job = client.load_table_from_file(
        open(tmp_file, "rb"),
        table_ref,
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        )
    )
    job.result()  # Wait for job to complete
    print("JSON Data inserted into BigQuery")
