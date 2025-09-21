import os
import json
import tempfile
from datetime import datetime
from google.cloud import bigquery

# Use ADC in production (Cloud Run service account)
# Do NOT set GOOGLE_APPLICATION_CREDENTIALS if using Cloud Run service account
if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
    print("Using default credentials from environment (Cloud Run service account)")

# BigQuery client
PROJECT_ID = os.environ.get("PROJECT_ID", "gen-ai-demo-468815")
DATASET_ID = os.environ.get("BQ_DATASET", "cement_plant")
TABLE_ID = os.environ.get("BQ_TABLE", "sensor_data")

client = bigquery.Client(project=PROJECT_ID)

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
    Save sensor data (JSON object) to BigQuery safely.
    Converts datetime to ISO format.
    """
    try:
        # Keep only valid schema columns and convert datetime
        row = {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in data.items() if k in COLUMNS
        }

        # Add created_at timestamp if not present
        if "created_at" not in row:
            row["created_at"] = datetime.utcnow().isoformat()

        # Use tempfile for safe production writes
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp_file:
            tmp_file.write(json.dumps(row) + "\n")
            tmp_file.flush()

            table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
            job = client.load_table_from_file(
                tmp_file,
                table_ref,
                job_config=bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
                )
            )
            job.result()  # Wait for job to complete

        print(f"Sensor data inserted into BigQuery: {row.get('timestamp')}")
    except Exception as e:
        print(f"Failed to insert sensor data: {e}")
