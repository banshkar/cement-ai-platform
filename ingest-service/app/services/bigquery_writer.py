from google.cloud import bigquery
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/service_account.json"

client = bigquery.Client()
dataset_id = "cement_plant"
table_id = "sensor_data"

def save_sensor_data(data: dict):
    table_ref = client.dataset(dataset_id).table(table_id)
    errors = client.insert_rows_json(table_ref, [data])
    if errors:
        print(f"BigQuery insert errors: {errors}")
    else:
        print("Data inserted into BigQuery")
