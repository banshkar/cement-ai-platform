from google.cloud import aiplatform
import os
from datetime import datetime

PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("VERTEX_REGION")
ENDPOINT_ID = os.environ.get("ENDPOINT_ID")
print(f"Vertex AI Config - Project: {PROJECT_ID}, Region: {REGION}, Endpoint: {ENDPOINT_ID}")
aiplatform.init(project=PROJECT_ID, location=REGION)
# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION)


def predict_sensor(sensor_dict):
    instances = [sensor_dict]  # Vertex AI always expects a list
    try:
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
        vertex_pred = endpoint.predict(instances=instances)
        # Extract the first prediction
        base_prediction = vertex_pred.predictions[0] if vertex_pred.predictions else None
        return base_prediction
    except Exception as e:
        print("Error calling Vertex AI endpoint:", e)
        return None