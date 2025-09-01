from google.cloud import aiplatform

PROJECT_ID = "your-gcp-project"
REGION = "us-central1"
ENDPOINT_ID = "your-vertex-endpoint"

aiplatform.init(project=PROJECT_ID, location=REGION)

def predict_sensor(sensor_data: dict):
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    instances = [sensor_data]
    prediction = endpoint.predict(instances=instances)
    return prediction
