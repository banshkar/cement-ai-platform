from google.cloud import aiplatform

PROJECT_ID = "gen-ai-demo-468815"       # your real GCP project ID
REGION = "us-central1"                  # your Vertex AI region
ENDPOINT_ID = "4902479356183445504" # the deployed Vertex AI endpoint ID

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

def predict_sensor(sensor_data: dict):
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    instances = [sensor_data]
    prediction = endpoint.predict(instances=instances)
    return prediction
