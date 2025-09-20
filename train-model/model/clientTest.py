import json
import requests
from google.auth import default
from google.auth.transport.requests import Request

# ==========================
# Config
# ==========================
PROJECT_ID = "gen-ai-demo-468815"
ENDPOINT_ID = "4902479356183445504"  # Your Vertex endpoint ID
REGION = "us-central1"
CSV_FILE = "test_rows.csv"  # Your CSV file with rows in the same order as training features

# ==========================
# Load CSV and convert to list of lists
# ==========================
import pandas as pd
df = pd.read_csv(CSV_FILE)
instances = df.values.tolist()  # Each row becomes a list

payload = {
    "instances": instances
}

# ==========================
# Get Google auth token
# ==========================
credentials, project = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
credentials.refresh(Request())
auth_token = credentials.token

# ==========================
# Call Vertex endpoint
# ==========================
url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"

headers = {
    "Authorization": f"Bearer {auth_token}",
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
result = response.json()

# ==========================
# Print predictions
# ==========================

predictions = result["predictions"]

# Print each prediction
for i, pred in enumerate(predictions, start=1):
    print(f"Row {i} → Prediction: {float(pred):.2f}")

