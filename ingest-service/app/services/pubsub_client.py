import json
from google.cloud import pubsub_v1
from app.models.sensor import SensorData
from app.services import bigquery_writer, vertex_ai, firebase_client, anomaly

PROJECT_ID = "your-gcp-project"
SUBSCRIPTION_ID = "cement-sensor-sub"

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

def callback(message):
    payload = json.loads(message.data.decode("utf-8"))
    sensor = SensorData(**payload)

    # Save raw sensor data
    bigquery_writer.save_sensor_data(sensor.dict())

    # Vertex AI prediction
    prediction = vertex_ai.predict_sensor(sensor.dict())

    # Local anomaly detection
    alerts = anomaly.detect_anomaly(sensor)

    # Push to Firebase
    firebase_client.push_prediction(sensor.sensor_id, sensor.dict(), prediction.predictions[0], alerts)

    print(f"[PubSub] Data processed for sensor {sensor.sensor_id}")
    message.ack()

def start_subscriber():
    future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"[PubSub] Listening on {subscription_path}...")
    return future
