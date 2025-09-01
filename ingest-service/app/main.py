from fastapi import FastAPI
from app.models.sensor import SensorData
from app.services import (
    pubsub_client,
    bigquery_writer,
    vertex_ai,
    firebase_client,
    anomaly,
    generative_ai
)
from app.ml import optimization, cross_process

app = FastAPI(title="Cement AI Predictive & Optimization Service")

@app.on_event("startup")
def startup():
    pubsub_client.start_subscriber()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(sensor: SensorData):
    # Save raw data
    bigquery_writer.save_sensor_data(sensor.dict())

    # Vertex AI predictions
    vertex_pred = vertex_ai.predict_sensor(sensor.dict())
    prediction = vertex_pred.predictions[0]

    # Local anomaly detection
    alerts = anomaly.detect_anomaly(sensor)

    # Optimization modules
    raw_mill_opt = optimization.raw_mill(sensor)
    fuel_opt = optimization.fuel_mix(sensor)
    holistic_opt = cross_process.holistic_optimization(sensor)

    # Generative AI recommendations
    gen_ai_rec = generative_ai.generate_strategy(sensor, prediction)

    co2_opt = co2_optimization(sensor)

    # Push all optimizations to Firebase
    firebase_client.push_prediction(
        sensor.sensor_id,
        sensor.dict(),
        prediction,
        alerts + list(raw_mill_opt.values()) + list(fuel_opt.values()) +
        [holistic_opt["recommendation"]] + gen_ai_rec + list(co2_opt.values())
    )

    return {
        "sensor_data": sensor.dict(),
        "vertex_prediction": prediction,
        "alerts": alerts,
        "raw_mill_optimization": raw_mill_opt,
        "fuel_optimization": fuel_opt,
        "holistic_optimization": holistic_opt,
        "co2_optimization": co2_opt,
        "recommendations": gen_ai_rec
    }