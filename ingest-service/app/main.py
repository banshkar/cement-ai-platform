from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.sensor import SensorData
from app.models.sensorInput import SensorInput
from app.services import (
    pubsub_client,
    bigquery_writer,
    vertex_ai,
    firebase_client,
    generative_ai
)
from app.ml import optimization, anomaly, cross_process
from app.utils.prepare import prepare_vertex_payload, sensor_to_list, process_sensor_row
from datetime import datetime
import threading, time, random
from fastapi.testclient import TestClient
from pydantic import BaseModel  # <-- add this
from typing import List, Optional  # <-- make sure Optional is imported
from app.utils.helpers import predicted_temp_with_af, required_fuel_rate_to_hold_heat
from fastapi.responses import JSONResponse
import uvicorn




# ChatBot request schema
class ChatBotRequest(BaseModel):
    message: str
    sensor: dict  # Pass current sensor readings from frontend

# ChatBot response schema
class ChatBotResponse(BaseModel):
    reply: str
    recommendations: list = []

app = FastAPI(title="Cement AI ChatBot Service")

allow_origins=[
    "https://cement-dashboard-2025.web.app",
    "https://cement-dashboard-2025.web.app/",
    "http://localhost:3000",
    "http://localhost:8080",
]

# Apply CORS middleware globally
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,        # allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],          # allow GET, POST, etc.
    allow_headers=["*"],          # allow all headers
)



@app.get("/health")
def health():
    return {"status": "ok", "message": "Service is healthy"}

if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)



# -------------------------
# Request/Response Models
# -------------------------
class ChatBotRequest(BaseModel):
    message: str
    sensor: Optional[dict] = None  # current dashboard sensor data

class Recommendation(BaseModel):
    recommendation: str
    priority: str

class ChatBotResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Recommendation]] = []

# -------------------------
# ChatBot Endpoint
# -------------------------
@app.post("/chatbot", response_model=ChatBotResponse)
def chatbot_endpoint(req: ChatBotRequest):
    user_message = req.message.lower()
    sensor_data = req.sensor or {}

    recommendations = []

    # Rule-based suggestions
    if "co2" in user_message:
        recommendations.append({"recommendation": "Reduce emissions", "priority": "high"})
    if "efficiency" in user_message:
        recommendations.append({"recommendation": "Adjust feed rate", "priority": "medium"})

    # Generative AI (Gemini) suggestions
    try:
        if sensor_data:
            gen_recs = generative_ai.generate_strategy(sensor_data, prediction=sensor_data.get("kiln_temp", 1150))
            recommendations.extend(gen_recs)
    except Exception as e:
        print("Generative AI error:", e)

    # Prepare bot reply
    if recommendations:
        reply_text = "Here are some recommendations based on your input."
    else:
        reply_text = "I don't have any specific suggestions for that. Try asking about CO₂ or efficiency."

    return ChatBotResponse(
        reply=reply_text,
        recommendations=recommendations
    )





# Helper: serialize datetime
# -------------------------
def serialize_datetime(data: dict):
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in data.items()}




# -------------------------
# Add Data Endpoint
# -------------------------
@app.post("/add_data")
def add_data(sensor: SensorData):
    sensor_dict = serialize_datetime(sensor.dict())
    bigquery_writer.save_sensor_data(sensor_dict)
    return {
        "status": "success",
        "message": "Sensor data saved successfully",
        "sensor_data": sensor_dict
    }

# -------------------------
# Process Single Sensor Row
# -------------------------
@app.post("/process-sensor")
def process_sensor(sensor: SensorInput):
    sensor_dict = process_sensor_row(sensor.dict())
    return {"sensor_data": sensor_dict}

# -------------------------
# Predict + Optimize
# -------------------------
@app.post("/predict")
def predict(sensor: SensorInput):
    sensor_dict = process_sensor_row(sensor.dict())
    sensor_dict = serialize_datetime(sensor_dict)  # <-- serialize here
    vertex_payload = sensor_to_list(sensor_dict)
    vertex_pred = vertex_ai.predict_sensor(vertex_payload)
    prediction = vertex_pred.predictions[0]

    alerts = anomaly.detect_anomaly(sensor)
    raw_mill_opt = optimization.raw_mill(sensor)
    fuel_opt = optimization.fuel_mix(sensor)
    holistic_opt = cross_process.holistic_optimization(sensor)
    gen_ai_rec = generative_ai.generate_strategy(sensor, prediction)

    return {
        "sensor_data": sensor_dict,
        "vertex_prediction": prediction,
        "alerts": alerts,
        "raw_mill_optimization": raw_mill_opt,
        "fuel_optimization": fuel_opt,
        "holistic_optimization": holistic_opt,
        "recommendations": gen_ai_rec
    }

# -------------------------
# Random payload generator
# -------------------------
def generate_payload():
    now = datetime.utcnow().isoformat()
    return {
        "timestamp": now,
        "kiln_temp": round(random.uniform(1100, 1200), 2),
        "motor_load": round(random.uniform(10, 20), 2),
        "feeder_rate": round(random.uniform(5, 10), 2),
        "emissions": round(random.uniform(0.05, 0.1), 4),
        "vibration": round(random.uniform(0.2, 0.3), 4),
        "pressure": round(random.uniform(100, 105), 2),
        "fuel_rate": round(random.uniform(330, 340), 2),
        "raw_feed": round(random.uniform(440, 450), 2),
        "grinding_power": round(random.uniform(310, 320), 2)
    }

# -------------------------
# Live Prediction Endpoint for React
# -------------------------
@app.get("/live-predict")
def live_predict():
    payload = generate_payload()
    sensor_input = SensorInput(**payload)

    # Process and predict
    sensor_dict = process_sensor_row(sensor_input.dict())
    vertex_payload = sensor_to_list(sensor_dict)
    vertex_pred = vertex_ai.predict_sensor(vertex_payload)

    # Access predictions correctly
    prediction = vertex_pred
    alerts = anomaly.detect_anomaly(sensor_input)
    raw_mill_opt = optimization.raw_mill(sensor_input)
    fuel_opt = optimization.fuel_mix(sensor_input)
    holistic_opt = cross_process.holistic_optimization(sensor_input)
    gen_ai_rec = generative_ai.generate_strategy(sensor_input, prediction)

    # Build response
    response = {
        "sensor_data": sensor_dict,
        "vertex_prediction": prediction,
        "alerts": alerts,
        "raw_mill_optimization": raw_mill_opt,
        "fuel_optimization": fuel_opt,
        "holistic_optimization": holistic_opt,
        "recommendations": gen_ai_rec
    }

    # Convert any datetime objects to ISO format
    return serialize_datetime(response)


# -------------------------
def serialize_datetime_vertex(sensor_dict):
    for k, v in sensor_dict.items():
        if isinstance(v, datetime):
            sensor_dict[k] = v.isoformat()  # convert to ISO string
    return sensor_dict


@app.post("/apply-af-correction")
def predict_apply_af(sensor: SensorInput):
    try:
        # --- Step A: ML model baseline prediction ---
        sensor_dict = process_sensor_row(sensor.dict())
        sensor_dict = serialize_datetime_vertex(sensor_dict)  # ensures no datetime leaks
        vertex_payload = sensor_to_list(sensor_dict)
        vertex_pred = vertex_ai.predict_sensor(vertex_payload)

        print("Vertex Prediction Raw:", vertex_pred)

        # --- If no prediction, short-circuit ---
      

        base_prediction = vertex_pred
        print("correction Prediction:", base_prediction)    
        # --- Other calculations ---
        alerts = anomaly.detect_anomaly(sensor)
        raw_mill_opt = optimization.raw_mill(sensor)
        fuel_opt = optimization.fuel_mix(sensor)
        holistic_opt = cross_process.holistic_optimization(sensor)
        gen_ai_rec = generative_ai.generate_strategy(sensor, base_prediction)

        # --- Step B: Apply AF correction ---
        adj_prediction = predicted_temp_with_af(
            base_temp=base_prediction,
            coal_cv=sensor.coal_calorific,
            af_cv=sensor.af_calorific,
            af_pct=sensor.af_share_pct
        )

        new_fuel_rate = required_fuel_rate_to_hold_heat(
            current_fuel_rate=sensor.fuel_rate,
            coal_cv=sensor.coal_calorific,
            af_cv=sensor.af_calorific,
            af_pct=sensor.af_share_pct
        )

        # --- Safe response building ---
        return {
            "base_prediction": round(float(base_prediction), 2),
            "adjusted_prediction": round(float(adj_prediction), 2) if adj_prediction else None,
            "required_fuel_rate": round(float(new_fuel_rate), 2) if new_fuel_rate else None,
            "fuel_rate_delta": round(float(new_fuel_rate - sensor.fuel_rate), 2) if new_fuel_rate else None,
            "af_share_pct": sensor.af_share_pct,
            "af_calorific": sensor.af_calorific
        }

    except Exception as e:
        print("Error in apply-af-correction:", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )



# Background loop (optional)
# -------------------------
def background_data_loop():
    client = TestClient(app)
    while True:
        payload = generate_payload()
        try:
            sensor_input = SensorInput(**payload)
            response = client.post("/predict", json=sensor_input.dict())
            print("Prediction Response:", response.json())
        except Exception as e:
            print("Error:", str(e))
        time.sleep(3)

@app.on_event("startup")
def start_background_task():
    threading.Thread(target=background_data_loop, daemon=True).start()
