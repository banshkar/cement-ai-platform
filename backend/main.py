from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Cement Plant AI Platform")

# Example Input model
class SensorData(BaseModel):
    kiln_temp: float
    motor_vibration: float
    feed_rate: float
    ambient_temp: float

@app.get("/")
def root():
    return {"message": "Cement Plant AI API is running 🚀"}

@app.post("/predict")
def predict(data: SensorData):
    # Dummy AI logic (replace with real ML/Generative AI)
    if data.kiln_temp > 1500:
        return {"recommendation": "Reduce kiln temp slightly to save fuel."}
    else:
        return {"recommendation": "Kiln operating within safe range."}

